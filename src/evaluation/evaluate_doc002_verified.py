import csv
import json
import re
import sys
import time
from pathlib import Path

import requests
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.hybrid_retriever import build_hybrid_index, retrieve_hybrid

CHUNKS_FILE = ROOT / "data/processed/DOC002_chunks.jsonl"
QUESTIONS_FILE = ROOT / "data/evaluation/doc002_qa_eval.jsonl"
OUTPUT_FILE = ROOT / "data/evaluation/doc002_hybrid_verified_results.csv"
SUMMARY_FILE = ROOT / "data/evaluation/doc002_hybrid_verified_summary.json"
COMPARISON_FILE = ROOT / "data/evaluation/doc002_retrieval_comparison.csv"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "aya-expanse:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"
TOP_K = 4
TEMPERATURE = 0.0
MAX_RETRIES = 2

ABSTENTION = (
    "INSUFFICIENT_EVIDENCE: The supplied document passages do not contain "
    "enough information to answer this question."
)


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ollama_chat(messages, num_predict=650):
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "messages": messages,
        "options": {"temperature": TEMPERATURE, "num_predict": num_predict},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=(15, 900))
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def parse_json_response(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Model did not return valid JSON: {raw[:500]}")


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def quote_exists(quote, page_text):
    q = normalize_text(quote)
    p = normalize_text(page_text)
    return bool(q) and q in p


def page_map(retrieved):
    result = {}
    for chunk in retrieved:
        page = int(chunk["page_number"])
        result.setdefault(page, []).append(chunk.get("text", ""))
    return {page: "\n".join(texts) for page, texts in result.items()}


def generate_structured_answer(question, retrieved):
    context_parts = []
    for chunk in retrieved:
        context_parts.append(
            f"[Page {int(chunk['page_number'])}]\n{chunk.get('text', '')}"
        )
    context = "\n\n".join(context_parts)

    system = """
You are a conservative document-grounded QA system.
Use ONLY the supplied passages. Do not use outside knowledge.
Return JSON only, matching this schema:
{
  "answerable": true,
  "claims": [
    {
      "text": "one short factual claim",
      "page": 12,
      "evidence_quote": "a short exact verbatim quote copied from that page"
    }
  ]
}
If the passages do not directly support an answer, return:
{"answerable": false, "claims": []}

Rules:
- Every factual claim must have one exact supporting quote from the cited page.
- Copy evidence_quote verbatim; do not paraphrase it.
- The quote must directly support the claim, not merely mention the same topic.
- Do not infer a company, author, date, statistic, or cause unless stated.
- Do not add advice, background knowledge, or details absent from the passages.
- Use only page numbers shown in the passages.
- Prefer fewer, precise claims over a long answer.
"""
    raw = ollama_chat(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Question: {question}\n\nDocument passages:\n{context}",
            },
        ],
        num_predict=700,
    )
    return parse_json_response(raw)


def verify_claims(question, claims, pages):
    """LLM entailment check plus deterministic exact-quote/page validation."""
    if not claims:
        return False, [], "No evidence-backed claims were returned."

    verified = []
    invalid_reasons = []

    for claim in claims:
        try:
            page = int(claim.get("page"))
            text = str(claim.get("text", "")).strip()
            quote = str(claim.get("evidence_quote", "")).strip()
        except (TypeError, ValueError):
            invalid_reasons.append("Malformed claim/page/quote.")
            continue

        if not text or page not in pages:
            invalid_reasons.append("Claim has no text or cites a non-retrieved page.")
            continue
        if not quote_exists(quote, pages[page]):
            invalid_reasons.append(f"Evidence quote does not exactly occur on page {page}.")
            continue

        verifier_system = """
You are a strict evidence entailment auditor.
Return JSON only: {"supported": true} or {"supported": false}.
Decide whether the quoted passage directly entails the claim in the context of
the user's question. Mere topic similarity is NOT support. Do not use outside
knowledge. If the quote is ambiguous, incomplete, or only indirectly related,
return false.
"""
        verifier_user = (
            f"Question: {question}\n"
            f"Claim: {text}\n"
            f"Source page: {page}\n"
            f"Exact source quote: {quote}\n"
            "Does this quote directly support this claim?"
        )
        try:
            verdict = parse_json_response(
                ollama_chat(
                    [
                        {"role": "system", "content": verifier_system},
                        {"role": "user", "content": verifier_user},
                    ],
                    num_predict=80,
                )
            )
        except Exception as exc:
            invalid_reasons.append(f"Verifier error: {exc}")
            continue

        if verdict.get("supported") is True:
            verified.append({"text": text, "page": page, "quote": quote})
        else:
            invalid_reasons.append(f"Claim failed entailment check on page {page}.")

    all_supported = len(verified) == len(claims) and len(claims) > 0
    return all_supported, verified, "; ".join(invalid_reasons)


def format_answer(verified_claims):
    return " ".join(
        f"{claim['text']} [Page {claim['page']}]"
        for claim in verified_claims
    )


def process_question(item, chunks, model, index):
    question = item["question"]
    expected_pages = set(int(p) for p in item.get("expected_pages", []))

    retrieved = retrieve_hybrid(
        question=question,
        chunks=chunks,
        embedding_model=model,
        dense_vectors=index[0],
        vectorizer=index[1],
        sparse_matrix=index[2],
        top_k=TOP_K,
    )

    retrieved_pages = sorted({int(c["page_number"]) for c in retrieved})
    pages = page_map(retrieved)
    answer = ABSTENTION
    error = ""
    verified_claims = []
    verification_reason = ""

    if retrieved:
        try:
            structured = generate_structured_answer(question, retrieved)
            if structured.get("answerable") is True:
                claims = structured.get("claims", [])
                supported, verified_claims, verification_reason = verify_claims(
                    question, claims, pages
                )
                if supported:
                    answer = format_answer(verified_claims)
                else:
                    answer = ABSTENTION
            else:
                answer = ABSTENTION
        except Exception as exc:
            error = f"Generation/verification error: {exc}"
            answer = ABSTENTION

    cited_pages = sorted({c["page"] for c in verified_claims})
    abstained = answer.startswith("INSUFFICIENT_EVIDENCE")
    citation_valid = bool(verified_claims) and not abstained and not error

    recall = ""
    if expected_pages:
        recall = round(
            len(expected_pages.intersection(retrieved_pages)) / len(expected_pages), 3
        )

    return {
        "id": item.get("id", ""),
        "question": question,
        "is_answerable": bool(item["is_answerable"]),
        "expected_pages": ",".join(map(str, sorted(expected_pages))),
        "retrieved_pages": ",".join(map(str, retrieved_pages)),
        "page_recall": recall,
        "answer": answer,
        "predicted_abstention": abstained,
        "citation_valid": citation_valid,
        "cited_pages": ",".join(map(str, cited_pages)),
        "verified_claim_count": len(verified_claims),
        "verification_reason": verification_reason,
        "error": error,
    }


def safe_read_csv(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def metric_summary(rows):
    valid = [r for r in rows if not r.get("error")]
    answerable = [r for r in valid if r["is_answerable"]]
    unanswerable = [r for r in valid if not r["is_answerable"]]
    recall_values = [
        float(r["page_recall"]) for r in answerable if r["page_recall"] != ""
    ]
    non_abstained = [r for r in valid if not r["predicted_abstention"]]
    return {
        "total_questions": len(rows),
        "successful_responses": len(valid),
        "failed_responses": len(rows) - len(valid),
        "average_expected_page_recall": (
            round(sum(recall_values) / len(recall_values), 3)
            if recall_values else None
        ),
        "answerability_accuracy": (
            round(
                (
                    sum(not r["predicted_abstention"] for r in answerable)
                    + sum(r["predicted_abstention"] for r in unanswerable)
                ) / len(valid),
                3,
            ) if valid else None
        ),
        "answerable_non_abstentions": sum(
            not r["predicted_abstention"] for r in answerable
        ),
        "answerable_total": len(answerable),
        "correct_unanswerable_abstentions": sum(
            r["predicted_abstention"] for r in unanswerable
        ),
        "unanswerable_total": len(unanswerable),
        "citation_consistency": (
            f"{sum(r['citation_valid'] for r in non_abstained)}/{len(non_abstained)}"
        ),
        "verified_non_abstained_answers": len(non_abstained),
    }


def main():
    print("=" * 68)
    print("DOCUTRUST AI - VERIFIED HYBRID EVALUATION")
    print("=" * 68)
    print("Checking Ollama connection and model...")
    requests.get("http://localhost:11434/api/tags", timeout=10).raise_for_status()

    chunks = load_jsonl(CHUNKS_FILE)
    questions = load_jsonl(QUESTIONS_FILE)
    if not chunks or not questions:
        raise ValueError("Chunks or evaluation questions are empty.")

    print(f"Chunks: {len(chunks)} | Questions: {len(questions)}")
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("Building hybrid retrieval index...")
    index = build_hybrid_index(chunks, model)

    results = []
    started = time.time()
    for i, item in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {item['question']}")
        row = process_question(item, chunks, model, index)
        results.append(row)
        print("Retrieved pages:", row["retrieved_pages"])
        print("Answer:", row["answer"])
        print("Citation verified:", row["citation_valid"])
        if row["error"]:
            print("ERROR:", row["error"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0].keys())
    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    summary = metric_summary(results)
    summary["elapsed_seconds"] = round(time.time() - started, 1)
    summary["model"] = OLLAMA_MODEL
    summary["embedding_model"] = EMBED_MODEL
    summary["top_k"] = TOP_K
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Compare retrieval/answerability metrics with existing result files.
    comparison = []
    candidates = [
        ("Dense baseline", ROOT / "data/evaluation/doc002_results.csv"),
        ("Previous hybrid", ROOT / "data/evaluation/doc002_hybrid_results.csv"),
        ("Verified hybrid", OUTPUT_FILE),
    ]
    for label, path in candidates:
        rows = results if path == OUTPUT_FILE else safe_read_csv(path)
        if not rows:
            comparison.append({"system": label, "status": "file unavailable"})
            continue
        # Normalize CSV booleans from previous experiment outputs.
        normalized = []
        for r in rows:
            is_answerable = str(r.get("is_answerable", "")).strip().lower() in (
                "true", "1", "yes"
            )
            abstained = str(r.get("predicted_abstention", "")).strip().lower() in (
                "true", "1", "yes"
            )
            citation = str(r.get("citation_valid", "")).strip().lower() in (
                "true", "1", "yes"
            )
            normalized.append({
                "is_answerable": is_answerable,
                "predicted_abstention": abstained,
                "citation_valid": citation,
                "page_recall": r.get("page_recall", ""),
                "error": r.get("error", ""),
            })
        m = metric_summary(normalized)
        comparison.append({"system": label, **m})

    with open(COMPARISON_FILE, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = sorted({key for row in comparison for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison)

    print("\n" + "=" * 68)
    print("VERIFIED HYBRID SUMMARY")
    print("=" * 68)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nComparison saved:", COMPARISON_FILE)
    print("Verified results saved:", OUTPUT_FILE)
    print("Summary saved:", SUMMARY_FILE)
    print("\nNote: This is a conservative automated evidence check, not a guarantee")
    print("of perfect semantic correctness. Manually review the remaining failures.")


if __name__ == "__main__":
    main()
