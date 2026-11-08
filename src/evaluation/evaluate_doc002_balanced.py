import csv
import json
import re
import sys
from pathlib import Path

import requests
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.hybrid_retriever import build_hybrid_index, retrieve_hybrid

CHUNKS = ROOT / "data/processed/DOC002_chunks.jsonl"
QUESTIONS = ROOT / "data/evaluation/doc002_qa_eval.jsonl"
OUT = ROOT / "data/evaluation/doc002_hybrid_balanced_results.csv"
SUMMARY = ROOT / "data/evaluation/doc002_hybrid_balanced_summary.json"
COMPARISON = ROOT / "data/evaluation/doc002_retrieval_comparison.csv"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "aya-expanse:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"
TOP_K = 4
ABSTENTION = "INSUFFICIENT_EVIDENCE: The supplied passages do not contain enough information to answer this question."


def read_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ask_ollama(messages, num_predict=400):
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": messages,
        "options": {"temperature": 0.0, "num_predict": num_predict},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=(15, 900))
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def is_abstention(answer):
    return answer.strip().lower().startswith("insufficient_evidence") or \
           answer.strip().lower().startswith("insufficient evidence")


def citations(answer):
    return sorted({int(x) for x in re.findall(
        r"\[\s*Page\s+(\d+)\s*\]", answer, flags=re.IGNORECASE
    )})


def load_context(retrieved):
    return "\n\n".join(
        f"[Page {int(c['page_number'])}]\n{c.get('text', '')}"
        for c in retrieved
    )


def generate_answer(question, context):
    system = """
You are a careful document-grounded question-answering assistant.
Answer ONLY from the supplied passages. Never use outside knowledge.
If the passages do not contain enough information, reply exactly:
INSUFFICIENT_EVIDENCE: The supplied passages do not contain enough information to answer this question.

If answerable:
- Give a concise, direct answer to the question.
- Cite every factual sentence with the exact format [Page N].
- N must be a page number shown in the supplied passages.
- Use the page that directly supports the sentence.
- Do not add unsupported explanations, recommendations, or background.
- Do not output quotation marks around passages unless needed.
"""
    return ask_ollama([
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n\nSupplied passages:\n{context}"},
    ])


def repair_citations(question, context, draft, allowed_pages):
    system = """
You are repairing a document-grounded answer.
Use ONLY the supplied passages. Preserve only claims directly supported by them.
Return a concise answer. Every factual sentence MUST end with [Page N],
where N is a page explicitly shown in the passages.
If the passages do not support an answer, return exactly:
INSUFFICIENT_EVIDENCE: The supplied passages do not contain enough information to answer this question.
Do not invent page numbers or facts.
"""
    return ask_ollama([
        {"role": "system", "content": system},
        {"role": "user", "content":
            f"Question: {question}\nAllowed page numbers: {allowed_pages}\n"
            f"Passages:\n{context}\n\nDraft needing citation repair:\n{draft}"},
    ])


def summarize(rows):
    ok = [r for r in rows if not r["error"]]
    ans = [r for r in ok if r["is_answerable"]]
    unans = [r for r in ok if not r["is_answerable"]]
    recalls = [float(r["page_recall"]) for r in ans if r["page_recall"] != ""]
    correct = sum(not r["predicted_abstention"] for r in ans) + \
              sum(r["predicted_abstention"] for r in unans)
    non_abs = [r for r in ok if not r["predicted_abstention"]]
    return {
        "total_questions": len(rows),
        "successful_responses": len(ok),
        "failed_responses": len(rows) - len(ok),
        "average_expected_page_recall": round(sum(recalls) / len(recalls), 3) if recalls else None,
        "answerability_accuracy": round(correct / len(ok), 3) if ok else None,
        "answerable_non_abstentions": f"{sum(not r['predicted_abstention'] for r in ans)}/{len(ans)}",
        "correct_unanswerable_abstentions": f"{sum(r['predicted_abstention'] for r in unans)}/{len(unans)}",
        "citation_consistency": f"{sum(r['citation_valid'] for r in non_abs)}/{len(non_abs)}",
    }


def read_csv(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    print("=" * 64)
    print("DOCUTRUST AI - BALANCED HYBRID EVALUATION")
    print("=" * 64)
    requests.get("http://localhost:11434/api/tags", timeout=10).raise_for_status()
    chunks = read_jsonl(CHUNKS)
    questions = read_jsonl(QUESTIONS)
    print(f"Chunks: {len(chunks)} | Questions: {len(questions)}")
    model = SentenceTransformer(EMBED_MODEL)
    index = build_hybrid_index(chunks, model)

    results = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        expected = set(int(p) for p in item.get("expected_pages", []))
        error = ""
        retrieved = []
        answer = ABSTENTION
        try:
            retrieved = retrieve_hybrid(
                question=q, chunks=chunks, embedding_model=model,
                dense_vectors=index[0], vectorizer=index[1],
                sparse_matrix=index[2], top_k=TOP_K,
            )
            pages = sorted({int(c["page_number"]) for c in retrieved})
            if retrieved:
                context = load_context(retrieved)
                answer = generate_answer(q, context)
                used = citations(answer)
                invalid = [p for p in used if p not in pages]
                # One repair attempt for missing/invalid citations only.
                if not is_abstention(answer) and (not used or invalid):
                    answer = repair_citations(q, context, answer, pages)
                    used = citations(answer)
                    invalid = [p for p in used if p not in pages]
                # Do not turn a citation-format failure into a false verified answer.
                if not is_abstention(answer) and (not used or invalid):
                    answer = ABSTENTION
            else:
                pages = []
        except Exception as exc:
            pages = sorted({int(c["page_number"]) for c in retrieved})
            error = str(exc)
            answer = ABSTENTION

        abstained = is_abstention(answer)
        used = citations(answer)
        citation_valid = bool(used) and not abstained and not error and all(p in pages for p in used)
        recall = ""
        if expected:
            recall = round(len(expected.intersection(pages)) / len(expected), 3)

        row = {
            "id": item.get("id", ""),
            "question": q,
            "is_answerable": bool(item["is_answerable"]),
            "expected_pages": ",".join(map(str, sorted(expected))),
            "retrieved_pages": ",".join(map(str, pages)),
            "page_recall": recall,
            "answer": answer,
            "predicted_abstention": abstained,
            "citation_valid": citation_valid,
            "cited_pages": ",".join(map(str, used)),
            "error": error,
        }
        results.append(row)
        print(f"\n[{i}/{len(questions)}] {q}")
        print("Retrieved:", pages)
        print("Answer:", answer)
        print("Citation valid:", citation_valid)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    metrics = summarize(results)
    metrics.update({"model": OLLAMA_MODEL, "embedding_model": EMBED_MODEL, "top_k": TOP_K})
    SUMMARY.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    comparison = []
    for label, path in [
        ("Dense baseline", ROOT / "data/evaluation/doc002_results.csv"),
        ("Previous hybrid", ROOT / "data/evaluation/doc002_hybrid_results.csv"),
        ("Verified hybrid", ROOT / "data/evaluation/doc002_hybrid_verified_results.csv"),
        ("Balanced hybrid", OUT),
    ]:
        if path == OUT:
            comparison.append({"system": label, **metrics})
            continue
        old = read_csv(path)
        if not old:
            comparison.append({"system": label, "status": "file unavailable"})
            continue
        norm = []
        for r in old:
            norm.append({
                "error": r.get("error", ""),
                "is_answerable": str(r.get("is_answerable", "")).lower() in ("true", "1"),
                "predicted_abstention": str(r.get("predicted_abstention", "")).lower() in ("true", "1"),
                "citation_valid": str(r.get("citation_valid", "")).lower() in ("true", "1"),
                "page_recall": r.get("page_recall", ""),
            })
        comparison.append({"system": label, **summarize(norm)})

    with open(COMPARISON, "w", encoding="utf-8-sig", newline="") as f:
        keys = sorted({k for row in comparison for k in row})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(comparison)

    print("\n" + "=" * 64)
    print("BALANCED HYBRID SUMMARY")
    print("=" * 64)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("\nSaved:", OUT)
    print("Saved:", SUMMARY)
    print("Saved:", COMPARISON)
    print("\nNote: Citation validation checks page references, not full semantic entailment.")


if __name__ == "__main__":
    main()
