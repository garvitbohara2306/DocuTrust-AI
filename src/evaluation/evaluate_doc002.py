import csv
import json
import re
from pathlib import Path

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = ROOT / "data/processed/DOC002_chunks.jsonl"
QUESTIONS_FILE = ROOT / "data/evaluation/doc002_qa_eval.jsonl"
RESULTS_FILE = ROOT / "data/evaluation/doc002_results.csv"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "aya-expanse:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"

TOP_K = 4


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_abstention(answer):
    text = (answer or "").strip().lower()
    return (
        text.startswith("insufficient_evidence")
        or text.startswith("insufficient evidence")
    )


def main():
    print("Loading evaluation data...")

    chunks = load_jsonl(CHUNKS_FILE)
    questions = load_jsonl(QUESTIONS_FILE)

    print(f"Chunks: {len(chunks)}")
    print(f"Questions: {len(questions)}")

    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    results = []

    for number, item in enumerate(questions, start=1):
        question = item["question"]
        expected_pages = set(item.get("expected_pages", []))

        print(f"\n[{number}/{len(questions)}] {question}")

        query_vector = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = index.search(query_vector, TOP_K)

        retrieved = [chunks[i] for i in indices[0] if i >= 0]
        retrieved_pages = sorted({
            int(c["page_number"]) for c in retrieved
        })

        context = "\n\n".join(
            f"[Page {c['page_number']} | {c.get('chunk_id', '')}]\n{c['text']}"
            for c in retrieved
        )

        system_prompt = """
You are a strict document-grounded question answering system.

Use ONLY the supplied document passages.
Do not use outside knowledge or make assumptions.

If the passages do not contain enough information to answer,
respond exactly with:
INSUFFICIENT_EVIDENCE: The supplied passages do not contain the answer.

If the answer is available:
- Give a concise answer.
- Cite every factual claim using [Page N].
- Only cite page numbers present in the supplied passages.
- Never invent information or citations.
"""

        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Document passages:\n{context}"
                    ),
                },
            ],
            "options": {
                "temperature": 0.0,
                "num_predict": 450,
            },
        }

        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=(15, 600),
            )
            response.raise_for_status()
            answer = response.json()["message"]["content"].strip()
            error = ""
        except Exception as exc:
            answer = ""
            error = str(exc)
            print(f"ERROR: {error}")

        abstained = is_abstention(answer)

        cited_pages = sorted({
            int(p) for p in re.findall(
                r"\[\s*Page\s+(\d+)",
                answer,
                re.IGNORECASE,
            )
        })

        invalid_citations = [
            p for p in cited_pages if p not in retrieved_pages
        ]

        citation_valid = (
            bool(cited_pages)
            and not invalid_citations
            and not abstained
        )

        recall = ""
        if expected_pages:
            recall = round(
                len(expected_pages.intersection(retrieved_pages))
                / len(expected_pages),
                3,
            )

        results.append({
            "id": item["id"],
            "question": question,
            "is_answerable": item["is_answerable"],
            "expected_pages": ",".join(map(str, sorted(expected_pages))),
            "retrieved_pages": ",".join(map(str, retrieved_pages)),
            "page_recall": recall,
            "answer": answer,
            "predicted_abstention": abstained,
            "citation_valid": citation_valid,
            "cited_pages": ",".join(map(str, cited_pages)),
            "error": error,
        })

        print("Abstained:", abstained)
        print("Retrieved pages:", retrieved_pages)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    answerable = [
        r for r in results if r["is_answerable"]
    ]
    unanswerable = [
        r for r in results if not r["is_answerable"]
    ]

    successful = [r for r in results if not r["error"]]

    recall_values = [
        r["page_recall"] for r in answerable
        if r["page_recall"] != ""
    ]

    print("\n========== EVALUATION SUMMARY ==========")
    print("Total questions:", len(results))
    print("Successful model responses:", len(successful))
    print("Failed model responses:", len(results) - len(successful))

    if recall_values:
        print(
            "Average expected-page Recall:",
            round(sum(recall_values) / len(recall_values), 3),
        )

    if successful:
        correct_abstentions = sum(
            r["predicted_abstention"] for r in unanswerable
        )
        correct_answers = sum(
            not r["predicted_abstention"] for r in answerable
        )

        print(
            "Answerability classification accuracy:",
            round(
                (
                    correct_abstentions + correct_answers
                ) / len(successful),
                3,
            ),
        )

        print(
            "Unanswerable correct abstentions:",
            f"{correct_abstentions}/{len(unanswerable)}",
        )

        print(
            "Answerable non-abstentions:",
            f"{correct_answers}/{len(answerable)}",
        )

        citation_count = sum(
            r["citation_valid"] for r in successful
            if not r["predicted_abstention"]
        )
        non_abstained = sum(
            not r["predicted_abstention"] for r in successful
        )

        print(
            "Citation consistency:",
            f"{citation_count}/{non_abstained}",
        )

    print("\nResults saved to:", RESULTS_FILE)


if __name__ == "__main__":
    main()
