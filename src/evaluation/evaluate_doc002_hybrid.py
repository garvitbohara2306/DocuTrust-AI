
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

from src.retrieval.hybrid_retriever import (
    build_hybrid_index,
    retrieve_hybrid,
)

CHUNKS_FILE = ROOT / "data/processed/DOC002_chunks.jsonl"
QUESTIONS_FILE = ROOT / "data/evaluation/doc002_qa_eval.jsonl"

# Separate output: original dense baseline remains untouched.
RESULTS_FILE = ROOT / "data/evaluation/doc002_hybrid_results.csv"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "aya-expanse:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"

TOP_K = 4


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def is_abstention(answer):
    text = (answer or "").strip().lower()

    return (
        text.startswith("insufficient_evidence")
        or text.startswith("insufficient evidence")
    )


def generate_answer(question, retrieved):
    context = "\n\n".join(
        (
            f"[Page {chunk['page_number']} | "
            f"{chunk.get('chunk_id', '')}]\n"
            f"{chunk['text']}"
        )
        for chunk in retrieved
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
            {
                "role": "system",
                "content": system_prompt,
            },
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

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=(15, 600),
    )

    response.raise_for_status()

    return response.json()["message"]["content"].strip()


def main():
    print("=" * 55)
    print("DOCUTRUST AI - HYBRID RETRIEVAL EVALUATION")
    print("=" * 55)

    print("\nLoading evaluation data...")

    chunks = load_jsonl(CHUNKS_FILE)
    questions = load_jsonl(QUESTIONS_FILE)

    if not chunks:
        raise ValueError("No document chunks found.")

    if not questions:
        raise ValueError("No evaluation questions found.")

    print(f"Chunks: {len(chunks)}")
    print(f"Questions: {len(questions)}")

    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBED_MODEL)

    print("\nBuilding hybrid retrieval index...")

    dense_vectors, vectorizer, sparse_matrix = build_hybrid_index(
        chunks,
        model,
    )

    print("Hybrid index ready.")

    results = []

    for number, item in enumerate(questions, start=1):

        question = item["question"]
        expected_pages = set(
            item.get("expected_pages", [])
        )

        print(
            f"\n[{number}/{len(questions)}] {question}"
        )

        error = ""

        # -----------------------------
        # Hybrid retrieval
        # -----------------------------

        try:
            retrieved = retrieve_hybrid(
                question=question,
                chunks=chunks,
                embedding_model=model,
                dense_vectors=dense_vectors,
                vectorizer=vectorizer,
                sparse_matrix=sparse_matrix,
                top_k=TOP_K,
            )

        except Exception as exc:
            retrieved = []
            error = f"Retrieval error: {exc}"
            print("ERROR:", error)

        retrieved_pages = sorted({
            int(chunk["page_number"])
            for chunk in retrieved
        })

        # -----------------------------
        # Answer generation
        # -----------------------------

        answer = ""

        if retrieved and not error:

            try:
                answer = generate_answer(
                    question,
                    retrieved,
                )

            except Exception as exc:
                error = f"Generation error: {exc}"
                print("ERROR:", error)

        elif not error:
            answer = (
                "INSUFFICIENT_EVIDENCE: "
                "No relevant passages were retrieved."
            )

        abstained = is_abstention(answer)

        # -----------------------------
        # Citation validation
        # -----------------------------

        cited_pages = sorted({
            int(page)
            for page in re.findall(
                r"\[\s*Page\s+(\d+)",
                answer,
                re.IGNORECASE,
            )
        })

        invalid_citations = [
            page
            for page in cited_pages
            if page not in retrieved_pages
        ]

        citation_valid = (
            bool(cited_pages)
            and not invalid_citations
            and not abstained
            and not error
        )

        # -----------------------------
        # Expected-page recall
        # -----------------------------

        recall = ""

        if expected_pages:
            recall = round(
                len(
                    expected_pages.intersection(
                        retrieved_pages
                    )
                ) / len(expected_pages),
                3,
            )

        # -----------------------------
        # Save result
        # -----------------------------

        results.append({
            "id": item["id"],
            "question": question,
            "is_answerable": item["is_answerable"],
            "expected_pages": ",".join(
                map(str, sorted(expected_pages))
            ),
            "retrieved_pages": ",".join(
                map(str, retrieved_pages)
            ),
            "page_recall": recall,
            "answer": answer,
            "predicted_abstention": abstained,
            "citation_valid": citation_valid,
            "cited_pages": ",".join(
                map(str, cited_pages)
            ),
            "error": error,
        })

        print("Retrieved pages:", retrieved_pages)
        print("Page recall:", recall)
        print("Abstained:", abstained)
        print("Citation valid:", citation_valid)

    # -----------------------------
    # Write separate CSV
    # -----------------------------

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "id",
        "question",
        "is_answerable",
        "expected_pages",
        "retrieved_pages",
        "page_recall",
        "answer",
        "predicted_abstention",
        "citation_valid",
        "cited_pages",
        "error",
    ]

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    # -----------------------------
    # Evaluation summary
    # -----------------------------

    answerable = [
        result
        for result in results
        if result["is_answerable"]
    ]

    unanswerable = [
        result
        for result in results
        if not result["is_answerable"]
    ]

    successful = [
        result
        for result in results
        if not result["error"]
    ]

    recall_values = [
        result["page_recall"]
        for result in answerable
        if result["page_recall"] != ""
    ]

    print("\n" + "=" * 55)
    print("HYBRID EVALUATION SUMMARY")
    print("=" * 55)

    print("Total questions:", len(results))
    print("Successful model responses:", len(successful))
    print("Failed responses:", len(results) - len(successful))

    if recall_values:
        average_recall = round(
            sum(recall_values) / len(recall_values),
            3,
        )

        print(
            "Average expected-page Recall:",
            average_recall,
        )

    if successful:

        correct_abstentions = sum(
            result["predicted_abstention"]
            for result in unanswerable
            if not result["error"]
        )

        correct_answers = sum(
            not result["predicted_abstention"]
            for result in answerable
            if not result["error"]
        )

        print(
            "Unanswerable correct abstentions:",
            f"{correct_abstentions}/{len(unanswerable)}",
        )

        print(
            "Answerable non-abstentions:",
            f"{correct_answers}/{len(answerable)}",
        )

        classified = (
            correct_abstentions + correct_answers
        )

        print(
            "Answerability accuracy:",
            round(
                classified / len(successful),
                3,
            ) if successful else 0,
        )

        citation_count = sum(
            result["citation_valid"]
            for result in successful
            if not result["predicted_abstention"]
        )

        non_abstained = sum(
            not result["predicted_abstention"]
            for result in successful
        )

        print(
            "Citation consistency:",
            f"{citation_count}/{non_abstained}",
        )

    print("\nResults saved to:")
    print(RESULTS_FILE)

    print("\nOriginal dense baseline was not modified.")


if __name__ == "__main__":
    main()
