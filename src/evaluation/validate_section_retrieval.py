import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS = "data/processed/DOC002_chunks.jsonl"
QUESTIONS = "data/evaluation/doc002_qa_eval.jsonl"

with open(CHUNKS, encoding="utf-8") as f:
    chunks = [json.loads(x) for x in f if x.strip()]

with open(QUESTIONS, encoding="utf-8") as f:
    questions = [json.loads(x) for x in f if x.strip()]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(
    [c["text"] for c in chunks],
    normalize_embeddings=True,
    show_progress_bar=True
)

baseline_hits = 0
improved_hits = 0
tested = 0

print("\nSECTION-AWARE RETRIEVAL VALIDATION\n")

for item in questions:
    query = item["question"]
    expected_pages = set(item.get("expected_pages", []))

    match = re.search(r"\bsection\s+(\d+\.\d+)\b", query, re.I)

    if not match or not expected_pages:
        continue

    section = match.group(1)

    q = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ q

    adjusted = scores.copy()

    pattern = re.compile(
        rf"(?m)^\s*{re.escape(section)}\.\s+",
        re.I
    )

    for i, chunk in enumerate(chunks):
        if pattern.search(chunk["text"]):
            adjusted[i] += 0.20

    baseline_top = np.argsort(scores)[::-1][:4]
    improved_top = np.argsort(adjusted)[::-1][:4]

    baseline_pages = {
        int(chunks[i]["page_number"]) for i in baseline_top
    }
    improved_pages = {
        int(chunks[i]["page_number"]) for i in improved_top
    }

    baseline_hit = bool(baseline_pages.intersection(expected_pages))
    improved_hit = bool(improved_pages.intersection(expected_pages))

    baseline_hits += baseline_hit
    improved_hits += improved_hit
    tested += 1

    print(f"\nQuestion: {query}")
    print("Expected pages:", sorted(expected_pages))
    print("Baseline hit:", baseline_hit)
    print("Section-aware hit:", improved_hit)

print("\n========== SUMMARY ==========")
print("Section-based questions tested:", tested)
print("Baseline Recall@4:", baseline_hits, "/", tested)
print("Section-aware Recall@4:", improved_hits, "/", tested)

if tested:
    print("Baseline rate:", round(baseline_hits / tested * 100, 2), "%")
    print("Improved rate:", round(improved_hits / tested * 100, 2), "%")
