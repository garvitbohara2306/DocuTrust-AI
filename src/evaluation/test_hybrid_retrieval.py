import json
import csv
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]

def load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

chunks = load(ROOT / "data/processed/DOC002_chunks.jsonl")
questions = load(ROOT / "data/evaluation/doc002_qa_eval.jsonl")

print("Loading semantic model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = [c["text"] for c in chunks]
dense_vectors = model.encode(
    texts, normalize_embeddings=True, convert_to_numpy=True
).astype("float32")

print("Building keyword index...")
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)
sparse_matrix = vectorizer.fit_transform(texts)

def rank_dense(query):
    q = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    scores = dense_vectors @ q
    return np.argsort(scores)[::-1]

def rank_sparse(query):
    q = vectorizer.transform([query])
    scores = (sparse_matrix @ q.T).toarray().ravel()
    return np.argsort(scores)[::-1]

def fuse(dense_rank, sparse_rank, k=60):
    scores = {}
    for ranking in [dense_rank, sparse_rank]:
        for rank, idx in enumerate(ranking, start=1):
            scores[int(idx)] = scores.get(int(idx), 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

results = []

for item in questions:
    q = item["question"]
    dense_rank = rank_dense(q)
    sparse_rank = rank_sparse(q)
    hybrid_rank = fuse(dense_rank, sparse_rank)

    expected = set(int(x) for x in item.get("expected_pages", []))

    dense_top = [int(chunks[int(i)]["page_number"]) for i in dense_rank[:4]]
    hybrid_top = [int(chunks[int(i)]["page_number"]) for i in hybrid_rank[:4]]

    dense_hit = bool(expected.intersection(dense_top)) if expected else None
    hybrid_hit = bool(expected.intersection(hybrid_top)) if expected else None

    results.append({
        "id": item["id"],
        "question": q,
        "is_answerable": item["is_answerable"],
        "expected_pages": ",".join(map(str, sorted(expected))),
        "dense_pages": ",".join(map(str, dense_top)),
        "hybrid_pages": ",".join(map(str, hybrid_top)),
        "dense_hit_at_4": dense_hit,
        "hybrid_hit_at_4": hybrid_hit
    })

    if item["id"] in ["DOC002_A01", "DOC002_A02", "DOC002_A04", "DOC002_A05", "DOC002_A09"]:
        print("\n", item["id"], q)
        print("Expected:", sorted(expected))
        print("Dense Top-4:", dense_top)
        print("Hybrid Top-4:", hybrid_top)

answerable = [r for r in results if r["is_answerable"]]
dense_hits = sum(r["dense_hit_at_4"] for r in answerable)
hybrid_hits = sum(r["hybrid_hit_at_4"] for r in answerable)

print("\n========== RETRIEVAL COMPARISON ==========")
print(f"Dense Page Hit@4:  {dense_hits}/{len(answerable)} = {dense_hits/len(answerable):.1%}")
print(f"Hybrid Page Hit@4: {hybrid_hits}/{len(answerable)} = {hybrid_hits/len(answerable):.1%}")

out = ROOT / "data/evaluation/doc002_hybrid_retrieval_results.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Saved:", out)
