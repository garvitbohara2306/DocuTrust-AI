import json
import numpy as np
from sentence_transformers import SentenceTransformer

with open("data/processed/DOC002_chunks.jsonl", encoding="utf-8") as f:
    chunks = [json.loads(x) for x in f if x.strip()]

with open("data/evaluation/doc002_qa_eval.jsonl", encoding="utf-8") as f:
    questions = [json.loads(x) for x in f if x.strip()]

ids = ["DOC002_A02", "DOC002_A04", "DOC002_A06",
       "DOC002_A17", "DOC002_A21", "DOC002_A22",
       "DOC002_A24", "DOC002_A25"]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(
    [c["text"] for c in chunks],
    normalize_embeddings=True
)

for item in questions:
    if item["id"] not in ids:
        continue

    q = model.encode(
        [item["question"]],
        normalize_embeddings=True
    )[0]

    scores = embeddings @ q
    top = np.argsort(scores)[::-1][:10]

    print("\n" + "=" * 65)
    print(item["id"], item["question"])
    print("Expected pages:", item.get("expected_pages"))

    for rank, i in enumerate(top, 1):
        c = chunks[i]
        print(
            f"Rank {rank}: Page {c['page_number']} | "
            f"{c['chunk_id']} | Score {scores[i]:.4f}"
        )

    for i, c in enumerate(chunks):
        if int(c["page_number"]) in item.get("expected_pages", []):
            print(
                f"EXPECTED CHUNK: {c['chunk_id']} | "
                f"Score: {scores[i]:.4f}"
            )
