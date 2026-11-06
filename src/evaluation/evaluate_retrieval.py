import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

ROOT = Path(__file__).resolve().parents[2]

CHUNKS = ROOT / "data/processed/DOC002_chunks.jsonl"
EVAL = ROOT / "data/evaluation/retrieval_eval.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


def read_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    chunks = read_jsonl(CHUNKS)
    tests = read_jsonl(EVAL)

    model = SentenceTransformer(MODEL_NAME)

    vectors = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    passed = 0

    for test in tests:
        query_vector = model.encode(
            [test["query"]],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        query_vector = np.ascontiguousarray(
            query_vector, dtype=np.float32
        )

        scores, positions = index.search(
            query_vector, min(TOP_K, len(chunks))
        )

        retrieved_ids = [
            chunks[int(pos)]["chunk_id"]
            for pos in positions[0] if pos >= 0
        ]

        expected = set(test["expected_chunk_ids"])
        found = expected.intersection(retrieved_ids)

        recall = len(found) / len(expected)
        passed += recall == 1.0

        print("\nQUERY:", test["query"])
        print("Expected:", sorted(expected))
        print("Retrieved:", retrieved_ids)
        print(f"Recall@{TOP_K}: {recall:.2f}")

    print("\n===== EVALUATION SUMMARY =====")
    print("Total queries:", len(tests))
    print("Queries with all expected chunks retrieved:", passed)
    print("Average Recall@3:",
          round(sum(
              len(set(t["expected_chunk_ids"]) &
                  set([
                      chunks[int(pos)]["chunk_id"]
                      for pos in index.search(
                          model.encode(
                              [t["query"]],
                              normalize_embeddings=True,
                              convert_to_numpy=True
                          ).astype("float32"),
                          TOP_K
                      )[1][0] if pos >= 0
                  ]))
              / len(t["expected_chunk_ids"])
              for t in tests
          ) / len(tests), 3)
          if tests else 0)


if __name__ == "__main__":
    main()
