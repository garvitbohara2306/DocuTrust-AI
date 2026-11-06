import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "processed" / "DOC002_chunks.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks():
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_search_index(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def search(query, chunks, model, index, top_k=5):
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

    scores, indices = index.search(query_vector, min(top_k, len(chunks)))

    print("\nSEARCH QUERY:", query)
    print("\nTOP MATCHING EVIDENCE")

    for rank, (score, position) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if position < 0:
            continue

        chunk = chunks[int(position)]

        print("\n" + "=" * 60)
        print(f"RANK: {rank}")
        print(f"SIMILARITY SCORE: {score:.4f}")
        print(f"DOCUMENT: {chunk['document_id']}")
        print(f"PAGE: {chunk['page_number']}")
        print(f"CHUNK ID: {chunk['chunk_id']}")
        print("EVIDENCE:")
        print(chunk["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        required=True,
        help="Question to search in the document"
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    chunks = load_chunks()

    if not chunks:
        raise ValueError("No chunks found in the dataset.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Building FAISS index...")
    index = build_search_index(chunks, model)

    print("Indexed chunks:", index.ntotal)

    search(args.query, chunks, model, index, args.top_k)


if __name__ == "__main__":
    main()
