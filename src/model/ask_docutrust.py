from pathlib import Path
import json
import urllib.request
import urllib.error
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = ROOT / "data" / "processed" / "DOC002_chunks.jsonl"

MODEL_NAME = "aya-expanse:8b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/chat"
TOP_K = 3

def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks not found: {CHUNKS_FILE}")

    with CHUNKS_FILE.open("r", encoding="utf-8-sig") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    if not chunks:
        raise ValueError("No document chunks found.")

    return chunks

def build_retriever(chunks):
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Creating document embeddings...")
    texts = [chunk["text"] for chunk in chunks]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    print(f"Retrieval ready: {len(chunks)} chunks indexed.")
    return model, index

def retrieve(question, model, index, chunks):
    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

    scores, positions = index.search(
        query_vector,
        min(TOP_K, len(chunks))
    )

    results = []

    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue

        chunk = chunks[int(position)].copy()
        chunk["similarity"] = float(score)
        results.append(chunk)

    return results

def generate_answer(question, retrieved_chunks):
    if not retrieved_chunks:
        return "I could not find sufficient evidence in the document."

    evidence = []

    for chunk in retrieved_chunks:
        evidence.append(
            f"[Page {chunk['page_number']} | {chunk['chunk_id']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(evidence)

    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are DocuTrust AI, a document-grounded assistant. "
                    "Answer ONLY using the supplied document evidence. "
                    "Never invent facts or use outside knowledge. "
                    "If the evidence does not answer the question, say: "
                    "'The provided document does not contain sufficient "
                    "information to answer this question.' "
                    "Cite the relevant page numbers using [Page N]. "
                    "Keep the answer clear and concise."
                )
            },
            {
                "role": "user",
                "content": (
                    f"DOCUMENT EVIDENCE:\n{context}\n\n"
                    f"QUESTION:\n{question}\n\n"
                    "Answer using the evidence and cite page numbers."
                )
            }
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": 350
        }
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["message"]["content"]

    except urllib.error.URLError as error:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running."
        ) from error

def main():
    print("=" * 60)
    print("             DOCUTRUST AI")
    print("      Document Grounded Question Answering")
    print("=" * 60)

    chunks = load_chunks()
    embedding_model, index = build_retriever(chunks)

    print("\nSystem ready!")
    print("Ask questions about the PDF.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ("exit", "quit"):
            print("DocuTrust AI session ended.")
            break

        if not question:
            continue

        try:
            retrieved = retrieve(
                question, embedding_model, index, chunks
            )

            print("\n--- RETRIEVED EVIDENCE ---")
            for item in retrieved:
                print(
                    f"Page {item['page_number']} | "
                    f"Score: {item['similarity']:.4f} | "
                    f"{item['chunk_id']}"
                )

            print("\n--- AI GENERATED ANSWER ---\n")
            answer = generate_answer(question, retrieved)
            print(answer)
            print("\n" + "-" * 60)

        except Exception as error:
            print(f"\nERROR: {error}")

if __name__ == "__main__":
    main()
