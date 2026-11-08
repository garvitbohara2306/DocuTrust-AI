from pathlib import Path
import json
import urllib.request
import urllib.error
import numpy as np
import sys
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.verify_answer import verify_answer
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
        return "The document does not contain sufficient evidence."

    import json
    import urllib.request
    import urllib.error

    context = "\n\n".join(
        f"[Page {c['page_number']} | {c['chunk_id']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are DocuTrust AI, a strict document-grounded assistant. "
                    "Follow these rules carefully:\n"
                    "1. Answer only from the supplied document evidence.\n"
                    "2. Never use outside knowledge or invent facts.\n"
                    "3. Cite every factual claim with its supporting page, "
                    "using exactly [Page N].\n"
                    "4. Only cite a page if its supplied passage supports "
                    "the specific claim.\n"
                    "5. Do not combine unrelated evidence into one claim.\n"
                    "6. Do not introduce standards, numbers, procedures, "
                    "or examples unless the evidence explicitly supports them.\n"
                    "7. If evidence is insufficient, clearly say so instead "
                    "of guessing.\n"
                    "8. Keep answers concise and easy to understand.\n"
                    "9. Never invent page numbers or citations."
                )
            },
            {
                "role": "user",
                "content": f"Evidence:\n{context}\n\nQuestion: {question}"
            }
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": 350
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["message"]["content"]

    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama HTTP {error.code}: {details}"
        ) from error

    except Exception as error:
        raise RuntimeError(
            f"Ollama request failed: {type(error).__name__}: {error}"
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

            answer = generate_answer(question, retrieved)

            print("\n--- GENERATED ANSWER (DRAFT) ---\n")
            print(answer)

            print("\n--- EVIDENCE VERIFICATION ---")
            verification = verify_answer(answer, retrieved)

            print("Retrieved pages:", verification["retrieved_pages"])
            print("Cited pages:", verification["cited_pages"])
            print("Missing citation:", verification["missing_citation"])
            print("Invalid citations:", verification["invalid_pages"])
            print("Text overlap:", verification["text_overlap"])

            # Verification gate
            if not verification["citations_valid"]:
                print("\nVERIFICATION GATE: UNVERIFIED")
                print("Reason: Missing or invalid page citations.")
                print("Do not present this answer as a verified answer.")

            else:
                print("\nVERIFICATION GATE: CITATION CHECK PASSED")
                print("Warning: Citation consistency does not prove factual correctness.")
                print("Claim-level evidence review is still required.")

            print("Note:", verification["verification_note"])
            print("\n" + "-" * 60)

        except Exception as error:
            print(f"\nERROR: {error}")

if __name__ == "__main__":
    main()
