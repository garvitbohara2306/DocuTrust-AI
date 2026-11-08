
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL_ID = "CohereLabs/tiny-aya-global:cohere"

def generate_answer(question, retrieved_chunks):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing from .env")

    if not retrieved_chunks:
        return "I could not find sufficient evidence in the supplied document."

    evidence = []
    for chunk in retrieved_chunks:
        evidence.append(
            f"[Page {chunk['page_number']} | {chunk['chunk_id']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(evidence)

    client = InferenceClient(token=token)

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are DocuTrust AI, a document-grounded assistant. "
                    "Answer using ONLY the supplied document evidence. "
                    "Do not use outside knowledge or invent facts. "
                    "If the evidence does not answer the question, clearly say "
                    "that the document does not provide sufficient information. "
                    "Cite supporting page numbers like [Page 14]. "
                    "Keep the answer clear and concise."
                )
            },
            {
                "role": "user",
                "content": (
                    f"DOCUMENT EVIDENCE:\n{context}\n\n"
                    f"QUESTION:\n{question}\n\n"
                    "Give a grounded answer with page citations."
                )
            }
        ],
        max_tokens=300,
        temperature=0.2
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    chunks_file = ROOT / "data" / "processed" / "DOC002_chunks.jsonl"

    if not chunks_file.exists():
        raise SystemExit(f"Chunks file not found: {chunks_file}")

    with chunks_file.open("r", encoding="utf-8-sig") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    selected = [
        chunk for chunk in chunks
        if chunk["chunk_id"] == "DOC002_P14_C1"
    ]

    if not selected:
        raise SystemExit("Test evidence chunk DOC002_P14_C1 was not found.")

    question = "What is an accessible PDF?"

    print("\nSending document evidence to the AI model...")
    answer = generate_answer(question, selected)

    print("\n" + "=" * 55)
    print("QUESTION:", question)
    print("=" * 55)
    print("\nGENERATED ANSWER:\n")
    print(answer)
    print("\n" + "=" * 55)
    print("MODEL CONNECTION TEST COMPLETED")
