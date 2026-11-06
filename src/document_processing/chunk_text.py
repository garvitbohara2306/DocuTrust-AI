import json
from pathlib import Path

INPUT = Path("data/processed/DOC002_pages.jsonl")
OUTPUT = Path("data/processed/DOC002_chunks.jsonl")

CHUNK_SIZE = 1200
OVERLAP = 200


def split_text(text):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(start + 1, end - OVERLAP)

    return chunks


def main():
    records = []

    with INPUT.open("r", encoding="utf-8") as file:
        for line in file:
            page = json.loads(line)

            if not page["has_text"]:
                continue

            page_chunks = split_text(page["text"])

            for index, chunk in enumerate(page_chunks, start=1):
                records.append({
                    "chunk_id": f"{page['document_id']}_P{page['page_number']}_C{index}",
                    "document_id": page["document_id"],
                    "source_file": page["source_file"],
                    "page_number": page["page_number"],
                    "chunk_index": index,
                    "text": chunk,
                    "character_count": len(chunk)
                })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nCHUNKING COMPLETE")
    print("Total chunks:", len(records))
    print("Average chunk size:",
          round(sum(r["character_count"] for r in records) / len(records), 2)
          if records else 0)
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
