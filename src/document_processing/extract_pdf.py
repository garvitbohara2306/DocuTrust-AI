import argparse
import json
import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(pdf_path: Path, document_id: str, output_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with_text = empty = 0

    with pymupdf.open(pdf_path) as doc, output_path.open("w", encoding="utf-8") as out:
        total = len(doc)

        for index, page in enumerate(doc):
            text = clean_text(page.get_text("text", sort=True))

            record = {
                "document_id": document_id,
                "source_file": pdf_path.name,
                "page_number": index + 1,
                "page_label": page.get_label() or str(index + 1),
                "text": text,
                "character_count": len(text),
                "has_text": bool(text),
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if text:
                with_text += 1
            else:
                empty += 1

    print("\nPDF EXTRACTION COMPLETE")
    print("Total pages:", total)
    print("Pages with text:", with_text)
    print("Empty pages:", empty)
    print("Output:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--doc-id", required=True)
    args = parser.parse_args()

    out = ROOT / "data" / "processed" / f"{args.doc_id}_pages.jsonl"
    extract_pdf(args.pdf, args.doc_id, out)
