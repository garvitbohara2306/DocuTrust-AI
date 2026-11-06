import json
from pathlib import Path

DATASET = Path("data/processed/pilot.jsonl")
SOURCE = Path("data/raw/sample_policy.txt")

required_fields = {
    "document_id",
    "question",
    "context",
    "answer",
    "page_number",
    "is_answerable",
    "evidence",
}

source_text = SOURCE.read_text(encoding="utf-8-sig").strip()
records = []

with DATASET.open("r", encoding="utf-8-sig") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Line {line_number}: Invalid JSON: {error}")

        missing = required_fields - record.keys()
        if missing:
            raise ValueError(f"Line {line_number}: Missing fields: {missing}")

        if not isinstance(record["is_answerable"], bool):
            raise ValueError(f"Line {line_number}: is_answerable must be true or false")

        if not isinstance(record["page_number"], int) or record["page_number"] < 1:
            raise ValueError(f"Line {line_number}: Invalid page number")

        if not record["question"].strip():
            raise ValueError(f"Line {line_number}: Empty question")

        if not record["context"].strip():
            raise ValueError(f"Line {line_number}: Empty context")

        if record["is_answerable"]:
            if not record["answer"].strip():
                raise ValueError(f"Line {line_number}: Answer is empty")

            if not record["evidence"].strip():
                raise ValueError(f"Line {line_number}: Evidence is empty")

            if record["evidence"] not in record["context"]:
                raise ValueError(f"Line {line_number}: Evidence is not present in context")

            if record["evidence"] not in source_text:
                raise ValueError(f"Line {line_number}: Evidence is not present in source document")

        else:
            if record["answer"] != "" or record["evidence"] != "":
                raise ValueError(
                    f"Line {line_number}: Unanswerable examples must have empty answer and evidence"
                )

            # Use the actual source document as context for unanswerable questions.
            record["context"] = source_text.strip()

        records.append(record)

with DATASET.open("w", encoding="utf-8") as file:
    for record in records:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

print("DATASET VALIDATION PASSED")
print("Total examples:", len(records))
print("Answerable:", sum(r["is_answerable"] for r in records))
print("Unanswerable:", sum(not r["is_answerable"] for r in records))
print("All evidence is present in the source document.")

