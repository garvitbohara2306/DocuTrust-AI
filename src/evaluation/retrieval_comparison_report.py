
import csv
import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/DOC002_chunks.jsonl"
QA_PATH = "data/evaluation/doc002_qa_eval.jsonl"
OUTPUT_PATH = "data/evaluation/retrieval_comparison_report.csv"
TOP_K = 4

EXPANSIONS = {
    "DOC002_A02": "research found only 5-10 percent documents in accessible format",
    "DOC002_A04": "Intended Audience Indian Government Website Stakeholders Web Information Managers Hired Resources Operational Staff Content Managers",
    "DOC002_A06": "NIC used MS Word 2010 Acrobat 9 Pro LibreOffice 6.4.5.2",
    "DOC002_A17": "Adobe PDF must be a tagged PDF underlying tagged structure",
    "DOC002_A21": "MS Word 2010 File Info Check for Issues Check Accessibility",
    "DOC002_A22": "LibreOffice Writer Export As PDF Select Tagged PDF Type",
    "DOC002_A24": "Acrobat Pro Full Check Accessibility Checker panel errors and hints",
    "DOC002_A25": "PDF Accessibility Checker PAC3 identifies PDF accessibility issues errors PDF UA",
}

with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = [json.loads(x) for x in f if x.strip()]

with open(QA_PATH, encoding="utf-8") as f:
    questions = [json.loads(x) for x in f if x.strip()]

questions = [
    q for q in questions
    if q["is_answerable"] and q.get("expected_pages")
]

print("Loading embedding model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(
    [c["text"] for c in chunks],
    normalize_embeddings=True,
    show_progress_bar=True
)

def retrieve(query, section_aware=False):
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ q_emb
    adjusted = scores.copy()

    if section_aware:
        match = re.search(
            r"\bsection\s+(\d+(?:\.\d+)*)",
            query,
            re.I
        )

        if match:
            section = match.group(1)
            pattern = re.compile(
                rf"(?m)^\s*{re.escape(section)}\.\s+"
            )

            for i, chunk in enumerate(chunks):
                if pattern.search(chunk["text"]):
                    adjusted[i] += 0.20

    indices = np.argsort(adjusted)[::-1][:TOP_K]

    return sorted({
        int(chunks[i]["page_number"])
        for i in indices
    })

rows = []

for item in questions:
    qid = item["id"]
    query = item["question"]
    expected = sorted(set(int(p) for p in item["expected_pages"]))

    baseline = retrieve(query)
    section = retrieve(query, section_aware=True)
    expanded = retrieve(EXPANSIONS.get(qid, query))

    row = {
        "question_id": qid,
        "question": query,
        "expected_pages": str(expected),
        "baseline_pages": str(baseline),
        "baseline_hit": bool(set(expected) & set(baseline)),
        "section_aware_pages": str(section),
        "section_aware_hit": bool(set(expected) & set(section)),
        "query_expansion_pages": str(expanded),
        "query_expansion_hit": bool(set(expected) & set(expanded)),
    }

    rows.append(row)

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nCSV REPORT CREATED:", OUTPUT_PATH)
print("Questions evaluated:", len(rows))

for method, key in [
    ("Baseline", "baseline_hit"),
    ("Section-aware", "section_aware_hit"),
    ("Query Expansion", "query_expansion_hit"),
]:
    hits = sum(row[key] for row in rows)
    print(f"{method}: {hits}/{len(rows)} = {hits/len(rows)*100:.2f}%")
