import csv
import json
import re
from pathlib import Path
from collections import Counter

RESULTS_PATH = Path("data/evaluation/doc002_results.csv")
CHUNKS_PATH = Path("data/processed/DOC002_chunks.jsonl")
OUTPUT_DIR = Path("data/evaluation")

REPORT_PATH = OUTPUT_DIR / "evidence_audit_report.csv"
SUMMARY_PATH = OUTPUT_DIR / "evidence_audit_summary.txt"
REVIEW_PATH = OUTPUT_DIR / "manual_review_required.csv"

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "to", "of", "in", "on", "at", "for", "from", "by",
    "with", "and", "or", "but", "as", "that", "this", "these",
    "those", "it", "its", "they", "them", "their", "there",
    "here", "which", "who", "what", "when", "where", "how",
    "why", "can", "could", "should", "would", "may", "might",
    "must", "will", "shall", "do", "does", "did", "have", "has",
    "had", "not", "only", "also", "according", "provided",
    "passages", "document", "documents", "page", "pages",
    "following", "based", "states", "state", "says", "said",
    "following", "answer", "answers", "include", "includes",
    "including", "such", "than", "then", "into", "about",
    "their", "through", "using", "used", "use"
}

ABSTENTION_MARKERS = (
    "insufficient_evidence",
    "insufficient evidence",
    "do not contain information",
    "does not contain information",
    "not provided in the passages",
    "cannot be determined from the passages",
    "unable to determine from the supplied passages"
)


def parse_pages(value):
    if not value:
        return set()

    value = str(value).strip()

    if value.startswith("["):
        try:
            parsed = json.loads(value)
            return {int(x) for x in parsed}
        except (ValueError, TypeError):
            pass

    return {
        int(x.strip())
        for x in value.split(",")
        if x.strip().isdigit()
    }


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return {
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    }


def remove_citations(text):
    text = re.sub(
        r"\[Page\s+\d+[^\]]*\]",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\(p\.?\s*\d+\)",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text


def split_claims(answer):
    answer = remove_citations(answer)

    lines = re.split(r"[\n]+", answer)

    claims = []

    for line in lines:
        line = line.strip()

        line = re.sub(
            r"^\s*(?:[-*•]|\d+[.)])\s*",
            "",
            line
        )

        if not line:
            continue

        # Split longer paragraphs into sentences.
        sentences = re.split(r"(?<=[.!?])\s+", line)

        for sentence in sentences:
            sentence = sentence.strip()

            words = tokenize(sentence)

            if len(words) >= 5:
                claims.append(sentence)

    return claims


def overlap_score(claim, evidence):
    claim_words = tokenize(claim)
    evidence_words = tokenize(evidence)

    if not claim_words:
        return 0.0

    matched = claim_words.intersection(evidence_words)

    return len(matched) / len(claim_words)


def find_best_evidence(claim, chunks):
    best_score = 0.0
    best_chunk = None

    for chunk in chunks:
        score = overlap_score(
            claim,
            chunk.get("text", "")
        )

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_score, best_chunk


def extract_citations(answer):
    return [
        int(page)
        for page in re.findall(
            r"\[Page\s+(\d+)",
            answer,
            re.IGNORECASE
        )
    ]


def is_abstention(answer):
    text = answer.strip().lower()

    return any(
        text.startswith(marker)
        for marker in (
            "insufficient_evidence",
            "insufficient evidence"
        )
    )


def has_abstention_wording(answer):
    text = answer.lower()

    return any(
        marker in text
        for marker in ABSTENTION_MARKERS
    )


def load_jsonl(path):
    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():

    print("\nLoading evaluation results...")
    with open(
        RESULTS_PATH,
        encoding="utf-8-sig",
        newline=""
    ) as f:
        results = list(csv.DictReader(f))

    print("Loading document chunks...")
    chunks = load_jsonl(CHUNKS_PATH)

    if not results:
        raise ValueError("Evaluation results are empty.")

    if not chunks:
        raise ValueError("Document chunks are empty.")

    print("Questions:", len(results))
    print("Document chunks:", len(chunks))

    report = []

    for index, row in enumerate(results, start=1):

        question_id = row.get("id") or row.get("question_id", "")
        question = row.get("question", "")
        answer = row.get("answer", "").strip()

        answerable = (
            row.get("is_answerable", "").lower() == "true"
        )

        expected_pages = parse_pages(
            row.get("expected_pages", "")
        )

        retrieved_pages = parse_pages(
            row.get("retrieved_pages", "")
        )

        citations = extract_citations(answer)

        retrieved_chunks = [
            c for c in chunks
            if c.get("page_number") in retrieved_pages
        ]

        claims = split_claims(answer)

        supported_claims = 0
        unsupported_claims = 0
        claim_details = []

        # Compare each answer claim with retrieved evidence
        # and the complete document.
        for claim in claims:

            retrieved_score, retrieved_match = find_best_evidence(
                claim,
                retrieved_chunks
            )

            document_score, document_match = find_best_evidence(
                claim,
                chunks
            )

            # Lexical overlap is only a diagnostic signal.
            if retrieved_score >= 0.45:
                supported_claims += 1
                claim_status = "LEXICALLY MATCHED RETRIEVED EVIDENCE"

            elif document_score >= 0.45:
                unsupported_claims += 1
                claim_status = "POSSIBLE RETRIEVAL MISS"

            else:
                unsupported_claims += 1
                claim_status = "LOW EVIDENCE OVERLAP"

            retrieved_page = (
                retrieved_match.get("page_number")
                if retrieved_match else ""
            )

            document_page = (
                document_match.get("page_number")
                if document_match else ""
            )

            claim_details.append(
                f"{claim_status}: {claim[:160]} "
                f"| Retrieved page: {retrieved_page} "
                f"| Best document page: {document_page}"
            )

        total_claims = len(claims)

        if total_claims:
            support_ratio = supported_claims / total_claims
        else:
            support_ratio = 0.0

        abstained = is_abstention(answer)
        abstention_wording = has_abstention_wording(answer)

        late_abstention = (
            abstention_wording and not abstained
        )

        expected_retrieved = bool(
            expected_pages.intersection(retrieved_pages)
        )

        # Citation membership check.
        if not citations:
            citation_status = "NO CITATION"

        elif all(
            page in retrieved_pages
            for page in citations
        ):
            citation_status = "VALID PAGE MEMBERSHIP"

        else:
            citation_status = "INVALID CITATION"

        # Overall diagnostic classification.
        if abstained:

            if answerable and not expected_retrieved:
                classification = "RETRIEVAL FAILURE - ABSTENTION"

            elif answerable:
                classification = "ANSWERABLE BUT ABSTAINED"

            else:
                classification = "CORRECT ABSTENTION"

        elif late_abstention:
            classification = "MIXED RESPONSE - REVIEW"

        elif not answerable:
            classification = "POSSIBLE UNSUPPORTED ANSWER"

        elif total_claims == 0:
            classification = "MANUAL REVIEW - NO CLAIMS"

        elif support_ratio >= 0.70:
            classification = "LIKELY SUPPORTED - VERIFY"

        elif support_ratio >= 0.40:
            classification = "PARTIALLY SUPPORTED - REVIEW"

        else:
            classification = "LOW EVIDENCE SUPPORT - REVIEW"

        # Retrieval diagnostics.
        if answerable and not expected_retrieved:
            retrieval_status = "EXPECTED PAGE NOT RETRIEVED"

        elif answerable:
            retrieval_status = "EXPECTED PAGE RETRIEVED"

        else:
            retrieval_status = "UNANSWERABLE QUESTION"

        # Manual review flags.
        needs_review = (
            "YES"
            if (
                classification not in [
                    "CORRECT ABSTENTION",
                    "LIKELY SUPPORTED - VERIFY"
                ]
                or citation_status == "INVALID CITATION"
            )
            else "NO"
        )

        report.append({
            "question_id": question_id,
            "question": question,
            "is_answerable": answerable,
            "expected_pages": sorted(expected_pages),
            "retrieved_pages": sorted(retrieved_pages),
            "retrieval_status": retrieval_status,
            "expected_page_retrieved": expected_retrieved,
            "response_abstained": abstained,
            "late_abstention": late_abstention,
            "total_claims": total_claims,
            "lexically_matched_claims": supported_claims,
            "low_overlap_claims": unsupported_claims,
            "lexical_support_ratio": round(support_ratio, 3),
            "citation_status": citation_status,
            "classification": classification,
            "needs_manual_review": needs_review,
            "claim_diagnostics": " || ".join(claim_details),
            "answer": answer
        })

        print(
            f"[{index}/{len(results)}] "
            f"{question_id}: {classification}"
        )

    # Save full report.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = list(report[0].keys())

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(report)

    # Save manual review subset.
    review_rows = [
        row for row in report
        if row["needs_manual_review"] == "YES"
    ]

    with open(
        REVIEW_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(review_rows)

    # Summary metrics.
    classifications = Counter(
        row["classification"]
        for row in report
    )

    citation_counts = Counter(
        row["citation_status"]
        for row in report
    )

    retrieval_counts = Counter(
        row["retrieval_status"]
        for row in report
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("DOCUTRUST AI - MASTER EVIDENCE AUDIT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total questions: {len(report)}\n")
        f.write(f"Document chunks: {len(chunks)}\n")
        f.write(f"Questions needing review: {len(review_rows)}\n\n")

        f.write("CLASSIFICATION SUMMARY\n")
        f.write("-" * 35 + "\n")

        for label, count in classifications.most_common():
            f.write(f"{label}: {count}\n")

        f.write("\nRETRIEVAL SUMMARY\n")
        f.write("-" * 35 + "\n")

        for label, count in retrieval_counts.items():
            f.write(f"{label}: {count}\n")

        f.write("\nCITATION SUMMARY\n")
        f.write("-" * 35 + "\n")

        for label, count in citation_counts.items():
            f.write(f"{label}: {count}\n")

        f.write("\nIMPORTANT LIMITATION\n")
        f.write(
            "Lexical overlap is not semantic entailment. "
            "A high score does not prove factual correctness, "
            "and a low score does not prove hallucination. "
            "All flagged claims require evidence review.\n"
        )

    print("\n" + "=" * 55)
    print("MASTER AUDIT COMPLETED")
    print("=" * 55)

    print("Total questions:", len(report))
    print("Questions needing review:", len(review_rows))

    print("\nClassification summary:")

    for label, count in classifications.most_common():
        print(f"{label}: {count}")

    print("\nGenerated files:")
    print(REPORT_PATH)
    print(REVIEW_PATH)
    print(SUMMARY_PATH)

    print(
        "\nReminder: Lexical matching is a screening diagnostic, "
        "not proof of factual correctness."
    )


if __name__ == "__main__":
    main()
