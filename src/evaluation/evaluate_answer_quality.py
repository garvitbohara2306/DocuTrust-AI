import csv
import re
import json
from pathlib import Path

RESULTS_PATH = "data/evaluation/doc002_results.csv"
OUTPUT_PATH = "data/evaluation/answer_quality_report.csv"


def parse_pages(value):
    """Parse comma-separated or JSON-formatted page numbers."""
    if not value or not value.strip():
        return set()

    value = value.strip()

    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return {int(p) for p in parsed}
        except (ValueError, TypeError):
            pass

    return {
        int(p.strip())
        for p in value.split(",")
        if p.strip().isdigit()
    }


def extract_citations(answer):
    """Extract citations such as [Page 17, DOC002_P17_C1]."""
    return [
        int(page)
        for page in re.findall(
            r"\[Page\s+(\d+)",
            answer,
            re.IGNORECASE
        )
    ]


def is_abstention(answer):
    """Detect an explicit abstention at the beginning."""
    text = answer.strip().lower()

    return (
        text.startswith("insufficient_evidence")
        or text.startswith("insufficient evidence")
    )


def has_late_abstention(answer):
    """Detect an abstention marker appearing after the beginning."""
    text = answer.strip().lower()

    markers = [
        "insufficient_evidence",
        "insufficient evidence",
    ]

    return (
        any(marker in text for marker in markers)
        and not is_abstention(answer)
    )


# Human-reviewed classifications.
MANUAL_REVIEW_OVERRIDES = {
    "DOC002_A02": {
        "response_status": "MIXED ANSWER AND ABSTENTION",
        "answerability_status": "NEEDS MANUAL REVIEW",
        "review_note": (
            "Response contains partial numerical claims and a later "
            "abstention. Retrieved evidence did not include the expected page."
        )
    },

    "DOC002_A13": {
        "response_status": "MIXED ANSWER AND ABSTENTION",
        "answerability_status": "NEEDS MANUAL REVIEW",
        "review_note": (
            "Answer is present, but the response also contains "
            "a contradictory abstention."
        )
    },

    "DOC002_A30": {
        "response_status": "ANSWER PROVIDED WITH LIMITATION",
        "answerability_status": "ANSWER PROVIDED",
        "review_note": (
            "The main question is answered; the final caveat concerns "
            "additional Hindi output quality."
        )
    },

    "DOC002_U14": {
        "response_status": "ABSTAINED",
        "answerability_status": "CORRECT ABSTENTION",
        "review_note": (
            "The answer explicitly states that the supplied passages "
            "do not contain the requested information."
        )
    },
}


# Load original evaluation results.
with open(
    RESULTS_PATH,
    encoding="utf-8-sig",
    newline=""
) as f:
    rows = list(csv.DictReader(f))


if not rows:
    raise ValueError("No records found in the original evaluation CSV.")


report = []

for row in rows:

    answer = row.get("answer", "").strip()
    question_id = row.get("id") or row.get("question_id", "")

    answerable = (
        row.get("is_answerable", "").strip().lower() == "true"
    )

    retrieved_pages = parse_pages(
        row.get("retrieved_pages", "")
    )

    citations = extract_citations(answer)

    abstained = is_abstention(answer)
    late_abstention = has_late_abstention(answer)

    # Citation page membership check.
    if not citations:
        citation_status = "NO CITATION"

    elif all(page in retrieved_pages for page in citations):
        citation_status = "VALID PAGE MEMBERSHIP"

    else:
        citation_status = "INVALID CITATION"

    # Default automatic classification.
    if abstained:
        response_status = "ABSTAINED"

    elif late_abstention:
        response_status = "MIXED ANSWER AND ABSTENTION"

    else:
        response_status = "ANSWER PROVIDED"

    if answerable:

        if abstained:
            answerability_status = "ANSWERABLE BUT ABSTAINED"

        elif late_abstention:
            answerability_status = "NEEDS MANUAL REVIEW"

        else:
            answerability_status = "ANSWER PROVIDED"

    else:

        if abstained:
            answerability_status = "CORRECT ABSTENTION"

        elif late_abstention:
            answerability_status = "MIXED - MANUAL REVIEW"

        else:
            answerability_status = "POSSIBLE UNSUPPORTED ANSWER"

    review_note = "Automatic classification"

    # Apply manually reviewed classifications.
    if question_id in MANUAL_REVIEW_OVERRIDES:

        override = MANUAL_REVIEW_OVERRIDES[question_id]

        response_status = override["response_status"]
        answerability_status = override["answerability_status"]
        review_note = override["review_note"]

    report.append({
        "question_id": question_id,
        "is_answerable": answerable,
        "response_status": response_status,
        "answerability_status": answerability_status,
        "late_abstention": late_abstention,
        "expected_pages": row.get("expected_pages", ""),
        "retrieved_pages": sorted(retrieved_pages),
        "citations": citations,
        "citation_status": citation_status,
        "review_note": review_note,
        "answer": answer,
    })


# Save detailed report.
Path(OUTPUT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=report[0].keys()
    )

    writer.writeheader()
    writer.writerows(report)


# Calculate metrics.
answerable_rows = [
    r for r in report
    if r["is_answerable"]
]

unanswerable_rows = [
    r for r in report
    if not r["is_answerable"]
]

correct_abstentions = sum(
    r["answerability_status"] == "CORRECT ABSTENTION"
    for r in unanswerable_rows
)

unsupported_answers = sum(
    r["answerability_status"] == "POSSIBLE UNSUPPORTED ANSWER"
    for r in unanswerable_rows
)

answerable_abstentions = sum(
    r["answerability_status"] == "ANSWERABLE BUT ABSTAINED"
    for r in answerable_rows
)

mixed_answerable = sum(
    r["answerability_status"] == "NEEDS MANUAL REVIEW"
    for r in answerable_rows
)

answers_provided = sum(
    r["answerability_status"] == "ANSWER PROVIDED"
    for r in answerable_rows
)

valid_citations = sum(
    r["citation_status"] == "VALID PAGE MEMBERSHIP"
    for r in report
)

invalid_citations = sum(
    r["citation_status"] == "INVALID CITATION"
    for r in report
)

no_citations = sum(
    r["citation_status"] == "NO CITATION"
    for r in report
)


print("\n========== ANSWER QUALITY DIAGNOSTIC ==========")

print("Total questions:", len(report))
print("Answerable questions:", len(answerable_rows))
print("Unanswerable questions:", len(unanswerable_rows))

print("\n--- Answerable Questions ---")

print(
    "Answers provided:",
    answers_provided,
    "/",
    len(answerable_rows)
)

print(
    "Answerable but abstained:",
    answerable_abstentions,
    "/",
    len(answerable_rows)
)

print(
    "Mixed responses needing review:",
    mixed_answerable,
    "/",
    len(answerable_rows)
)

print("\n--- Unanswerable Questions ---")

print(
    "Correct abstentions:",
    correct_abstentions,
    "/",
    len(unanswerable_rows)
)

print(
    "Possible unsupported answers:",
    unsupported_answers,
    "/",
    len(unanswerable_rows)
)

print("\n========== CITATION DIAGNOSTIC ==========")

print("Valid page membership:", valid_citations)
print("Invalid citations:", invalid_citations)
print("No citations:", no_citations)

print("\n========== MANUALLY REVIEWED QUESTIONS ==========")

for question_id in MANUAL_REVIEW_OVERRIDES:
    for r in report:
        if r["question_id"] == question_id:
            print(
                r["question_id"],
                "|",
                r["response_status"],
                "|",
                r["answerability_status"]
            )

print("\n========== QUESTIONS NEEDING REVIEW ==========")

for r in report:
    if (
        r["answerability_status"] in [
            "ANSWERABLE BUT ABSTAINED",
            "NEEDS MANUAL REVIEW",
            "POSSIBLE UNSUPPORTED ANSWER",
            "MIXED - MANUAL REVIEW"
        ]
        or r["citation_status"] == "INVALID CITATION"
    ):
        print(
            r["question_id"],
            "|",
            r["answerability_status"],
            "|",
            r["citation_status"],
            "| Retrieved:",
            r["retrieved_pages"],
            "| Cited:",
            r["citations"]
        )

print("\nReport saved:", OUTPUT_PATH)

print(
    "\nNOTE: Citation page membership does not prove factual support. "
    "Manual evidence review is still required."
)