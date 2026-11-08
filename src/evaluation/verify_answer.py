"""Conservative citation-consistency checks; not a factuality/entailment guarantee."""
import re
import string

PAGE_CITATION = re.compile(r"\[\s*Page\s+(\d+)", re.IGNORECASE)

def extract_cited_pages(answer: str):
    return sorted({int(page) for page in PAGE_CITATION.findall(answer or "")})

def verify_citations(answer: str, retrieved_chunks):
    retrieved_pages = sorted({int(c["page_number"]) for c in retrieved_chunks})
    cited_pages = extract_cited_pages(answer)
    invalid_pages = [p for p in cited_pages if p not in retrieved_pages]
    return {
        "retrieved_pages": retrieved_pages,
        "cited_pages": cited_pages,
        "missing_citation": not bool(cited_pages),
        "invalid_pages": invalid_pages,
        "citations_valid": bool(cited_pages) and not invalid_pages,
    }

def calculate_text_overlap(answer: str, retrieved_chunks):
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "of",
        "to", "in", "for", "on", "with", "by", "as", "it", "this", "that",
        "these", "those", "be", "can", "also", "from", "according",
    }
    def words(text):
        text = (text or "").lower().translate(str.maketrans("", "", string.punctuation))
        return {w for w in text.split() if len(w) > 2 and w not in stop_words}
    answer_words = words(answer)
    evidence_words = words(" ".join(c.get("text", "") for c in retrieved_chunks))
    if not answer_words:
        return 0.0
    return round(len(answer_words & evidence_words) / len(answer_words), 3)

def verify_answer(answer: str, retrieved_chunks):
    citations = verify_citations(answer, retrieved_chunks)
    return {
        **citations,
        "text_overlap": calculate_text_overlap(answer, retrieved_chunks),
        "verification_note": (
            "These are preliminary citation/lexical checks only. They do not establish "
            "claim-level support, entailment, or factual correctness."
        ),
    }
