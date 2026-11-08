
from __future__ import annotations

import re
from pathlib import Path

import faiss
import numpy as np
import pymupdf
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "aya-expanse:8b"
OLLAMA_CHAT = "http://localhost:11434/api/chat"

CHUNK_SIZE = 1200
OVERLAP = 200
TOP_K = 6
CANDIDATE_K = 30


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="DocuTrust AI",
    page_icon="📄",
    layout="wide",
)

# Improve readability
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 17px !important;
    }

    .stMarkdown, .stText, .stCaption, label {
        font-size: 17px !important;
    }

    h1 { font-size: 2.1rem !important; }
    h2 { font-size: 1.65rem !important; }
    h3 { font-size: 1.3rem !important; }

    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 DocuTrust AI")
st.caption(
    "Document-grounded question answering with source-page references"
)


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(text: str) -> list[str]:

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):

            boundary = text.rfind(" ", start, end)

            if boundary > start + CHUNK_SIZE // 2:
                end = boundary

        part = text[start:end].strip()

        if part:
            chunks.append(part)

        if end >= len(text):
            break

        start = max(start + 1, end - OVERLAP)

    return chunks


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_chunks(
    pdf_bytes: bytes,
    filename: str
) -> list[dict]:

    records = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    ) as doc:

        for page_idx, page in enumerate(doc):

            text = page.get_text("text", sort=True)

            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            if not text:
                continue

            for chunk_idx, part in enumerate(
                split_text(text),
                start=1,
            ):

                records.append({
                    "document_id": Path(filename).stem,
                    "source_file": filename,
                    "page_number": page_idx + 1,
                    "chunk_id": (
                        f"{Path(filename).stem}"
                        f"_P{page_idx + 1}"
                        f"_C{chunk_idx}"
                    ),
                    "text": part,
                })

    return records


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(
    show_spinner="Loading the embedding model..."
)
def load_embedding_model():

    return SentenceTransformer(EMBED_MODEL)


# ============================================================
# FAISS INDEX
# ============================================================

def build_index(chunks: list[dict], model):

    if not chunks:
        raise ValueError(
            "No text chunks were extracted from this PDF."
        )

    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    vectors = np.ascontiguousarray(
        vectors,
        dtype=np.float32,
    )

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index


# ============================================================
# HYBRID RETRIEVAL
# Semantic + Keyword + Section-aware ranking
# ============================================================

STOPWORDS = {
    "what", "are", "the", "is", "in", "on",
    "of", "for", "to", "and", "a", "an",
    "which", "listed", "mentioned", "from",
    "does", "do", "was", "were", "this",
    "that", "section", "about", "explain",
    "describe", "tell", "me",
}


def tokenize(value: str) -> set[str]:

    return {
        token.lower()
        for token in re.findall(
            r"[a-zA-Z0-9]+",
            value,
        )
        if token.lower() not in STOPWORDS
    }


def retrieve(
    question: str,
    chunks: list[dict],
    model,
    index,
) -> list[dict]:

    if not chunks:
        return []

    # --------------------------------------
    # 1. Semantic search
    # --------------------------------------

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_vector = np.ascontiguousarray(
        query_vector,
        dtype=np.float32,
    )

    semantic_k = min(CANDIDATE_K, len(chunks))

    semantic_scores, semantic_positions = index.search(
        query_vector,
        semantic_k,
    )

    semantic_rank = {}
    semantic_score_map = {}

    for rank, (score, position) in enumerate(
        zip(semantic_scores[0], semantic_positions[0]),
        start=1,
    ):

        if position >= 0:

            position = int(position)

            semantic_rank[position] = rank
            semantic_score_map[position] = float(score)

    # --------------------------------------
    # 2. Keyword search
    # --------------------------------------

    query_tokens = tokenize(question)
    expanded_tokens = set(query_tokens)

    if query_tokens.intersection({
        "accessibility",
        "standard",
        "standards",
        "guideline",
        "guidelines",
    }):

        expanded_tokens.update({
            "accessibility",
            "standard",
            "standards",
            "guidelines",
            "guideline",
        })

    lexical_scores = {}

    section_match = re.search(
        r"\bsection\s+(\d+(?:\.\d+)*)\b",
        question,
        flags=re.IGNORECASE,
    )

    section_number = (
        section_match.group(1)
        if section_match
        else None
    )

    for i, chunk in enumerate(chunks):

        text = chunk["text"].lower()
        chunk_tokens = tokenize(text)

        overlap = len(expanded_tokens & chunk_tokens)

        coverage = (
            overlap / max(len(expanded_tokens), 1)
        )

        section_bonus = 0.0

        if section_number:

            heading_pattern = (
                rf"(?m)^\s*{re.escape(section_number)}(?:\s|[.)])"
            )

            if re.search(heading_pattern, text):
                section_bonus = 0.20

        lexical_scores[i] = coverage + section_bonus

    lexical_order = sorted(
        lexical_scores,
        key=lambda i: lexical_scores[i],
        reverse=True,
    )

    lexical_k = min(CANDIDATE_K, len(chunks))

    lexical_rank = {
        position: rank
        for rank, position in enumerate(
            lexical_order[:lexical_k],
            start=1,
        )
    }

    # --------------------------------------
    # 3. Reciprocal Rank Fusion
    # --------------------------------------

    candidate_positions = (
        set(semantic_rank.keys())
        | set(lexical_rank.keys())
    )

    fused_results = []

    for position in candidate_positions:

        semantic_r = semantic_rank.get(
            position,
            len(chunks) + 1,
        )

        lexical_r = lexical_rank.get(
            position,
            len(chunks) + 1,
        )

        fused_score = (
            1.0 / (60 + semantic_r)
            + 1.0 / (60 + lexical_r)
        )

        fused_results.append(
            (fused_score, position)
        )

    fused_results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    # --------------------------------------
    # 4. Return top evidence chunks
    # --------------------------------------

    found = []

    for fused_score, position in fused_results[:TOP_K]:

        item = dict(chunks[position])

        item["similarity"] = semantic_score_map.get(
            position,
            0.0,
        )

        item["retrieval_score"] = fused_score

        found.append(item)

    return found


# ============================================================
# OLLAMA GENERATION
# ============================================================

def call_ollama(
    messages: list[dict],
    num_predict: int = 500,
) -> str:

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": 0.0,
            "num_predict": num_predict,
        },
    }

    try:

        response = requests.post(
            OLLAMA_CHAT,
            json=payload,
            timeout=(15, 600),
        )

        response.raise_for_status()

        result = response.json()

        answer = result.get(
            "message", {}
        ).get(
            "content", ""
        ).strip()

        if not answer:
            raise RuntimeError(
                "Ollama returned an empty answer."
            )

        return answer

    except requests.RequestException as exc:

        raise RuntimeError(
            "Could not connect to Ollama. "
            "Check that Ollama is running and the model is available."
        ) from exc


def ask_ollama(
    question: str,
    evidence: list[dict],
) -> str:

    if not evidence:

        return (
            "INSUFFICIENT_EVIDENCE: "
            "No relevant document passages were retrieved."
        )

    context = "\n\n".join(
        f"[Page {chunk['page_number']} | "
        f"{chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in evidence
    )

    allowed_pages = sorted({
        int(chunk["page_number"])
        for chunk in evidence
    })

    allowed_pages_text = ", ".join(
        map(str, allowed_pages)
    )

    system_prompt = f"""
You are DocuTrust AI, a strict document-grounded assistant.

RULES:

1. Use ONLY the supplied evidence.
2. Never use outside knowledge.
3. Do not guess or invent facts.
4. If evidence is insufficient, respond:
   INSUFFICIENT_EVIDENCE: followed by one short explanation.
5. If evidence is sufficient, answer in concise numbered points.
6. EVERY factual numbered point MUST end with a page citation.
7. Use this exact citation format: [Page 14]
8. Replace 14 with the actual page number from the evidence.
9. Only cite these page numbers: {allowed_pages_text}
10. Do not invent line numbers.
11. Do not write factual points without citations.
12. Keep the answer focused on the question.
13. Do not claim that a source says something unless the
    supplied text supports it.

EXAMPLE FORMAT:

1. PDF/UA stands for PDF/Universal Accessibility. [Page 14]
2. PDF/UA is formally associated with ISO 14289. [Page 14]

Before responding, check that every factual point has a citation.
"""

    user_prompt = (
        f"EVIDENCE:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer only from the evidence and follow the required "
        "citation format."
    )

    answer = call_ollama(
        [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
    )

    # One repair attempt if citations are missing or invalid.
    check = citation_check(answer, evidence)

    if check["status"] == "UNVERIFIED":

        repair_prompt = f"""
Rewrite the answer below using ONLY the supplied evidence.

Do not add new facts or change its meaning.

Every factual numbered point must end with a citation
exactly like [Page 14].

Only these page numbers are allowed:
{allowed_pages_text}

If a factual statement cannot be supported, remove it.

If the evidence is insufficient, respond:
INSUFFICIENT_EVIDENCE: followed by a short explanation.

DRAFT ANSWER:
{answer}

EVIDENCE:
{context}

QUESTION:
{question}

Return only the corrected answer.
"""

        repaired = call_ollama(
            [
                {
                    "role": "system",
                    "content": (
                        "You correct citation formatting for a "
                        "document-grounded answer. Never invent evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": repair_prompt,
                },
            ],
            num_predict=500,
        )

        repaired_check = citation_check(
            repaired,
            evidence,
        )

        if repaired_check["status"] != "UNVERIFIED":
            return repaired

    return answer


# ============================================================
# CITATION CHECK
# ============================================================

def citation_check(
    answer: str,
    evidence: list[dict],
) -> dict:

    # Accept [Page 14] and [Page 14, line 12].
    cited = sorted({
        int(page)
        for page in re.findall(
            r"\[\s*Page\s+(\d+)(?:\s*,[^\]]*)?\s*\]",
            answer or "",
            flags=re.IGNORECASE,
        )
    })

    retrieved = sorted({
        int(chunk["page_number"])
        for chunk in evidence
    })

    invalid = [
        page
        for page in cited
        if page not in retrieved
    ]

    normalized = (answer or "").strip().lower()

    abstention = (
        normalized.startswith("insufficient_evidence:")
        or normalized.startswith("insufficient evidence")
    )


    if abstention:
       status = "INSUFFICIENT EVIDENCE"
    elif not cited or invalid:
       status = "UNVERIFIED"
    else:
       status = "CITATION CHECK PASSED"

    return {
        "cited_pages": cited,
        "retrieved_pages": retrieved,
        "missing": not bool(cited),
        "invalid": invalid,
        "passed": status == "CITATION CHECK PASSED",
        "status": status,
    }


# ============================================================
# STREAMLIT APPLICATION
# ============================================================

uploaded = st.file_uploader(
    "Upload a text-based PDF",
    type=["pdf"],
)

if uploaded:

    try:

        signature = (
            uploaded.name,
            uploaded.size,
        )

        if (
            st.session_state.get("document_signature")
            != signature
        ):

            with st.spinner(
                "Extracting PDF text and creating chunks..."
            ):

                chunks = extract_chunks(
                    uploaded.getvalue(),
                    uploaded.name,
                )

                if not chunks:

                    st.error(
                        "No selectable text found. "
                        "This may be a scanned PDF; OCR is not included."
                    )

                    st.stop()

                model = load_embedding_model()

                index = build_index(
                    chunks,
                    model,
                )

                st.session_state["chunks"] = chunks
                st.session_state["index"] = index

                st.session_state["document_signature"] = (
                    signature
                )

                st.session_state["document_name"] = (
                    uploaded.name
                )

        st.success(
            f"Loaded {st.session_state['document_name']} — "
            f"{len(st.session_state['chunks'])} text chunks."
        )

    except Exception as exc:

        st.error(
            f"Could not process this PDF: "
            f"{type(exc).__name__}: {exc}"
        )

        st.stop()


# ============================================================
# QUESTION ANSWERING
# ============================================================

if "chunks" in st.session_state:

    question = st.text_input(
        "Ask a question about your PDF"
    )

    if st.button(
        "Ask DocuTrust",
        type="primary",
        disabled=not bool(question.strip()),
    ):

        try:

            model = load_embedding_model()

            evidence = retrieve(
                question.strip(),
                st.session_state["chunks"],
                model,
                st.session_state["index"],
            )

            with st.expander(
                "Retrieved evidence",
                expanded=True,
            ):

                for item in evidence:

                    st.markdown(
                        f"**Page {item['page_number']} · "
                        f"semantic score {item['similarity']:.3f} · "
                        f"{item['chunk_id']}**"
                    )

                    st.write(item["text"])

            with st.spinner(
                "Generating an answer with Aya Expanse..."
            ):

                answer = ask_ollama(
                    question.strip(),
                    evidence,
                )

            check = citation_check(
                answer,
                evidence,
            )

            st.subheader("Answer")

            if check["status"] == "INSUFFICIENT EVIDENCE":

                st.info(
                    "INSUFFICIENT EVIDENCE — "
                    "the system declined to guess."
                )

                cleaned_answer = re.sub(
                    r"^\s*INSUFFICIENT_EVIDENCE:\s*",
                    "",
                    answer,
                    flags=re.IGNORECASE,
                )

                st.markdown(cleaned_answer)

            else:

                st.markdown(answer)

            st.subheader("Evidence status")

            if check["status"] == "CITATION CHECK PASSED":

                st.success("CITATION CHECK PASSED")

                st.caption(
                    "Citations refer to pages that were retrieved. "
                    "This does not prove factual support."
                )

            elif check["status"] == "INSUFFICIENT EVIDENCE":

                st.info("INSUFFICIENT EVIDENCE")

                st.caption(
                    "The model abstained instead of guessing."
                )

            else:

                st.warning(
                    "UNVERIFIED — citations are missing or "
                    "refer to pages outside retrieved evidence."
                )

            col1, col2 = st.columns(2)

            col1.metric(
                "Retrieved pages",
                ", ".join(
                    map(str, check["retrieved_pages"])
                ) or "None",
            )

            col2.metric(
                "Cited pages",
                ", ".join(
                    map(str, check["cited_pages"])
                ) or "None",
            )

            if check["invalid"]:

                st.error(
                    f"Invalid page citations: {check['invalid']}"
                )

            st.caption(
                "This automated check validates citation presence "
                "and page membership only. It does NOT prove "
                "claim-level support or factual correctness. "
                "Human review is required."
            )

        except Exception as exc:

            st.error(
                f"Error: {type(exc).__name__}: {exc}"
            )

else:

    st.info(
        "Upload a PDF to begin. "
        "The first run may download the MiniLM embedding model."
    )

    st.markdown(
        "**Local model:** Aya Expanse 8B via Ollama  ·  "
        "**Retrieval:** MiniLM + FAISS hybrid search"
    )
