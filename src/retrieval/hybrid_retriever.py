
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def build_hybrid_index(chunks, embedding_model):
    """
    Build dense semantic and sparse TF-IDF indexes.
    """

    if not chunks:
        raise ValueError("Cannot index an empty document.")

    texts = [item["text"] for item in chunks]

    # Dense semantic embeddings
    dense_vectors = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")

    # Sparse lexical index
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    sparse_matrix = vectorizer.fit_transform(texts)

    return dense_vectors, vectorizer, sparse_matrix


def extract_section_number(question):
    """
    Extract section references such as:
    section 1.3
    section 2.2.1
    """

    match = re.search(
        r"\bsection\s+(\d+(?:\.\d+)+)\b",
        question,
        re.IGNORECASE
    )

    return match.group(1) if match else None


def has_section_heading(text, section_number):
    """
    Identify an actual section heading.

    Ignore table-of-contents entries containing
    dotted leaders or page-number references.
    """

    if not section_number:
        return False

    lines = text.splitlines()

    pattern = (
        r"^\s*"
        + re.escape(section_number)
        + r"\s*[.)\-:]?\s+(.+?)\s*$"
    )

    for i, line in enumerate(lines):

        line = line.strip()

        match = re.match(pattern, line)

        if not match:
            continue

        # Ignore TOC dotted leaders.
        if re.search(r"\.{3,}|…{2,}", line):
            continue

        heading_text = match.group(1).strip()

        # Ignore entries ending with page numbers.
        if re.search(r"\.{2,}\s*\d+\s*$", line):
            continue

        # A heading must have meaningful text.
        if len(heading_text) < 2:
            continue

        # Require supporting body text after the heading.
        following_lines = lines[i + 1:i + 6]

        following_text = " ".join(
            item.strip() for item in following_lines
        ).strip()

        if len(following_text) < 20:
            continue

        return True

    return False


def retrieve_hybrid(
    question,
    chunks,
    embedding_model,
    dense_vectors,
    vectorizer,
    sparse_matrix,
    top_k=4,
    rrf_k=60
):
    """
    Hybrid retrieval using:
    1. Dense semantic similarity
    2. Sparse TF-IDF similarity
    3. Reciprocal Rank Fusion
    4. Exact section-heading priority
    """

    if not chunks:
        return []

    # -----------------------------
    # 1. Dense retrieval
    # -----------------------------

    query_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    dense_scores = dense_vectors @ query_vector

    dense_rank = np.argsort(dense_scores)[::-1]

    # -----------------------------
    # 2. Sparse retrieval
    # -----------------------------

    sparse_query = vectorizer.transform([question])

    sparse_scores = (
        sparse_matrix @ sparse_query.T
    ).toarray().ravel()

    # Ignore zero lexical similarity.
    nonzero_sparse = np.flatnonzero(
        sparse_scores > 0
    )

    sparse_rank = sorted(
        nonzero_sparse,
        key=lambda position: sparse_scores[position],
        reverse=True
    )

    # -----------------------------
    # 3. Reciprocal Rank Fusion
    # -----------------------------

    fused_scores = {}

    for ranking in (dense_rank, sparse_rank):

        for rank, position in enumerate(
            ranking,
            start=1
        ):

            position = int(position)

            fused_scores[position] = (
                fused_scores.get(position, 0.0)
                + 1.0 / (rrf_k + rank)
            )

    # -----------------------------
    # 4. Exact section priority
    # -----------------------------

    requested_section = extract_section_number(
        question
    )

    section_matches = {}

    if requested_section:

        for position, chunk in enumerate(chunks):

            matched = has_section_heading(
                chunk.get("text", ""),
                requested_section
            )

            section_matches[position] = matched

            if matched:

                # Section heading boost.
                fused_scores[position] = (
                    fused_scores.get(position, 0.0)
                    + 0.10
                )

    # -----------------------------
    # 5. Final ranking
    # -----------------------------

    ranked_positions = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )

    results = []

    for position in ranked_positions[:top_k]:

        item = dict(chunks[position])

        item["similarity"] = float(
            dense_scores[position]
        )

        item["retrieval_score"] = float(
            fused_scores[position]
        )

        item["section_match"] = section_matches.get(
            position,
            False
        )

        results.append(item)

    return results
