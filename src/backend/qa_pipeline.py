
"""
DocuTrust AI: reusable, per-upload PDF question-answering pipeline.

Supports UI chunks:
    {"id": "...", "page": 1, "text": "..."}

And evaluation chunks:
    {"chunk_id": "...", "page_number": 1, "text": "..."}

Citation validation checks page membership only. It does NOT prove
that every generated claim is semantically entailed by the evidence.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from sentence_transformers import SentenceTransformer

from src.retrieval.hybrid_retriever import (
    build_hybrid_index,
    retrieve_hybrid,
)
from src.evaluation.verify_answer import verify_answer


class DocumentQAPipeline:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        ollama_model: str = "aya-expanse:8b",
        ollama_url: str = "http://localhost:11434/api/chat",
        top_k: int = 4,
        embedding_model: Any | None = None,
    ) -> None:
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.top_k = top_k
        self.embedding_model = embedding_model or SentenceTransformer(
            embedding_model_name
        )
        self.chunks: list[dict[str, Any]] = []
        self.dense_vectors = None
        self.vectorizer = None
        self.sparse_matrix = None
        self.ready = False

    @staticmethod
    def _normalise_chunks(
        chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        normalised = []

        for index, source in enumerate(chunks):
            text = str(source.get("text", "")).strip()
            if not text:
                continue

            page = source.get("page_number", source.get("page", 1))

            try:
                page = int(page)
            except (TypeError, ValueError):
                page = 1

            chunk_id = (
                source.get("chunk_id")
                or source.get("id")
                or f"UPLOAD_P{page}_C{index + 1}"
            )

            item = dict(source)
            item["text"] = text
            item["page_number"] = page
            item["chunk_id"] = str(chunk_id)
            normalised.append(item)

        if not normalised:
            raise ValueError("No usable text chunks were supplied.")

        return normalised

    def index_document(self, chunks: list[dict[str, Any]]) -> int:
        """Build a fresh hybrid index for one uploaded PDF."""
        self.chunks = self._normalise_chunks(chunks)

        (
            self.dense_vectors,
            self.vectorizer,
            self.sparse_matrix,
        ) = build_hybrid_index(self.chunks, self.embedding_model)

        self.ready = True
        return len(self.chunks)

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        if not self.ready:
            raise RuntimeError("Index a PDF before asking questions.")

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        return retrieve_hybrid(
            question=question.strip(),
            chunks=self.chunks,
            embedding_model=self.embedding_model,
            dense_vectors=self.dense_vectors,
            vectorizer=self.vectorizer,
            sparse_matrix=self.sparse_matrix,
            top_k=self.top_k,
        )

    def _ollama_chat(
        self,
        messages: list[dict[str, str]],
        num_predict: int = 350,
    ) -> str:
        """Send a non-streaming chat request to Ollama."""
        payload = {
            "model": self.ollama_model,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": 0.1,
                "num_predict": num_predict,
            },
        }

        request = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))

            return result["message"]["content"].strip()

        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama HTTP {exc.code}: {details}"
            ) from exc

        except Exception as exc:
            raise RuntimeError(
                f"Ollama request failed ({type(exc).__name__}): {exc}"
            ) from exc

    @staticmethod
    def _build_context(evidence: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            f"[Page {item['page_number']} | {item['chunk_id']}]\n"
            f"{item['text']}"
            for item in evidence
        )

    def _generate(
        self,
        question: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        context = self._build_context(evidence)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are DocuTrust AI, a strict document-grounded assistant. "
                    "Answer only from the supplied evidence. Do not use outside "
                    "knowledge or invent facts. Place an inline citation immediately after every factual sentence using "
                    "the exact format [Page N]. Only cite pages whose supplied "
                    "passages support the claim. Never put citations in a separate section. Never add standards, names, numbers, or details unless they appear in the supplied evidence. If the evidence is insufficient, "
                    "say so clearly. Keep the answer concise. Do not invent page "
                    "numbers, standards, statistics, or procedures."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"DOCUMENT EVIDENCE:\n{context}\n\n"
                    f"QUESTION:\n{question}"
                ),
            },
        ]

        return self._ollama_chat(messages, num_predict=350)

    def _repair_citations(
        self,
        question: str,
        answer: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        """
        Make one repair attempt when citations are missing or invalid.
        The model must use only page numbers present in the evidence.
        """
        context = self._build_context(evidence)

        allowed_pages = sorted(
            {int(item["page_number"]) for item in evidence}
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are repairing citations in a document-grounded answer. "
                    "Use ONLY the supplied document evidence. Preserve the "
                    "original answer's meaning, but remove any claim that cannot "
                    "be supported by the supplied evidence. Every factual claim "
                    "must have a citation in the exact format [Page N]. "
                    f"Allowed page numbers: {allowed_pages}. "
                    "Never invent a page number. Do not add new facts. "
                    "If the evidence cannot answer the question, respond with "
                    "INSUFFICIENT_EVIDENCE. Return only the corrected answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"ORIGINAL ANSWER:\n{answer}\n\n"
                    f"DOCUMENT EVIDENCE:\n{context}\n\n"
                    "Repair the answer's citations using the evidence above."
                ),
            },
        ]

        return self._ollama_chat(messages, num_predict=350)

    def ask(self, question: str) -> dict[str, Any]:
        """Retrieve evidence, generate an answer, repair citations and verify."""
        evidence = self.retrieve(question)

        if not evidence:
            return {
                "answer": (
                    "The uploaded document does not contain "
                    "sufficient evidence."
                ),
                "sources": [],
                "verification": {
                    "citations_valid": False,
                    "missing_citation": True,
                    "invalid_pages": [],
                    "text_overlap": 0.0,
                    "verification_note": "No evidence was retrieved.",
                },
                "citation_repair_attempted": False,
                "status": "insufficient_evidence",
            }

        answer = self._generate(question, evidence)

        if ("INSUFFICIENT_EVIDENCE" in answer.upper() or "INSUFFICIENT EVIDENCE" in answer.upper() or "does not contain any information" in answer.lower() or "does not include any information" in answer.lower()):
            verification = verify_answer(answer, evidence)
            status = "insufficient_evidence"
            repair_attempted = False

        else:
            verification = verify_answer(answer, evidence)
            repair_attempted = False

            if (
                verification["missing_citation"]
                or verification["invalid_pages"]
            ):
                repair_attempted = True

                try:
                    repaired_answer = self._repair_citations(
                        question,
                        answer,
                        evidence,
                    )

                    if repaired_answer:
                        answer = repaired_answer

                    verification = verify_answer(answer, evidence)

                except Exception as exc:
                    # Preserve the original answer if repair fails.
                    verification["citation_repair_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

            if ("INSUFFICIENT_EVIDENCE" in answer.upper() or "INSUFFICIENT EVIDENCE" in answer.upper() or "does not contain any information" in answer.lower() or "does not include any information" in answer.lower()):
                status = "insufficient_evidence"
            elif (
                verification["missing_citation"]
                or verification["invalid_pages"]
            ):
                status = "unverified"
            else:
                status = "citation_check_passed"

        sources = [
            {
                "chunk_id": item["chunk_id"],
                "page": item["page_number"],
                "text": item["text"],
                "retrieval_score": item.get("retrieval_score"),
            }
            for item in evidence
        ]

        return {
            "answer": answer,
            "sources": sources,
            "verification": verification,
            "citation_repair_attempted": repair_attempted,
            "status": status,
        }
