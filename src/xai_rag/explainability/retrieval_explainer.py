"""Retrieval explainer — attributes each selected chunk back to search stages."""

from __future__ import annotations

import logging
import re

from xai_rag.models import RankedResult, RetrievalExplanation, SearchResult

logger = logging.getLogger(__name__)

# Minimum term length to consider for matching (filters stopwords/noise).
_MIN_TERM_LENGTH = 3


def _tokenize(text: str) -> set[str]:
    """Lowercase tokenization with basic punctuation stripping."""
    return {t for t in re.findall(r"\b\w+\b", text.lower()) if len(t) >= _MIN_TERM_LENGTH}


class RetrievalExplainer:
    """Generates per-chunk explanations tracing back through the retrieval pipeline.

    For every chunk that survives reranking, the explainer records its score
    at each stage (vector, BM25, RRF, reranker) and produces a human-readable
    ``selection_reason`` summarising *why* the chunk was selected.
    """

    def explain(
        self,
        query: str,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        reranked_results: list[RankedResult],
    ) -> list[RetrievalExplanation]:
        """Build explanations for each reranked chunk.

        Parameters
        ----------
        query:
            The original user query.
        vector_results:
            Raw results from dense vector search.
        bm25_results:
            Raw results from BM25 / Elasticsearch.
        reranked_results:
            Final results after hybrid fusion and cross-encoder reranking.

        Returns
        -------
        list[RetrievalExplanation]
            One explanation per reranked chunk, in the same order.
        """
        # Index original stage scores by chunk id for O(1) lookup.
        vector_scores: dict[str, float] = {r.id: r.score for r in vector_results}
        bm25_scores: dict[str, float] = {r.id: r.score for r in bm25_results}

        query_terms = _tokenize(query)

        explanations: list[RetrievalExplanation] = []
        for ranked in reranked_results:
            chunk_terms = _tokenize(ranked.content)
            matching = sorted(query_terms & chunk_terms)

            v_score = vector_scores.get(ranked.id, 0.0)
            b_score = bm25_scores.get(ranked.id, 0.0)

            reason = self._build_reason(
                chunk_id=ranked.id,
                v_score=v_score,
                b_score=b_score,
                reranker_score=ranked.reranker_score,
                matching_terms=matching,
            )

            explanations.append(
                RetrievalExplanation(
                    chunk_id=ranked.id,
                    content_preview=ranked.content[:200],
                    vector_score=v_score,
                    bm25_score=b_score,
                    rrf_rank=ranked.rrf_rank,
                    reranker_score=ranked.reranker_score,
                    matching_terms=matching,
                    selection_reason=reason,
                )
            )

        logger.debug("Generated %d retrieval explanations", len(explanations))
        return explanations

    @staticmethod
    def _build_reason(
        *,
        chunk_id: str,
        v_score: float,
        b_score: float,
        reranker_score: float,
        matching_terms: list[str],
    ) -> str:
        """Compose a human-readable selection reason."""
        parts: list[str] = []

        # Identify which stages contributed.
        in_vector = v_score > 0.0
        in_bm25 = b_score > 0.0

        if in_vector and in_bm25:
            parts.append(
                f"Chunk {chunk_id} appeared in both vector search "
                f"(similarity={v_score:.4f}) and BM25 (score={b_score:.2f}), "
                "boosting its RRF fusion rank."
            )
        elif in_vector:
            parts.append(
                f"Chunk {chunk_id} was found via semantic vector search "
                f"(similarity={v_score:.4f}) but not by keyword BM25."
            )
        elif in_bm25:
            parts.append(
                f"Chunk {chunk_id} was found via BM25 keyword search "
                f"(score={b_score:.2f}) but not by dense vector search."
            )

        if matching_terms:
            terms_str = ", ".join(f'"{t}"' for t in matching_terms[:10])
            parts.append(f"Matching query terms: {terms_str}.")

        parts.append(f"Cross-encoder reranker confirmed relevance with score {reranker_score:.4f}.")

        return " ".join(parts)
