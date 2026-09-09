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
        # Keep fallbacks for callers deserializing older ranked records.
        vector_by_id = {result.id: result for result in vector_results}
        bm25_by_id = {result.id: result for result in bm25_results}

        query_terms = _tokenize(query)

        explanations: list[RetrievalExplanation] = []
        for ranked in reranked_results:
            chunk_terms = _tokenize(ranked.content)
            matching = sorted(query_terms & chunk_terms)

            vector_fallback = vector_by_id.get(ranked.id)
            bm25_fallback = bm25_by_id.get(ranked.id)
            v_score = ranked.vector_score
            v_rank = ranked.vector_rank
            b_score = ranked.bm25_score
            b_rank = ranked.bm25_rank
            if v_rank is None and vector_fallback is not None:
                v_score = vector_fallback.score
                v_rank = vector_fallback.rank or vector_results.index(vector_fallback) + 1
            if b_rank is None and bm25_fallback is not None:
                b_score = bm25_fallback.score
                b_rank = bm25_fallback.rank or bm25_results.index(bm25_fallback) + 1

            reason = self._build_reason(
                chunk_id=ranked.id,
                v_score=v_score,
                v_rank=v_rank,
                b_score=b_score,
                b_rank=b_rank,
                reranker_score=ranked.reranker_score or 0.0,
                matching_terms=matching,
            )

            explanations.append(
                RetrievalExplanation(
                    chunk_id=ranked.id,
                    content_preview=ranked.content[:200],
                    vector_score=v_score,
                    vector_rank=v_rank,
                    bm25_score=b_score,
                    bm25_rank=b_rank,
                    rrf_score=ranked.rrf_score,
                    rrf_rank=ranked.rrf_rank,
                    reranker_score=ranked.reranker_score,
                    reranker_rank=ranked.reranker_rank,
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
        v_score: float | None,
        v_rank: int | None,
        b_score: float | None,
        b_rank: int | None,
        reranker_score: float,
        matching_terms: list[str],
    ) -> str:
        """Compose a human-readable selection reason."""
        parts: list[str] = []

        # Identify which stages contributed.
        in_vector = v_rank is not None
        in_bm25 = b_rank is not None

        if in_vector and in_bm25:
            parts.append(
                f"Chunk {chunk_id} appeared in both vector search "
                f"(similarity={v_score or 0.0:.4f}) and BM25 (score={b_score or 0.0:.2f}), "
                "boosting its RRF fusion rank."
            )
        elif in_vector:
            parts.append(
                f"Chunk {chunk_id} was found via semantic vector search "
                f"(similarity={v_score or 0.0:.4f}) but not by keyword BM25."
            )
        elif in_bm25:
            parts.append(
                f"Chunk {chunk_id} was found via BM25 keyword search "
                f"(score={b_score or 0.0:.2f}) but not by dense vector search."
            )

        if matching_terms:
            terms_str = ", ".join(f'"{t}"' for t in matching_terms[:10])
            parts.append(f"Matching query terms: {terms_str}.")

        parts.append(f"Cross-encoder reranker assigned relevance score {reranker_score:.4f}.")

        return " ".join(parts)
