"""Cross-encoder reranker for second-stage ranking."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import TYPE_CHECKING

from xai_rag.models import RankedResult, SearchResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Shared thread pool for CPU-bound model inference.
_RERANKER_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reranker")

from xai_rag.config import settings

_DEFAULT_MODEL = settings.reranker_model


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    """Lazily load and cache the CrossEncoder model (singleton)."""
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder model: %s", model_name)
    return CrossEncoder(model_name)


class Reranker:
    """Cross-encoder reranker wrapping ``sentence-transformers.CrossEncoder``.

    The underlying model is loaded lazily on first call to :meth:`rerank` and
    cached for the process lifetime.  Inference runs in a thread-pool executor
    so the async event loop is never blocked.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name

    @property
    def model(self) -> CrossEncoder:
        """Return the (lazily-loaded) CrossEncoder instance."""
        return _load_cross_encoder(self._model_name)

    def _score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """Run cross-encoder inference synchronously (called inside executor)."""
        pairs = [[query, p] for p in passages]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RankedResult]:
        """Rerank search results with the cross-encoder model.

        Parameters
        ----------
        query:
            The user's natural-language query.
        results:
            Candidate ``SearchResult`` objects (typically from hybrid search).
        top_k:
            Number of top results to return after reranking.

        Returns
        -------
        list[RankedResult]
            Top *top_k* results sorted by descending reranker score, with
            original scores preserved for explainability.
        """
        if not results:
            return []

        passages = [r.content for r in results]
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            _RERANKER_POOL,
            self._score_pairs,
            query,
            passages,
        )

        # Build ranked results preserving original retrieval metadata.
        ranked: list[RankedResult] = []
        for idx, (result, reranker_score) in enumerate(zip(results, scores)):
            ranked.append(
                RankedResult(
                    id=result.id,
                    content=result.content,
                    metadata=result.metadata,
                    vector_score=result.score if "vector" not in result.metadata else result.score,
                    bm25_score=0.0,
                    rrf_rank=idx + 1,
                    reranker_score=reranker_score,
                )
            )

        ranked.sort(key=lambda r: r.reranker_score, reverse=True)
        top_results = ranked[:top_k]

        # Update rrf_rank to reflect final ordering.
        for pos, r in enumerate(top_results, start=1):
            r.rrf_rank = pos

        logger.debug(
            "Reranker: %d candidates -> top %d (best=%.4f)",
            len(results),
            len(top_results),
            top_results[0].reranker_score if top_results else 0.0,
        )
        return top_results
