"""Hybrid search with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from xai_rag.models import SearchResult
from xai_rag.retrieval.bm25_search import bm25_search
from xai_rag.retrieval.vector_search import vector_search

if TYPE_CHECKING:
    import asyncpg
    from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)


def rrf_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    For each document *d* appearing in any list, the fused score is::

        score(d) = sum( 1 / (k + rank_i(d)) )  for each list i where d appears

    A higher fused score means the document consistently ranks well across
    multiple retrieval strategies.

    Parameters
    ----------
    result_lists:
        Two or more ranked result lists (e.g. vector + BM25).
    k:
        RRF constant that dampens the contribution of low-ranked results.
        Default 60 matches the value proposed in the original RRF paper.

    Returns
    -------
    list[SearchResult]
        Deduplicated results sorted by descending RRF score.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    result_map: dict[str, SearchResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            rrf_scores[result.id] += 1.0 / (k + rank)
            # Keep the copy with the highest original score for content/metadata.
            if result.id not in result_map or result.score > result_map[result.id].score:
                result_map[result.id] = result

    # Build final list with RRF score replacing the original score.
    fused: list[SearchResult] = []
    for doc_id, fused_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        original = result_map[doc_id]
        fused.append(
            SearchResult(
                id=original.id,
                content=original.content,
                metadata=original.metadata,
                score=fused_score,
            )
        )

    logger.debug(
        "rrf_fusion merged %d lists -> %d unique documents",
        len(result_lists),
        len(fused),
    )
    return fused


async def hybrid_search(
    pool: asyncpg.Pool,
    es_client: AsyncElasticsearch,
    query: str,
    query_embedding: list[float],
    k: int = 100,
    *,
    es_index: str = "chunks",
    rrf_k: int = 60,
) -> tuple[list[SearchResult], list[SearchResult], list[SearchResult]]:
    """Run vector + BM25 retrieval and fuse with RRF.

    Parameters
    ----------
    pool:
        asyncpg connection pool for pgvector.
    es_client:
        Async Elasticsearch client.
    query:
        Natural-language query string.
    query_embedding:
        Dense embedding of *query*.
    k:
        Number of candidates from each retrieval stage.
    es_index:
        Elasticsearch index name.
    rrf_k:
        RRF constant.

    Returns
    -------
    tuple[list[SearchResult], list[SearchResult], list[SearchResult]]
        ``(vector_results, bm25_results, fused_results)`` so callers can
        use individual stage outputs for explainability.
    """
    import asyncio

    vector_results, bm25_results = await asyncio.gather(
        vector_search(pool, query_embedding, k=k),
        bm25_search(es_client, query, index=es_index, k=k),
    )

    fused_results = rrf_fusion([vector_results, bm25_results], k=rrf_k)

    logger.info(
        "hybrid_search: vector=%d, bm25=%d, fused=%d",
        len(vector_results),
        len(bm25_results),
        len(fused_results),
    )
    return vector_results, bm25_results, fused_results
