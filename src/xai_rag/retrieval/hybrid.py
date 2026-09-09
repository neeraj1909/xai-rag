"""Hybrid search with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from xai_rag.config import settings
from xai_rag.models import FusedResult, RetrievalStage, SearchResult
from xai_rag.retrieval.bm25_search import bm25_search
from xai_rag.retrieval.vector_search import vector_search

if TYPE_CHECKING:
    import chromadb
    from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)


def rrf_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[FusedResult]:
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
    list[FusedResult]
        Deduplicated results sorted by descending RRF score.
    """
    if k < 1:
        raise ValueError("k must be greater than zero")

    rrf_scores: dict[str, float] = defaultdict(float)
    result_map: dict[str, SearchResult] = {}
    vector_values: dict[str, tuple[float, int]] = {}
    bm25_values: dict[str, tuple[float, int]] = {}

    for list_index, result_list in enumerate(result_lists):
        seen_in_list: set[str] = set()
        for rank, result in enumerate(result_list, start=1):
            if result.id in seen_in_list:
                continue
            seen_in_list.add(result.id)

            stage_rank = result.rank or rank
            rrf_scores[result.id] += 1.0 / (k + stage_rank)
            if result.id not in result_map:
                result_map[result.id] = result

            stage = result.stage
            if stage is None and list_index == 0:
                stage = RetrievalStage.VECTOR
            elif stage is None and list_index == 1:
                stage = RetrievalStage.BM25

            if stage == RetrievalStage.VECTOR and result.id not in vector_values:
                vector_values[result.id] = (result.score, stage_rank)
            elif stage == RetrievalStage.BM25 and result.id not in bm25_values:
                bm25_values[result.id] = (result.score, stage_rank)

    ordered = sorted(rrf_scores.items(), key=lambda item: (-item[1], item[0]))
    fused: list[FusedResult] = []
    for rrf_rank, (doc_id, fused_score) in enumerate(ordered, start=1):
        original = result_map[doc_id]
        vector_score, vector_rank = vector_values.get(doc_id, (None, None))
        bm25_score, bm25_rank = bm25_values.get(doc_id, (None, None))
        fused.append(
            FusedResult(
                id=original.id,
                content=original.content,
                context_content=original.context_content,
                metadata=original.metadata,
                vector_score=vector_score,
                vector_rank=vector_rank,
                bm25_score=bm25_score,
                bm25_rank=bm25_rank,
                rrf_score=fused_score,
                rrf_rank=rrf_rank,
            )
        )

    logger.debug(
        "rrf_fusion merged %d lists -> %d unique documents",
        len(result_lists),
        len(fused),
    )
    return fused


async def hybrid_search(
    collection: chromadb.Collection,
    es_client: AsyncElasticsearch,
    query: str,
    query_embedding: list[float],
    k: int = 100,
    *,
    es_index: str | None = None,
    rrf_k: int = 60,
) -> tuple[list[SearchResult], list[SearchResult], list[FusedResult]]:
    """Run vector + BM25 retrieval and fuse with RRF.

    Parameters
    ----------
    collection:
        ChromaDB collection for vector search.
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
    tuple[list[SearchResult], list[SearchResult], list[FusedResult]]
        ``(vector_results, bm25_results, fused_results)`` so callers can
        use individual stage outputs for explainability.
    """
    resolved_index = es_index or settings.elasticsearch_index
    vector_results, bm25_results = await asyncio.gather(
        asyncio.to_thread(vector_search, collection, query_embedding, k),
        bm25_search(es_client, query, index=resolved_index, k=k),
    )

    fused_results = rrf_fusion([vector_results, bm25_results], k=rrf_k)

    logger.info(
        "hybrid_search: vector=%d, bm25=%d, fused=%d",
        len(vector_results),
        len(bm25_results),
        len(fused_results),
    )
    return vector_results, bm25_results, fused_results
