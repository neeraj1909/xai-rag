"""BM25 full-text search via Elasticsearch async client."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from xai_rag.models import RetrievalStage, SearchResult

if TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)


async def bm25_search(
    es_client: AsyncElasticsearch,
    query: str,
    index: str,
    k: int = 100,
) -> list[SearchResult]:
    """Run a BM25 multi-match query against the Elasticsearch index.

    Parameters
    ----------
    es_client:
        An ``elasticsearch.AsyncElasticsearch`` instance.
    query:
        The user's natural-language query string.
    index:
        Elasticsearch index name containing document chunks.
    k:
        Maximum number of results.

    Returns
    -------
    list[SearchResult]
        Chunks ordered by descending BM25 score.
    """
    response = await es_client.search(
        index=index,
        size=k,
        query={
            "multi_match": {
                "query": query,
                "fields": ["content"],
                "type": "best_fields",
                "tie_breaker": 0.3,
            }
        },
        source=["content", "metadata"],
    )

    results: list[SearchResult] = []
    for rank, hit in enumerate(response["hits"]["hits"], start=1):
        source = hit["_source"]
        metadata = source.get("metadata", {})
        results.append(
            SearchResult(
                id=hit["_id"],
                content=source.get("content", ""),
                context_content=metadata.get("parent_content"),
                metadata=metadata,
                score=float(hit["_score"]),
                stage=RetrievalStage.BM25,
                rank=rank,
            )
        )

    logger.debug("bm25_search returned %d results (requested k=%d)", len(results), k)
    return results
