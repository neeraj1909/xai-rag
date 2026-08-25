"""BM25 full-text search via Elasticsearch async client."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tenacity import retry, stop_after_attempt, wait_exponential

from xai_rag.models import SearchResult

if TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    reraise=True,
)
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
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        results.append(
            SearchResult(
                id=hit["_id"],
                content=source.get("content", ""),
                metadata=source.get("metadata", {}),
                score=float(hit["_score"]),
            )
        )

    logger.debug("bm25_search returned %d results (requested k=%d)", len(results), k)
    return results
