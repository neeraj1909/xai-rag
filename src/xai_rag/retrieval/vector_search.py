"""Dense vector search using pgvector cosine distance."""

from __future__ import annotations

import json 
import logging
from typing import TYPE_CHECKING

from tenacity import retry, stop_after_attempt, wait_exponential

from xai_rag.models import SearchResult

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    reraise=True,
)
async def vector_search(
    pool: asyncpg.Pool,
    query_embedding: list[float],
    k: int = 100,
    *,
    table: str = "documents",
    embedding_column: str = "embedding",
) -> list[SearchResult]:
    """Search for the *k* nearest documents by cosine distance.

    Uses the pgvector ``<=>`` operator which returns ``1 - cosine_similarity``
    so lower values are better.  We convert to a similarity score in [0, 1].

    Parameters
    ----------
    pool:
        An ``asyncpg`` connection pool connected to a pgvector-enabled database.
    query_embedding:
        The dense vector for the query (same dimensionality as stored embeddings).
    k:
        Maximum number of results to return.
    table:
        Name of the documents table.
    embedding_column:
        Name of the embedding column.

    Returns
    -------
    list[SearchResult]
        Results ordered by descending cosine similarity.
    """
    # pgvector <=> returns cosine *distance* (1 - similarity).
    query = f"""
        SELECT id::text, content, metadata,
               1 - ({embedding_column} <=> $1::vector) AS similarity
        FROM {table}
        ORDER BY {embedding_column} <=> $1::vector
        LIMIT $2
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, str(query_embedding), k)

    results: list[SearchResult] = []
    for row in rows:
        results.append(
            SearchResult(
                id=row["id"],
                content=row["content"],
                metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
                score=float(row["similarity"]),
            )
        )

    logger.debug("vector_search returned %d results (requested k=%d)", len(results), k)
    return results
