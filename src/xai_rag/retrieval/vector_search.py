"""Dense vector search using ChromaDB cosine similarity."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tenacity import retry, stop_after_attempt, wait_exponential

from xai_rag.models import SearchResult

if TYPE_CHECKING:
    import chromadb

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    reraise=True,
)
def vector_search(
    collection: chromadb.Collection,
    query_embedding: list[float],
    k: int = 100,
) -> list[SearchResult]:
    """Search for the *k* nearest documents by cosine similarity.

    Uses ChromaDB's built-in HNSW index with cosine distance.

    Parameters
    ----------
    collection:
        A ChromaDB collection configured with cosine similarity.
    query_embedding:
        The dense vector for the query (same dimensionality as stored embeddings).
    k:
        Maximum number of results to return.

    Returns
    -------
    list[SearchResult]
        Results ordered by descending cosine similarity.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    search_results: list[SearchResult] = []
    if not results["ids"] or not results["ids"][0]:
        return search_results

    for doc_id, content, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        strict=True,
    ):
        # ChromaDB cosine distance is in [0, 2]; similarity = 1 - distance
        similarity = 1.0 - distance
        search_results.append(
            SearchResult(
                id=doc_id,
                content=content,
                metadata=metadata or {},
                score=similarity,
            )
        )

    logger.debug("vector_search returned %d results (requested k=%d)", len(search_results), k)
    return search_results
