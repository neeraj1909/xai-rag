"""Dense vector search using ChromaDB cosine similarity."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from xai_rag.models import RetrievalStage, SearchResult

if TYPE_CHECKING:
    import chromadb

logger = logging.getLogger(__name__)


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

    rows = zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        strict=True,
    )
    for rank, (doc_id, content, metadata, distance) in enumerate(rows, start=1):
        # ChromaDB cosine distance is in [0, 2]; similarity = 1 - distance
        similarity = 1.0 - distance
        result_metadata = metadata or {}
        search_results.append(
            SearchResult(
                id=doc_id,
                content=content,
                context_content=result_metadata.get("parent_content"),
                metadata=result_metadata,
                score=similarity,
                stage=RetrievalStage.VECTOR,
                rank=rank,
            )
        )

    logger.debug("vector_search returned %d results (requested k=%d)", len(search_results), k)
    return search_results
