"""Storage — persist chunks + embeddings to ChromaDB and Elasticsearch."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

import chromadb
from elasticsearch import AsyncElasticsearch, NotFoundError

from xai_rag.config import settings

if TYPE_CHECKING:
    from xai_rag.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)


# --- ChromaDB ---


def get_chroma_client() -> chromadb.HttpClient:
    """Create an HTTP client to ChromaDB."""
    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)


def get_chroma_collection(client: chromadb.HttpClient) -> chromadb.Collection:
    """Get or create the documents collection."""
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks_chromadb(
    collection: chromadb.Collection,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
    source_file: str,
) -> list[str]:
    """Store chunks with their embeddings in ChromaDB. Returns list of chunk IDs."""
    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [chunk.content for chunk in chunks]
    metadatas = [
        {
            "chunk_index": chunk.index,
            "source_file": source_file,
            "chunk_strategy": chunk.strategy,
            "has_parent": chunk.parent_content is not None,
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(f"Stored {len(ids)} chunks from {source_file} in ChromaDB")
    return ids


def delete_by_source(collection: chromadb.Collection, source_file: str) -> int:
    """Delete all chunks for a given source file. Returns count deleted."""
    # Get matching IDs first to report count
    results = collection.get(where={"source_file": source_file})
    count = len(results["ids"])
    if count > 0:
        collection.delete(where={"source_file": source_file})
    logger.info(f"Deleted {count} chunks for {source_file}")
    return count


# --- Elasticsearch ---


async def get_es_client() -> AsyncElasticsearch:
    """Create an Elasticsearch async client."""
    client_kwargs = {}
    if settings.elasticsearch_api_key:
        client_kwargs["api_key"] = settings.elasticsearch_api_key
    return AsyncElasticsearch(settings.elasticsearch_url, **client_kwargs)


async def ensure_es_index(es: AsyncElasticsearch, index: str | None = None) -> None:
    index = index or settings.elasticsearch_index

    try:
        await es.indices.get(index=index)
        return  # index exists, do nothing
    except NotFoundError:
        logger.info("Elasticsearch index does not exist; creating: %s", index)

    await es.indices.create(
        index=index,
        settings={
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
        },
        mappings={
            "properties": {
                "content": {"type": "text", "analyzer": "standard"},
                "chunk_id": {"type": "keyword"},
                "source_file": {"type": "keyword"},
                "metadata": {"type": "object", "enabled": False},
            }
        },
    )
    logger.info(f"Created Elasticsearch index: {index}")


async def store_chunks_elasticsearch(
    es: AsyncElasticsearch,
    chunks: list[TextChunk],
    chunk_ids: list[str],
    source_file: str,
    index: str | None = None,
) -> None:
    """Index chunks in Elasticsearch for BM25 search."""
    index = index or settings.elasticsearch_index
    await ensure_es_index(es, index)

    actions = []
    for chunk, chunk_id in zip(chunks, chunk_ids, strict=True):
        actions.append({"index": {"_index": index, "_id": chunk_id}})
        actions.append(
            {
                "content": chunk.content,
                "chunk_id": chunk_id,
                "source_file": source_file,
                "metadata": {},
            }
        )

    if actions:
        await es.bulk(operations=actions, refresh=True)
        logger.info(f"Indexed {len(chunk_ids)} chunks in Elasticsearch")


async def delete_es_by_source(
    es: AsyncElasticsearch, source_file: str, index: str | None = None
) -> None:
    """Delete all Elasticsearch documents for a source file."""
    index = index or settings.elasticsearch_index
    await es.delete_by_query(
        index=index,
        query={"term": {"source_file": source_file}},
    )
