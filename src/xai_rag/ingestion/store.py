"""Storage — persist chunks + embeddings to pgvector and Elasticsearch."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import asyncpg
from elasticsearch import AsyncElasticsearch

from xai_rag.config import settings
from xai_rag.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)


# --- pgvector ---


async def get_pg_pool() -> asyncpg.Pool:
    """Create a connection pool to PostgreSQL with pgvector."""
    dsn = settings.database_url_sync  # asyncpg uses non-async DSN format
    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
    return pool


async def store_chunks_pgvector(
    pool: asyncpg.Pool,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
    source_file: str,
) -> list[str]:
    """Store chunks with their embeddings in pgvector. Returns list of chunk IDs."""
    ids = []
    async with pool.acquire() as conn:
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

            await conn.execute(
                """
                INSERT INTO documents (id, content, chunk_index, parent_id, embedding,
                                       metadata, source_file, chunk_strategy)
                VALUES ($1::uuid, $2, $3, $4, $5::vector, $6::jsonb, $7, $8)
                """,
                uuid.UUID(chunk_id),
                chunk.content,
                chunk.index,
                None,  # parent_id — set in a second pass for parent_doc strategy
                embedding_str,
                json.dumps({"has_parent": chunk.parent_content is not None}),
                source_file,
                chunk.strategy,
            )
            ids.append(chunk_id)

    logger.info(f"Stored {len(ids)} chunks from {source_file} in pgvector")
    return ids


async def delete_by_source(pool: asyncpg.Pool, source_file: str) -> int:
    """Delete all chunks for a given source file. Returns count deleted."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM documents WHERE source_file = $1", source_file
        )
        count = int(result.split()[-1])
        logger.info(f"Deleted {count} chunks for {source_file}")
        return count


# --- Elasticsearch ---


async def get_es_client() -> AsyncElasticsearch:
    """Create an Elasticsearch async client."""
    return AsyncElasticsearch(settings.elasticsearch_url)


async def ensure_es_index(es: AsyncElasticsearch, index: str | None = None) -> None:
    index = index or settings.elasticsearch_index
    
    try:
        await es.indices.get(index=index)
        return  # index exists, do nothing
    except Exception:
        pass  # index doesn't exist, create it

    await es.indices.create(
        index=index,
        settings={                          # ? direct kwarg, not body=
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
        },
        mappings={                          # ? direct kwarg, not body=
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
    for chunk, chunk_id in zip(chunks, chunk_ids):
        actions.append({"index": {"_index": index, "_id": chunk_id}})
        actions.append({
            "content": chunk.content,
            "chunk_id": chunk_id,
            "source_file": source_file,
            "metadata": {},
        })

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
