"""Storage — persist chunks + embeddings to ChromaDB and Elasticsearch."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import chromadb
from elasticsearch import AsyncElasticsearch, NotFoundError

from xai_rag.config import settings

if TYPE_CHECKING:
    from xai_rag.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)

_CHUNK_ID_SCHEMA_VERSION = 1


class StorageWriteError(RuntimeError):
    """Raised when a backing index reports a partial or failed write."""


class IndexCompatibilityError(RuntimeError):
    """Raised when stored embeddings were built with incompatible settings."""


class IndexNotReadyError(RuntimeError):
    """Raised when the Elasticsearch index has no active primary shard."""


@dataclass(frozen=True)
class IndexParityReport:
    """Comparison of deterministic chunk records in both retrieval stores."""

    chroma_count: int
    elasticsearch_count: int
    missing_in_chroma: tuple[str, ...]
    missing_in_elasticsearch: tuple[str, ...]
    content_mismatches: tuple[str, ...]

    @property
    def consistent(self) -> bool:
        return not (
            self.missing_in_chroma or self.missing_in_elasticsearch or self.content_mismatches
        )

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "consistent": self.consistent}


def build_chunk_id(source_file: str, chunk: TextChunk) -> str:
    """Build a deterministic content/provenance identity for idempotent upserts."""
    identity = {
        "schema_version": _CHUNK_ID_SCHEMA_VERSION,
        "source_file": source_file,
        "chunk_index": chunk.index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "chunk_strategy": chunk.strategy,
        "content": chunk.content,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def get_index_fingerprint() -> str:
    """Fingerprint the settings that define vector-index compatibility."""
    manifest = {
        "schema_version": _CHUNK_ID_SCHEMA_VERSION,
        "embedding_model": settings.embedding_model,
        "embedding_model_revision": settings.embedding_model_revision,
        "embedding_dimension": settings.embedding_dimension,
        "distance": "cosine",
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def require_compatible_fingerprint(actual: str | None, *, store_name: str) -> None:
    """Fail closed when an existing index lacks or mismatches its manifest."""
    expected = get_index_fingerprint()
    if actual != expected:
        observed = actual or "missing"
        raise IndexCompatibilityError(
            f"{store_name} index fingerprint is {observed}; expected {expected}. Reindex required."
        )


def _chunk_metadata(chunk: TextChunk, source_file: str) -> dict[str, str | int | bool]:
    metadata: dict[str, str | int | bool] = {
        "schema_version": _CHUNK_ID_SCHEMA_VERSION,
        "chunk_index": chunk.index,
        "source_file": source_file,
        "chunk_strategy": chunk.strategy,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "has_parent": chunk.parent_content is not None,
        "content_sha256": hashlib.sha256(chunk.content.encode("utf-8")).hexdigest(),
        "embedding_model": settings.embedding_model,
        "embedding_model_revision": settings.embedding_model_revision,
        "embedding_dimension": settings.embedding_dimension,
        "index_fingerprint": get_index_fingerprint(),
    }
    if chunk.parent_content is not None:
        metadata["parent_content"] = chunk.parent_content
    if chunk.parent_start_char is not None:
        metadata["parent_start_char"] = chunk.parent_start_char
    if chunk.parent_end_char is not None:
        metadata["parent_end_char"] = chunk.parent_end_char
    return metadata


def compare_index_records(
    chroma_records: dict[str, str | None],
    elasticsearch_records: dict[str, str | None],
) -> IndexParityReport:
    """Compare ID/content fingerprints without mutating either index."""
    chroma_ids = set(chroma_records)
    elasticsearch_ids = set(elasticsearch_records)
    common_ids = chroma_ids & elasticsearch_ids
    mismatches = sorted(
        chunk_id
        for chunk_id in common_ids
        if chroma_records[chunk_id] != elasticsearch_records[chunk_id]
    )
    return IndexParityReport(
        chroma_count=len(chroma_ids),
        elasticsearch_count=len(elasticsearch_ids),
        missing_in_chroma=tuple(sorted(elasticsearch_ids - chroma_ids)),
        missing_in_elasticsearch=tuple(sorted(chroma_ids - elasticsearch_ids)),
        content_mismatches=tuple(mismatches),
    )


# --- ChromaDB ---


def get_chroma_client() -> chromadb.HttpClient:
    """Create an HTTP client to ChromaDB."""
    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)


def get_chroma_collection(client: chromadb.HttpClient) -> chromadb.Collection:
    """Get or create the documents collection."""
    expected_fingerprint = get_index_fingerprint()
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine", "index_fingerprint": expected_fingerprint},
    )
    actual_fingerprint = (collection.metadata or {}).get("index_fingerprint")
    if actual_fingerprint is None and collection.count() == 0:
        collection.modify(
            metadata={**(collection.metadata or {}), "index_fingerprint": expected_fingerprint}
        )
    else:
        require_compatible_fingerprint(actual_fingerprint, store_name="ChromaDB")
    return collection


def store_chunks_chromadb(
    collection: chromadb.Collection,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
    source_file: str,
) -> list[str]:
    """Store chunks with their embeddings in ChromaDB. Returns list of chunk IDs."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    ids = [build_chunk_id(source_file, chunk) for chunk in chunks]
    documents = [chunk.content for chunk in chunks]
    metadatas = [_chunk_metadata(chunk, source_file) for chunk in chunks]
    existing = collection.get(where={"source_file": source_file})
    existing_ids = set(existing.get("ids", []))

    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
    stale_ids = sorted(existing_ids - set(ids))
    if stale_ids:
        collection.delete(ids=stale_ids)

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
        response = await es.indices.get(index=index)
        if response:
            index_definition = next(iter(response.values()))
            actual_fingerprint = (
                index_definition.get("mappings", {}).get("_meta", {}).get("index_fingerprint")
            )
            require_compatible_fingerprint(actual_fingerprint, store_name="Elasticsearch")
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
                "_meta": {"index_fingerprint": get_index_fingerprint()},
                "properties": {
                    "content": {"type": "text", "analyzer": "standard"},
                    "chunk_id": {"type": "keyword"},
                    "source_file": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": False},
                },
            },
        )
        logger.info("Created Elasticsearch index: %s", index)

    health = await es.cluster.health(
        index=index,
        wait_for_status="yellow",
        timeout="10s",
    )
    if health.get("timed_out") or str(health.get("status", "red")) == "red":
        raise IndexNotReadyError(f"Elasticsearch index {index} has no active primary shard")


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
                "metadata": _chunk_metadata(chunk, source_file),
            }
        )

    if not actions:
        await delete_es_by_source(es, source_file, index)
        return

    response = await es.bulk(operations=actions, refresh="wait_for")
    if response.get("errors", False):
        error_types: list[str] = []
        for item in response.get("items", []):
            operation = next(iter(item.values()), {})
            error = operation.get("error")
            if isinstance(error, dict):
                error_types.append(str(error.get("type", "unknown")))
            elif error:
                error_types.append(type(error).__name__)
        summary = ", ".join(sorted(set(error_types))) or "unknown"
        raise StorageWriteError(f"Elasticsearch bulk write failed: {summary}")

    await es.delete_by_query(
        index=index,
        query={
            "bool": {
                "filter": [{"term": {"source_file": source_file}}],
                "must_not": [{"ids": {"values": chunk_ids}}],
            }
        },
        conflicts="proceed",
        refresh=True,
    )
    logger.info("Indexed %d chunks in Elasticsearch", len(chunk_ids))


async def delete_es_by_source(
    es: AsyncElasticsearch, source_file: str, index: str | None = None
) -> None:
    """Delete all Elasticsearch documents for a source file."""
    index = index or settings.elasticsearch_index
    await es.delete_by_query(
        index=index,
        query={"term": {"source_file": source_file}},
    )


async def _elasticsearch_records(
    es: AsyncElasticsearch,
    *,
    index: str,
    source_file: str | None,
) -> dict[str, str | None]:
    """Read all chunk IDs and fingerprints using stable search-after pagination."""
    query = {"term": {"source_file": source_file}} if source_file else {"match_all": {}}
    search_after: list[object] | None = None
    records: dict[str, str | None] = {}

    while True:
        kwargs: dict[str, object] = {
            "index": index,
            "size": 1_000,
            "query": query,
            "sort": [{"chunk_id": "asc"}],
            "source": ["chunk_id", "metadata"],
        }
        if search_after is not None:
            kwargs["search_after"] = search_after
        response = await es.search(**kwargs)
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            break
        for hit in hits:
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            chunk_id = str(source.get("chunk_id") or hit["_id"])
            records[chunk_id] = metadata.get("content_sha256")
        search_after = hits[-1].get("sort")
        if search_after is None or len(hits) < 1_000:
            break

    return records


async def check_index_parity(
    collection: chromadb.Collection,
    es: AsyncElasticsearch,
    *,
    source_file: str | None = None,
    index: str | None = None,
) -> IndexParityReport:
    """Read and compare the Chroma and Elasticsearch chunk manifests."""
    get_kwargs: dict[str, object] = {"include": ["metadatas"]}
    if source_file:
        get_kwargs["where"] = {"source_file": source_file}
    chroma_response = await asyncio.to_thread(collection.get, **get_kwargs)
    chroma_ids = chroma_response.get("ids", [])
    chroma_metadatas = chroma_response.get("metadatas") or [{} for _ in chroma_ids]
    chroma_records = {
        str(chunk_id): metadata.get("content_sha256")
        for chunk_id, metadata in zip(chroma_ids, chroma_metadatas, strict=True)
    }
    elasticsearch_records = await _elasticsearch_records(
        es,
        index=index or settings.elasticsearch_index,
        source_file=source_file,
    )
    return compare_index_records(chroma_records, elasticsearch_records)
