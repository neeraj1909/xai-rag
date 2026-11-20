"""Content fingerprints for an explicitly scoped, quiescent XAI-RAG index.

This module has no credentials, defaults, output writes, or access-control checks.
Caller must supply already-authorized store handles and hold writers quiescent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record(chunk_id: Any, text: Any, metadata: Any, sources: set[str]) -> tuple[str, str]:
    if (
        not isinstance(chunk_id, str)
        or not chunk_id
        or not isinstance(text, str)
        or not isinstance(metadata, Mapping)
    ):
        raise ValueError("invalid record")
    if metadata.get("source_file") not in sources:
        raise ValueError("record outside declared source scope")
    if metadata.get("content_sha256") != _sha(text.encode("utf-8")):
        raise ValueError("stored content checksum missing or invalid")
    return chunk_id, _sha(_canonical({"content": text, "metadata": metadata}))


def _vector_digest(vector: Any) -> str:
    if vector is None or not hasattr(vector, "__iter__"):
        raise ValueError("missing vector")
    values = [float(x) for x in vector]
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("invalid vector")
    # Canonical IEEE-754 single precision little endian, length framed.
    return _sha(struct.pack("<I", len(values)) + struct.pack("<" + "f" * len(values), *values))


def _manifest(records: dict[str, str]) -> str:
    return _sha(_canonical([[key, records[key]] for key in sorted(records)]))


async def snapshot(
    collection: Any,
    es: Any,
    *,
    tenant: str,
    sources: Sequence[str],
    es_index: str,
    page_size: int = 100,
) -> dict[str, Any]:
    """Return actual retrievable-record and vector digests; caller enforces access.

    Namespace binding is an explicit assertion by caller, not verified policy.
    No valid atomic snapshot exists across these stores: externally stop writers
    and refresh Elasticsearch before calling, and repeat after the evaluation.
    """
    collection_name = getattr(collection, "name", None)
    if (
        not tenant
        or not es_index
        or not collection_name
        or not sources
        or len(set(sources)) != len(sources)
        or not 1 <= page_size <= 1000
    ):
        raise ValueError("tenant, index, unique nonempty sources and valid page size required")
    if any(not isinstance(x, str) or not x for x in sources):
        raise ValueError("invalid source scope")
    source_set = set(sources)
    seen_sources: set[str] = set()
    chroma: dict[str, str] = {}
    vectors: dict[str, str] = {}
    offset = 0
    while True:
        batch = await asyncio.to_thread(
            collection.get,
            where={"source_file": {"$in": sorted(source_set)}},
            include=["documents", "metadatas", "embeddings"],
            limit=page_size,
            offset=offset,
        )
        ids, texts, metadata, embeddings = (
            batch.get(k) for k in ("ids", "documents", "metadatas", "embeddings")
        )
        if any(x is None for x in (ids, texts, metadata, embeddings)) or any(
            len(x) != len(ids) for x in (texts, metadata, embeddings)
        ):
            raise ValueError("invalid Chroma page")
        for chunk_id, text, data, vector in zip(ids, texts, metadata, embeddings, strict=True):
            chunk_id, digest = _record(chunk_id, text, data, source_set)
            seen_sources.add(data["source_file"])
            if chunk_id in chroma:
                raise ValueError("duplicate Chroma identity or unstable pagination")
            chroma[chunk_id] = digest
            vectors[chunk_id] = _vector_digest(vector)
        if len(ids) < page_size:
            break
        offset += len(ids)
    elastic: dict[str, str] = {}
    after = None
    while True:
        kwargs: dict[str, Any] = {
            "index": es_index,
            "size": page_size,
            "query": {"terms": {"source_file": sorted(source_set)}},
            "sort": [{"chunk_id": "asc"}],
            "source": ["chunk_id", "source_file", "content", "metadata"],
            "allow_partial_search_results": False,
        }
        if after is not None:
            kwargs["search_after"] = after
        result = await es.search(**kwargs)
        shards, hits_obj = result.get("_shards"), result.get("hits")
        if (
            result.get("timed_out") is not False
            or not isinstance(shards, Mapping)
            or shards.get("failed") != 0
            or not isinstance(hits_obj, Mapping)
            or not isinstance(hits_obj.get("hits"), list)
        ):
            raise ValueError("incomplete Elasticsearch search")
        hits = hits_obj["hits"]
        for hit in hits:
            source = hit.get("_source", {})
            chunk_id, digest = _record(
                source.get("chunk_id"), source.get("content"), source.get("metadata"), source_set
            )
            if (
                chunk_id != hit.get("_id")
                or source.get("source_file") != source["metadata"].get("source_file")
                or chunk_id in elastic
            ):
                raise ValueError("invalid Elasticsearch identity or source")
            elastic[chunk_id] = digest
        if len(hits) < page_size:
            break
        next_after = hits[-1].get("sort")
        if not isinstance(next_after, list) or not next_after or next_after == after:
            raise ValueError("incomplete Elasticsearch pagination")
        after = next_after
    if not chroma or not elastic:
        raise ValueError("empty scoped index")
    if seen_sources != source_set:
        raise ValueError("not every expected source appears in Chroma")
    scope_hash = _sha(
        _canonical(
            {
                "tenant": tenant,
                "sources": sorted(source_set),
                "chroma_collection": collection_name,
                "elasticsearch_index": es_index,
            }
        )
    )
    return {
        "schema_version": "xai-index-content-snapshot-v1",
        "scope_sha256": scope_hash,
        "chroma_count": len(chroma),
        "elasticsearch_count": len(elastic),
        "chroma_records_sha256": _manifest(chroma),
        "elasticsearch_records_sha256": _manifest(elastic),
        "chroma_vectors_float32_le_sha256": _manifest(vectors),
        "record_parity": chroma == elastic,
        "atomic_cross_store_snapshot": False,
        "writers_quiescent": "caller assertion; not independently enforced",
    }
