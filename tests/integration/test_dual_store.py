"""Integration proof for Chroma/Elasticsearch persistence and retrieval."""

from __future__ import annotations

import os

import pytest

from xai_rag.config import settings
from xai_rag.ingestion.chunker import TextChunk
from xai_rag.ingestion.store import (
    check_index_parity,
    get_chroma_client,
    get_chroma_collection,
    get_es_client,
    store_chunks_chromadb,
    store_chunks_elasticsearch,
)
from xai_rag.retrieval.bm25_search import bm25_search
from xai_rag.retrieval.hybrid import hybrid_search
from xai_rag.retrieval.vector_search import vector_search

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("XAI_RAG_RUN_INTEGRATION") != "1",
        reason="set XAI_RAG_RUN_INTEGRATION=1 against disposable services",
    ),
]


def _embedding(*values: float) -> list[float]:
    """Build a deterministic vector matching the configured index dimension."""
    return [*values, *([0.0] * (settings.embedding_dimension - len(values)))]


@pytest.mark.asyncio
async def test_dual_store_idempotency_parity_and_real_retrieval() -> None:
    """Exercise both persistence engines without model or provider calls."""
    chroma_client = get_chroma_client()
    collection = get_chroma_collection(chroma_client)
    es_client = await get_es_client()

    source_file = "integration/document.txt"
    chunks = [
        TextChunk(
            content="Quantum entanglement correlates measurements across particles.",
            index=0,
            start_char=0,
            end_char=64,
            strategy="fixed",
            parent_content="A short reference about quantum physics.",
            parent_start_char=0,
            parent_end_char=40,
        ),
        TextChunk(
            content="Photosynthesis converts light energy into chemical energy.",
            index=1,
            start_char=65,
            end_char=124,
            strategy="fixed",
        ),
    ]
    embeddings = [_embedding(1.0, 0.0), _embedding(0.0, 1.0)]

    try:
        chunk_ids = store_chunks_chromadb(collection, chunks, embeddings, source_file)
        await store_chunks_elasticsearch(
            es_client,
            chunks,
            chunk_ids,
            source_file,
            settings.elasticsearch_index,
        )

        parity = await check_index_parity(
            collection,
            es_client,
            source_file=source_file,
            index=settings.elasticsearch_index,
        )
        assert parity.consistent
        assert parity.chroma_count == parity.elasticsearch_count == 2

        vector_results = vector_search(collection, _embedding(1.0, 0.0), k=2)
        bm25_results = await bm25_search(
            es_client,
            "quantum entanglement particles",
            settings.elasticsearch_index,
            k=2,
        )
        assert vector_results[0].id == chunk_ids[0]
        assert vector_results[0].context_content == "A short reference about quantum physics."
        assert bm25_results[0].id == chunk_ids[0]

        vector_stage, bm25_stage, fused = await hybrid_search(
            collection,
            es_client,
            "quantum entanglement particles",
            _embedding(1.0, 0.0),
            k=2,
            es_index=settings.elasticsearch_index,
        )
        assert vector_stage[0].id == bm25_stage[0].id == fused[0].id == chunk_ids[0]
        assert fused[0].vector_rank == fused[0].bm25_rank == 1

        replayed_ids = store_chunks_chromadb(collection, chunks[:1], embeddings[:1], source_file)
        await store_chunks_elasticsearch(
            es_client,
            chunks[:1],
            replayed_ids,
            source_file,
            settings.elasticsearch_index,
        )
        replayed_parity = await check_index_parity(
            collection,
            es_client,
            source_file=source_file,
            index=settings.elasticsearch_index,
        )
        assert replayed_ids == chunk_ids[:1]
        assert replayed_parity.consistent
        assert replayed_parity.chroma_count == replayed_parity.elasticsearch_count == 1
    finally:
        await es_client.indices.delete(index=settings.elasticsearch_index, ignore_unavailable=True)
        await es_client.close()
        chroma_client.delete_collection(settings.chroma_collection)
