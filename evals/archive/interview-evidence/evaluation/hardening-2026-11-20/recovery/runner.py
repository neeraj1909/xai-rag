"""Disposable real-store recovery check; no provider or model calls."""
import asyncio
import os
import sys
from pathlib import Path
from xai_rag.authorization import AccessScope
from xai_rag.config import settings
from xai_rag.ingestion.chunker import TextChunk
from xai_rag.ingestion.store import (
    check_index_parity, get_chroma_client, get_chroma_collection,
    get_es_client, store_chunks_elasticsearch,
)
from xai_rag.service import RAGService

root = Path(sys.argv[2])
source = root / "docs" / "original.txt"
state = root / "journal"
scope = AccessScope("test-tenant", ("original.txt",), True)

def chunk(text, strategy, *_args):
    return [TextChunk(text, 0, 0, len(text), strategy)]

async def embed(texts):
    return [[0.01] * settings.embedding_dimension for _ in texts]

async def crash_after_chroma(*_args):
    os._exit(29)

async def run():
    collection = get_chroma_collection(get_chroma_client(), "journal-integration-test")
    es = (await get_es_client()).options(request_timeout=60)
    try:
        service = RAGService(
            collection=collection, es_client=es, index_name="journal-integration-test",
            tenant="test-tenant", ingestion_journal_root=state,
            parse_file_fn=lambda path: path.read_text(), chunk_text_fn=chunk, embed_texts_fn=embed,
            store_elasticsearch_fn=crash_after_chroma if sys.argv[1] == "crash" else store_chunks_elasticsearch,
        )
        if sys.argv[1] == "crash":
            await service.ingest(source, document_root=source.parent, strategy="fixed", access_scope=scope)
            raise AssertionError("process did not crash")
        await service.recover_ingestion(access_scope=scope, acknowledge_quiescence=True)
        data = collection.get(include=["documents"])
        assert data["documents"] == ["original staged data\n"], data
        report = await check_index_parity(collection, es, index="journal-integration-test")
        assert report.consistent, report
        print("REAL_STORE_RECOVERY_OK", report.to_dict())
    finally:
        await es.close()

asyncio.run(run())
