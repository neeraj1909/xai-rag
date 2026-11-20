"""Concurrent ingestion must not mix versions of one logical source in one process."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from xai_rag.ingestion.chunker import TextChunk
from xai_rag.service import IngestionBusyError, RAGService

pytestmark = pytest.mark.unit


def _service(chroma, elastic, parity_fn=None):
    async def embed(texts):
        return [[1.0] for _ in texts]

    async def parity(collection, es):
        return SimpleNamespace(consistent=True)

    return RAGService(
        collection=object(),
        es_client=object(),
        parse_file_fn=lambda path: path.read_text(),
        chunk_text_fn=lambda text, strategy, *_: [TextChunk(text, 0, 0, len(text), strategy)],
        embed_texts_fn=embed,
        store_chroma_fn=chroma,
        store_elasticsearch_fn=elastic,
        parity_fn=parity_fn or parity,
    )


def _document(root: Path, name: str, text: str) -> Path:
    root.mkdir(exist_ok=True)
    p = root / name
    p.write_text(text)
    return p


@pytest.mark.asyncio
async def test_batch_lock_covers_writes_and_whole_index_parity(tmp_path):
    root = tmp_path / "docs"
    a = _document(root, "a.txt", "A")
    b = _document(root, "b.txt", "B")
    parity_started = asyncio.Event()
    release_parity = asyncio.Event()
    writes = []
    chroma_records = set()
    elastic_records = set()
    parity_calls = 0

    def chroma(_collection, chunks, _embeddings, source):
        writes.append(source)
        chroma_records.add(source)
        return [source]

    async def elastic(_es, _chunks, _ids, source, _index):
        elastic_records.add(source)

    async def parity(_collection, _es):
        nonlocal parity_calls
        parity_calls += 1
        if parity_calls == 1:
            parity_started.set()
            await release_parity.wait()
        return SimpleNamespace(consistent=chroma_records == elastic_records)

    first = asyncio.create_task(_service(chroma, elastic, parity).ingest(a, document_root=root))
    await asyncio.wait_for(parity_started.wait(), 2)
    try:
        with pytest.raises(IngestionBusyError):
            await _service(chroma, elastic, parity).ingest(b, document_root=root)
        assert writes == ["a.txt"]
    finally:
        release_parity.set()
        result = await first
    retry = await _service(chroma, elastic, parity).ingest(b, document_root=root)
    results = [result, retry]
    assert all(result.status == "success" and result.parity_consistent for result in results)
    assert writes == ["a.txt", "b.txt"]
    assert chroma_records == elastic_records == {"a.txt", "b.txt"}


@pytest.mark.asyncio
async def test_cancelled_ingest_drains_chroma_before_next_version(tmp_path):
    root = tmp_path / "docs"
    source = _document(root, "shared.txt", "A")
    thread_started = threading.Event()
    release_thread = threading.Event()
    writes = []

    def chroma(_collection, chunks, _embeddings, _source):
        writes.append(chunks[0].content)
        if len(writes) == 1:
            thread_started.set()
            assert release_thread.wait(timeout=5)
        return ["id"]

    async def elastic(*_args):
        pass

    first = asyncio.create_task(_service(chroma, elastic).ingest(source, document_root=root))
    assert await asyncio.to_thread(thread_started.wait, 2)
    first.cancel()
    source.write_text("B")
    try:
        with pytest.raises(IngestionBusyError):
            await _service(chroma, elastic).ingest(source, document_root=root)
        assert writes == ["A"]
    finally:
        release_thread.set()
        (first_result,) = await asyncio.gather(first, return_exceptions=True)
    second_result = await _service(chroma, elastic).ingest(source, document_root=root)
    assert isinstance(first_result, asyncio.CancelledError)
    assert not isinstance(second_result, BaseException)
    assert second_result.status == "success" and second_result.parity_consistent
    assert writes == ["A", "B"]


@pytest.mark.asyncio
@pytest.mark.parametrize("journaled", [False, True])
async def test_cancelled_chunking_holds_admission_until_thread_finishes(tmp_path, journaled):
    root = tmp_path / "docs"
    source = _document(root, "one.txt", "A")
    started = threading.Event()
    release = threading.Event()
    writes = []
    journal = tmp_path / "journal"
    journal.mkdir(mode=0o700)

    def chunk(text, strategy, *_args):
        started.set()
        assert release.wait(timeout=5)
        return [TextChunk(text, 0, 0, len(text), strategy)]

    service = RAGService(
        collection=SimpleNamespace(name="test-collection"),
        es_client=object(),
        ingestion_journal_root=journal if journaled else None,
        parse_file_fn=lambda path: path.read_text(),
        chunk_text_fn=chunk,
        store_chroma_fn=lambda *_: writes.append("write"),
    )
    first = asyncio.create_task(service.ingest(source, document_root=root))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(IngestionBusyError):
            await service.ingest(source, document_root=root)
        assert not first.done()
        assert writes == []
    finally:
        release.set()
        (result,) = await asyncio.gather(first, return_exceptions=True)
    assert isinstance(result, asyncio.CancelledError)
    assert writes == []


@pytest.mark.asyncio
async def test_cancelled_discovery_holds_admission_until_thread_finishes(tmp_path, monkeypatch):
    from xai_rag import service as service_module

    root = tmp_path / "docs"
    source = _document(root, "one.txt", "A")
    started = threading.Event()
    release = threading.Event()

    def discover(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=5)
        return [source]

    monkeypatch.setattr(service_module, "discover_document_paths", discover)

    async def elastic(*_args):
        pass

    service = _service(lambda *_: [], elastic)
    first = asyncio.create_task(service.ingest(source, document_root=root))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        first.cancel()
        await asyncio.sleep(0)
        with pytest.raises(IngestionBusyError):
            await service.ingest(source, document_root=root)
        assert not first.done()
    finally:
        release.set()
        (result,) = await asyncio.gather(first, return_exceptions=True)
    assert isinstance(result, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_cancelled_ingest_drains_elasticsearch_before_next_version(tmp_path):
    root = tmp_path / "docs"
    source = _document(root, "shared.txt", "A")
    first_in_es = asyncio.Event()
    release = asyncio.Event()
    writes = []

    def chroma(_collection, chunks, _embeddings, _source):
        writes.append(chunks[0].content)
        return [chunks[0].content]

    async def elastic(_es, chunks, _ids, _source, _index):
        if chunks[0].content == "A":
            first_in_es.set()
            await release.wait()

    first = asyncio.create_task(_service(chroma, elastic).ingest(source, document_root=root))
    await asyncio.wait_for(first_in_es.wait(), 2)
    first.cancel()
    source.write_text("B")
    try:
        with pytest.raises(IngestionBusyError):
            await _service(chroma, elastic).ingest(source, document_root=root)
        assert writes == ["A"]
    finally:
        release.set()
        (first_result,) = await asyncio.gather(first, return_exceptions=True)
    second_result = await _service(chroma, elastic).ingest(source, document_root=root)
    assert isinstance(first_result, asyncio.CancelledError)
    assert not isinstance(second_result, BaseException)
    assert second_result.status == "success" and second_result.parity_consistent
    assert writes == ["A", "B"]


@pytest.mark.asyncio
async def test_idle_batch_lock_registry_releases_unused_entries():
    from xai_rag.service import _ingestion_lock, _ingestion_locks

    loop = asyncio.get_running_loop()
    lock = _ingestion_lock()
    async with lock:
        assert loop in _ingestion_locks
    del lock
    assert loop not in _ingestion_locks


@pytest.mark.asyncio
async def test_many_contenders_are_rejected_without_queueing(tmp_path):
    root = tmp_path / "docs"
    source = _document(root, "shared.txt", "A")
    first_in_es = asyncio.Event()
    release = asyncio.Event()

    def chroma(*_args):
        return ["id"]

    async def elastic(*_args):
        first_in_es.set()
        await release.wait()

    service = _service(chroma, elastic)
    first = asyncio.create_task(service.ingest(source, document_root=root))
    await asyncio.wait_for(first_in_es.wait(), 2)
    try:
        attempts = await asyncio.gather(
            *(service.ingest(source, document_root=root) for _ in range(30)),
            return_exceptions=True,
        )
        assert all(isinstance(result, IngestionBusyError) for result in attempts)
        assert not first.done()
    finally:
        release.set()
        await first


def test_http_busy_is_retryable_without_internal_details(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from xai_rag.api import app as api
    from xai_rag.config import settings

    class BusyService:
        async def ingest(self, *_args, **_kwargs):
            raise IngestionBusyError("private internal details")

    root = tmp_path / "docs"
    source = _document(root, "shared.txt", "A")
    monkeypatch.setattr(settings, "api_key", "a" * 32)
    monkeypatch.setattr(settings, "auth_policy_file", "")
    monkeypatch.setattr(settings, "ingest_root", str(root))
    monkeypatch.setattr(api, "_build_service", lambda: BusyService())
    monkeypatch.setattr(api, "_RATE_LIMIT_BUCKETS", {})
    client = TestClient(api.app)
    response = client.post(
        "/ingest", params={"source_path": str(source)}, headers={"X-API-Key": "a" * 32}
    )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert "private internal details" not in response.text


@pytest.mark.asyncio
async def test_injected_generator_requires_prompt_evidence_contract():
    from xai_rag.models import QueryRequest

    class OldGenerator:
        async def generate(self, query, chunks):
            raise AssertionError("unsafe legacy generation must not be used")

    class EmptyReranker:
        async def rerank(self, query, results, top_k):
            return []

    async def embed_query(query):
        return [1.0]

    async def bm25(client, query, index, k):
        return []

    service = RAGService(
        collection=object(),
        es_client=object(),
        embed_query_fn=embed_query,
        vector_search_fn=lambda *_args: [],
        bm25_search_fn=bm25,
        fusion_fn=lambda *_args: [],
        reranker_factory=EmptyReranker,
        generator_factory=OldGenerator,
    )
    with pytest.raises(
        TypeError, match="generator_factory must provide async generate_with_evidence"
    ):
        await service.query(QueryRequest(query="question", include_explanations=False))
