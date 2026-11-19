"""Cancellation must not free admission while inference is still running."""

import asyncio
import threading

import pytest

from xai_rag.api import app as api
from xai_rag.config import settings
from xai_rag.ingestion import embedder
from xai_rag.models import QueryRequest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_query_timeout_retains_slot_until_inference_finishes(monkeypatch):
    release = threading.Event()
    started = threading.Event()
    active_lock = threading.Lock()
    active = 0
    maximum_active = 0
    later_stages = []

    def blocked_embedding(texts, batch_size=32):
        nonlocal active, maximum_active
        with active_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        started.set()
        try:
            if not release.wait(timeout=5):
                raise TimeoutError("test worker was not released")
            return [[0.0]]
        finally:
            with active_lock:
                active -= 1

    class Service:
        async def query(self, request, *, event_sink=None):
            await embedder.embed_query(request.query)
            later_stages.append(request.query)
            return "answer"

    monkeypatch.setattr(embedder, "embed_texts_sync", blocked_embedding)
    monkeypatch.setattr(api, "_QUERY_SEMAPHORE", asyncio.Semaphore(1))
    monkeypatch.setattr(settings, "query_timeout_seconds", 0.05)
    request = QueryRequest(query="safe test")
    try:
        with pytest.raises(TimeoutError):
            await api._run_bounded_query(Service(), request)
        assert started.is_set()
        # More requests can time out waiting, but must not start more inference.
        for _ in range(2):
            with pytest.raises(TimeoutError):
                await api._run_bounded_query(Service(), request)
        assert maximum_active == 1
        assert later_stages == []
    finally:
        release.set()
        await asyncio.sleep(0.1)
    assert not api._QUERY_SEMAPHORE.locked()
    assert not api._QUERY_TASKS

    class FastService:
        async def query(self, request, *, event_sink=None):
            return "answer"

    assert await api._run_bounded_query(FastService(), request) == "answer"


@pytest.mark.asyncio
async def test_disconnect_retains_slot_and_never_continues_to_generation(monkeypatch):
    release = threading.Event()
    started = threading.Event()
    later_stages = []

    def blocked_embedding(texts, batch_size=32):
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test worker was not released")
        return [[0.0]]

    class Service:
        async def query(self, request, *, event_sink=None):
            await event_sink("retrieval", {"status": "started"})
            await embedder.embed_query(request.query)
            later_stages.append("generation")
            return "answer"

    monkeypatch.setattr(embedder, "embed_texts_sync", blocked_embedding)
    monkeypatch.setattr(api, "_QUERY_SEMAPHORE", asyncio.Semaphore(1))
    monkeypatch.setattr(api, "_build_service", Service)
    monkeypatch.setattr(settings, "query_timeout_seconds", 0.05)
    response = await api.query_stream(QueryRequest(query="safe test"))
    stream = response.body_iterator
    try:
        assert (await anext(stream))["event"] == "retrieval"
        while not started.is_set():
            await asyncio.sleep(0.001)
        await stream.aclose()
        assert api._QUERY_SEMAPHORE.locked()
        await asyncio.sleep(0.01)
        assert later_stages == []
    finally:
        release.set()
        await asyncio.sleep(0.1)
    assert not api._QUERY_SEMAPHORE.locked()
    assert later_stages == []


@pytest.mark.asyncio
async def test_repeated_cancel_drains_worker_exception_without_starting_next_stage(monkeypatch):
    release = threading.Event()
    started = threading.Event()
    later_stages = []

    def failed_embedding(texts, batch_size=32):
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test worker was not released")
        raise ValueError("private worker failure")

    class Service:
        async def query(self, request, *, event_sink=None):
            await embedder.embed_query(request.query)
            later_stages.append("generation")

    monkeypatch.setattr(embedder, "embed_texts_sync", failed_embedding)
    monkeypatch.setattr(api, "_QUERY_SEMAPHORE", asyncio.Semaphore(1))
    monkeypatch.setattr(settings, "query_timeout_seconds", 10)
    caller = asyncio.create_task(api._run_bounded_query(Service(), QueryRequest(query="safe")))
    try:
        while not started.is_set():
            await asyncio.sleep(0.001)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        for worker in list(api._QUERY_TASKS):
            worker.cancel()  # A second cancellation must not free the slot early.
        await asyncio.sleep(0.01)
        assert api._QUERY_SEMAPHORE.locked()
        assert later_stages == []
    finally:
        release.set()
        await asyncio.sleep(0.1)
    assert not api._QUERY_SEMAPHORE.locked()
    assert not api._QUERY_TASKS
    assert later_stages == []


@pytest.mark.asyncio
async def test_shutdown_cancels_admitted_work_before_closing_connections(monkeypatch):
    from xai_rag.ingestion import store

    release = threading.Event()
    started = threading.Event()
    finished = threading.Event()
    later_stages = []
    closed_after_worker = []

    def blocked_embedding(texts, batch_size=32):
        started.set()
        try:
            if not release.wait(timeout=5):
                raise TimeoutError("test worker was not released")
            return [[0.0]]
        finally:
            finished.set()

    class Service:
        async def query(self, request, *, event_sink=None):
            await embedder.embed_query(request.query)
            later_stages.append("generation")

    class Elastic:
        async def info(self):
            return {}

        async def close(self):
            closed_after_worker.append(finished.is_set())

    async def ensure_index(client):
        return None

    monkeypatch.setattr(embedder, "embed_texts_sync", blocked_embedding)
    monkeypatch.setattr(api, "_QUERY_SEMAPHORE", asyncio.Semaphore(1))
    monkeypatch.setattr(settings, "query_timeout_seconds", 10)
    monkeypatch.setattr(settings, "otel_endpoint", "")
    monkeypatch.setattr(api.chromadb, "HttpClient", lambda **kwargs: object())
    monkeypatch.setattr(api, "AsyncElasticsearch", lambda *args, **kwargs: Elastic())
    monkeypatch.setattr(store, "get_chroma_collection", lambda client: object())
    monkeypatch.setattr(store, "ensure_es_index", ensure_index)
    caller = None
    try:
        async with api.lifespan(api.app):
            caller = asyncio.create_task(
                api._run_bounded_query(Service(), QueryRequest(query="safe"))
            )
            while not started.is_set():
                await asyncio.sleep(0.001)
            asyncio.get_running_loop().call_later(0.05, release.set)
        with pytest.raises(asyncio.CancelledError):
            await caller
    finally:
        release.set()
        if caller is not None and not caller.done():
            caller.cancel()
    assert closed_after_worker == [True]
    assert later_stages == []
    assert not api._QUERY_TASKS


@pytest.mark.asyncio
async def test_actual_service_vector_work_remains_admitted_after_timeout(monkeypatch):
    from xai_rag.service import RAGService

    release = threading.Event()
    started = threading.Event()
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    later_stages = []

    async def embed(query):
        return [0.0]

    def vector(collection, embedding, k):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        started.set()
        try:
            if not release.wait(timeout=5):
                raise TimeoutError("test worker was not released")
            return []
        finally:
            with lock:
                active -= 1

    async def bm25(client, query, index, k):
        return []

    def fusion(result_lists, k):
        later_stages.append("fusion")
        return []

    service = RAGService(
        collection=object(),
        es_client=object(),
        embed_query_fn=embed,
        vector_search_fn=vector,
        bm25_search_fn=bm25,
        fusion_fn=fusion,
    )
    monkeypatch.setattr(api, "_QUERY_SEMAPHORE", asyncio.Semaphore(1))
    monkeypatch.setattr(settings, "query_timeout_seconds", 0.05)
    request = QueryRequest(query="safe test", include_explanations=False)
    try:
        with pytest.raises(TimeoutError):
            await api._run_bounded_query(service, request)
        assert started.is_set()
        with pytest.raises(TimeoutError):
            await api._run_bounded_query(service, request)
        assert maximum_active == 1
        assert later_stages == []
    finally:
        release.set()
        await asyncio.sleep(0.1)
    assert not api._QUERY_SEMAPHORE.locked()
    assert later_stages == []
