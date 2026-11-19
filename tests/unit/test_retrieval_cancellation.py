"""Retrieval failures must not leave untracked vector work running."""

import asyncio
import threading

import pytest

from xai_rag.models import QueryRequest
from xai_rag.service import RAGService

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_failed_bm25_waits_for_running_vector_work_before_returning():
    started = threading.Event()
    release = threading.Event()
    bm25_failed = asyncio.Event()

    async def embed(_query):
        return [0.1]

    def vector(*_args):
        started.set()
        assert release.wait(5), "test did not release the vector worker"
        return []

    async def failed_bm25(*_args):
        assert await asyncio.to_thread(started.wait, 3)
        bm25_failed.set()
        raise RuntimeError("BM25 unavailable")

    service = RAGService(
        collection=object(),
        es_client=object(),
        embed_query_fn=embed,
        vector_search_fn=vector,
        bm25_search_fn=failed_bm25,
    )
    task = asyncio.create_task(service.query(QueryRequest(query="example question")))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        await asyncio.wait_for(bm25_failed.wait(), 3)
        await asyncio.sleep(0)
        assert not task.done(), "retrieval returned while vector work was still running"
    finally:
        release.set()
        if task.done() and not task.cancelled():
            task.exception()
    with pytest.raises(RuntimeError, match="BM25 unavailable"):
        await task


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_vector_worker():
    started = threading.Event()
    release = threading.Event()

    async def embed(_query):
        return [0.1]

    def vector(*_args):
        started.set()
        assert release.wait(5), "test did not release the vector worker"
        return []

    async def bm25(*_args):
        await asyncio.Event().wait()
        return []

    service = RAGService(
        collection=object(),
        es_client=object(),
        embed_query_fn=embed,
        vector_search_fn=vector,
        bm25_search_fn=bm25,
    )
    task = asyncio.create_task(service.query(QueryRequest(query="example question")))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done(), "cancellation returned while the vector worker was active"
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
