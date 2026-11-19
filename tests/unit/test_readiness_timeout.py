"""Readiness must fail promptly without accumulating stuck health probes."""

import asyncio
import json
import threading

import pytest

from xai_rag.api import app as api

pytestmark = pytest.mark.unit


def test_stalled_chroma_probe_is_bounded_and_shared(monkeypatch) -> None:
    released = threading.Event()
    calls = []

    class SlowChroma:
        def heartbeat(self):
            calls.append(1)
            released.wait(timeout=2)
            return 1

    class Cluster:
        async def health(self, **kwargs):
            return {"status": "yellow", "timed_out": False}

    class HealthyElasticsearch:
        cluster = Cluster()

        async def info(self):
            return {}

    monkeypatch.setattr(api.state, "chroma_client", SlowChroma())
    monkeypatch.setattr(api.state, "es_client", HealthyElasticsearch())
    monkeypatch.setattr(api, "_READINESS_TIMEOUT_SECONDS", 0.05, raising=False)

    async def run():
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(api.health(), api.health(), api.health()), timeout=0.5
            )
            assert [response.status_code for response in responses] == [503, 503, 503]
            assert all(
                json.loads(response.body)["checks"]["chromadb"] == "error" for response in responses
            )
            assert len(calls) == 1
            released.set()
            await asyncio.sleep(0.05)
            recovered = await asyncio.wait_for(api.health(), timeout=0.5)
            assert recovered.status_code == 200
            assert len(calls) == 2
        finally:
            released.set()
            await asyncio.sleep(0.05)

    asyncio.run(run())


def test_stalled_elasticsearch_probe_returns_degraded(monkeypatch) -> None:
    class HealthyChroma:
        def heartbeat(self):
            return 1

    class SlowElasticsearch:
        async def info(self):
            await asyncio.sleep(5)

    monkeypatch.setattr(api.state, "chroma_client", HealthyChroma())
    monkeypatch.setattr(api.state, "es_client", SlowElasticsearch())
    monkeypatch.setattr(api, "_READINESS_TIMEOUT_SECONDS", 0.05, raising=False)

    async def run():
        response = await asyncio.wait_for(api.health(), timeout=0.5)
        assert response.status_code == 503
        assert json.loads(response.body)["checks"]["elasticsearch"] == "error"

    asyncio.run(run())
