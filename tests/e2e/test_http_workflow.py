"""Black-box checks against the deployed Compose HTTP boundary."""

from __future__ import annotations

import asyncio
import os
import statistics
import time

import httpx
import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.getenv("XAI_RAG_RUN_E2E") != "1",
        reason="set XAI_RAG_RUN_E2E=1 against the disposable Compose stack",
    ),
]

BASE_URL = os.getenv("XAI_RAG_E2E_BASE_URL", "http://127.0.0.1:18000")
API_KEY = os.getenv(
    "XAI_RAG_E2E_API_KEY",
    "e2e-only-api-key-with-at-least-32-characters",
)
HEADERS = {"X-API-Key": API_KEY}
QUERY = "What does the document explain about retrieval-augmented generation?"


@pytest.mark.asyncio
async def test_ingest_query_stream_and_bounded_concurrency() -> None:
    timeout = httpx.Timeout(240.0)
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=timeout) as client:
        assert (await client.get("/live")).json() == {"status": "alive"}
        ready = await client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "healthy"

        unauthorized = await client.post("/query", json={"query": QUERY})
        assert unauthorized.status_code == 401

        ingest_params = {
            "source_path": "/app/sample_docs/rag-interview-notes.md",
            "strategy": "fixed",
        }
        first_ingest = await client.post("/ingest", params=ingest_params, headers=HEADERS)
        assert first_ingest.status_code == 200, first_ingest.text
        first_ingest_body = first_ingest.json()
        assert first_ingest_body["status"] == "success"
        assert first_ingest_body["total_chunks"] > 0
        assert first_ingest_body["parity_consistent"] is True

        replay = await client.post("/ingest", params=ingest_params, headers=HEADERS)
        assert replay.status_code == 200, replay.text
        assert replay.json()["total_chunks"] == first_ingest_body["total_chunks"]
        assert replay.json()["parity_consistent"] is True

        response = await client.post(
            "/query",
            headers=HEADERS,
            json={"query": QUERY, "top_k": 2, "include_explanations": True},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        explanation_ids = {
            explanation["chunk_id"] for explanation in body["retrieval_explanations"]
        }
        assert body["sources"]
        assert set(body["sources"]) <= explanation_ids
        assert f"[{body['sources'][0]}]" in body["answer"]
        assert body["input_tokens"] == 32
        assert body["output_tokens"] == 16
        assert {
            "embed",
            "vector",
            "bm25",
            "retrieve",
            "fusion",
            "rerank",
            "explain",
            "generate",
        } <= set(body["stage_latency_ms"])

        async with client.stream(
            "POST",
            "/query/stream",
            headers=HEADERS,
            json={"query": QUERY, "top_k": 2},
        ) as stream:
            assert stream.status_code == 200
            stream_body = "".join([chunk async for chunk in stream.aiter_text()])
        assert "event: retrieval" in stream_body
        assert "event: generation" in stream_body
        assert "event: done" in stream_body
        assert "event: error" not in stream_body

        disabled_evaluation = await client.post(
            "/query",
            headers=HEADERS,
            json={"query": QUERY, "include_ragas": True},
        )
        assert disabled_evaluation.status_code == 400

        async def timed_query() -> tuple[int, float]:
            started = time.perf_counter()
            result = await client.post(
                "/query",
                headers=HEADERS,
                json={"query": QUERY, "top_k": 1, "include_explanations": False},
            )
            return result.status_code, (time.perf_counter() - started) * 1000

        load_results = await asyncio.gather(*(timed_query() for _ in range(4)))
        assert [status for status, _ in load_results] == [200, 200, 200, 200]
        latencies = sorted(latency for _, latency in load_results)
        print(
            "e2e_concurrency_ms",
            {"p50": statistics.median(latencies), "max": max(latencies)},
        )
