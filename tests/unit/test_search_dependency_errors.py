"""Search outages have distinct, sanitized HTTP and streaming outcomes."""

import httpx
import pytest
from elasticsearch import ConnectionError as ESConnectionError
from elasticsearch import ConnectionTimeout as ESConnectionTimeout

from xai_rag.api import app as api

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status", "stream_type"),
    [
        (ESConnectionError("PRIVATE_CANARY"), 503, "dependency_unavailable"),
        (ESConnectionTimeout("PRIVATE_CANARY"), 503, "dependency_unavailable"),
        (TimeoutError("PRIVATE_CANARY"), 504, "timeout"),
        (ValueError("PRIVATE_CANARY"), 500, "ValueError"),
    ],
)
async def test_query_dependency_errors_are_classified_and_sanitized(
    monkeypatch, caplog, error, expected_status, stream_type
):
    class BrokenService:
        async def query(self, request, *, event_sink=None):
            raise error

    monkeypatch.setitem(api.app.dependency_overrides, api.require_api_key, lambda: None)
    monkeypatch.setattr(api, "_build_service", BrokenService)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.post("/query", json={"query": "What is the policy?"})
        stream = await client.post("/query/stream", json={"query": "What is the policy?"})

    assert response.status_code == expected_status
    assert "PRIVATE_CANARY" not in response.text
    assert stream.status_code == 200
    assert '"error_type": "' + stream_type + '"' in stream.text
    assert "PRIVATE_CANARY" not in stream.text
    assert "PRIVATE_CANARY" not in caplog.text
