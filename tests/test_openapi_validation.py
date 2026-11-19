"""The API's documented validation response matches its privacy-safe behavior."""

import httpx
import pytest

from xai_rag.api import app as api
from xai_rag.config import settings

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/query", "/query/stream", "/ingest"])
async def test_validation_schema_matches_http_response(monkeypatch, path):
    monkeypatch.setattr(settings, "api_key", "a" * 32)
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        if path == "/ingest":
            response = await client.post(
                path,
                headers={"X-API-Key": "a" * 32},
                params={"source_path": "/tmp/source", "strategy": "PRIVATE_CANARY"},
            )
        else:
            response = await client.post(
                path,
                headers={"X-API-Key": "a" * 32},
                json={"query": "PRIVATE_CANARY" * 500},
            )

    assert response.status_code == 422
    body = response.json()
    schema = api.app.openapi()["paths"][path]["post"]["responses"]["422"]["content"][
        "application/json"
    ]["schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["detail"]
    assert schema["properties"]["detail"] == {"type": "string", "const": "Invalid request"}
    assert body == {"detail": schema["properties"]["detail"]["const"]}
    assert "PRIVATE_CANARY" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["authentication", "not_ready", "disconnected"])
async def test_query_503_schema_matches_http_response(monkeypatch, failure):
    monkeypatch.setattr(settings, "api_key", "" if failure == "authentication" else "a" * 32)
    monkeypatch.setattr(api.state, "chroma_collection", None)
    monkeypatch.setattr(api.state, "es_client", None)

    if failure == "disconnected":

        class BrokenService:
            async def query(self, request, *, event_sink=None):
                raise api.ElasticsearchConnectionError("PRIVATE_PROVIDER_CANARY")

        monkeypatch.setattr(api, "_build_service", BrokenService)

    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query", headers={"X-API-Key": "a" * 32}, json={"query": "hello"}
        )

    assert response.status_code == 503
    schema = api.app.openapi()["paths"]["/query"]["post"]["responses"]["503"]["content"][
        "application/json"
    ]["schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["detail"]
    assert response.json()["detail"] in schema["properties"]["detail"]["enum"]
    assert "PRIVATE_PROVIDER_CANARY" not in response.text
