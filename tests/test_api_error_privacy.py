"""HTTP validation and startup failures do not expose sensitive input."""

import traceback
from pathlib import Path

import httpx
import pytest
from fastapi.exceptions import RequestValidationError

from xai_rag.api import app as api
from xai_rag.config import settings

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "request_kwargs", "canary"),
    [
        ("/query", {"json": {"query": "QUERY_CANARY_" + "x" * 4001}}, "QUERY_CANARY_"),
        (
            "/ingest",
            {"params": {"source_path": "/tmp/document", "strategy": "STRATEGY_CANARY"}},
            "STRATEGY_CANARY",
        ),
    ],
)
async def test_validation_response_omits_input(monkeypatch, endpoint, request_kwargs, canary):
    monkeypatch.setattr(settings, "api_key", "a" * 32)
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(endpoint, headers={"X-API-Key": "a" * 32}, **request_kwargs)

    assert response.status_code == 422
    assert canary not in response.text
    assert response.json() == {"detail": "Invalid request"}


@pytest.mark.asyncio
async def test_validation_handler_does_not_reflect_arbitrary_error_details():
    error = RequestValidationError(
        [{"loc": ("body", "LOC_CANARY"), "msg": "MSG_CANARY", "type": "TYPE_CANARY"}]
    )
    response = await api.validation_error_handler(None, error)
    assert b"CANARY" not in response.body


@pytest.mark.asyncio
async def test_lifespan_traceback_suppresses_provider_error(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("PROVIDER_SECRET_CANARY")

    async def no_delay(*args):
        pass

    monkeypatch.setattr(api.chromadb, "HttpClient", broken)
    monkeypatch.setattr(api.asyncio, "sleep", no_delay)
    monkeypatch.setattr(api.state, "es_client", None)
    with pytest.raises(RuntimeError) as caught:
        async with api.lifespan(api.app):
            pass

    assert "PROVIDER_SECRET_CANARY" not in "".join(traceback.format_exception(caught.value))
    assert "PROVIDER_SECRET_CANARY" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_make_serve_disables_access_log():
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
    serve = makefile.split("\nserve:\n", 1)[1].split("\n\n", 1)[0]
    assert "--no-access-log" in serve
