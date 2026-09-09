"""Regression tests for the API request/response model contract."""

import asyncio
import json

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from xai_rag.api import app as api
from xai_rag.config import Settings, settings
from xai_rag.models import QueryRequest, XAIRAGResponse

pytestmark = pytest.mark.unit


def test_query_request_defaults_match_api_usage() -> None:
    request = QueryRequest(query="What is hybrid search?")

    assert request.query == "What is hybrid search?"
    assert request.top_k == 5
    assert request.include_explanations is True
    assert request.include_faithfulness is False
    assert request.include_ragas is False


def test_xai_rag_response_accepts_pipeline_output() -> None:
    response = XAIRAGResponse(
        query="What is hybrid search?",
        answer="It combines dense and lexical retrieval.",
        latency_ms=12.5,
    )

    assert response.answer.startswith("It combines")
    assert response.claims == []
    assert response.sources == []
    assert response.retrieval_explanations == []
    assert response.faithfulness_report == []
    assert response.ragas_scores == {}


def test_api_module_imports_with_declared_models() -> None:
    assert api.app.title == "XAI-RAG"


def test_query_endpoint_delegates_to_shared_service(monkeypatch) -> None:
    calls: list[QueryRequest] = []

    class FakeService:
        async def query(self, request, *, event_sink=None):
            calls.append(request)
            return XAIRAGResponse(query=request.query, answer="test answer")

    monkeypatch.setattr(api, "_build_service", FakeService)

    response = asyncio.run(
        api.query_endpoint(
            QueryRequest(query="test query", include_explanations=False),
        )
    )

    assert response.answer == "test answer"
    assert calls[0].query == "test query"


def test_query_failure_log_does_not_capture_exception_content(monkeypatch, caplog) -> None:
    class BrokenService:
        async def query(self, request, *, event_sink=None):
            raise RuntimeError("private query content")

    monkeypatch.setattr(api, "_build_service", BrokenService)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api.query_endpoint(QueryRequest(query="test query")))

    assert exc_info.value.status_code == 500
    assert "private query content" not in caplog.text


def test_service_factory_fails_with_explicit_readiness_error(monkeypatch) -> None:
    monkeypatch.setattr(api.state, "chroma_collection", None)
    monkeypatch.setattr(api.state, "es_client", None)

    with pytest.raises(HTTPException) as exc_info:
        api._build_service()

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Search dependencies are not ready"


def test_liveness_does_not_depend_on_search_services() -> None:
    assert asyncio.run(api.live()) == {"status": "alive"}


def test_query_request_rejects_unbounded_query() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="x" * 4001)


def test_api_key_dependency_fails_closed_without_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", "")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api.require_api_key(None))

    assert exc_info.value.status_code == 503


def test_api_key_dependency_rejects_invalid_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", "s" * 32)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api.require_api_key("wrong-key"))

    assert exc_info.value.status_code == 401


def test_health_response_does_not_expose_dependency_errors(monkeypatch, caplog) -> None:
    class BrokenChroma:
        def heartbeat(self):
            raise RuntimeError("internal chroma secret")

    class BrokenElasticsearch:
        async def info(self):
            raise RuntimeError("internal elasticsearch secret")

    monkeypatch.setattr(api.state, "chroma_client", BrokenChroma())
    monkeypatch.setattr(api.state, "es_client", BrokenElasticsearch())

    response = asyncio.run(api.health())
    body = json.loads(response.body)

    assert response.status_code == 503
    assert body["checks"] == {"chromadb": "error", "elasticsearch": "error"}
    assert "secret" not in response.body.decode()
    assert "secret" not in caplog.text


def test_readiness_rejects_an_elasticsearch_index_without_active_primaries(monkeypatch) -> None:
    class HealthyChroma:
        def heartbeat(self):
            return 1

    class RedCluster:
        async def health(self, **kwargs):
            return {"status": "red", "timed_out": True}

    class ReachableButRedElasticsearch:
        cluster = RedCluster()

        async def info(self):
            return {"version": {"number": "8.17.0"}}

    monkeypatch.setattr(api.state, "chroma_client", HealthyChroma())
    monkeypatch.setattr(api.state, "es_client", ReachableButRedElasticsearch())

    response = asyncio.run(api.health())
    body = json.loads(response.body)

    assert response.status_code == 503
    assert body["checks"]["elasticsearch"] == "error"


def test_ingest_path_is_confined_to_configured_root(tmp_path, monkeypatch) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    inside = root / "guide.md"
    inside.write_text("guide", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    monkeypatch.setattr(settings, "ingest_root", str(root))

    assert api._resolve_ingest_path(str(inside)) == inside.resolve()

    with pytest.raises(HTTPException) as exc_info:
        api._resolve_ingest_path(str(outside))

    assert exc_info.value.status_code == 400


def test_settings_validate_operational_limits() -> None:
    with pytest.raises(ValidationError):
        Settings(rate_limit_per_minute=0)

    with pytest.raises(ValidationError):
        Settings(llm_timeout_seconds=0)
