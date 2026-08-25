"""Regression tests for the API request/response model contract."""

import asyncio
import importlib
import json

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from xai_rag.api import app as api
from xai_rag.config import Settings, settings
from xai_rag.models import QueryRequest, XAIRAGResponse


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


def test_query_endpoint_passes_configured_index_to_bm25(monkeypatch) -> None:
    """The API must pass its configured index to Elasticsearch retrieval."""
    calls: dict[str, str] = {}

    async def fake_embed_query(query: str) -> list[float]:
        return [0.1]

    def fake_vector_search(collection, query_embedding, k):
        from xai_rag.models import SearchResult

        return [SearchResult(id="vector-1", content="vector result")]

    async def fake_bm25_search(client, query: str, index: str, k: int):
        from xai_rag.models import SearchResult

        calls["index"] = index
        return [SearchResult(id="bm25-1", content="BM25 result")]

    def fake_rrf_fusion(result_lists):
        return result_lists[0] + result_lists[1]

    class FakeReranker:
        async def rerank(self, query, results, top_k):
            from xai_rag.models import RankedResult

            calls["candidate_count"] = len(results)
            return [
                RankedResult(
                    id=results[0].id,
                    content=results[0].content,
                )
            ][:top_k]

    class FakeGenerator:
        async def generate(self, query, ranked):
            from xai_rag.models import RAGGenerationResult

            return RAGGenerationResult(answer="test answer")

    embedder = importlib.import_module("xai_rag.ingestion.embedder")
    vector_search = importlib.import_module("xai_rag.retrieval.vector_search")
    bm25_search = importlib.import_module("xai_rag.retrieval.bm25_search")
    hybrid = importlib.import_module("xai_rag.retrieval.hybrid")
    reranker = importlib.import_module("xai_rag.retrieval.reranker")
    generator = importlib.import_module("xai_rag.generation.generator")
    monkeypatch.setattr(embedder, "embed_query", fake_embed_query)
    monkeypatch.setattr(vector_search, "vector_search", fake_vector_search)
    monkeypatch.setattr(bm25_search, "bm25_search", fake_bm25_search)
    monkeypatch.setattr(hybrid, "rrf_fusion", fake_rrf_fusion)
    monkeypatch.setattr(reranker, "Reranker", FakeReranker)
    monkeypatch.setattr(generator, "RAGGenerator", FakeGenerator)
    monkeypatch.setattr(settings, "reranker_candidate_k", 1, raising=False)
    api.state.chroma_collection = object()
    api.state.es_client = object()

    response = asyncio.run(
        api.query_endpoint(
            QueryRequest(query="test query", include_explanations=False),
        )
    )

    assert response.answer == "test answer"
    assert calls["index"] == settings.elasticsearch_index
    assert calls["candidate_count"] == 1


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


def test_health_response_does_not_expose_dependency_errors(monkeypatch) -> None:
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
