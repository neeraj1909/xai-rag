"""Tenant and document authorization must hold before retrieval and ingestion."""

import hashlib
import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from xai_rag.api import app as api
from xai_rag.authorization import AccessScope, load_policy, tenant_store_name
from xai_rag.config import settings
from xai_rag.models import QueryRequest, RetrievalStage, SearchResult
from xai_rag.retrieval.bm25_search import bm25_search
from xai_rag.retrieval.vector_search import vector_search
from xai_rag.service import RAGService

pytestmark = pytest.mark.unit


def policy_file(tmp_path, *, key="k" * 32, tenant="one", sources=None, role="reader"):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "principals": [
                    {
                        "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                        "tenant": tenant,
                        "sources": sources if sources is not None else ["allowed.txt"],
                        "role": role,
                    }
                ]
            }
        )
    )
    return str(path)


def test_policy_authentication_and_tenant_names(tmp_path):
    policy = load_policy(policy_file(tmp_path))
    assert policy.authenticate("k" * 32) == AccessScope("one", ("allowed.txt",), False)
    assert policy.authenticate("x" * 32) is None
    assert policy.authenticate("short") is None
    assert tenant_store_name("base", "one") != tenant_store_name("base", "two")
    assert tenant_store_name("base", "one") != "base"


@pytest.mark.parametrize("sources", [["../secret"], ["/absolute"], ["./relative"], ["x", "x"]])
def test_policy_rejects_ambiguous_sources(tmp_path, sources):
    with pytest.raises(ValueError):
        load_policy(policy_file(tmp_path, sources=sources))


def test_missing_policy_cannot_fall_back_to_legacy_key(monkeypatch):
    monkeypatch.setattr(settings, "auth_policy_file", "missing")
    monkeypatch.setattr(settings, "api_key", "k" * 32)
    monkeypatch.setattr(api.state, "auth_policy", None)
    with pytest.raises(HTTPException) as error:
        api.require_api_key("k" * 32)
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_filters_execute_in_backends_before_top_k():
    collection = Mock()
    collection.query.return_value = {"ids": [[]]}
    assert vector_search(collection, [0.1], 2, allowed_sources=("allowed.txt",)) == []
    assert collection.query.call_args.kwargs["where"] == {"source_file": {"$in": ["allowed.txt"]}}
    es = Mock()
    es.search = AsyncMock(return_value={"hits": {"hits": []}})
    assert await bm25_search(es, "q", "tenant-index", 2, allowed_sources=("allowed.txt",)) == []
    assert es.search.call_args.kwargs["query"]["bool"]["filter"] == [
        {"terms": {"source_file": ["allowed.txt"]}}
    ]
    collection.reset_mock()
    es.search.reset_mock()
    assert vector_search(collection, [0.1], allowed_sources=()) == []
    assert await bm25_search(es, "q", "tenant-index", allowed_sources=()) == []
    collection.query.assert_not_called()
    es.search.assert_not_called()


@pytest.mark.asyncio
async def test_cross_tenant_scope_rejected_before_embedding(monkeypatch):
    monkeypatch.setattr(settings, "auth_policy_file", "policy")
    embed = AsyncMock(return_value=[0.1])
    service = RAGService(
        collection=object(), es_client=object(), tenant="one", embed_query_fn=embed
    )
    for scope in [None, AccessScope("two", ("allowed.txt",), False)]:
        with pytest.raises(PermissionError):
            await service.query(QueryRequest(query="q"), access_scope=scope)
    embed.assert_not_called()


@pytest.mark.asyncio
async def test_retrieval_defense_in_depth_blocks_unauthorized_source(monkeypatch):
    monkeypatch.setattr(settings, "auth_policy_file", "policy")
    leaked = SearchResult(
        id="secret",
        content="private",
        metadata={"source_file": "secret.txt"},
        score=1,
        stage=RetrievalStage.VECTOR,
        rank=1,
    )
    vector = Mock(return_value=[leaked])
    bm25 = AsyncMock(return_value=[])
    service = RAGService(
        collection=object(),
        es_client=object(),
        tenant="one",
        index_name="one-index",
        embed_query_fn=AsyncMock(return_value=[0.1]),
        vector_search_fn=vector,
        bm25_search_fn=bm25,
    )
    with pytest.raises(PermissionError):
        await service.query(
            QueryRequest(query="q"), access_scope=AccessScope("one", ("allowed.txt",), False)
        )
    assert vector.call_args.kwargs["allowed_sources"] == ("allowed.txt",)
    assert bm25.call_args.args[2] == "one-index"
    assert bm25.call_args.kwargs["allowed_sources"] == ("allowed.txt",)


@pytest.mark.asyncio
async def test_ingestion_checks_all_sources_before_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "auth_policy_file", "policy")
    (tmp_path / "allowed.txt").write_text("allowed")
    (tmp_path / "secret.txt").write_text("secret")
    parse = Mock()
    service = RAGService(collection=object(), es_client=object(), tenant="one", parse_file_fn=parse)
    with pytest.raises(PermissionError):
        await service.ingest(
            tmp_path / "secret.txt",
            document_root=tmp_path,
            access_scope=AccessScope("one", ("allowed.txt",), True),
        )
    parse.assert_not_called()
    with pytest.raises(PermissionError):
        await service.ingest(
            tmp_path / "allowed.txt",
            document_root=tmp_path,
            access_scope=AccessScope("one", ("allowed.txt",), False),
        )
    parse.assert_not_called()


@pytest.mark.asyncio
async def test_ungranted_ingestion_never_probes_path_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "auth_policy_file", "policy")
    monkeypatch.setattr(settings, "ingest_root", str(tmp_path))
    (tmp_path / "existing.txt").write_text("small")
    (tmp_path / "large.txt").write_bytes(b"x" * 100)
    monkeypatch.setattr(settings, "max_document_bytes", 1)
    build = Mock()
    monkeypatch.setattr(api, "_build_scoped_service", build)
    scope = AccessScope("one", ("allowed.txt",), True)
    responses = []
    for path in ("missing.txt", "existing.txt", "large.txt"):
        with pytest.raises(HTTPException) as exc:
            await api.ingest_endpoint(path, scope=scope)
        responses.append((exc.value.status_code, exc.value.detail))
    assert responses == [(403, "Ingestion is not authorized")] * 3
    build.assert_not_called()


def test_directory_request_selects_only_granted_paths_without_discovery(tmp_path):
    from xai_rag.authorization import granted_ingestion_paths

    scope = AccessScope("one", ("folder/a.txt", "elsewhere/b.txt"), True)
    assert granted_ingestion_paths(tmp_path, tmp_path / "folder", scope) == [
        tmp_path / "folder/a.txt"
    ]
    with pytest.raises(PermissionError):
        granted_ingestion_paths(tmp_path, tmp_path / "missing", scope)


@pytest.mark.asyncio
async def test_ingestion_rejects_symlink_pivot_after_authorization(tmp_path, monkeypatch):
    from xai_rag import service as service_module

    monkeypatch.setattr(settings, "auth_policy_file", "policy")
    allowed = tmp_path / "allowed.txt"
    allowed.write_text("public")
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    real_stage = service_module.stage_document

    def pivot(root, path, max_bytes, *, budget):
        allowed.unlink()
        allowed.symlink_to(secret)
        return real_stage(root, path, max_bytes, budget=budget)

    monkeypatch.setattr(service_module, "stage_document", pivot)
    parse = Mock()
    store = Mock()
    service = RAGService(
        collection=object(),
        es_client=object(),
        tenant="one",
        parse_file_fn=parse,
        store_chroma_fn=store,
    )
    result = await service.ingest(
        allowed,
        document_root=tmp_path,
        access_scope=AccessScope("one", ("allowed.txt",), True),
    )
    assert result.documents[0].status == "failed"
    parse.assert_not_called()
    store.assert_not_called()


@pytest.mark.asyncio
async def test_scoped_ingest_parity_never_checks_ungranted_sources(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(settings, "auth_policy_file", "policy")
    allowed = tmp_path / "allowed.txt"
    allowed.write_text("allowed")

    async def check_parity(_collection, _es, *, index, source_file):
        assert index == "tenant-index"
        # A tenant-wide check would discover an unrelated private divergence.
        assert source_file == "allowed.txt"
        return SimpleNamespace(consistent=True)

    service = RAGService(
        collection=object(),
        es_client=object(),
        tenant="one",
        index_name="tenant-index",
        parse_file_fn=lambda path: path.read_text(),
        chunk_text_fn=lambda *_: [],
        store_chroma_fn=lambda *_: [],
        store_elasticsearch_fn=AsyncMock(),
        parity_fn=check_parity,
    )
    result = await service.ingest(
        allowed,
        document_root=tmp_path,
        access_scope=AccessScope("one", ("allowed.txt",), True),
    )
    assert result.status == "success"
    assert result.parity_consistent is True


@pytest.mark.asyncio
async def test_http_auth_controls_query_and_stream_even_with_spoofed_tenant(tmp_path, monkeypatch):
    import httpx

    from xai_rag.models import XAIRAGResponse

    policy = load_policy(policy_file(tmp_path))
    monkeypatch.setattr(settings, "auth_policy_file", "policy")
    monkeypatch.setattr(api.state, "auth_policy", policy)
    monkeypatch.setattr(api, "_RATE_LIMIT_BUCKETS", {})
    observed = []

    class Service:
        async def query(self, request, *, event_sink=None, access_scope=None):
            observed.append(access_scope)
            return XAIRAGResponse(query=request.query, answer="ok")

    monkeypatch.setattr(api, "_build_scoped_service", lambda scope: Service())
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for endpoint in ("/query", "/query/stream"):
            response = await client.post(
                endpoint, headers={"X-API-Key": "k" * 32}, json={"query": "q", "tenant": "two"}
            )
            assert response.status_code == 200
        assert (
            await client.post("/query", headers={"X-API-Key": "x" * 32}, json={"query": "q"})
        ).status_code == 401
        assert (
            await client.post(
                "/ingest", headers={"X-API-Key": "k" * 32}, params={"source_path": "allowed.txt"}
            )
        ).status_code == 403
    assert observed == [AccessScope("one", ("allowed.txt",), False)] * 2


def test_real_chroma_isolates_tenants_and_applies_source_filter():
    import os
    import uuid

    import chromadb

    port = os.environ.get("XAI_AUTH_TEST_CHROMA_PORT")
    if not port:
        pytest.skip("requires isolated Chroma service")
    client = chromadb.HttpClient(host="127.0.0.1", port=int(port))
    base = uuid.uuid4().hex
    first = client.get_or_create_collection(tenant_store_name(base, "one"))
    second = client.get_or_create_collection(tenant_store_name(base, "two"))
    first.add(
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.9, 0.1]],
        documents=["allowed", "secret"],
        metadatas=[{"source_file": "allowed.txt"}, {"source_file": "secret.txt"}],
    )
    second.add(
        ids=["a"],
        embeddings=[[1.0, 0.0]],
        documents=["other tenant"],
        metadatas=[{"source_file": "allowed.txt"}],
    )
    own = vector_search(first, [1.0, 0.0], 2, allowed_sources=("allowed.txt",))
    other = vector_search(second, [1.0, 0.0], 2, allowed_sources=("allowed.txt",))
    assert [result.content for result in own] == ["allowed"]
    assert [result.content for result in other] == ["other tenant"]


@pytest.mark.asyncio
async def test_real_elasticsearch_isolates_tenants_and_applies_source_filter():
    import os
    import uuid

    from elasticsearch import AsyncElasticsearch

    from xai_rag.ingestion.store import ensure_es_index

    port = os.environ.get("XAI_AUTH_TEST_ES_PORT")
    if not port:
        pytest.skip("requires isolated Elasticsearch service")
    client = AsyncElasticsearch(f"http://127.0.0.1:{int(port)}")
    suffix = uuid.uuid4().hex
    one = tenant_store_name(suffix, "one")
    two = tenant_store_name(suffix, "two")
    try:
        await ensure_es_index(client, one)
        await ensure_es_index(client, two)
        for index, doc_id, source, content in [
            (one, "a", "allowed.txt", "target allowed"),
            (one, "b", "secret.txt", "target secret"),
            (two, "a", "allowed.txt", "target other tenant"),
        ]:
            await client.index(
                index=index,
                id=doc_id,
                document={
                    "content": content,
                    "source_file": source,
                    "metadata": {"source_file": source},
                },
                refresh=True,
            )
        own = await bm25_search(client, "target", one, allowed_sources=("allowed.txt",))
        other = await bm25_search(client, "target", two, allowed_sources=("allowed.txt",))
        assert [result.content for result in own] == ["target allowed"]
        assert [result.content for result in other] == ["target other tenant"]
    finally:
        await client.indices.delete(index=[one, two], ignore_unavailable=True)
        await client.close()


def test_scoped_relative_ingestion_path_uses_root_and_rejects_escape(tmp_path, monkeypatch):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "allowed.txt").write_text("allowed")
    (tmp_path / "secret.txt").write_text("secret")
    (root / "link.txt").symlink_to(tmp_path / "secret.txt")
    monkeypatch.setattr(settings, "ingest_root", str(root))
    assert api._resolve_ingest_path("allowed.txt", relative_to_root=True) == root / "allowed.txt"
    for path in ("../secret.txt", "link.txt"):
        with pytest.raises(HTTPException):
            api._resolve_ingest_path(path, relative_to_root=True)
