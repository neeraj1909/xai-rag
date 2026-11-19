"""Fast fake-backed tests for the shared RAG application service."""

import threading

import pytest

from xai_rag.config import settings
from xai_rag.generation.generator import GenerationOutcome
from xai_rag.models import (
    Claim,
    FusedResult,
    QueryRequest,
    RAGGenerationResult,
    RetrievalStage,
    SearchResult,
)
from xai_rag.service import RAGService, RequestEvaluationDisabledError

pytestmark = pytest.mark.unit


class FakeReranker:
    async def rerank(self, query, results, top_k):
        return [
            result.to_ranked(reranker_score=0.9, reranker_rank=index)
            for index, result in enumerate(results[:top_k], start=1)
        ]


class FakeGenerator:
    async def generate_with_evidence(self, query, chunks):
        return GenerationOutcome(
            result=RAGGenerationResult(
                answer=f"Grounded answer [{chunks[0].id}].",
                sources=[chunks[0].id],
            ),
            evidence=tuple(chunks),
        )


def _service(thread_names: list[str], **overrides) -> RAGService:
    async def embed(query: str) -> list[float]:
        return [0.1, 0.2]

    def vector(collection, embedding, k):
        thread_names.append(threading.current_thread().name)
        return [
            SearchResult(
                id="chunk-1",
                content="evidence",
                score=-0.1,
                stage=RetrievalStage.VECTOR,
                rank=1,
            )
        ]

    async def bm25(client, query, index, k):
        return []

    def fusion(result_lists, k):
        return [
            FusedResult(
                id="chunk-1",
                content="evidence",
                vector_score=-0.1,
                vector_rank=1,
                rrf_score=1 / 61,
                rrf_rank=1,
            )
        ]

    return RAGService(
        collection=object(),
        es_client=object(),
        embed_query_fn=embed,
        vector_search_fn=vector,
        bm25_search_fn=bm25,
        fusion_fn=fusion,
        reranker_factory=FakeReranker,
        generator_factory=FakeGenerator,
        **overrides,
    )


@pytest.mark.asyncio
async def test_service_runs_one_pipeline_and_emits_streamable_events() -> None:
    thread_names: list[str] = []
    events: list[str] = []

    async def sink(event: str, data: dict) -> None:
        events.append(event)

    response = await _service(thread_names).query(
        QueryRequest(query="question", include_explanations=False),
        event_sink=sink,
    )

    assert response.answer == "Grounded answer [chunk-1]."
    assert response.sources == ["chunk-1"]
    assert events == ["retrieval", "generation", "done"]
    assert set(response.stage_latency_ms) >= {"embed", "retrieve", "fusion", "rerank", "generate"}
    assert all(name != threading.current_thread().name for name in thread_names)


@pytest.mark.asyncio
async def test_request_scoped_external_evaluation_is_disabled_by_default() -> None:
    with pytest.raises(RequestEvaluationDisabledError):
        await _service([]).query(QueryRequest(query="question", include_ragas=True))


@pytest.mark.asyncio
async def test_ragas_evaluation_is_returned_and_supplies_faithfulness_score() -> None:
    calls: list[tuple[str, str, list[str]]] = []

    async def evaluate(query: str, answer: str, contexts: list[str]):
        calls.append((query, answer, contexts))
        return {
            "status": "computed",
            "scores": {
                "faithfulness": 0.91,
                "answer_relevancy": 0.82,
                "context_precision": 0.73,
            },
            "error_type": None,
        }

    response = await _service(
        [],
        evaluator_fn=evaluate,
        allow_request_evaluation=True,
    ).query(
        QueryRequest(
            query="question",
            include_explanations=False,
            include_ragas=True,
        )
    )

    assert response.ragas_scores["scores"] == {
        "faithfulness": 0.91,
        "answer_relevancy": 0.82,
        "context_precision": 0.73,
    }
    assert response.overall_confidence == 0.91
    assert calls == [
        (
            "question",
            "Grounded answer [chunk-1].",
            ["evidence"],
        )
    ]


@pytest.mark.asyncio
async def test_service_uses_prompt_visible_evidence_for_nli_and_ragas() -> None:
    seen = []

    class EvidenceGenerator:
        async def generate_with_evidence(self, query, chunks):
            visible = chunks[0].model_copy(
                update={"content": "shown &lt;fragment&gt; literal &amp;lt;"}
            )
            return GenerationOutcome(
                RAGGenerationResult(
                    answer="Answer [chunk-1].",
                    claims=[Claim(text="Answer", source_chunk_ids=["chunk-1"])],
                    sources=["chunk-1"],
                ),
                (visible,),
            )

    class Checker:
        async def check_claims(self, claims, chunks):
            seen.append(("nli", [chunk.content for chunk in chunks]))
            return []

    async def evaluate(query, answer, contexts):
        seen.append(("ragas", contexts))
        return {"status": "skipped", "scores": {}, "error_type": None}

    service = _service(
        [], faithfulness_factory=Checker, evaluator_fn=evaluate, allow_request_evaluation=True
    )
    service._generator_factory = EvidenceGenerator
    response = await service.query(
        QueryRequest(
            query="question",
            include_explanations=False,
            include_faithfulness=True,
            include_ragas=True,
        )
    )

    assert seen == [
        ("nli", ["shown <fragment> literal &lt;"]),
        ("ragas", ["shown &lt;fragment&gt; literal &amp;lt;"]),
    ]
    assert "evidence" not in response.model_dump()


@pytest.mark.asyncio
async def test_requested_top_k_is_not_silently_limited_by_candidate_default(monkeypatch) -> None:
    seen: list[tuple[int, int]] = []

    async def embed(query: str) -> list[float]:
        return [0.1, 0.2]

    def vector(collection, embedding, k):
        return []

    async def bm25(client, query, index, k):
        return []

    def fusion(result_lists, k):
        return [
            FusedResult(id=f"chunk-{index}", content="evidence", rrf_score=0.1)
            for index in range(5)
        ]

    class RecordingReranker(FakeReranker):
        async def rerank(self, query, results, top_k):
            seen.append((len(results), top_k))
            return await super().rerank(query, results, top_k)

    monkeypatch.setattr(settings, "reranker_candidate_k", 2)
    service = RAGService(
        collection=object(),
        es_client=object(),
        embed_query_fn=embed,
        vector_search_fn=vector,
        bm25_search_fn=bm25,
        fusion_fn=fusion,
        reranker_factory=RecordingReranker,
        generator_factory=FakeGenerator,
    )

    await service.query(QueryRequest(query="question", top_k=3, include_explanations=False))

    assert seen == [(3, 3)]
