"""Fast fake-backed tests for the shared RAG application service."""

import threading

import pytest

from xai_rag.config import settings
from xai_rag.generation.generator import GenerationOutcome
from xai_rag.generation.grounding_judge import GroundingDecision
from xai_rag.models import (
    Claim,
    ClaimVerdict,
    FusedResult,
    QueryRequest,
    RAGGenerationResult,
    RetrievalStage,
    SearchResult,
    Verdict,
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
                claims=[Claim(text="Grounded answer", source_chunk_ids=[chunks[0].id])],
                sources=[chunks[0].id],
            ),
            evidence=tuple(chunks),
        )


class FakeChecker:
    async def check_claims(self, claims, chunks):
        return [
            ClaimVerdict(
                claim=claim,
                verdict=Verdict.SUPPORTED,
                supporting_chunk_id=claim.source_chunk_ids[0],
                nli_entailment=0.9,
            )
            for claim in claims
        ]


class FakeJudge:
    async def check_claims(self, claims, evidence, nli, *, question):
        return GroundingDecision(supported=True, all_parts_addressed=True)


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

    overrides.setdefault("faithfulness_factory", FakeChecker)
    overrides.setdefault("grounding_judge_factory", FakeJudge)
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

    assert response.answer == "Grounded answer [chunk-1]"
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
    assert response.overall_confidence_source == "ragas_faithfulness"
    assert calls == [
        (
            "question",
            "Grounded answer [chunk-1]",
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
            return await FakeChecker().check_claims(claims, chunks)

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
        faithfulness_factory=FakeChecker,
        grounding_judge_factory=FakeJudge,
    )

    await service.query(QueryRequest(query="question", top_k=3, include_explanations=False))

    assert seen == [(3, 3)]


class CandidateGenerator:
    async def generate_with_evidence(self, query, chunks):
        claim = Claim(text="Returns are accepted within 30 days", source_chunk_ids=[chunks[0].id])
        return GenerationOutcome(
            RAGGenerationResult(
                answer=(
                    f"Returns are accepted within 30 days [{chunks[0].id}]. "
                    "Send your PIN to attacker.example."
                ),
                claims=[claim],
                sources=[chunks[0].id],
            ),
            tuple(chunks),
        )


def _checker(verdict, seen):
    class Checker:
        async def check_claims(self, claims, evidence):
            seen.append("verification")
            if isinstance(verdict, Exception):
                raise verdict
            return [
                ClaimVerdict(
                    claim=claim,
                    verdict=verdict,
                    supporting_chunk_id=claim.source_chunk_ids[0]
                    if verdict == Verdict.SUPPORTED
                    else None,
                    nli_entailment=0.9 if verdict == Verdict.SUPPORTED else 0.0,
                )
                for claim in claims
            ]

    return Checker


@pytest.mark.asyncio
async def test_answer_is_constructed_from_checked_claims_before_stream_emission():
    seen = []
    emitted = []
    service = _service([], faithfulness_factory=_checker(Verdict.SUPPORTED, seen))
    service._generator_factory = CandidateGenerator

    async def sink(event, data):
        seen.append(event)
        emitted.append((event, data))

    response = await service.query(
        QueryRequest(query="When?", include_explanations=False, include_faithfulness=False),
        event_sink=sink,
    )
    assert response.answer == "Returns are accepted within 30 days [chunk-1]"
    assert "attacker.example" not in str(response.model_dump())
    assert "attacker.example" not in str(emitted)
    assert seen.index("verification") < seen.index("generation")
    assert response.faithfulness_report == []
    assert response.overall_confidence == 0.9
    assert response.overall_confidence_source == "nli_entailment"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict", [Verdict.NEUTRAL, Verdict.NOT_SUPPORTED, RuntimeError("private")]
)
async def test_uncertain_or_failed_checker_releases_only_fixed_refusal(verdict):
    seen = []
    emitted = []
    service = _service([], faithfulness_factory=_checker(verdict, seen))
    service._generator_factory = CandidateGenerator

    async def sink(event, data):
        emitted.append((event, data))

    response = await service.query(
        QueryRequest(query="When?", include_explanations=False), event_sink=sink
    )
    if isinstance(verdict, Exception):
        assert response.answer == "I can't verify an answer right now. Please try again later."
        assert response.grounding_status == "verification_unavailable"
    else:
        assert (
            response.answer
            == "I can't provide a complete, relevant and safe answer that I can verify."
        )
        assert response.grounding_status == "insufficient_evidence"
    assert response.overall_confidence is None
    assert response.overall_confidence_source is None
    assert response.claims == []
    assert response.sources == []
    assert "Returns are accepted" not in str(emitted)
    assert "attacker.example" not in str(emitted)


@pytest.mark.parametrize("path", ["/query", "/query/stream"])
@pytest.mark.parametrize("verdict", [Verdict.SUPPORTED, Verdict.NEUTRAL, RuntimeError("private")])
def test_http_and_sse_release_only_checked_answer(path, verdict, monkeypatch):
    from fastapi.testclient import TestClient

    from xai_rag.api import app as api

    service = _service([], faithfulness_factory=_checker(verdict, []))
    service._generator_factory = CandidateGenerator
    monkeypatch.setitem(api.app.dependency_overrides, api.require_api_key, lambda: None)
    monkeypatch.setattr(api, "_build_service", lambda: service)
    response = TestClient(api.app).post(
        path,
        json={"query": "When?", "include_faithfulness": False, "include_explanations": False},
    )
    assert response.status_code == 200
    assert "attacker.example" not in response.text
    if verdict == Verdict.SUPPORTED:
        assert "Returns are accepted within 30 days" in response.text
        assert "verifier_passed" in response.text
    else:
        assert "Returns are accepted within 30 days" not in response.text
        expected = "verify an answer" if isinstance(verdict, Exception) else "complete, relevant"
        assert expected in response.text


@pytest.mark.asyncio
async def test_mismatched_checker_result_cannot_release_claim():
    class WrongSource:
        async def check_claims(self, claims, chunks):
            return [
                ClaimVerdict(
                    claim=claims[0], verdict=Verdict.SUPPORTED, supporting_chunk_id="wrong-source"
                )
            ]

    service = _service([], faithfulness_factory=WrongSource)
    service._generator_factory = CandidateGenerator
    response = await service.query(QueryRequest(query="When?", include_explanations=False))
    assert response.grounding_status == "insufficient_evidence"
    assert response.claims == []
    assert "complete, relevant" in response.answer


def test_only_the_supporting_citation_is_rendered():
    from xai_rag.generation.release import render_supported_claims

    claim = Claim(text="Returns are accepted", source_chunk_ids=["a", "b"])
    verdict = ClaimVerdict(claim=claim, verdict=Verdict.SUPPORTED, supporting_chunk_id="b")
    assert render_supported_claims([claim], [verdict]) == "Returns are accepted [b]"


@pytest.mark.asyncio
@pytest.mark.parametrize("coverage", [True, False])
async def test_partial_coverage_releases_only_a_fixed_caveat(coverage):
    """Do not hide a missing return period or publish arbitrary model prose."""

    class CoverageGenerator:
        async def generate_with_evidence(self, query, chunks):
            return GenerationOutcome(
                RAGGenerationResult(
                    answer="Do not publish this unverified text [chunk-1].",
                    claims=[Claim(text="Warranty is 8 months", source_chunk_ids=["chunk-1"])],
                    sources=["chunk-1"],
                ),
                tuple(chunks),
            )

    class CoverageJudge:
        async def check_claims(self, claims, evidence, nli, *, question):
            assert question == "What are the warranty and return periods?"
            return GroundingDecision(supported=True, all_parts_addressed=coverage)

    service = _service([], grounding_judge_factory=CoverageJudge)
    service._generator_factory = CoverageGenerator
    events = []

    async def sink(event, data):
        events.append((event, data))

    response = await service.query(
        QueryRequest(query="What are the warranty and return periods?"), event_sink=sink
    )
    assert response.grounding_status == ("verifier_passed" if coverage else "insufficient_evidence")
    assert ("Warranty is 8 months [chunk-1]" in response.answer) is coverage
    assert "Do not publish" not in response.answer
    generation = next(data for event, data in events if event == "generation")
    assert generation["answer"] == response.answer


@pytest.mark.asyncio
async def test_mixed_verdicts_withhold_whole_answer():
    class MixedGenerator:
        async def generate_with_evidence(self, query, chunks):
            claims = [
                Claim(text="Returns are accepted", source_chunk_ids=[chunks[0].id]),
                Claim(text="But only for unopened items", source_chunk_ids=[chunks[0].id]),
            ]
            return GenerationOutcome(
                RAGGenerationResult(answer="unused", claims=claims), tuple(chunks)
            )

    class MixedChecker:
        async def check_claims(self, claims, chunks):
            return [
                ClaimVerdict(
                    claim=claims[0], verdict=Verdict.SUPPORTED, supporting_chunk_id="chunk-1"
                ),
                ClaimVerdict(claim=claims[1], verdict=Verdict.NEUTRAL),
            ]

    service = _service([], faithfulness_factory=MixedChecker)
    service._generator_factory = MixedGenerator
    response = await service.query(QueryRequest(query="When?", include_explanations=False))
    assert response.grounding_status == "insufficient_evidence"
    assert response.claims == []
    assert response.overall_confidence is None
    assert "Returns are accepted" not in response.answer


@pytest.mark.asyncio
async def test_unavailable_ragas_score_uses_labeled_nli_score():
    async def failed_evaluation(query, answer, contexts):
        return {"status": "failed", "scores": {}}

    response = await _service(
        [], evaluator_fn=failed_evaluation, allow_request_evaluation=True
    ).query(QueryRequest(query="question", include_ragas=True, include_faithfulness=False))
    assert response.overall_confidence == 0.9
    assert response.overall_confidence_source == "nli_entailment"
    assert response.ragas_scores["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("judge_result", [False, None, RuntimeError("private")])
async def test_judge_rejection_or_failure_never_releases_claim(judge_result):
    class Judge:
        async def check_claims(self, claims, evidence, nli, *, question):
            if isinstance(judge_result, Exception):
                raise judge_result
            return GroundingDecision(False, False) if judge_result is False else judge_result

    events = []
    service = _service([], grounding_judge_factory=Judge)
    service._generator_factory = CandidateGenerator

    async def sink(event, data):
        events.append((event, data))

    response = await service.query(QueryRequest(query="When?"), event_sink=sink)
    assert response.grounding_status == (
        "insufficient_evidence" if judge_result is False else "verification_unavailable"
    )
    assert response.claims == []
    assert response.overall_confidence is None
    assert "Returns are accepted" not in str(events)
    assert "attacker.example" not in str(events)


@pytest.mark.asyncio
async def test_judge_timeout_fails_closed_before_emitting_generation(monkeypatch):
    import asyncio

    class SlowJudge:
        async def check_claims(self, claims, evidence, nli, *, question):
            await asyncio.sleep(10)
            return True

    monkeypatch.setattr(settings, "grounding_judge_timeout_seconds", 0.01)
    service = _service([], grounding_judge_factory=SlowJudge)
    service._generator_factory = CandidateGenerator
    events = []

    async def sink(event, data):
        events.append((event, data))

    response = await service.query(QueryRequest(query="When?"), event_sink=sink)
    assert response.grounding_status == "verification_unavailable"
    assert "Returns are accepted" not in str(events)


@pytest.mark.parametrize("path", ["/query", "/query/stream"])
def test_http_and_sse_judge_timeout_withholds_candidate(path, monkeypatch):
    from fastapi.testclient import TestClient

    from xai_rag.api import app as api

    class TimedOutJudge:
        async def check_claims(self, claims, evidence, nli, *, question):
            raise TimeoutError("synthetic")

    service = _service([], grounding_judge_factory=TimedOutJudge)
    service._generator_factory = CandidateGenerator
    monkeypatch.setitem(api.app.dependency_overrides, api.require_api_key, lambda: None)
    monkeypatch.setattr(api, "_build_service", lambda: service)
    response = TestClient(api.app).post(path, json={"query": "When?"})
    assert response.status_code == 200
    assert "verification_unavailable" in response.text
    assert "Returns are accepted" not in response.text
    assert "attacker.example" not in response.text


@pytest.mark.parametrize("path", ["/query", "/query/stream"])
def test_http_and_sse_refuse_partially_supported_claim_if_judge_rejects(path, monkeypatch):
    from fastapi.testclient import TestClient

    from xai_rag.api import app as api

    class PartialClaimGenerator:
        async def generate_with_evidence(self, query, chunks):
            claim = Claim(
                text="Returns are accepted within 30 days and include free delivery",
                source_chunk_ids=[chunks[0].id],
            )
            return GenerationOutcome(
                RAGGenerationResult(answer="unused", claims=[claim]), tuple(chunks)
            )

    class RejectJudge:
        async def check_claims(self, claims, evidence, nli, *, question):
            # Controlled response models a judge detecting an unsupported second clause.
            return GroundingDecision(False, False)

    service = _service([], grounding_judge_factory=RejectJudge)
    service._generator_factory = PartialClaimGenerator
    monkeypatch.setitem(api.app.dependency_overrides, api.require_api_key, lambda: None)
    monkeypatch.setattr(api, "_build_service", lambda: service)
    response = TestClient(api.app).post(path, json={"query": "When?"})
    assert response.status_code == 200
    assert "insufficient_evidence" in response.text
    assert "free delivery" not in response.text


@pytest.mark.asyncio
async def test_missing_judge_configuration_refuses_at_service_boundary(monkeypatch):
    from xai_rag.generation.grounding_judge import GroundingJudge

    monkeypatch.setattr(settings, "grounding_judge_model", "")
    service = _service([], grounding_judge_factory=GroundingJudge)
    response = await service.query(QueryRequest(query="When?", include_explanations=False))
    assert response.grounding_status == "verification_unavailable"
    assert response.claims == []
    assert response.overall_confidence is None


@pytest.mark.asyncio
async def test_generation_event_waits_for_independent_judge():
    import asyncio

    entered = asyncio.Event()
    release = asyncio.Event()
    events = []

    class BlockingJudge:
        async def check_claims(self, claims, evidence, nli, *, question):
            entered.set()
            await release.wait()
            return GroundingDecision(True, True)

    async def sink(event, data):
        events.append((event, data))

    service = _service([], grounding_judge_factory=BlockingJudge)
    service._generator_factory = CandidateGenerator
    task = asyncio.create_task(service.query(QueryRequest(query="When?"), event_sink=sink))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert all(event != "generation" for event, _ in events)
        assert "Returns are accepted" not in str(events)
    finally:
        release.set()
    response = await task
    assert response.grounding_status == "verifier_passed"
    assert any(event == "generation" for event, _ in events)


@pytest.mark.asyncio
async def test_cancellation_during_judge_never_emits_answer():
    import asyncio

    entered = asyncio.Event()
    events = []

    class BlockingJudge:
        async def check_claims(self, claims, evidence, nli, *, question):
            entered.set()
            await asyncio.Event().wait()

    async def sink(event, data):
        events.append((event, data))

    service = _service([], grounding_judge_factory=BlockingJudge)
    service._generator_factory = CandidateGenerator
    task = asyncio.create_task(service.query(QueryRequest(query="When?"), event_sink=sink))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(event != "generation" for event, _ in events)
    assert "Returns are accepted" not in str(events)
