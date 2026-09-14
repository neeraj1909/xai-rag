"""The optional external evaluator must never fake a successful empty result."""

import builtins
from types import SimpleNamespace

import pytest

from xai_rag.config import settings
from xai_rag.evaluation.ragas_eval import _build_scorers, evaluate_response

pytestmark = pytest.mark.unit


class FakeMetric:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[dict] = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value)


def test_current_ragas_collection_metrics_can_be_constructed(monkeypatch) -> None:
    from ragas.metrics.collections import AnswerRelevancy, ContextUtilization, Faithfulness

    monkeypatch.setattr(settings, "openai_api_key", "test-key")

    scorers = _build_scorers(reference=None)

    assert isinstance(scorers.faithfulness, Faithfulness)
    assert isinstance(scorers.answer_relevancy, AnswerRelevancy)
    assert isinstance(scorers.context_precision, ContextUtilization)
    assert scorers.context_uses_reference is False


@pytest.mark.asyncio
async def test_evaluation_returns_all_requested_ragas_metrics() -> None:
    faithfulness = FakeMetric(0.91)
    answer_relevancy = FakeMetric(0.82)
    context_precision = FakeMetric(0.73)

    def scorer_factory(reference: str | None):
        assert reference is None
        return SimpleNamespace(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_uses_reference=False,
        )

    result = await evaluate_response(
        "question",
        "answer",
        ["first context", "second context"],
        scorer_factory=scorer_factory,
    )

    assert result == {
        "status": "computed",
        "scores": {
            "faithfulness": 0.91,
            "answer_relevancy": 0.82,
            "context_precision": 0.73,
        },
        "error_type": None,
    }
    assert faithfulness.calls == [
        {
            "user_input": "question",
            "response": "answer",
            "retrieved_contexts": ["first context", "second context"],
        }
    ]
    assert answer_relevancy.calls == [{"user_input": "question", "response": "answer"}]
    assert context_precision.calls == [
        {
            "user_input": "question",
            "response": "answer",
            "retrieved_contexts": ["first context", "second context"],
        }
    ]


@pytest.mark.asyncio
async def test_context_precision_uses_reference_answer_when_provided() -> None:
    context_precision = FakeMetric(0.88)

    def scorer_factory(reference: str | None):
        assert reference == "reference answer"
        return SimpleNamespace(
            faithfulness=FakeMetric(0.9),
            answer_relevancy=FakeMetric(0.8),
            context_precision=context_precision,
            context_uses_reference=True,
        )

    result = await evaluate_response(
        "question",
        "answer",
        ["context"],
        ground_truth="reference answer",
        scorer_factory=scorer_factory,
    )

    assert result["scores"]["context_precision"] == 0.88
    assert context_precision.calls == [
        {
            "user_input": "question",
            "reference": "reference answer",
            "retrieved_contexts": ["context"],
        }
    ]


@pytest.mark.asyncio
async def test_missing_ragas_dependency_has_explicit_status(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"datasets", "ragas"} or name.startswith("ragas."):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = await evaluate_response("query", "answer", ["context"])

    assert result == {
        "status": "unavailable",
        "scores": {},
        "error_type": "dependency_unavailable",
    }
