"""The optional external evaluator must never fake a successful empty result."""

import asyncio
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


@pytest.mark.asyncio
async def test_current_ragas_collection_metrics_can_be_constructed(monkeypatch) -> None:
    from ragas.metrics.collections import AnswerRelevancy, ContextUtilization, Faithfulness

    monkeypatch.setattr(settings, "openai_api_key", "test-key")

    scorers = await _build_scorers(reference=None)

    assert isinstance(scorers.faithfulness, Faithfulness)
    assert isinstance(scorers.answer_relevancy, AnswerRelevancy)
    assert isinstance(scorers.context_precision, ContextUtilization)
    assert scorers.context_uses_reference is False
    await scorers.owned_client.close()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_owned_ragas_client_closes_after_work_finishes(monkeypatch, outcome) -> None:
    import xai_rag.evaluation.ragas_eval as module

    started = asyncio.Event()
    drained = []
    closed = []

    class Metric:
        def __init__(self, fail_one=False):
            self.fail_one = fail_one

        async def ascore(self, **kwargs):
            started.set()
            try:
                if outcome == "failure" and self.fail_one:
                    raise RuntimeError("synthetic")
                if outcome in {"failure", "cancel"}:
                    await asyncio.Event().wait()
                return SimpleNamespace(value=1.0)
            finally:
                drained.append(True)

    class OwnedClient:
        async def close(self):
            assert len(drained) == 3
            closed.append(True)

    scorers = SimpleNamespace(
        faithfulness=Metric(fail_one=True),
        answer_relevancy=Metric(),
        context_precision=Metric(),
        context_uses_reference=False,
        owned_client=OwnedClient(),
    )

    async def build(_reference):
        return scorers

    monkeypatch.setattr(module, "_build_scorers", build)
    task = asyncio.create_task(module.evaluate_response("q", "a", ["c"]))
    await started.wait()
    if outcome == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        assert result["status"] == ("computed" if outcome == "success" else "error")
    assert closed == [True]


@pytest.mark.asyncio
async def test_injected_scorer_client_remains_caller_owned() -> None:
    class CallerClient:
        async def close(self):
            pytest.fail("must not close caller-owned client")

    scorers = SimpleNamespace(
        faithfulness=FakeMetric(1.0),
        answer_relevancy=FakeMetric(1.0),
        context_precision=FakeMetric(1.0),
        context_uses_reference=False,
        owned_client=CallerClient(),
    )
    result = await evaluate_response("q", "a", ["c"], scorer_factory=lambda _: scorers)
    assert result["status"] == "computed"


@pytest.mark.asyncio
async def test_scorer_setup_failure_closes_constructed_client(monkeypatch) -> None:
    import openai
    import ragas.llms

    closed = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def close(self):
            closed.append(True)

    def fail_setup(*args, **kwargs):
        raise RuntimeError("synthetic setup failure")

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    monkeypatch.setattr(ragas.llms, "llm_factory", fail_setup)
    result = await evaluate_response("q", "a", ["c"])
    assert result["status"] == "error"
    assert result["error_type"] == "RuntimeError"
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["drain", "close"])
async def test_repeated_cancel_waits_for_cleanup_and_propagates(monkeypatch, phase) -> None:
    import xai_rag.evaluation.ragas_eval as module

    cleaning = asyncio.Event()
    release = asyncio.Event()
    close_done = []

    class Metric:
        def __init__(self, fail=False):
            self.fail = fail

        async def ascore(self, **kwargs):
            if self.fail:
                await asyncio.sleep(0)
                raise ValueError("synthetic")
            try:
                await asyncio.Event().wait()
            finally:
                if phase == "drain":
                    cleaning.set()
                    await release.wait()

    class Client:
        async def close(self):
            if phase == "close":
                cleaning.set()
                await release.wait()
            close_done.append(True)

    scorers = SimpleNamespace(
        faithfulness=Metric(fail=True),
        answer_relevancy=Metric(),
        context_precision=Metric(),
        context_uses_reference=False,
        owned_client=Client(),
    )

    async def build(_reference):
        return scorers

    monkeypatch.setattr(module, "_build_scorers", build)
    task = asyncio.create_task(module.evaluate_response("q", "a", ["c"]))
    await asyncio.wait_for(cleaning.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert close_done == [True]


@pytest.mark.asyncio
async def test_repeated_cancel_during_setup_failure_waits_for_client_close(monkeypatch) -> None:
    import openai
    import ragas.llms

    closing = asyncio.Event()
    release = asyncio.Event()
    closed = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def close(self):
            closing.set()
            await release.wait()
            closed.append(True)

    def fail_setup(*args, **kwargs):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    monkeypatch.setattr(ragas.llms, "llm_factory", fail_setup)
    task = asyncio.create_task(evaluate_response("q", "a", ["c"]))
    await asyncio.wait_for(closing.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert closed == [True]
