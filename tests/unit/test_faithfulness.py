"""Unit contracts for NLI label handling and claim aggregation."""

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest
import torch

from xai_rag.explainability import faithfulness as faithfulness_module
from xai_rag.explainability.faithfulness import FaithfulnessChecker
from xai_rag.models import Claim, RankedResult, Verdict

pytestmark = pytest.mark.unit


def _chunk(chunk_id: str, content: str) -> RankedResult:
    return RankedResult(id=chunk_id, content=content, rrf_score=0.1, rrf_rank=1)


def test_concurrent_cold_requests_share_one_model_load(monkeypatch) -> None:
    start = threading.Barrier(2)
    second_load = threading.Event()
    loads = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            loads.append("tokenizer")
            if len(loads) > 1:
                second_load.set()
            else:
                second_load.wait(1)
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            loads.append("model")
            return cls()

        def eval(self):
            return self

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizer
    transformers.AutoModelForSequenceClassification = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    faithfulness_module._load_nli_model.cache_clear()
    try:

        def get_model(_):
            start.wait(timeout=3)
            return FaithfulnessChecker()._get_model_and_tokenizer()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = executor.map(get_model, range(2))
        assert loads.count("tokenizer") == 1
        assert loads.count("model") == 1
        assert first[0] is second[0]
        assert first[1] is second[1]
    finally:
        faithfulness_module._load_nli_model.cache_clear()


def test_model_label_mapping_is_derived_from_model_config(monkeypatch) -> None:
    checker = FaithfulnessChecker()

    class FakeTokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1]])}

    class FakeModel:
        config = SimpleNamespace(id2label={0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"})

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[-4.0, -2.0, 5.0]]))

    monkeypatch.setattr(
        checker,
        "_get_model_and_tokenizer",
        lambda: (FakeModel(), FakeTokenizer()),
    )

    scores = checker._score_pair("premise", "hypothesis")

    assert scores["entailment"] > 0.99
    assert scores["contradiction"] < 0.01


def test_stronger_contradiction_in_another_chunk_is_not_lost(monkeypatch) -> None:
    checker = FaithfulnessChecker()
    scores = {
        "support": {"entailment": 0.60, "neutral": 0.35, "contradiction": 0.05},
        "conflict": {"entailment": 0.02, "neutral": 0.03, "contradiction": 0.95},
    }
    monkeypatch.setattr(
        checker,
        "_score_pair",
        lambda premise, hypothesis: scores[premise],
    )

    verdict = checker._check_single_claim(
        Claim(text="claim"),
        [_chunk("s", "support"), _chunk("c", "conflict")],
    )

    assert verdict.verdict == Verdict.NOT_SUPPORTED
    assert verdict.nli_entailment == 0.60
    assert verdict.nli_contradiction == 0.95
    assert verdict.supporting_chunk_id == "s"
    assert verdict.contradicting_chunk_id == "c"
    assert verdict.confidence == 0.95


def test_claim_sources_scope_the_evidence(monkeypatch) -> None:
    checker = FaithfulnessChecker()
    seen: list[str] = []

    def score(premise: str, hypothesis: str) -> dict[str, float]:
        seen.append(premise)
        return {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}

    monkeypatch.setattr(checker, "_score_pair", score)
    claim = Claim(text="claim", source_chunk_ids=["declared"])

    verdict = checker._check_single_claim(
        claim,
        [_chunk("undeclared", "wrong"), _chunk("declared", "right")],
    )

    assert seen == ["right"]
    assert verdict.supporting_chunk_id == "declared"
    assert verdict.contradicting_chunk_id is None


def test_strong_contradiction_does_not_mark_weak_entailment_as_support(monkeypatch) -> None:
    checker = FaithfulnessChecker()
    monkeypatch.setattr(
        checker,
        "_score_pair",
        lambda premise, hypothesis: {"entailment": 0.05, "neutral": 0.05, "contradiction": 0.9},
    )

    verdict = checker._check_single_claim(Claim(text="claim"), [_chunk("c1", "conflicting")])

    assert verdict.verdict == Verdict.NOT_SUPPORTED
    assert verdict.supporting_chunk_id is None
    assert verdict.contradicting_chunk_id == "c1"


def test_neutral_scores_do_not_identify_evidence_as_support_or_conflict(monkeypatch) -> None:
    checker = FaithfulnessChecker()
    monkeypatch.setattr(
        checker,
        "_score_pair",
        lambda premise, hypothesis: {"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1},
    )

    verdict = checker._check_single_claim(Claim(text="claim"), [_chunk("c1", "unrelated")])

    assert verdict.verdict == Verdict.NEUTRAL
    assert verdict.supporting_chunk_id is None
    assert verdict.contradicting_chunk_id is None


@pytest.mark.asyncio
async def test_cancelled_nli_waits_for_already_started_thread(monkeypatch) -> None:
    checker = FaithfulnessChecker()
    started = threading.Event()
    release = threading.Event()

    def score(premise: str, hypothesis: str) -> dict[str, float]:
        started.set()
        if not release.wait(3):
            raise TimeoutError("test did not release NLI worker")
        return {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}

    monkeypatch.setattr(checker, "_score_pair", score)
    task = asyncio.create_task(checker.check_claims([Claim(text="claim")], [_chunk("c1", "text")]))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_failed_nli_waits_for_other_started_threads(monkeypatch) -> None:
    checker = FaithfulnessChecker()
    started = threading.Event()
    release = threading.Event()

    def score(premise: str, hypothesis: str) -> dict[str, float]:
        if hypothesis == "blocked":
            started.set()
            if not release.wait(3):
                raise TimeoutError("test did not release NLI worker")
            return {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}
        if not started.wait(2):
            raise TimeoutError("test worker did not start")
        raise ValueError("NLI failure")

    monkeypatch.setattr(checker, "_score_pair", score)
    task = asyncio.create_task(
        checker.check_claims([Claim(text="blocked"), Claim(text="failure")], [_chunk("c1", "text")])
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        await asyncio.sleep(0.05)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(ValueError, match="NLI failure"):
            await task


@pytest.mark.asyncio
async def test_nli_submits_only_bounded_claim_work_and_keeps_all_verdicts(monkeypatch) -> None:
    checker = FaithfulnessChecker()
    started = threading.Event()
    release = threading.Event()

    class TrackingExecutor(ThreadPoolExecutor):
        submitted = 0

        def submit(self, fn, /, *args, **kwargs):
            self.submitted += 1
            if self.submitted >= 2:
                started.set()
            return super().submit(fn, *args, **kwargs)

    def score(premise: str, hypothesis: str) -> dict[str, float]:
        if not release.wait(3):
            raise TimeoutError("test did not release NLI workers")
        return {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}

    executor = TrackingExecutor(max_workers=2)
    monkeypatch.setattr(faithfulness_module, "_NLI_POOL", executor)
    monkeypatch.setattr(checker, "_score_pair", score)
    task = asyncio.create_task(
        checker.check_claims([Claim(text=f"claim {i}") for i in range(7)], [_chunk("c1", "text")])
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        await asyncio.sleep(0.05)
        assert executor.submitted == 2
    finally:
        release.set()
        verdicts = await task
        executor.shutdown(wait=True)
    assert len(verdicts) == 7
    assert all(verdict.verdict == Verdict.SUPPORTED for verdict in verdicts)
