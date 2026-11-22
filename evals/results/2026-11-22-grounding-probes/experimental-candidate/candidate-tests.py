"""Behavioral checks for bounded NLI evidence recovery."""

from types import SimpleNamespace

import pytest
import torch

from xai_rag.explainability.faithfulness import FaithfulnessChecker, _EvidenceTooLongError
from xai_rag.models import Claim, RankedResult, Verdict

pytestmark = pytest.mark.unit


def test_supported_evidence_in_markdown_survives_false_full_chunk_contradiction(monkeypatch):
    checker = FaithfulnessChecker()
    claim = Claim(text="The library closes at six.", source_chunk_ids=["source"])
    evidence = RankedResult(id="source", content="# Library\n\nThe library closes at six.\n")

    def score(premise, hypothesis):
        assert hypothesis == claim.text
        if premise.strip() in {claim.text, "# Library\n" + claim.text}:
            return {"entailment": 0.98, "neutral": 0.01, "contradiction": 0.01}
        if premise.strip() == "# Library":
            return {"entailment": 0.01, "neutral": 0.98, "contradiction": 0.01}
        return {"entailment": 0.03, "neutral": 0.07, "contradiction": 0.90}

    monkeypatch.setattr(checker, "_score_pair", score)
    monkeypatch.setattr(
        checker,
        "_score_pairs",
        lambda premises, hypothesis: [score(premise, hypothesis) for premise in premises],
    )
    result = checker._check_single_claim(claim, [evidence])
    assert result.verdict == Verdict.SUPPORTED
    assert result.supporting_chunk_id == "source"


def test_conflicting_cited_sources_cannot_be_overridden_by_higher_support(monkeypatch):
    checker = FaithfulnessChecker()
    claim = Claim(text="The fee is 10.", source_chunk_ids=["support", "conflict"])
    evidence = [
        RankedResult(id="support", content="The fee is 10."),
        RankedResult(id="conflict", content="The fee is not 10."),
    ]

    def score(premise, hypothesis):
        if premise == "The fee is 10.":
            return {"entailment": 0.99, "neutral": 0.005, "contradiction": 0.005}
        return {"entailment": 0.05, "neutral": 0.15, "contradiction": 0.8}

    monkeypatch.setattr(checker, "_score_pair", score)
    result = checker._check_single_claim(claim, evidence)
    assert result.verdict == Verdict.NOT_SUPPORTED
    assert result.contradicting_chunk_id == "conflict"


def test_fallback_does_not_discard_contradiction_in_another_cited_source(monkeypatch):
    checker = FaithfulnessChecker()
    claim = Claim(text="The fee is 10.", source_chunk_ids=["support", "conflict"])
    evidence = [
        RankedResult(id="support", content="# Guide\nThe fee is 10."),
        RankedResult(id="conflict", content="# Correction\nThe fee is not 10."),
    ]
    monkeypatch.setattr(
        checker,
        "_score_pair",
        lambda **_: {"entailment": 0.05, "neutral": 0.05, "contradiction": 0.9},
    )
    monkeypatch.setattr(checker, "_score_pairs", lambda *_: pytest.fail("conflict was discarded"))
    assert checker._check_single_claim(claim, evidence).verdict == Verdict.NOT_SUPPORTED


def test_window_recovery_rejects_a_refuted_quotation(monkeypatch):
    checker = FaithfulnessChecker()
    claim = Claim(text="The fee is 10.", source_chunk_ids=["source"])
    evidence = RankedResult(
        id="source", content="# Incorrect statement\nThe fee is 10.\nThe fee is 20."
    )

    monkeypatch.setattr(
        checker,
        "_score_pair",
        lambda **_: {"entailment": 0.05, "neutral": 0.05, "contradiction": 0.9},
    )

    def score_windows(premises, hypothesis):
        return [
            {"entailment": 0.99, "neutral": 0.005, "contradiction": 0.005}
            if premise == "The fee is 10."
            else {"entailment": 0.001, "neutral": 0.001, "contradiction": 0.998}
            if "Incorrect statement" in premise
            else {"entailment": 0.05, "neutral": 0.9, "contradiction": 0.05}
            for premise in premises
        ]

    monkeypatch.setattr(checker, "_score_pairs", score_windows)
    result = checker._check_single_claim(claim, [evidence])
    assert result.verdict == Verdict.NOT_SUPPORTED
    assert result.contradicting_chunk_id == "source"


def test_missing_cited_source_never_falls_back_to_other_evidence(monkeypatch):
    checker = FaithfulnessChecker()
    claim = Claim(text="The fee is 10.", source_chunk_ids=["missing"])
    evidence = RankedResult(id="other", content="The fee is 10.")
    monkeypatch.setattr(checker, "_score_pair", lambda *_: pytest.fail("unscoped evidence used"))
    result = checker._check_single_claim(claim, [evidence])
    assert result.verdict != Verdict.SUPPORTED
    assert result.supporting_chunk_id is None


def test_partial_cited_evidence_is_withheld(monkeypatch):
    checker = FaithfulnessChecker()
    claim = Claim(text="The fee is 10.", source_chunk_ids=["present", "missing"])
    evidence = RankedResult(id="present", content="The fee is 10.")
    monkeypatch.setattr(checker, "_score_pair", lambda **_: pytest.fail("partial sources used"))
    assert checker._check_single_claim(claim, [evidence]).verdict != Verdict.SUPPORTED


def test_excessive_number_of_windows_is_withheld_without_unbounded_model_work(monkeypatch):
    checker = FaithfulnessChecker()
    claim = Claim(text="The fee is 10.", source_chunk_ids=["source"])
    evidence = RankedResult(id="source", content="\n".join(f"Section {i}." for i in range(40)))
    monkeypatch.setattr(
        checker,
        "_score_pair",
        lambda **_: {"entailment": 0.05, "neutral": 0.05, "contradiction": 0.9},
    )
    monkeypatch.setattr(checker, "_score_pairs", lambda *_: pytest.fail("window cap exceeded"))
    assert checker._check_single_claim(claim, [evidence]).verdict == Verdict.NOT_SUPPORTED


def test_token_limit_is_enforced_before_inference(monkeypatch):
    checker = FaithfulnessChecker()

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.ones((1, 513), dtype=torch.int64)}

    class Model:
        config = SimpleNamespace(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})

        def __call__(self, **kwargs):
            pytest.fail("oversized input must not reach NLI inference")

    monkeypatch.setattr(checker, "_get_model_and_tokenizer", lambda: (Model(), Tokenizer()))
    with pytest.raises(ValueError, match="token limit"):
        checker._score_pair("long evidence", "claim")


def test_oversized_source_is_withheld_when_its_windows_also_exceed_limit(monkeypatch):
    checker = FaithfulnessChecker()
    claim = Claim(text="claim", source_chunk_ids=["short", "long"])
    evidence = [
        RankedResult(id="short", content="supported"),
        RankedResult(id="long", content="first passage\nsecond passage"),
    ]

    def score(premise, hypothesis):
        if premise == "supported":
            return {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}
        raise _EvidenceTooLongError("NLI evidence exceeds the model token limit")

    def oversized(*args):
        raise _EvidenceTooLongError("NLI evidence exceeds the model token limit")

    monkeypatch.setattr(checker, "_score_pair", score)
    monkeypatch.setattr(checker, "_score_pairs", oversized)
    assert checker._check_single_claim(claim, evidence).verdict != Verdict.SUPPORTED


def test_many_windows_are_inferred_in_small_batches(monkeypatch):
    checker = FaithfulnessChecker()
    batches = []

    class Tokenizer:
        def __call__(self, premises, hypotheses, **kwargs):
            batches.append(len(premises))
            return {"input_ids": torch.ones((len(premises), 3), dtype=torch.int64)}

    class Model:
        config = SimpleNamespace(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})

        def __call__(self, input_ids):
            return SimpleNamespace(logits=torch.ones((len(input_ids), 3)))

    monkeypatch.setattr(checker, "_get_model_and_tokenizer", lambda: (Model(), Tokenizer()))
    assert len(checker._score_pairs([str(i) for i in range(10)], "claim")) == 10
    assert batches == [4, 4, 2]
