"""Unit contracts for NLI label handling and claim aggregation."""

from types import SimpleNamespace

import pytest
import torch

from xai_rag.explainability.faithfulness import FaithfulnessChecker
from xai_rag.models import Claim, RankedResult, Verdict

pytestmark = pytest.mark.unit


def _chunk(chunk_id: str, content: str) -> RankedResult:
    return RankedResult(id=chunk_id, content=content, rrf_score=0.1, rrf_rank=1)


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
