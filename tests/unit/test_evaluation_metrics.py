"""Hand-computed contracts for production RAG component metrics."""

import math

import pytest

from xai_rag.evaluation.metrics import (
    answer_token_f1,
    average_precision_at_k,
    citation_coverage,
    citation_validity,
    hit_rate_at_k,
    ndcg_at_k,
    normalized_exact_match,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

pytestmark = pytest.mark.unit

RELEVANCE = {"a": 1.0, "c": 3.0, "d": 2.0}
RETRIEVED = ["a", "b", "c"]


def test_binary_retrieval_metrics_match_hand_calculation() -> None:
    assert precision_at_k(RETRIEVED, RELEVANCE, 3) == pytest.approx(2 / 3)
    assert recall_at_k(RETRIEVED, RELEVANCE, 3) == pytest.approx(2 / 3)
    assert hit_rate_at_k(RETRIEVED, RELEVANCE, 3) == 1.0
    assert reciprocal_rank(RETRIEVED, RELEVANCE) == 1.0
    assert average_precision_at_k(RETRIEVED, RELEVANCE, 3) == pytest.approx((1 + 2 / 3) / 3)


def test_ndcg_uses_graded_relevance_and_log_discount() -> None:
    actual_dcg = (2**1 - 1) / math.log2(2) + (2**3 - 1) / math.log2(4)
    ideal_dcg = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4)

    assert ndcg_at_k(RETRIEVED, RELEVANCE, 3) == pytest.approx(actual_dcg / ideal_dcg)


def test_duplicate_retrieval_ids_do_not_double_count_relevance() -> None:
    assert precision_at_k(["a", "a"], RELEVANCE, 2) == 0.5
    assert recall_at_k(["a", "a"], RELEVANCE, 2) == pytest.approx(1 / 3)


def test_unjudged_case_is_not_silently_treated_as_zero_quality() -> None:
    with pytest.raises(ValueError, match="relevance judgments"):
        recall_at_k(["a"], {}, 1)


def test_citation_validity_and_claim_coverage_are_separate() -> None:
    assert citation_validity(["known", "missing"], {"known"}) == 0.5
    assert citation_validity([], {"known"}) == 0.0
    assert citation_coverage([["known"], [], ["missing"]], {"known"}) == pytest.approx(1 / 3)
    assert citation_coverage([], {"known"}) == 0.0


def test_percentile_uses_linear_interpolation() -> None:
    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 0.5) == 25.0
    assert percentile(values, 0.95) == pytest.approx(38.5)


def test_reference_answer_metrics_are_deterministic_and_case_insensitive() -> None:
    assert normalized_exact_match("R", "R/yes") == 0.0
    assert normalized_exact_match("  Reciprocal-Rank Fusion! ", "reciprocal rank fusion") == 1.0
    assert answer_token_f1("alpha beta", "alpha gamma") == 0.5


@pytest.mark.parametrize("k", [0, -1])
def test_retrieval_metrics_reject_invalid_cutoff(k: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        precision_at_k(RETRIEVED, RELEVANCE, k)
