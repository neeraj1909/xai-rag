"""Dependency-free metrics for individual RAG pipeline components."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero")


def _relevant_ids(relevance: dict[str, float]) -> set[str]:
    return {document_id for document_id, grade in relevance.items() if grade > 0.0}


def _require_judgments(relevance: dict[str, float]) -> set[str]:
    relevant = _relevant_ids(relevance)
    if not relevant:
        raise ValueError("relevance judgments must include at least one positive grade")
    return relevant


def precision_at_k(retrieved: Sequence[str], relevance: dict[str, float], k: int) -> float:
    """Return unique relevant hits in the first *k* ranks divided by *k*.

    Dividing by the requested cutoff deliberately penalizes short result lists.
    Duplicate IDs cannot earn relevance twice.
    """
    _validate_k(k)
    relevant = _require_judgments(relevance)
    seen: set[str] = set()
    hits = 0
    for document_id in retrieved[:k]:
        if document_id not in seen and document_id in relevant:
            hits += 1
        seen.add(document_id)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevance: dict[str, float], k: int) -> float:
    """Return the fraction of all judged-relevant IDs retrieved by rank *k*."""
    _validate_k(k)
    relevant = _require_judgments(relevance)
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def hit_rate_at_k(retrieved: Sequence[str], relevance: dict[str, float], k: int) -> float:
    """Return one when at least one relevant ID appears by rank *k*."""
    _validate_k(k)
    relevant = _require_judgments(relevance)
    return float(bool(set(retrieved[:k]) & relevant))


def reciprocal_rank(retrieved: Sequence[str], relevance: dict[str, float]) -> float:
    """Return reciprocal rank of the first relevant result, or zero."""
    relevant = _require_judgments(relevance)
    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    retrieved: Sequence[str],
    relevance: dict[str, float],
    k: int,
) -> float:
    """Return AP@k using binary relevance and unique retrieved IDs."""
    _validate_k(k)
    relevant = _require_judgments(relevance)
    seen: set[str] = set()
    hits = 0
    precision_sum = 0.0
    for rank, document_id in enumerate(retrieved[:k], start=1):
        if document_id in relevant and document_id not in seen:
            hits += 1
            precision_sum += hits / rank
        seen.add(document_id)
    return precision_sum / min(len(relevant), k)


def ndcg_at_k(retrieved: Sequence[str], relevance: dict[str, float], k: int) -> float:
    """Return normalized discounted cumulative gain at *k* for graded judgments."""
    _validate_k(k)
    positive_grades = [grade for grade in relevance.values() if grade > 0.0]
    if not positive_grades:
        raise ValueError("relevance judgments must include at least one positive grade")

    seen: set[str] = set()
    dcg = 0.0
    for rank, document_id in enumerate(retrieved[:k], start=1):
        grade = 0.0 if document_id in seen else max(0.0, relevance.get(document_id, 0.0))
        dcg += (2**grade - 1) / math.log2(rank + 1)
        seen.add(document_id)

    ideal_grades = sorted(positive_grades, reverse=True)[:k]
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1)
    )
    return dcg / ideal_dcg


def citation_validity(cited_ids: Sequence[str], available_ids: Collection[str]) -> float:
    """Return the valid-citation fraction, or zero when no citation was emitted."""
    unique_citations = set(cited_ids)
    if not unique_citations:
        return 0.0
    available = set(available_ids)
    return len(unique_citations & available) / len(unique_citations)


def citation_coverage(
    claim_source_ids: Sequence[Sequence[str]],
    available_ids: Collection[str],
) -> float:
    """Return valid claim coverage, or zero when no claim attribution was captured."""
    if not claim_source_ids:
        return 0.0
    available = set(available_ids)
    covered = sum(bool(set(source_ids) & available) for source_ids in claim_source_ids)
    return covered / len(claim_source_ids)


def normalized_exact_match(answer: str, reference: str) -> float:
    """Compare case-folded alphanumeric tokens after whitespace normalization."""

    def normalize(value: str) -> str:
        return " ".join(re.findall(r"\w+", value.casefold()))

    return float(normalize(answer) == normalize(reference))


def answer_token_f1(answer: str, reference: str) -> float:
    """Return bag-of-token F1 against a human reference answer."""
    answer_tokens = re.findall(r"\w+", answer.casefold())
    reference_tokens = re.findall(r"\w+", reference.casefold())
    if not answer_tokens and not reference_tokens:
        return 1.0
    if not answer_tokens or not reference_tokens:
        return 0.0
    common = sum((Counter(answer_tokens) & Counter(reference_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(answer_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for quantile in ``[0, 1]``."""
    if not values:
        raise ValueError("at least one value is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
