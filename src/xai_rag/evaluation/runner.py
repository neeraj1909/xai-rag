"""Aggregate component metrics and enforce explicit evaluation gates."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
from xai_rag.evaluation.models import (
    EvaluationComparison,
    EvaluationDataset,
    EvaluationReport,
    GateResult,
    GateStatus,
    IngestionStatus,
    MetricComparison,
    MetricResult,
    MetricStatus,
    QueryEvaluationCase,
)
from xai_rag.models import Verdict

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class GateThreshold(BaseModel):
    """Inclusive minimum and/or maximum accepted value for one metric."""

    minimum: float | None = Field(default=None, alias="min")
    maximum: float | None = Field(default=None, alias="max")
    owner: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def require_boundary(self) -> GateThreshold:
        if self.minimum is None and self.maximum is None:
            raise ValueError("a gate requires min and/or max")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("gate min cannot exceed max")
        return self


class EvaluationGates(BaseModel):
    """Blocking and advisory metric thresholds."""

    blocking: dict[str, GateThreshold] = Field(default_factory=dict)
    advisory: dict[str, GateThreshold] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_toml(cls, path: Path) -> EvaluationGates:
        with path.open("rb") as handle:
            return cls.model_validate(tomllib.load(handle))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _metric(values: Sequence[float], *, detail: str = "") -> MetricResult:
    if not values:
        return MetricResult(
            value=None,
            sample_size=0,
            status=MetricStatus.NOT_COMPUTED,
            detail=detail or "No eligible observations",
        )
    return MetricResult(
        value=_mean(values),
        sample_size=len(values),
        status=MetricStatus.COMPUTED,
        detail=detail,
    )


def _value_metric(value: float, sample_size: int, *, detail: str = "") -> MetricResult:
    return MetricResult(
        value=value,
        sample_size=sample_size,
        status=MetricStatus.COMPUTED,
        detail=detail,
    )


def _stage_ids(case: QueryEvaluationCase, stage: str) -> list[str]:
    return getattr(case, f"{stage}_ids")


def _retrieval_metrics(
    cases: list[QueryEvaluationCase],
    k_values: tuple[int, ...],
) -> dict[str, MetricResult]:
    metrics: dict[str, MetricResult] = {}
    judged = [case for case in cases if any(grade > 0 for grade in case.relevance.values())]
    for stage in ("vector", "bm25", "fused", "reranked"):
        reciprocal_ranks = [
            reciprocal_rank(_stage_ids(case, stage), case.relevance) for case in judged
        ]
        metrics[f"retrieval.{stage}.mrr"] = _metric(
            reciprocal_ranks,
            detail="Macro average across queries with positive relevance judgments",
        )
        duplicate_rates = []
        for case in cases:
            result_ids = _stage_ids(case, stage)
            if result_ids:
                duplicate_rates.append(1.0 - len(set(result_ids)) / len(result_ids))
        metrics[f"retrieval.{stage}.duplicate_rate"] = _metric(duplicate_rates)

        for k in k_values:
            prefix = f"retrieval.{stage}"
            metrics[f"{prefix}.precision@{k}"] = _metric(
                [precision_at_k(_stage_ids(case, stage), case.relevance, k) for case in judged]
            )
            metrics[f"{prefix}.recall@{k}"] = _metric(
                [recall_at_k(_stage_ids(case, stage), case.relevance, k) for case in judged]
            )
            metrics[f"{prefix}.hit_rate@{k}"] = _metric(
                [hit_rate_at_k(_stage_ids(case, stage), case.relevance, k) for case in judged]
            )
            metrics[f"{prefix}.map@{k}"] = _metric(
                [
                    average_precision_at_k(_stage_ids(case, stage), case.relevance, k)
                    for case in judged
                ]
            )
            metrics[f"{prefix}.ndcg@{k}"] = _metric(
                [ndcg_at_k(_stage_ids(case, stage), case.relevance, k) for case in judged]
            )

    for k in k_values:
        metrics[f"retrieval.fusion.ndcg_delta@{k}"] = _metric(
            [
                ndcg_at_k(case.fused_ids, case.relevance, k)
                - max(
                    ndcg_at_k(case.vector_ids, case.relevance, k),
                    ndcg_at_k(case.bm25_ids, case.relevance, k),
                )
                for case in judged
            ],
            detail="Fused NDCG minus the better individual retriever per query",
        )
        metrics[f"retrieval.reranker.ndcg_delta@{k}"] = _metric(
            [
                ndcg_at_k(case.reranked_ids, case.relevance, k)
                - ndcg_at_k(case.fused_ids, case.relevance, k)
                for case in judged
            ],
            detail="Reranked NDCG minus fused NDCG per query",
        )
    return metrics


def _generation_metrics(cases: list[QueryEvaluationCase]) -> dict[str, MetricResult]:
    answered = [
        case for case in cases if case.answer is not None and case.actual_refusal is not True
    ]
    validity_values = [
        citation_validity(
            [*case.cited_ids, *(item for sources in case.claim_source_ids for item in sources)],
            case.reranked_ids,
        )
        for case in answered
    ]
    coverage_values = [
        citation_coverage(case.claim_source_ids, case.reranked_ids) for case in answered
    ]
    verdicts = [verdict for case in cases for verdict in case.claim_verdicts]
    reference_cases = [
        case for case in cases if case.answer is not None and case.expected_answer is not None
    ]
    metrics = {
        "generation.citation_validity": _metric(validity_values),
        "generation.citation_coverage": _metric(coverage_values),
        "generation.answer_exact_match": _metric(
            [
                normalized_exact_match(case.answer or "", case.expected_answer or "")
                for case in reference_cases
            ]
        ),
        "generation.answer_token_f1": _metric(
            [
                answer_token_f1(case.answer or "", case.expected_answer or "")
                for case in reference_cases
            ]
        ),
        "generation.claim_support_rate": _metric(
            [float(verdict == Verdict.SUPPORTED) for verdict in verdicts]
        ),
        "generation.claim_not_supported_rate": _metric(
            [float(verdict == Verdict.NOT_SUPPORTED) for verdict in verdicts]
        ),
        "generation.claim_neutral_rate": _metric(
            [float(verdict == Verdict.NEUTRAL) for verdict in verdicts]
        ),
    }
    judge_names = sorted({name for case in cases for name in case.judge_scores})
    for name in judge_names:
        metrics[f"generation.judge.{name}"] = _metric(
            [case.judge_scores[name] for case in cases if name in case.judge_scores]
        )
    human_names = sorted({name for case in cases for name in case.human_scores})
    for name in human_names:
        metrics[f"generation.human.{name}"] = _metric(
            [case.human_scores[name] for case in cases if name in case.human_scores]
        )
    return metrics


def _latency_metrics(cases: list[QueryEvaluationCase]) -> dict[str, MetricResult]:
    metrics: dict[str, MetricResult] = {}
    total_values = [case.total_latency_ms for case in cases if case.total_latency_ms is not None]
    for name, quantile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
        metrics[f"latency.total.{name}_ms"] = (
            _value_metric(percentile(total_values, quantile), len(total_values))
            if total_values
            else _metric([])
        )

    stage_names = sorted({stage for case in cases for stage in case.stage_latency_ms})
    for stage in stage_names:
        values = [case.stage_latency_ms[stage] for case in cases if stage in case.stage_latency_ms]
        for name, quantile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
            metrics[f"latency.{stage}.{name}_ms"] = _value_metric(
                percentile(values, quantile), len(values)
            )
    return metrics


def _ingestion_metrics(dataset: EvaluationDataset) -> dict[str, MetricResult]:
    records = dataset.ingestion_records
    statuses = [float(record.status == IngestionStatus.SUCCESS) for record in records]
    parity = [
        float(record.parity_consistent)
        for record in records
        if record.parity_consistent is not None
    ]
    chunk_lengths = [length for record in records for length in record.chunk_lengths]
    chunk_overlaps = [overlap for record in records for overlap in record.chunk_overlaps]
    total_chunks = len(chunk_lengths)
    valid_spans = sum(record.valid_span_count for record in records)
    provenance = sum(record.provenance_count for record in records)
    latency = [record.latency_ms for record in records if record.latency_ms is not None]

    metrics = {
        "ingestion.success_rate": _metric(statuses),
        "ingestion.failure_rate": _metric(
            [float(record.status == IngestionStatus.FAILED) for record in records]
        ),
        "ingestion.skip_rate": _metric(
            [float(record.status == IngestionStatus.SKIPPED) for record in records]
        ),
        "ingestion.index_parity_rate": _metric(parity),
        "ingestion.valid_span_coverage": (
            _value_metric(valid_spans / total_chunks, total_chunks) if total_chunks else _metric([])
        ),
        "ingestion.provenance_coverage": (
            _value_metric(provenance / total_chunks, total_chunks) if total_chunks else _metric([])
        ),
    }
    for name, quantile in (("p50", 0.50), ("p95", 0.95)):
        metrics[f"ingestion.chunk_length.{name}"] = (
            _value_metric(percentile(chunk_lengths, quantile), total_chunks)
            if chunk_lengths
            else _metric([])
        )
        metrics[f"ingestion.latency.{name}_ms"] = (
            _value_metric(percentile(latency, quantile), len(latency)) if latency else _metric([])
        )
        metrics[f"ingestion.chunk_overlap.{name}"] = (
            _value_metric(percentile(chunk_overlaps, quantile), len(chunk_overlaps))
            if chunk_overlaps
            else _metric([])
        )
    return metrics


def _apply_gates(
    metrics: dict[str, MetricResult],
    gates: EvaluationGates | None,
) -> tuple[list[GateResult], bool]:
    if gates is None:
        return [], True

    results: list[GateResult] = []
    passed = True
    for blocking, configured in ((True, gates.blocking), (False, gates.advisory)):
        for metric_name, threshold in configured.items():
            metric = metrics.get(metric_name)
            value = metric.value if metric and metric.status == MetricStatus.COMPUTED else None
            if value is None:
                status = GateStatus.NOT_COMPUTED
            elif (
                threshold.minimum is not None
                and value < threshold.minimum
                or threshold.maximum is not None
                and value > threshold.maximum
            ):
                status = GateStatus.FAIL
            else:
                status = GateStatus.PASS
            if blocking and status != GateStatus.PASS:
                passed = False
            results.append(
                GateResult(
                    metric=metric_name,
                    blocking=blocking,
                    status=status,
                    value=value,
                    minimum=threshold.minimum,
                    maximum=threshold.maximum,
                    owner=threshold.owner,
                    rationale=threshold.rationale,
                )
            )
    return results, passed


def evaluate_dataset(
    dataset: EvaluationDataset,
    *,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    gates: EvaluationGates | None = None,
) -> EvaluationReport:
    """Calculate a complete component report without calling models or services."""
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive cutoffs")

    metrics = _retrieval_metrics(dataset.query_cases, tuple(sorted(set(k_values))))
    metrics.update(_generation_metrics(dataset.query_cases))
    metrics.update(_latency_metrics(dataset.query_cases))
    metrics.update(_ingestion_metrics(dataset))

    cases = dataset.query_cases
    metrics["end_to_end.error_rate"] = _metric(
        [float(case.error_type is not None) for case in cases]
    )
    refusal_cases = [
        case
        for case in cases
        if case.expected_refusal is not None and case.actual_refusal is not None
    ]
    metrics["end_to_end.refusal_accuracy"] = _metric(
        [float(case.expected_refusal == case.actual_refusal) for case in refusal_cases]
    )
    input_tokens = [case.input_tokens for case in cases if case.input_tokens is not None]
    output_tokens = [case.output_tokens for case in cases if case.output_tokens is not None]
    costs = [case.estimated_cost_usd for case in cases if case.estimated_cost_usd is not None]
    metrics["usage.input_tokens.mean"] = _metric(input_tokens)
    metrics["usage.output_tokens.mean"] = _metric(output_tokens)
    metrics["usage.estimated_cost_usd.mean"] = _metric(costs)
    metrics["usage.estimated_cost_usd.total"] = (
        _value_metric(sum(costs), len(costs)) if costs else _metric([])
    )

    gate_results, passed = _apply_gates(metrics, gates)
    return EvaluationReport(
        manifest=dataset.manifest,
        dataset_name=dataset.manifest.dataset_name,
        dataset_revision=dataset.manifest.dataset_revision,
        dataset_digest=dataset.digest(),
        query_case_count=len(dataset.query_cases),
        ingestion_record_count=len(dataset.ingestion_records),
        metrics=metrics,
        gates=gate_results,
        passed=passed,
    )


def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
) -> EvaluationComparison:
    """Return candidate-minus-baseline deltas and expose missing metrics explicitly."""
    comparisons: dict[str, MetricComparison] = {}
    for metric_name in sorted(set(baseline.metrics) | set(candidate.metrics)):
        baseline_metric = baseline.metrics.get(metric_name)
        candidate_metric = candidate.metrics.get(metric_name)
        baseline_value = baseline_metric.value if baseline_metric else None
        candidate_value = candidate_metric.value if candidate_metric else None
        if baseline_value is None or candidate_value is None:
            comparisons[metric_name] = MetricComparison(
                baseline=baseline_value,
                candidate=candidate_value,
                delta=None,
                status=MetricStatus.NOT_COMPUTED,
            )
        else:
            comparisons[metric_name] = MetricComparison(
                baseline=baseline_value,
                candidate=candidate_value,
                delta=candidate_value - baseline_value,
                status=MetricStatus.COMPUTED,
            )
    return EvaluationComparison(
        baseline_digest=baseline.dataset_digest,
        candidate_digest=candidate.dataset_digest,
        metrics=comparisons,
    )
