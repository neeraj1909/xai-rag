"""Evaluation dataset, report, and gate contracts."""

import pytest

from xai_rag.evaluation.models import (
    EvaluationDataset,
    EvaluationManifest,
    IngestionEvaluationRecord,
    IngestionStatus,
    MetricResult,
    MetricStatus,
    QueryEvaluationCase,
)
from xai_rag.evaluation.runner import (
    EvaluationGates,
    _apply_gates,
    compare_reports,
    evaluate_dataset,
)
from xai_rag.models import Verdict

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_gate_threshold_is_rejected(number: float) -> None:
    with pytest.raises(ValueError):
        EvaluationGates.model_validate(
            {
                "blocking": {
                    "metric": {"min": number, "owner": "engineering", "rationale": "Invalid number"}
                }
            }
        )


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_observations_in_jsonl_are_rejected(number: float) -> None:
    import json

    records = [json.loads(line) for line in _dataset().to_jsonl().splitlines()]
    records[1]["data"]["relevance"]["a"] = number
    with pytest.raises(ValueError):
        EvaluationDataset.from_jsonl("\n".join(map(json.dumps, records)))


def test_nonfinite_metric_cannot_pass_even_if_validation_was_bypassed() -> None:
    metric = MetricResult.model_construct(
        value=float("nan"), sample_size=1, status=MetricStatus.COMPUTED
    )
    gates = EvaluationGates.model_validate(
        {
            "blocking": {
                "metric": {"min": 0.9, "owner": "engineering", "rationale": "Must be finite"}
            }
        }
    )
    results, passed = _apply_gates({"metric": metric}, gates)
    assert not passed
    assert results[0].status == "not_computed"


def test_nonfinite_mutated_gate_cannot_pass() -> None:
    gates = EvaluationGates.model_validate(
        {
            "blocking": {
                "metric": {"min": 0.9, "owner": "engineering", "rationale": "Must be finite"}
            }
        }
    )
    gates.blocking["metric"].minimum = float("nan")
    metric = MetricResult(value=1.0, sample_size=1, status=MetricStatus.COMPUTED)
    results, passed = _apply_gates({"metric": metric}, gates)
    assert not passed
    assert results[0].status == "not_computed"


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        manifest=EvaluationManifest(
            dataset_name="unit",
            dataset_revision="v1",
            corpus_revision="corpus-1",
            config_revision="config-1",
            model_revisions={"embedding": "embed-1"},
        ),
        query_cases=[
            QueryEvaluationCase(
                case_id="q1",
                query="What is RRF?",
                relevance={"a": 3.0, "b": 1.0},
                vector_ids=["b", "x"],
                bm25_ids=["a", "x"],
                fused_ids=["a", "b"],
                reranked_ids=["a", "b"],
                prompt_visible_ids=["a", "b"],
                answer="RRF combines ranks [a].",
                expected_answer="RRF combines ranks [a].",
                cited_ids=["a"],
                claim_source_ids=[["a"]],
                claim_verdicts=[Verdict.SUPPORTED],
                expected_refusal=False,
                actual_refusal=False,
                judge_scores={"answer_relevance": 0.9},
                human_scores={"answer_correctness": 1.0},
                stage_latency_ms={"retrieve": 20.0, "generate": 30.0},
                total_latency_ms=55.0,
                input_tokens=100,
                output_tokens=20,
                estimated_cost_usd=0.002,
            ),
            QueryEvaluationCase(
                case_id="q2",
                query="Unknown question",
                relevance={},
                reranked_ids=[],
                answer="I don't know.",
                expected_refusal=True,
                actual_refusal=True,
                total_latency_ms=10.0,
            ),
        ],
        ingestion_records=[
            IngestionEvaluationRecord(
                document_id="doc-1",
                status=IngestionStatus.SUCCESS,
                source_chars=100,
                chunk_lengths=[40, 60],
                chunk_overlaps=[0, 5],
                valid_span_count=2,
                provenance_count=2,
                parity_consistent=True,
                latency_ms=5.0,
            )
        ],
    )


def test_dataset_jsonl_round_trip_and_digest_are_stable() -> None:
    dataset = _dataset()

    round_tripped = EvaluationDataset.from_jsonl(dataset.to_jsonl())

    assert round_tripped == dataset
    assert round_tripped.digest() == dataset.digest()


def test_runner_reports_component_metrics_and_sample_sizes() -> None:
    report = evaluate_dataset(_dataset(), k_values=(1, 2))

    assert report.metrics["retrieval.vector.recall@2"].value == 0.5
    assert report.metrics["retrieval.vector.recall@2"].sample_size == 1
    assert report.metrics["retrieval.reranked.ndcg@2"].value == 1.0
    assert report.metrics["generation.citation_validity"].value == 1.0
    assert report.metrics["generation.citation_validity"].sample_size == 1
    assert report.metrics["generation.answer_exact_match"].value == 1.0
    assert report.metrics["generation.judge.answer_relevance"].value == 0.9
    assert report.metrics["generation.claim_support_rate"].value == 1.0
    assert report.metrics["generation.claim_not_supported_rate"].value == 0.0
    assert "generation.claim_contradiction_rate" not in report.metrics
    assert report.metrics["end_to_end.refusal_accuracy"].value == 1.0
    assert report.metrics["end_to_end.error_rate"].value == 0.0
    assert report.metrics["latency.total.p95_ms"].value == pytest.approx(52.75)
    assert report.metrics["ingestion.index_parity_rate"].value == 1.0
    assert report.metrics["ingestion.provenance_coverage"].value == 1.0
    assert report.metrics["usage.estimated_cost_usd.total"].value == 0.002
    assert report.manifest.corpus_revision == "corpus-1"


def test_unavailable_metric_has_explicit_not_computed_status() -> None:
    dataset = _dataset()
    dataset.query_cases[0].claim_verdicts = []

    report = evaluate_dataset(dataset, k_values=(1,))
    metric = report.metrics["generation.claim_support_rate"]

    assert metric.status == "not_computed"
    assert metric.value is None
    assert metric.sample_size == 0


def test_blocking_and_advisory_gates_are_distinguished() -> None:
    gates = EvaluationGates.model_validate(
        {
            "blocking": {
                "generation.citation_validity": {
                    "min": 1.0,
                    "owner": "engineering",
                    "rationale": "Citations must resolve",
                },
                "end_to_end.error_rate": {
                    "max": 0.0,
                    "owner": "engineering",
                    "rationale": "Smoke cases must run",
                },
            },
            "advisory": {
                "latency.total.p95_ms": {
                    "max": 1.0,
                    "owner": "product",
                    "rationale": "Candidate SLO",
                }
            },
        }
    )

    report = evaluate_dataset(_dataset(), k_values=(1,), gates=gates)

    assert report.passed is True
    assert {gate.status for gate in report.gates if not gate.blocking} == {"fail"}


def test_missing_blocking_metric_does_not_pass() -> None:
    gates = EvaluationGates.model_validate(
        {
            "blocking": {
                "retrieval.vector.recall@99": {
                    "min": 0.5,
                    "owner": "engineering",
                    "rationale": "Missing metrics cannot pass",
                }
            }
        }
    )

    report = evaluate_dataset(_dataset(), k_values=(1,), gates=gates)

    assert report.passed is False
    assert report.gates[0].status == "not_computed"


def test_answer_without_citation_evidence_fails_blocking_gates() -> None:
    dataset = _dataset()
    dataset.query_cases[0].cited_ids = []
    dataset.query_cases[0].claim_source_ids = []
    gates = EvaluationGates.model_validate(
        {
            "blocking": {
                "generation.citation_validity": {
                    "min": 1.0,
                    "owner": "engineering",
                    "rationale": "An answer must emit only resolvable citations",
                },
                "generation.citation_coverage": {
                    "min": 1.0,
                    "owner": "engineering",
                    "rationale": "An answer must attribute every claim",
                },
            }
        }
    )

    report = evaluate_dataset(dataset, k_values=(1,), gates=gates)

    assert report.passed is False
    assert {gate.status for gate in report.gates} == {"fail"}


def test_report_comparison_exposes_candidate_minus_baseline_delta() -> None:
    baseline = evaluate_dataset(_dataset(), k_values=(1,))
    candidate_dataset = _dataset()
    candidate_dataset.query_cases[0].reranked_ids = ["x"]
    candidate = evaluate_dataset(candidate_dataset, k_values=(1,))

    comparison = compare_reports(baseline, candidate)

    metric = comparison.metrics["retrieval.reranked.recall@1"]
    assert metric.baseline == 0.5
    assert metric.candidate == 0.0
    assert metric.delta == -0.5


def test_citation_gate_rejects_ranked_but_not_visible_chunk() -> None:
    dataset = _dataset()
    case = dataset.query_cases[0]
    case.prompt_visible_ids = ["b"]
    gates = EvaluationGates.model_validate(
        {
            "blocking": {
                "generation.citation_validity": {
                    "min": 1.0,
                    "owner": "engineering",
                    "rationale": "Only visible citations are valid",
                }
            }
        }
    )

    report = evaluate_dataset(dataset, gates=gates)

    assert report.metrics["generation.citation_validity"].value == 0.0
    assert report.gates[0].status == "fail"
    assert not report.passed


def test_legacy_unknown_prompt_visibility_fails_closed() -> None:
    dataset = _dataset()
    dataset.query_cases[0].prompt_visible_ids = None
    gates = EvaluationGates.model_validate(
        {
            "blocking": {
                "generation.citation_validity": {
                    "min": 1.0,
                    "owner": "engineering",
                    "rationale": "Need visible evidence",
                }
            }
        }
    )

    report = evaluate_dataset(dataset, gates=gates)

    assert report.metrics["generation.citation_validity"].status == "not_computed"
    assert report.gates[0].status == "not_computed"
    assert not report.passed


def test_legacy_jsonl_without_prompt_visibility_remains_readable() -> None:
    import json

    records = [json.loads(line) for line in _dataset().to_jsonl().splitlines()]
    for record in records:
        if record["record_type"] == "query":
            record["data"].pop("prompt_visible_ids")
    dataset = EvaluationDataset.from_jsonl("\n".join(map(json.dumps, records)))

    assert dataset.query_cases[0].prompt_visible_ids is None
    assert (
        evaluate_dataset(dataset).metrics["generation.citation_validity"].status == "not_computed"
    )


def test_citation_metrics_do_not_hide_missing_visibility_among_answered_cases() -> None:
    dataset = _dataset()
    dataset.query_cases[0].prompt_visible_ids = None
    second = dataset.query_cases[0].model_copy(deep=True)
    second.case_id = "q3"
    second.prompt_visible_ids = ["a"]
    dataset.query_cases.append(second)

    report = evaluate_dataset(dataset)

    assert report.metrics["generation.citation_validity"].status == "not_computed"
    assert report.metrics["generation.citation_coverage"].status == "not_computed"


def test_explicit_empty_prompt_visibility_is_not_unknown() -> None:
    dataset = _dataset()
    dataset.query_cases[0].prompt_visible_ids = []

    metric = evaluate_dataset(dataset).metrics["generation.citation_validity"]

    assert metric.status == "computed"
    assert metric.value == 0.0


def test_blocking_gate_requires_minimum_eligible_samples() -> None:
    gates = EvaluationGates.model_validate(
        {
            "blocking": {
                "end_to_end.error_rate": {
                    "max": 0.0,
                    "min_samples": 3,
                    "owner": "engineering",
                    "rationale": "Sufficient cases",
                }
            }
        }
    )

    report = evaluate_dataset(_dataset(), gates=gates)

    assert report.metrics["end_to_end.error_rate"].sample_size == 2
    assert report.gates[0].status == "not_computed"
    assert report.gates[0].sample_size == 2
    assert report.gates[0].min_samples == 3
    assert not report.passed


def test_minimum_sample_gate_accepts_boundary() -> None:
    gates = EvaluationGates.model_validate(
        {
            "blocking": {
                "end_to_end.error_rate": {
                    "max": 0.0,
                    "min_samples": 2,
                    "owner": "engineering",
                    "rationale": "Sufficient cases",
                }
            }
        }
    )

    assert evaluate_dataset(_dataset(), gates=gates).passed


def test_minimum_sample_gate_rejects_zero() -> None:
    with pytest.raises(ValueError):
        EvaluationGates.model_validate(
            {
                "blocking": {
                    "end_to_end.error_rate": {
                        "max": 0.0,
                        "min_samples": 0,
                        "owner": "engineering",
                        "rationale": "Invalid",
                    }
                }
            }
        )
