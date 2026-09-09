"""Versioned, serializable contracts for offline and production RAG evaluation."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum, auto
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from xai_rag.models import Verdict  # noqa: TC001 - Pydantic resolves this runtime enum


class EvaluationModel(BaseModel):
    """Strict base so misspelled evidence fields cannot be silently discarded."""

    model_config = ConfigDict(extra="forbid")


class EvaluationManifest(EvaluationModel):
    """Reproducibility metadata shared by every case in an evaluation dataset."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    corpus_revision: str = Field(min_length=1)
    config_revision: str = Field(min_length=1)
    evaluator_version: str = "1.0"
    model_revisions: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryEvaluationCase(EvaluationModel):
    """One judged query and the observed outputs from every RAG stage."""

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevance: dict[str, float] = Field(default_factory=dict)
    expected_answer: str | None = None
    expected_refusal: bool | None = None
    vector_ids: list[str] = Field(default_factory=list)
    bm25_ids: list[str] = Field(default_factory=list)
    fused_ids: list[str] = Field(default_factory=list)
    reranked_ids: list[str] = Field(default_factory=list)
    answer: str | None = None
    cited_ids: list[str] = Field(default_factory=list)
    claim_source_ids: list[list[str]] = Field(default_factory=list)
    claim_verdicts: list[Verdict] = Field(default_factory=list)
    actual_refusal: bool | None = None
    judge_scores: dict[str, float] = Field(default_factory=dict)
    human_scores: dict[str, float] = Field(default_factory=dict)
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float | None = Field(default=None, ge=0.0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    error_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("relevance")
    @classmethod
    def validate_relevance(cls, relevance: dict[str, float]) -> dict[str, float]:
        if any(grade < 0.0 for grade in relevance.values()):
            raise ValueError("relevance grades must be non-negative")
        return relevance

    @field_validator("stage_latency_ms")
    @classmethod
    def validate_stage_latency(cls, latencies: dict[str, float]) -> dict[str, float]:
        if any(value < 0.0 for value in latencies.values()):
            raise ValueError("stage latency values must be non-negative")
        return latencies

    @field_validator("judge_scores", "human_scores")
    @classmethod
    def validate_scores(cls, scores: dict[str, float]) -> dict[str, float]:
        if any(not 0.0 <= value <= 1.0 for value in scores.values()):
            raise ValueError("judge and human scores must be between zero and one")
        return scores


class IngestionStatus(StrEnum):
    """Terminal outcome for one document ingestion attempt."""

    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


class IngestionEvaluationRecord(EvaluationModel):
    """Observed parsing, chunking, provenance, and store outcome for one document."""

    document_id: str = Field(min_length=1)
    status: IngestionStatus
    source_chars: int = Field(ge=0)
    chunk_lengths: list[int] = Field(default_factory=list)
    chunk_overlaps: list[int] = Field(default_factory=list)
    valid_span_count: int = Field(default=0, ge=0)
    provenance_count: int = Field(default=0, ge=0)
    parity_consistent: bool | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    error_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_lengths")
    @classmethod
    def validate_chunk_lengths(cls, lengths: list[int]) -> list[int]:
        if any(length <= 0 for length in lengths):
            raise ValueError("chunk lengths must be greater than zero")
        return lengths

    @field_validator("chunk_overlaps")
    @classmethod
    def validate_chunk_overlaps(cls, overlaps: list[int]) -> list[int]:
        if any(overlap < 0 for overlap in overlaps):
            raise ValueError("chunk overlaps must be non-negative")
        return overlaps

    @model_validator(mode="after")
    def validate_counts(self) -> IngestionEvaluationRecord:
        chunk_count = len(self.chunk_lengths)
        if self.valid_span_count > chunk_count or self.provenance_count > chunk_count:
            raise ValueError("valid span and provenance counts cannot exceed chunk count")
        return self


class EvaluationDataset(EvaluationModel):
    """A complete versioned evaluation dataset and captured observations."""

    manifest: EvaluationManifest
    query_cases: list[QueryEvaluationCase] = Field(default_factory=list)
    ingestion_records: list[IngestionEvaluationRecord] = Field(default_factory=list)

    def digest(self) -> str:
        """Return a deterministic digest over the complete dataset."""
        canonical = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_jsonl(self) -> str:
        """Serialize as typed JSONL records without losing the shared manifest."""
        records = [{"record_type": "manifest", "data": self.manifest.model_dump(mode="json")}]
        records.extend(
            {"record_type": "query", "data": case.model_dump(mode="json")}
            for case in self.query_cases
        )
        records.extend(
            {"record_type": "ingestion", "data": record.model_dump(mode="json")}
            for record in self.ingestion_records
        )
        return "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records
        )

    @classmethod
    def from_jsonl(cls, content: str) -> EvaluationDataset:
        """Deserialize the typed JSONL representation."""
        manifest: EvaluationManifest | None = None
        query_cases: list[QueryEvaluationCase] = []
        ingestion_records: list[IngestionEvaluationRecord] = []

        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_type = record["record_type"]
                data = record["data"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid evaluation JSONL record on line {line_number}") from exc
            if record_type == "manifest":
                if manifest is not None:
                    raise ValueError("evaluation JSONL contains more than one manifest")
                manifest = EvaluationManifest.model_validate(data)
            elif record_type == "query":
                query_cases.append(QueryEvaluationCase.model_validate(data))
            elif record_type == "ingestion":
                ingestion_records.append(IngestionEvaluationRecord.model_validate(data))
            else:
                raise ValueError(f"unknown evaluation record type: {record_type}")

        if manifest is None:
            raise ValueError("evaluation JSONL is missing its manifest")
        return cls(
            manifest=manifest,
            query_cases=query_cases,
            ingestion_records=ingestion_records,
        )


class MetricStatus(StrEnum):
    """Whether a metric had enough evidence to be calculated."""

    COMPUTED = auto()
    NOT_COMPUTED = auto()


class MetricResult(EvaluationModel):
    """One metric value with its evidence count and computation status."""

    value: float | None = None
    sample_size: int = Field(default=0, ge=0)
    status: MetricStatus
    detail: str = ""


class GateStatus(StrEnum):
    """Outcome of evaluating one quality threshold."""

    PASS = auto()
    FAIL = auto()
    NOT_COMPUTED = auto()


class GateResult(EvaluationModel):
    """Evaluation of one blocking or advisory gate."""

    metric: str
    blocking: bool
    status: GateStatus
    value: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    owner: str
    rationale: str


class EvaluationReport(EvaluationModel):
    """Machine-readable component and end-to-end evaluation output."""

    schema_version: Literal["1.0"] = "1.0"
    manifest: EvaluationManifest
    dataset_name: str
    dataset_revision: str
    dataset_digest: str
    query_case_count: int = Field(ge=0)
    ingestion_record_count: int = Field(ge=0)
    metrics: dict[str, MetricResult]
    gates: list[GateResult] = Field(default_factory=list)
    passed: bool


class MetricComparison(EvaluationModel):
    """Raw candidate-minus-baseline delta without assuming metric direction."""

    baseline: float | None = None
    candidate: float | None = None
    delta: float | None = None
    status: MetricStatus


class EvaluationComparison(EvaluationModel):
    """Metric-by-metric comparison of two evaluation reports."""

    baseline_digest: str
    candidate_digest: str
    metrics: dict[str, MetricComparison]
