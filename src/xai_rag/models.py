"""Domain models for the XAI-RAG pipeline.

All data classes use Pydantic for validation, serialization, and
OpenAPI schema generation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievalStage(StrEnum):
    """Retrieval stage that produced a native score and rank."""

    VECTOR = "vector"
    BM25 = "bm25"


class SearchResult(BaseModel):
    """A single document chunk returned by a retrieval stage."""

    id: str
    content: str
    context_content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    stage: RetrievalStage | None = None
    rank: int | None = Field(default=None, ge=1)


class FusedResult(BaseModel):
    """A result after rank fusion with every native score preserved."""

    id: str
    content: str
    context_content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector_score: float | None = None
    vector_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    bm25_rank: int | None = Field(default=None, ge=1)
    rrf_score: float = Field(default=0.0, ge=0.0)
    rrf_rank: int | None = Field(default=None, ge=1)

    def to_ranked(
        self,
        *,
        reranker_score: float | None = None,
        reranker_rank: int | None = None,
    ) -> RankedResult:
        """Promote a fused result without changing any retrieval-stage values."""
        return RankedResult(
            **self.model_dump(),
            reranker_score=reranker_score,
            reranker_rank=reranker_rank,
        )


class RankedResult(FusedResult):
    """A fused result after cross-encoder reranking."""

    reranker_score: float | None = None
    reranker_rank: int | None = Field(default=None, ge=1)


class RetrievalExplanation(BaseModel):
    """Human-readable explanation of why a chunk was selected."""

    chunk_id: str
    content_preview: str
    vector_score: float | None = None
    vector_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    bm25_rank: int | None = Field(default=None, ge=1)
    rrf_score: float = Field(default=0.0, ge=0.0)
    rrf_rank: int | None = Field(default=None, ge=1)
    reranker_score: float | None = None
    reranker_rank: int | None = Field(default=None, ge=1)
    matching_terms: list[str] = Field(default_factory=list)
    selection_reason: str = ""


class Claim(BaseModel):
    """A factual claim extracted from the generated answer."""

    text: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class Verdict(StrEnum):
    """NLI-based faithfulness verdict."""

    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    NEUTRAL = "neutral"


class ClaimVerdict(BaseModel):
    """Faithfulness check result for a single claim."""

    claim: Claim
    verdict: Verdict
    nli_entailment: float = 0.0
    nli_contradiction: float = 0.0
    supporting_chunk_id: str | None = None
    contradicting_chunk_id: str | None = None
    confidence: float = 0.0


class RAGGenerationResult(BaseModel):
    """Structured output from the RAG generator."""

    answer: str
    claims: list[Claim] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class QueryRequest(BaseModel):
    """Input contract for a full XAI-RAG query."""

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=100)
    include_explanations: bool = True
    include_faithfulness: bool = False
    include_ragas: bool = False


class XAIRAGResponse(BaseModel):
    """API response containing the answer and explainability artefacts."""

    query: str
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    retrieval_explanations: list[RetrievalExplanation] = Field(default_factory=list)
    faithfulness_report: list[ClaimVerdict] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ragas_scores: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = Field(default=0.0, ge=0.0)
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class IngestionDocumentResult(BaseModel):
    """Terminal outcome for one document in an ingestion request."""

    source_file: str
    status: Literal["success", "skipped", "failed"]
    chunk_count: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error_type: str | None = None


class IngestionResponse(BaseModel):
    """Bounded ingestion outcome with explicit partial-failure and parity state."""

    status: Literal["success", "partial", "failed"]
    discovered_files: int = Field(ge=0)
    ingested_files: int = Field(ge=0)
    skipped_files: int = Field(ge=0)
    failed_files: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    parity_consistent: bool | None = None
    documents: list[IngestionDocumentResult] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0.0)
