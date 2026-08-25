"""Domain models for the XAI-RAG pipeline.

All data classes use Pydantic for validation, serialization, and
OpenAPI schema generation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single document chunk returned by a retrieval stage."""

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class RankedResult(BaseModel):
    """A search result after hybrid fusion and cross-encoder reranking."""

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rrf_rank: int = 0
    reranker_score: float = 0.0


class RetrievalExplanation(BaseModel):
    """Human-readable explanation of why a chunk was selected."""

    chunk_id: str
    content_preview: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rrf_rank: int = 0
    reranker_score: float = 0.0
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
    confidence: float = 0.0


class RAGGenerationResult(BaseModel):
    """Structured output from the RAG generator."""

    answer: str
    claims: list[Claim] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


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
