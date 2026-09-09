"""Application service shared by HTTP, SSE, and command-line adapters."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from xai_rag.config import settings
from xai_rag.evaluation.ragas_eval import evaluate_response
from xai_rag.explainability.faithfulness import FaithfulnessChecker
from xai_rag.explainability.retrieval_explainer import RetrievalExplainer
from xai_rag.generation.generator import RAGGenerator
from xai_rag.ingestion.chunker import chunk_text
from xai_rag.ingestion.embedder import embed_query, embed_texts
from xai_rag.ingestion.parser import SUPPORTED_SUFFIXES, parse_file, source_identity
from xai_rag.ingestion.store import (
    StorageWriteError,
    check_index_parity,
    store_chunks_chromadb,
    store_chunks_elasticsearch,
)
from xai_rag.models import (
    IngestionDocumentResult,
    IngestionResponse,
    QueryRequest,
    XAIRAGResponse,
)
from xai_rag.observability.tracing import hash_text, pipeline_stage
from xai_rag.retrieval.bm25_search import bm25_search
from xai_rag.retrieval.hybrid import rrf_fusion
from xai_rag.retrieval.reranker import Reranker
from xai_rag.retrieval.vector_search import vector_search

if TYPE_CHECKING:
    from pathlib import Path

    import chromadb
    from elasticsearch import AsyncElasticsearch

EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]


class RequestEvaluationDisabledError(ValueError):
    """Raised when expensive request-path evaluation has not been enabled."""


class IngestionLimitError(ValueError):
    """Raised before parsing when an ingestion request exceeds a configured bound."""


def discover_document_paths(
    source: Path,
    document_root: Path,
    *,
    max_files: int,
    max_document_bytes: int,
    max_total_bytes: int,
) -> list[Path]:
    """Resolve supported files under a root and enforce count/size bounds."""
    root = document_root.expanduser().resolve()
    resolved_source = source.expanduser().resolve()
    try:
        resolved_source.relative_to(root)
    except ValueError as exc:
        raise IngestionLimitError("ingestion source is outside the document root") from exc

    if resolved_source.is_file():
        if resolved_source.suffix.casefold() not in SUPPORTED_SUFFIXES:
            raise IngestionLimitError("ingestion source has an unsupported file type")
        candidates = [resolved_source]
    elif resolved_source.is_dir():
        candidates = sorted(
            path.resolve()
            for path in resolved_source.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
        )
    else:
        raise IngestionLimitError("ingestion source does not exist")

    if len(candidates) > max_files:
        raise IngestionLimitError(f"ingestion request exceeds the {max_files}-file limit")

    total_bytes = 0
    for path in candidates:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise IngestionLimitError("resolved document is outside the document root") from exc
        size = path.stat().st_size
        if size > max_document_bytes:
            raise IngestionLimitError(
                f"document exceeds the {max_document_bytes}-byte per-file limit"
            )
        total_bytes += size
        if total_bytes > max_total_bytes:
            raise IngestionLimitError(
                f"ingestion request exceeds the {max_total_bytes}-byte total limit"
            )
    return candidates


class RAGService:
    """Coordinate one query while keeping transports and dependencies replaceable."""

    def __init__(
        self,
        *,
        collection: chromadb.Collection,
        es_client: AsyncElasticsearch,
        embed_query_fn=embed_query,
        vector_search_fn=vector_search,
        bm25_search_fn=bm25_search,
        fusion_fn=rrf_fusion,
        reranker_factory=Reranker,
        generator_factory=RAGGenerator,
        explainer_factory=RetrievalExplainer,
        faithfulness_factory=FaithfulnessChecker,
        evaluator_fn=evaluate_response,
        parse_file_fn=parse_file,
        chunk_text_fn=chunk_text,
        embed_texts_fn=embed_texts,
        store_chroma_fn=store_chunks_chromadb,
        store_elasticsearch_fn=store_chunks_elasticsearch,
        parity_fn=check_index_parity,
        allow_request_evaluation: bool | None = None,
    ) -> None:
        self._collection = collection
        self._es_client = es_client
        self._embed_query = embed_query_fn
        self._vector_search = vector_search_fn
        self._bm25_search = bm25_search_fn
        self._fusion = fusion_fn
        self._reranker_factory = reranker_factory
        self._generator_factory = generator_factory
        self._explainer_factory = explainer_factory
        self._faithfulness_factory = faithfulness_factory
        self._evaluator = evaluator_fn
        self._parse_file = parse_file_fn
        self._chunk_text = chunk_text_fn
        self._embed_texts = embed_texts_fn
        self._store_chroma = store_chroma_fn
        self._store_elasticsearch = store_elasticsearch_fn
        self._check_parity = parity_fn
        self._allow_request_evaluation = (
            settings.allow_request_evaluation
            if allow_request_evaluation is None
            else allow_request_evaluation
        )

    async def _timed(
        self,
        name: str,
        operation: Callable[[], Awaitable[Any]],
        timings: dict[str, float],
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        start = time.perf_counter()
        with pipeline_stage(f"rag.{name}", attributes):
            result = await operation()
        timings[name] = (time.perf_counter() - start) * 1000
        return result

    async def _emit(
        self,
        event_sink: EventSink | None,
        event: str,
        data: dict[str, Any],
    ) -> None:
        if event_sink is not None:
            await event_sink(event, data)

    async def query(
        self,
        request: QueryRequest,
        *,
        event_sink: EventSink | None = None,
    ) -> XAIRAGResponse:
        """Run the canonical query pipeline and optionally publish stage events."""
        if request.include_ragas and not self._allow_request_evaluation:
            raise RequestEvaluationDisabledError(
                "request-scoped external evaluation is disabled; use the offline evaluator"
            )

        started = time.perf_counter()
        timings: dict[str, float] = {}
        with pipeline_stage(
            "rag.query",
            {
                "xai_rag.query_hash": hash_text(request.query),
                "xai_rag.top_k": request.top_k,
                "xai_rag.faithfulness_enabled": request.include_faithfulness,
            },
        ):
            query_embedding = await self._timed(
                "embed",
                lambda: self._embed_query(request.query),
                timings,
                attributes={
                    "xai_rag.model": settings.embedding_model,
                    "xai_rag.model_revision": settings.embedding_model_revision,
                },
            )

            retrieval_started = time.perf_counter()
            with pipeline_stage("rag.retrieve"):
                vector_results, bm25_results = await asyncio.gather(
                    self._timed(
                        "vector",
                        lambda: asyncio.to_thread(
                            self._vector_search,
                            self._collection,
                            query_embedding,
                            settings.vector_search_k,
                        ),
                        timings,
                    ),
                    self._timed(
                        "bm25",
                        lambda: self._bm25_search(
                            self._es_client,
                            request.query,
                            settings.elasticsearch_index,
                            settings.bm25_search_k,
                        ),
                        timings,
                    ),
                )
            timings["retrieve"] = (time.perf_counter() - retrieval_started) * 1000

            fusion_started = time.perf_counter()
            with pipeline_stage("rag.fusion"):
                fused = self._fusion([vector_results, bm25_results], settings.rrf_k)
            timings["fusion"] = (time.perf_counter() - fusion_started) * 1000

            candidate_k = max(request.top_k, settings.reranker_candidate_k)
            reranker = self._reranker_factory()
            ranked = await self._timed(
                "rerank",
                lambda: reranker.rerank(
                    request.query,
                    fused[:candidate_k],
                    top_k=request.top_k,
                ),
                timings,
                attributes={
                    "xai_rag.model": settings.reranker_model,
                    "xai_rag.model_revision": settings.reranker_model_revision,
                    "xai_rag.candidate_count": min(len(fused), candidate_k),
                },
            )
            await self._emit(
                event_sink,
                "retrieval",
                {
                    "vector_count": len(vector_results),
                    "bm25_count": len(bm25_results),
                    "fused_count": len(fused),
                    "ranked_ids": [result.id for result in ranked],
                },
            )

            explanations = []
            if request.include_explanations:
                explain_started = time.perf_counter()
                with pipeline_stage("rag.explain"):
                    explanations = self._explainer_factory().explain(
                        request.query,
                        vector_results,
                        bm25_results,
                        ranked,
                    )
                timings["explain"] = (time.perf_counter() - explain_started) * 1000

            generator = self._generator_factory()
            generated = await self._timed(
                "generate",
                lambda: generator.generate(request.query, ranked),
                timings,
                attributes={"xai_rag.model": settings.llm_model},
            )
            await self._emit(
                event_sink,
                "generation",
                {
                    "answer": generated.answer,
                    "claims_count": len(generated.claims),
                    "sources": generated.sources,
                },
            )

            faithfulness_report = []
            if request.include_faithfulness and generated.claims:
                checker = self._faithfulness_factory()
                faithfulness_report = await self._timed(
                    "faithfulness",
                    lambda: checker.check_claims(generated.claims, ranked),
                    timings,
                    attributes={
                        "xai_rag.model": settings.nli_model,
                        "xai_rag.model_revision": settings.nli_model_revision,
                        "xai_rag.claim_count": len(generated.claims),
                    },
                )
                await self._emit(
                    event_sink,
                    "faithfulness",
                    {
                        "verdicts": [
                            {
                                "claim_index": index,
                                "verdict": verdict.verdict.value,
                                "confidence": verdict.confidence,
                            }
                            for index, verdict in enumerate(faithfulness_report)
                        ]
                    },
                )

            external_scores: dict[str, Any] = {}
            if request.include_ragas:
                external_scores = await self._timed(
                    "external_evaluation",
                    lambda: self._evaluator(
                        request.query,
                        generated.answer,
                        [result.context_content or result.content for result in ranked],
                    ),
                    timings,
                )

            overall_confidence = (
                sum(verdict.nli_entailment for verdict in faithfulness_report)
                / len(faithfulness_report)
                if faithfulness_report
                else 0.0
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            response = XAIRAGResponse(
                query=request.query,
                answer=generated.answer,
                claims=generated.claims,
                sources=generated.sources,
                retrieval_explanations=explanations,
                faithfulness_report=faithfulness_report,
                overall_confidence=overall_confidence,
                ragas_scores=external_scores,
                latency_ms=elapsed_ms,
                stage_latency_ms=timings,
                input_tokens=generated.input_tokens,
                output_tokens=generated.output_tokens,
            )

        await self._emit(
            event_sink,
            "done",
            {"status": "complete", "latency_ms": response.latency_ms},
        )
        return response

    async def ingest(
        self,
        source: Path,
        *,
        document_root: Path,
        strategy: str = "semantic",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestionResponse:
        """Run bounded parsing, chunking, embedding, storage, and parity validation."""
        started = time.perf_counter()
        paths = await asyncio.to_thread(
            discover_document_paths,
            source,
            document_root,
            max_files=settings.max_ingest_files,
            max_document_bytes=settings.max_document_bytes,
            max_total_bytes=settings.max_ingest_total_bytes,
        )
        outcomes: list[IngestionDocumentResult] = []
        total_chunks = 0

        for path in paths:
            document_started = time.perf_counter()
            source_file = source_identity(path, document_root)
            try:
                with pipeline_stage(
                    "rag.ingest.document",
                    {"xai_rag.source_hash": hash_text(source_file)},
                ):
                    text = await asyncio.to_thread(self._parse_file, path)
                    chunks = await asyncio.to_thread(
                        self._chunk_text,
                        text,
                        strategy,
                        settings.chunk_size if chunk_size is None else chunk_size,
                        chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
                        settings.semantic_threshold,
                    )
                    embeddings = (
                        await self._embed_texts([chunk.content for chunk in chunks])
                        if chunks
                        else []
                    )
                    chunk_ids = await asyncio.to_thread(
                        self._store_chroma,
                        self._collection,
                        chunks,
                        embeddings,
                        source_file,
                    )
                    await self._store_elasticsearch(
                        self._es_client,
                        chunks,
                        chunk_ids,
                        source_file,
                        settings.elasticsearch_index,
                    )
                status = "success" if chunks else "skipped"
                total_chunks += len(chunks)
                outcomes.append(
                    IngestionDocumentResult(
                        source_file=source_file,
                        status=status,
                        chunk_count=len(chunks),
                        latency_ms=(time.perf_counter() - document_started) * 1000,
                    )
                )
            except Exception as exc:
                outcomes.append(
                    IngestionDocumentResult(
                        source_file=source_file,
                        status="failed",
                        error_type=type(exc).__name__,
                        latency_ms=(time.perf_counter() - document_started) * 1000,
                    )
                )

        parity_consistent: bool | None = None
        try:
            parity = await self._check_parity(self._collection, self._es_client)
            parity_consistent = parity.consistent
            if not parity.consistent:
                raise StorageWriteError("retrieval indexes diverged after ingestion")
        except Exception:
            parity_consistent = False

        succeeded = sum(outcome.status == "success" for outcome in outcomes)
        skipped = sum(outcome.status == "skipped" for outcome in outcomes)
        failed = sum(outcome.status == "failed" for outcome in outcomes)
        if failed or parity_consistent is False:
            status = "failed" if succeeded == 0 else "partial"
        else:
            status = "success"
        return IngestionResponse(
            status=status,
            discovered_files=len(paths),
            ingested_files=succeeded,
            skipped_files=skipped,
            failed_files=failed,
            total_chunks=total_chunks,
            parity_consistent=parity_consistent,
            documents=outcomes,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
