"""FastAPI application — main entry point for the XAI-RAG API."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import chromadb
from elasticsearch import AsyncElasticsearch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from sse_starlette.sse import EventSourceResponse

from xai_rag.config import settings
from xai_rag.models import QueryRequest, XAIRAGResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


# --- Application state ---


class AppState:
    chroma_client: chromadb.HttpClient | None = None
    chroma_collection: chromadb.Collection | None = None
    es_client: AsyncElasticsearch | None = None


state = AppState()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_RATE_LIMIT_BUCKET: deque[float] = deque()
_RATE_LIMIT_WINDOW_SECONDS = 60.0


def require_api_key(provided_api_key: str | None = Depends(api_key_header)) -> None:
    """Authenticate API requests and enforce a small per-process rate limit."""
    expected_api_key = settings.api_key
    if len(expected_api_key) < 32:
        raise HTTPException(status_code=503, detail="API authentication is not configured")

    if not provided_api_key or not secrets.compare_digest(provided_api_key, expected_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    now = time.monotonic()
    window_start = now - _RATE_LIMIT_WINDOW_SECONDS
    while _RATE_LIMIT_BUCKET and _RATE_LIMIT_BUCKET[0] <= window_start:
        _RATE_LIMIT_BUCKET.popleft()

    if len(_RATE_LIMIT_BUCKET) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded", headers={"Retry-After": "60"}
        )

    _RATE_LIMIT_BUCKET.append(now)


def _parse_cors_origins(raw_origins: str) -> list[str]:
    """Parse a comma-separated CORS allowlist without permitting wildcards."""
    return [
        origin.strip()
        for origin in raw_origins.split(",")
        if origin.strip() and origin.strip() != "*"
    ]


def _resolve_ingest_path(source_path: str) -> Path:
    """Resolve an ingestion path and keep it inside the configured document root."""
    root = Path(settings.ingest_root).expanduser().resolve()
    path = Path(source_path).expanduser().resolve()

    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Path is outside the configured ingest root"
        ) from exc

    if not path.exists():
        raise HTTPException(status_code=400, detail="Source path not found")
    return path


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Manage application lifecycle — connect on startup, disconnect on shutdown."""
    # Startup: tolerate dependency startup races in local Compose and rolling deploys.
    for attempt in range(5):
        try:
            logger.info("Connecting to ChromaDB (attempt %d/5)...", attempt + 1)
            state.chroma_client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
            state.chroma_collection = state.chroma_client.get_or_create_collection(
                name=settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Connecting to Elasticsearch...")
            client_kwargs = {}
            if settings.elasticsearch_api_key:
                client_kwargs["api_key"] = settings.elasticsearch_api_key
            state.es_client = AsyncElasticsearch(settings.elasticsearch_url, **client_kwargs)
            await state.es_client.info()
            break
        except Exception as exc:
            if state.es_client:
                await state.es_client.close()
                state.es_client = None
            if attempt == 4:
                raise RuntimeError("Required search dependencies are unavailable") from exc
            await asyncio.sleep(2**attempt)

    # Optional: setup OTEL tracing
    if settings.otel_endpoint:
        try:
            from xai_rag.observability.tracing import setup_tracing

            setup_tracing(
                settings.otel_service_name,
                settings.otel_endpoint,
                insecure=settings.otel_insecure,
            )
            logger.info("OpenTelemetry tracing configured")
        except Exception:
            logger.warning("OTEL setup skipped", exc_info=True)

    yield

    # Shutdown
    if state.es_client:
        await state.es_client.close()
    logger.info("Connections closed")


# --- FastAPI app ---

app = FastAPI(
    title="XAI-RAG",
    description=(
        "Explainable Retrieval-Augmented Generation — answers with retrieval attribution, "
        "NLI faithfulness, and optional isolated evaluation"
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.expose_docs else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.expose_docs else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(settings.cors_origins),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type", "X-API-Key"],
    allow_credentials=False,
)


# --- Endpoints ---


@app.get("/health")
async def health() -> JSONResponse:
    """Health check — verify all dependencies are reachable."""
    checks: dict[str, str] = {}

    # ChromaDB
    if state.chroma_client is None:
        checks["chromadb"] = "unavailable"
    else:
        try:
            state.chroma_client.heartbeat()
            checks["chromadb"] = "ok"
        except Exception:
            logger.warning("ChromaDB health check failed", exc_info=True)
            checks["chromadb"] = "error"

    # Elasticsearch
    if state.es_client is None:
        checks["elasticsearch"] = "unavailable"
    else:
        try:
            await state.es_client.info()
            checks["elasticsearch"] = "ok"
        except Exception:
            logger.warning("Elasticsearch health check failed", exc_info=True)
            checks["elasticsearch"] = "error"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        content={"status": "healthy" if all_ok else "degraded", "checks": checks},
        status_code=200 if all_ok else 503,
    )


@app.post("/query", response_model=XAIRAGResponse, dependencies=[Depends(require_api_key)])
async def query_endpoint(request: QueryRequest):
    """Run a full XAI-RAG query — retrieve, explain, generate, check faithfulness."""
    start = time.perf_counter()

    try:
        from xai_rag.explainability.faithfulness import FaithfulnessChecker
        from xai_rag.explainability.retrieval_explainer import RetrievalExplainer
        from xai_rag.generation.generator import RAGGenerator
        from xai_rag.ingestion.embedder import embed_query
        from xai_rag.retrieval.bm25_search import bm25_search
        from xai_rag.retrieval.hybrid import rrf_fusion
        from xai_rag.retrieval.reranker import Reranker
        from xai_rag.retrieval.vector_search import vector_search

        # 1. Embed query
        query_embedding = await embed_query(request.query)

        # 2. Hybrid retrieval
        vec_results = vector_search(
            state.chroma_collection,
            query_embedding,
            k=settings.vector_search_k,
        )
        bm25_results = await bm25_search(
            state.es_client,
            request.query,
            index=settings.elasticsearch_index,
            k=settings.bm25_search_k,
        )
        fused = rrf_fusion([vec_results, bm25_results])

        # 3. Re-rank
        reranker = Reranker()
        ranked = await reranker.rerank(
            request.query,
            fused[: settings.reranker_candidate_k],
            top_k=request.top_k,
        )

        # 4. Retrieval explanations
        explanations = []
        if request.include_explanations:
            explainer = RetrievalExplainer()
            explanations = explainer.explain(request.query, vec_results, bm25_results, ranked)

        # 5. Generate answer
        generator = RAGGenerator()
        gen_result = await generator.generate(request.query, ranked)

        # 6. Faithfulness check
        faithfulness_report = []
        if request.include_faithfulness and gen_result.claims:
            checker = FaithfulnessChecker()
            faithfulness_report = await checker.check_claims(gen_result.claims, ranked)

        # 7. RAGAS evaluation (optional, expensive)
        ragas_scores = {}
        if request.include_ragas:
            try:
                from xai_rag.evaluation.ragas_eval import evaluate_response

                contexts = [r.content for r in ranked]
                ragas_scores = await evaluate_response(request.query, gen_result.answer, contexts)
            except Exception:
                logger.warning("RAGAS evaluation failed", exc_info=True)

        # Compute overall confidence
        if faithfulness_report:
            avg_entailment = sum(v.nli_entailment for v in faithfulness_report) / len(
                faithfulness_report
            )
        else:
            avg_entailment = 0.0

        elapsed = (time.perf_counter() - start) * 1000

        return XAIRAGResponse(
            query=request.query,
            answer=gen_result.answer,
            claims=gen_result.claims,
            sources=gen_result.sources,
            retrieval_explanations=explanations,
            faithfulness_report=faithfulness_report,
            overall_confidence=avg_entailment,
            ragas_scores=ragas_scores,
            latency_ms=elapsed,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query failed") from exc


@app.post("/query/stream", dependencies=[Depends(require_api_key)])
async def query_stream(request: QueryRequest):
    """Stream query results via Server-Sent Events (SSE).

    Events: retrieval, generation, faithfulness, done
    """

    async def event_generator():
        import json

        try:
            async for event in query_events():
                yield event
        except Exception:
            logger.exception("Streaming query failed")
            yield {
                "event": "error",
                "data": json.dumps({"status": "error", "message": "Query failed"}),
            }

    async def query_events():
        import json

        from xai_rag.ingestion.embedder import embed_query
        from xai_rag.retrieval.bm25_search import bm25_search
        from xai_rag.retrieval.hybrid import rrf_fusion
        from xai_rag.retrieval.reranker import Reranker
        from xai_rag.retrieval.vector_search import vector_search

        # 1. Retrieve
        query_embedding = await embed_query(request.query)
        vec_results = vector_search(
            state.chroma_collection,
            query_embedding,
            k=settings.vector_search_k,
        )
        bm25_results = await bm25_search(
            state.es_client,
            request.query,
            index=settings.elasticsearch_index,
            k=settings.bm25_search_k,
        )
        fused = rrf_fusion([vec_results, bm25_results])

        reranker = Reranker()
        ranked = await reranker.rerank(
            request.query,
            fused[: settings.reranker_candidate_k],
            top_k=request.top_k,
        )

        yield {
            "event": "retrieval",
            "data": json.dumps(
                {
                    "chunks_found": len(fused),
                    "top_k": len(ranked),
                    "top_chunk_preview": ranked[0].content[:200] if ranked else "",
                }
            ),
        }

        # 2. Generate
        from xai_rag.generation.generator import RAGGenerator

        generator = RAGGenerator()
        gen_result = await generator.generate(request.query, ranked)

        yield {
            "event": "generation",
            "data": json.dumps(
                {
                    "answer": gen_result.answer,
                    "claims_count": len(gen_result.claims),
                    "sources": gen_result.sources,
                }
            ),
        }

        # 3. Faithfulness
        if request.include_faithfulness and gen_result.claims:
            from xai_rag.explainability.faithfulness import FaithfulnessChecker

            checker = FaithfulnessChecker()
            verdicts = await checker.check_claims(gen_result.claims, ranked)
            yield {
                "event": "faithfulness",
                "data": json.dumps(
                    {
                        "verdicts": [
                            {
                                "claim": v.claim.text,
                                "verdict": v.verdict.value,
                                "confidence": v.confidence,
                            }
                            for v in verdicts
                        ],
                    }
                ),
            }

        yield {"event": "done", "data": json.dumps({"status": "complete"})}

    return EventSourceResponse(event_generator())


@app.post("/ingest", dependencies=[Depends(require_api_key)])
async def ingest_endpoint(
    source_path: str,
    strategy: Literal["fixed", "semantic", "parent_doc"] = "semantic",
):
    """Ingest documents from a path (server-side)."""
    from xai_rag.ingestion.chunker import chunk_text
    from xai_rag.ingestion.embedder import embed_texts
    from xai_rag.ingestion.parser import parse_directory, parse_file
    from xai_rag.ingestion.store import (
        ensure_es_index,
        store_chunks_chromadb,
        store_chunks_elasticsearch,
    )

    path = _resolve_ingest_path(source_path)

    docs = [(path, parse_file(path))] if path.is_file() else parse_directory(path)
    await ensure_es_index(state.es_client)

    total = 0
    for file_path, text in docs:
        chunks = chunk_text(text, strategy=strategy)
        embeddings = await embed_texts([c.content for c in chunks])
        ids = store_chunks_chromadb(state.chroma_collection, chunks, embeddings, file_path.name)
        await store_chunks_elasticsearch(state.es_client, chunks, ids, file_path.name)
        total += len(chunks)

    return {"ingested_files": len(docs), "total_chunks": total}
