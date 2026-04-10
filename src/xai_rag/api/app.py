"""FastAPI application — main entry point for the XAI-RAG API."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import chromadb
from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from xai_rag.config import settings
from xai_rag.models import QueryRequest, XAIRAGResponse, SearchResult

logger = logging.getLogger(__name__)


# --- Application state ---

class AppState:
    chroma_client: chromadb.HttpClient | None = None
    chroma_collection: chromadb.Collection | None = None
    es_client: AsyncElasticsearch | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Manage application lifecycle — connect on startup, disconnect on shutdown."""
    # Startup
    logger.info("Connecting to ChromaDB...")
    state.chroma_client = chromadb.HttpClient(
        host=settings.chroma_host, port=settings.chroma_port
    )
    state.chroma_collection = state.chroma_client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Connecting to Elasticsearch...")
    state.es_client = AsyncElasticsearch(settings.elasticsearch_url)

    # Optional: setup OTEL tracing
    try:
        from xai_rag.observability.tracing import setup_tracing
        setup_tracing(settings.otel_service_name)
        logger.info("OpenTelemetry tracing configured")
    except Exception as e:
        logger.warning(f"OTEL setup skipped: {e}")

    yield

    # Shutdown
    if state.es_client:
        await state.es_client.close()
    logger.info("Connections closed")


# --- FastAPI app ---

app = FastAPI(
    title="XAI-RAG",
    description="Explainable Retrieval-Augmented Generation — answers with retrieval attribution, NLI faithfulness, and RAGAS evaluation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---

@app.get("/health")
async def health():
    """Health check — verify all dependencies are reachable."""
    checks = {}

    # ChromaDB
    try:
        state.chroma_client.heartbeat()
        checks["chromadb"] = "ok"
    except Exception as e:
        checks["chromadb"] = f"error: {e}"

    # Elasticsearch
    try:
        info = await state.es_client.info()
        checks["elasticsearch"] = "ok"
    except Exception as e:
        checks["elasticsearch"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "healthy" if all_ok else "degraded", "checks": checks}


@app.post("/query", response_model=XAIRAGResponse)
async def query_endpoint(request: QueryRequest):
    """Run a full XAI-RAG query — retrieve, explain, generate, check faithfulness."""
    start = time.perf_counter()

    try:
        from xai_rag.ingestion.embedder import embed_query
        from xai_rag.retrieval.vector_search import vector_search
        from xai_rag.retrieval.bm25_search import bm25_search
        from xai_rag.retrieval.hybrid import rrf_fusion
        from xai_rag.retrieval.reranker import Reranker
        from xai_rag.explainability.retrieval_explainer import RetrievalExplainer
        from xai_rag.generation.generator import RAGGenerator
        from xai_rag.explainability.faithfulness import FaithfulnessChecker

        # 1. Embed query
        query_embedding = await embed_query(request.query)

        # 2. Hybrid retrieval
        vec_results = vector_search(state.chroma_collection, query_embedding)
        bm25_results = await bm25_search(state.es_client, request.query)
        fused = rrf_fusion([vec_results, bm25_results])

        # 3. Re-rank
        reranker = Reranker()
        ranked = await reranker.rerank(request.query, fused[:100], top_k=request.top_k)

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
                ragas_scores = await evaluate_response(
                    request.query, gen_result.answer, contexts
                )
            except Exception as e:
                logger.warning(f"RAGAS evaluation failed: {e}")

        # Compute overall confidence
        if faithfulness_report:
            avg_entailment = sum(v.nli_entailment for v in faithfulness_report) / len(faithfulness_report)
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

    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream query results via Server-Sent Events (SSE).

    Events: retrieval, generation, faithfulness, done
    """
    async def event_generator():
        import json
        from xai_rag.ingestion.embedder import embed_query
        from xai_rag.retrieval.vector_search import vector_search
        from xai_rag.retrieval.bm25_search import bm25_search
        from xai_rag.retrieval.hybrid import rrf_fusion
        from xai_rag.retrieval.reranker import Reranker

        # 1. Retrieve
        query_embedding = await embed_query(request.query)
        vec_results = vector_search(state.chroma_collection, query_embedding)
        bm25_results = await bm25_search(state.es_client, request.query)
        fused = rrf_fusion([vec_results, bm25_results])

        reranker = Reranker()
        ranked = await reranker.rerank(request.query, fused[:100], top_k=request.top_k)

        yield {"event": "retrieval", "data": json.dumps({
            "chunks_found": len(fused),
            "top_k": len(ranked),
            "top_chunk_preview": ranked[0].content[:200] if ranked else "",
        })}

        # 2. Generate
        from xai_rag.generation.generator import RAGGenerator
        generator = RAGGenerator()
        gen_result = await generator.generate(request.query, ranked)

        yield {"event": "generation", "data": json.dumps({
            "answer": gen_result.answer,
            "claims_count": len(gen_result.claims),
            "sources": gen_result.sources,
        })}

        # 3. Faithfulness
        if request.include_faithfulness and gen_result.claims:
            from xai_rag.explainability.faithfulness import FaithfulnessChecker
            checker = FaithfulnessChecker()
            verdicts = await checker.check_claims(gen_result.claims, ranked)
            yield {"event": "faithfulness", "data": json.dumps({
                "verdicts": [{"claim": v.claim.text, "verdict": v.verdict.value,
                              "confidence": v.confidence} for v in verdicts],
            })}

        yield {"event": "done", "data": json.dumps({"status": "complete"})}

    return EventSourceResponse(event_generator())


@app.post("/ingest")
async def ingest_endpoint(source_path: str, strategy: str = "semantic"):
    """Ingest documents from a path (server-side)."""
    from pathlib import Path
    from xai_rag.ingestion.parser import parse_file, parse_directory
    from xai_rag.ingestion.chunker import chunk_text
    from xai_rag.ingestion.embedder import embed_texts
    from xai_rag.ingestion.store import store_chunks_chromadb, store_chunks_elasticsearch, ensure_es_index

    path = Path(source_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {source_path}")

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
