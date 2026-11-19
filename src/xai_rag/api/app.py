"""FastAPI application — main entry point for the XAI-RAG API."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import chromadb
from elasticsearch import AsyncElasticsearch
from elasticsearch import ConnectionError as ElasticsearchConnectionError
from elasticsearch import ConnectionTimeout as ElasticsearchConnectionTimeout
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
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
    tracer_provider: object | None = None


state = AppState()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = {}
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_WINDOW_SECONDS = 60.0
_QUERY_SEMAPHORE = asyncio.Semaphore(settings.max_concurrent_queries)
_QUERY_TASKS: set[asyncio.Task] = set()
_READINESS_TIMEOUT_SECONDS = 2.0
_CHROMA_PROBE: asyncio.Task | None = None
_CHROMA_PROBE_CLIENT: object | None = None


async def _probe_chroma(client) -> None:
    """Share one in-flight synchronous probe, including after a request times out."""
    global _CHROMA_PROBE, _CHROMA_PROBE_CLIENT
    if _CHROMA_PROBE is None or _CHROMA_PROBE.done():
        _CHROMA_PROBE_CLIENT = client
        _CHROMA_PROBE = asyncio.create_task(asyncio.to_thread(client.heartbeat))

        def consume_result(task: asyncio.Task) -> None:
            if not task.cancelled():
                task.exception()

        _CHROMA_PROBE.add_done_callback(consume_result)
    elif _CHROMA_PROBE_CLIENT is not client:
        raise RuntimeError("A previous Chroma readiness probe is still running")

    # Cancelling to_thread does not stop its thread. Shield the shared task so
    # repeated health checks cannot enqueue more blocked heartbeat calls.
    async with asyncio.timeout(_READINESS_TIMEOUT_SECONDS):
        await asyncio.shield(_CHROMA_PROBE)


def _build_service():
    """Create the shared application service only when dependencies are ready."""
    if state.chroma_collection is None or state.es_client is None:
        raise HTTPException(status_code=503, detail="Search dependencies are not ready")

    from xai_rag.service import RAGService

    return RAGService(collection=state.chroma_collection, es_client=state.es_client)


async def _run_bounded_query(service, request: QueryRequest, *, event_sink=None):
    """Bound total queue+execution time and per-process query concurrency."""

    async def run_admitted():
        async with _QUERY_SEMAPHORE:
            return await service.query(request, event_sink=event_sink)

    task = asyncio.create_task(run_admitted())
    _QUERY_TASKS.add(task)

    def finish(completed: asyncio.Task) -> None:
        _QUERY_TASKS.discard(completed)
        if not completed.cancelled():
            completed.exception()

    task.add_done_callback(finish)
    try:
        async with asyncio.timeout(settings.query_timeout_seconds):
            return await asyncio.shield(task)
    except BaseException:
        # The request returns promptly, but the task retains its admission slot
        # until any non-cancellable worker thread has actually finished.
        task.cancel()
        raise


def require_api_key(provided_api_key: str | None = Depends(api_key_header)) -> None:
    """Authenticate API requests and enforce a small per-process rate limit."""
    expected_api_key = settings.api_key
    if len(expected_api_key) < 32:
        raise HTTPException(status_code=503, detail="API authentication is not configured")

    if not provided_api_key or not secrets.compare_digest(
        provided_api_key.encode("utf-8"), expected_api_key.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")

    principal = hashlib.sha256(expected_api_key.encode("utf-8")).hexdigest()
    with _RATE_LIMIT_LOCK:
        bucket = _RATE_LIMIT_BUCKETS.setdefault(principal, deque())
        now = time.monotonic()
        window_start = now - _RATE_LIMIT_WINDOW_SECONDS
        while bucket and bucket[0] <= window_start:
            bucket.popleft()

        if len(bucket) >= settings.rate_limit_per_minute:
            raise HTTPException(
                status_code=429, detail="Rate limit exceeded", headers={"Retry-After": "60"}
            )

        bucket.append(now)


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
            from xai_rag.ingestion.store import ensure_es_index, get_chroma_collection

            logger.info("Connecting to ChromaDB (attempt %d/5)...", attempt + 1)
            state.chroma_client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
            state.chroma_collection = get_chroma_collection(state.chroma_client)
            logger.info("Connecting to Elasticsearch...")
            client_kwargs = {}
            if settings.elasticsearch_api_key:
                client_kwargs["api_key"] = settings.elasticsearch_api_key
            state.es_client = AsyncElasticsearch(settings.elasticsearch_url, **client_kwargs)
            await state.es_client.info()
            await ensure_es_index(state.es_client)
            break
        except Exception as exc:
            if state.es_client:
                await state.es_client.close()
                state.es_client = None
            state.chroma_collection = None
            state.chroma_client = None
            if attempt == 4:
                logger.error("Search dependency startup failed (%s)", type(exc).__name__)
                raise RuntimeError("Required search dependencies are unavailable") from None
            await asyncio.sleep(2**attempt)

    # Optional: setup OTEL tracing
    if settings.otel_endpoint:
        try:
            from xai_rag.observability.tracing import setup_tracing

            state.tracer_provider = setup_tracing(
                settings.otel_service_name,
                settings.otel_endpoint,
                insecure=settings.otel_insecure,
            )
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
            logger.info("OpenTelemetry tracing configured")
        except Exception as exc:
            logger.warning("OTEL setup skipped (%s)", type(exc).__name__)

    yield

    # Ask admitted work to stop before closing the shared dependencies. Native
    # worker threads cannot be killed; do not wait for a stuck worker forever.
    active_queries = set(_QUERY_TASKS)
    for task in active_queries:
        task.cancel()
    if active_queries:
        _, pending = await asyncio.wait(active_queries, timeout=5)
        if pending:
            logger.warning("Query workers still stopping at shutdown: %d", len(pending))

    # Shutdown
    if state.es_client:
        await state.es_client.close()
    state.es_client = None
    state.chroma_collection = None
    state.chroma_client = None
    if state.tracer_provider:
        from xai_rag.observability.tracing import shutdown_observability

        shutdown_observability(state.tracer_provider)
        state.tracer_provider = None
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
    responses={
        422: {
            "description": "Invalid request",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["detail"],
                        "properties": {"detail": {"type": "string", "const": "Invalid request"}},
                    }
                }
            },
        }
    },
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, _exc: RequestValidationError) -> JSONResponse:
    """Avoid reflecting submitted input or validator-controlled error details."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request"},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(settings.cors_origins),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type", "X-API-Key"],
    allow_credentials=False,
)


# --- Endpoints ---


@app.get("/live")
async def live() -> dict[str, str]:
    """Process liveness check that does not depend on downstream services."""
    return {"status": "alive"}


@app.get("/ready")
@app.get("/health")
async def health() -> JSONResponse:
    """Readiness check — verify all required search dependencies are reachable."""
    checks: dict[str, str] = {}

    # ChromaDB
    if state.chroma_client is None:
        checks["chromadb"] = "unavailable"
    else:
        try:
            await _probe_chroma(state.chroma_client)
            checks["chromadb"] = "ok"
        except Exception as exc:
            logger.warning("ChromaDB health check failed (%s)", type(exc).__name__)
            checks["chromadb"] = "error"

    # Elasticsearch
    if state.es_client is None:
        checks["elasticsearch"] = "unavailable"
    else:
        try:
            async with asyncio.timeout(_READINESS_TIMEOUT_SECONDS):
                await state.es_client.info()
                index_health = await state.es_client.cluster.health(
                    index=settings.elasticsearch_index,
                    wait_for_status="yellow",
                    timeout="2s",
                )
            if index_health.get("timed_out") or str(index_health.get("status", "red")) == "red":
                raise RuntimeError("required Elasticsearch index has no active primary")
            checks["elasticsearch"] = "ok"
        except Exception as exc:
            logger.warning("Elasticsearch health check failed (%s)", type(exc).__name__)
            checks["elasticsearch"] = "error"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        content={"status": "healthy" if all_ok else "degraded", "checks": checks},
        status_code=200 if all_ok else 503,
    )


@app.post(
    "/query",
    response_model=XAIRAGResponse,
    dependencies=[Depends(require_api_key)],
    responses={
        503: {
            "description": "Authentication or search dependency unavailable",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["detail"],
                        "properties": {
                            "detail": {
                                "type": "string",
                                "enum": [
                                    "API authentication is not configured",
                                    "Search dependencies are not ready",
                                    "Search dependency unavailable",
                                ],
                            }
                        },
                    }
                }
            },
        }
    },
)
async def query_endpoint(request: QueryRequest):
    """Run a full XAI-RAG query — retrieve, explain, generate, check faithfulness."""
    from xai_rag.service import RequestEvaluationDisabledError

    try:
        return await _run_bounded_query(_build_service(), request)
    except RequestEvaluationDisabledError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Query timed out") from exc
    except (ElasticsearchConnectionError, ElasticsearchConnectionTimeout) as exc:
        logger.warning("Search dependency unavailable (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Search dependency unavailable") from exc
    except Exception as exc:
        logger.error("Query failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Query failed") from exc


@app.post("/query/stream", dependencies=[Depends(require_api_key)])
async def query_stream(request: QueryRequest):
    """Stream query results via Server-Sent Events (SSE).

    Events: retrieval, generation, faithfulness, done
    """

    import json

    from xai_rag.service import RequestEvaluationDisabledError

    service = _build_service()
    if request.include_ragas and not settings.allow_request_evaluation:
        raise HTTPException(
            status_code=400,
            detail="request-scoped external evaluation is disabled; use the offline evaluator",
        )

    async def event_generator():
        queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

        async def publish(event: str, data: dict) -> None:
            await queue.put((event, data))

        async def run_pipeline() -> None:
            try:
                await _run_bounded_query(service, request, event_sink=publish)
            except RequestEvaluationDisabledError:
                await queue.put(("error", {"status": "error", "error_type": "disabled"}))
            except TimeoutError:
                await queue.put(("error", {"status": "error", "error_type": "timeout"}))
            except (ElasticsearchConnectionError, ElasticsearchConnectionTimeout) as exc:
                logger.warning("Search dependency unavailable (%s)", type(exc).__name__)
                await queue.put(
                    ("error", {"status": "error", "error_type": "dependency_unavailable"})
                )
            except Exception as exc:
                logger.error("Streaming query failed (%s)", type(exc).__name__)
                await queue.put(
                    (
                        "error",
                        {"status": "error", "error_type": type(exc).__name__},
                    )
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_pipeline())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event, data = item
                yield {"event": event, "data": json.dumps(data)}
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return EventSourceResponse(event_generator())


@app.post("/ingest", dependencies=[Depends(require_api_key)])
async def ingest_endpoint(
    source_path: str,
    strategy: Literal["fixed", "semantic", "parent_doc"] = "semantic",
):
    """Run bounded server-side ingestion through the shared application service."""
    from xai_rag.service import IngestionLimitError

    path = _resolve_ingest_path(source_path)
    document_root = Path(settings.ingest_root).expanduser().resolve()
    try:
        result = await _build_service().ingest(
            path,
            document_root=document_root,
            strategy=strategy,
        )
    except IngestionLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    status_code = 200 if result.status == "success" else 207 if result.status == "partial" else 500
    return JSONResponse(status_code=status_code, content=result.model_dump(mode="json"))
