"""OpenTelemetry tracing setup and ``@traced`` decorator for async functions."""

from __future__ import annotations

import functools
import hashlib
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, TypeVar

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_TRACER_NAME = "xai-rag"
_METER_PROVIDER: MeterProvider | None = None
_SENSITIVE_ATTRIBUTE_PARTS = frozenset(
    {"content", "document", "prompt", "query", "secret", "token"}
)


@lru_cache(maxsize=1)
def _metric_instruments():
    meter = metrics.get_meter(_TRACER_NAME)
    return (
        meter.create_histogram(
            "xai_rag.pipeline.stage.duration",
            unit="ms",
            description="Duration of one RAG pipeline stage",
        ),
        meter.create_counter(
            "xai_rag.pipeline.stage.outcomes",
            description="Pipeline stage terminal outcomes",
        ),
    )


def hash_text(value: str) -> str:
    """Return a stable one-way identifier for correlation without content capture."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in (attributes or {}).items():
        normalized = key.casefold()
        if any(
            part in normalized for part in _SENSITIVE_ATTRIBUTE_PARTS
        ) and not normalized.endswith("_hash"):
            continue
        if isinstance(value, str | bool | int | float):
            safe[key] = value
    return safe


@contextmanager
def pipeline_stage(name: str, attributes: dict[str, Any] | None = None):
    """Trace and meter a stage while recording no raw inputs or error messages."""
    tracer = trace.get_tracer(_TRACER_NAME)
    metric_attributes = {"xai_rag.stage": name}
    latency_histogram, outcome_counter = _metric_instruments()
    start = time.perf_counter()
    with tracer.start_as_current_span(name) as span:
        for key, value in _safe_attributes(attributes).items():
            span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            error_type = type(exc).__name__
            span.set_attribute("error.type", error_type)
            span.set_status(Status(StatusCode.ERROR))
            outcome_counter.add(1, {**metric_attributes, "xai_rag.outcome": "error"})
            raise
        else:
            span.set_status(Status(StatusCode.OK))
            outcome_counter.add(1, {**metric_attributes, "xai_rag.outcome": "ok"})
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("xai_rag.duration_ms", elapsed_ms)
            latency_histogram.record(elapsed_ms, metric_attributes)


def setup_tracing(
    service_name: str = "xai-rag",
    otlp_endpoint: str | None = None,
    insecure: bool = False,
) -> TracerProvider:
    """Configure OpenTelemetry with an OTLP gRPC exporter.

    Call once at application startup (e.g. in your FastAPI lifespan).

    Parameters
    ----------
    service_name:
        The ``service.name`` resource attribute.
    otlp_endpoint:
        OTLP collector endpoint.  Defaults to the ``OTEL_EXPORTER_OTLP_ENDPOINT``
        env-var or ``http://localhost:4317``.

    Returns
    -------
    TracerProvider
        The configured provider (also set as the global provider).
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    exporter_kwargs: dict[str, Any] = {}
    if otlp_endpoint is not None:
        exporter_kwargs["endpoint"] = otlp_endpoint
    exporter_kwargs["insecure"] = insecure

    exporter = OTLPSpanExporter(**exporter_kwargs)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    metric_exporter = OTLPMetricExporter(**exporter_kwargs)
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    global _METER_PROVIDER
    _METER_PROVIDER = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(_METER_PROVIDER)
    _metric_instruments.cache_clear()
    logger.info(
        "OTEL tracing configured: service=%s endpoint=%s",
        service_name,
        otlp_endpoint or "(env default)",
    )
    return provider


def shutdown_observability(tracer_provider: TracerProvider | None) -> None:
    """Flush and close trace and metric exporters during graceful shutdown."""
    if tracer_provider is not None:
        tracer_provider.shutdown()
    if _METER_PROVIDER is not None:
        _METER_PROVIDER.shutdown()


def traced(
    name: str | None = None,
    *,
    record_query: bool = False,
    record_num_results: bool = False,
    record_model_name: bool = False,
) -> Callable[[F], F]:
    """Decorator that wraps an async function with an OpenTelemetry span.

    Automatically records ``latency_ms`` on every span.  Opt-in attributes
    can be enabled via keyword flags; the decorator extracts values from
    the function's keyword arguments when present.

    Parameters
    ----------
    name:
        Span name.  Defaults to the decorated function's qualified name.
    record_query:
        If ``True``, hash a ``query`` kwarg before attaching it. Raw query text
        is never recorded.
    record_num_results:
        If ``True``, attach the length of the return value (if it is a list)
        as ``num_results``.
    record_model_name:
        If ``True``, look for a ``model`` or ``model_name`` kwarg and record it.

    Usage
    -----
    ::

        @traced("retrieval.vector_search", record_query=True, record_num_results=True)
        async def vector_search(pool, query_embedding, k=100):
            ...
    """

    def decorator(fn: F) -> F:
        span_name = name or fn.__qualname__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(_TRACER_NAME)

            with tracer.start_as_current_span(span_name) as span:
                start = time.perf_counter()

                # Record optional attributes from kwargs.
                if record_query and "query" in kwargs:
                    span.set_attribute("xai_rag.query_hash", hash_text(str(kwargs["query"])))

                if record_model_name:
                    for key in ("model", "model_name"):
                        if key in kwargs:
                            span.set_attribute("model_name", str(kwargs[key]))
                            break

                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    span.set_attribute("error.type", type(exc).__name__)
                    span.set_status(Status(StatusCode.ERROR))
                    raise
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    span.set_attribute("latency_ms", elapsed_ms)

                if record_num_results and isinstance(result, list):
                    span.set_attribute("num_results", len(result))

                return result

        return wrapper  # type: ignore[return-value]

    return decorator
