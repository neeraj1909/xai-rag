"""OpenTelemetry tracing setup and ``@traced`` decorator for async functions."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_TRACER_NAME = "xai-rag"


def setup_tracing(
    service_name: str = "xai-rag",
    otlp_endpoint: str | None = None,
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
    exporter_kwargs["insecure"] = True

    exporter = OTLPSpanExporter(**exporter_kwargs)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    logger.info(
        "OTEL tracing configured: service=%s endpoint=%s",
        service_name,
        otlp_endpoint or "(env default)",
    )
    return provider


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
        If ``True``, look for a ``query`` kwarg and attach it as a span attribute.
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
                    span.set_attribute("query", str(kwargs["query"])[:500])

                if record_model_name:
                    for key in ("model", "model_name"):
                        if key in kwargs:
                            span.set_attribute("model_name", str(kwargs[key]))
                            break

                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(exc)[:500])
                    raise
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    span.set_attribute("latency_ms", elapsed_ms)

                if record_num_results and isinstance(result, list):
                    span.set_attribute("num_results", len(result))

                return result

        return wrapper  # type: ignore[return-value]

    return decorator
