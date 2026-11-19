"""Privacy and outcome contracts for pipeline telemetry."""

import pytest

from xai_rag.observability import tracing

pytestmark = pytest.mark.unit


class FakeSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_status(self, status) -> None:
        self.status = status


class FakeSpanContext:
    def __init__(self, span: FakeSpan) -> None:
        self.span = span

    def __enter__(self) -> FakeSpan:
        return self.span

    def __exit__(self, *args) -> None:
        return None


class FakeTracer:
    def __init__(self, span: FakeSpan) -> None:
        self.span = span

    def start_as_current_span(self, name: str, **kwargs):
        self.name = name
        return FakeSpanContext(self.span)


@pytest.mark.asyncio
async def test_query_recording_is_hashed_and_errors_do_not_capture_messages(monkeypatch) -> None:
    span = FakeSpan()
    monkeypatch.setattr(tracing.trace, "get_tracer", lambda name: FakeTracer(span))

    @tracing.traced("test", record_query=True)
    async def broken(*, query: str) -> None:
        raise RuntimeError(f"secret leaked by {query}")

    with pytest.raises(RuntimeError):
        await broken(query="private customer question")

    serialized = repr(span.attributes)
    assert "private customer question" not in serialized
    assert "secret leaked" not in serialized
    assert len(str(span.attributes["xai_rag.query_hash"])) == 64
    assert span.attributes["error.type"] == "RuntimeError"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["pipeline", "decorator"])
async def test_real_span_does_not_export_sensitive_exception(kind, monkeypatch) -> None:
    """The SDK records raised exceptions unless its context manager is configured."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing.trace, "get_tracer", lambda name: provider.get_tracer(name))
    canary = "synthetic-private-error-canary"

    with pytest.raises(RuntimeError, match=canary):
        if kind == "pipeline":
            with tracing.pipeline_stage("test-stage"):
                raise RuntimeError(canary)
        else:

            @tracing.traced("test-decorator", record_query=True)
            async def broken(*, query: str) -> None:
                raise RuntimeError(canary)

            await broken(query="synthetic-private-query")

    span = exporter.get_finished_spans()[0]
    exported = (
        repr(span.attributes)
        + repr([dict(event.attributes or {}) for event in span.events])
        + repr(span.status)
    )
    assert canary not in exported
    assert "synthetic-private-query" not in exported
    assert span.attributes["error.type"] == "RuntimeError"
    assert span.status.status_code == StatusCode.ERROR
    provider.shutdown()
