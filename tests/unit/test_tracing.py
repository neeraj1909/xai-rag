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

    def start_as_current_span(self, name: str):
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
