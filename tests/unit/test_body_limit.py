"""Boundary checks for the pre-authentication request body limit."""

import asyncio

import pytest
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

from xai_rag.api.body_limit import RequestBodyLimit


@pytest.mark.unit
def test_chunked_oversize_is_rejected_before_body_parser_and_authentication():
    app = FastAPI()
    calls = []

    def authenticate(x_api_key: str = Header(default="")) -> None:
        calls.append("auth")
        if not x_api_key:
            raise HTTPException(status_code=401)

    class Payload(BaseModel):
        query: str

    @app.post("/query", dependencies=[Depends(authenticate)])
    def query(payload: Payload):
        calls.append("handler")
        return payload

    app.add_middleware(RequestBodyLimit, max_bytes=32)
    with TestClient(app) as client:
        response = client.post("/query", content=iter([b'{"query":"', b"x" * 64, b'"}']))
        assert response.status_code == 413
        assert response.json() == {"detail": "Request body too large"}
        assert calls == []
        assert client.post("/query", json={"query": "ok"}).status_code == 401
        response = client.post("/query", json={"query": "ok"}, headers={"x-api-key": "valid"})
        assert response.status_code == 200
        assert response.json() == {"query": "ok"}


@pytest.mark.unit
def test_actual_api_rejects_oversize_before_unauthenticated_json_parsing():
    from xai_rag.api.app import app
    from xai_rag.config import settings

    # No lifespan or backend services are needed: the outer middleware rejects
    # the body before routing or dependencies. It must not reflect input.
    client = TestClient(app)  # no context manager: do not start the lifespan
    response = client.post("/query", content=b"x" * (settings.max_request_body_bytes + 1))
    client.close()
    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}


@pytest.mark.unit
def test_oversize_stops_consuming_frames_at_limit():
    async def exercise():
        calls = []
        sent = []
        frames = iter(
            [
                {"type": "http.request", "body": b"abc", "more_body": True},
                {"type": "http.request", "body": b"def", "more_body": True},
                {"type": "http.request", "body": b"unread", "more_body": False},
            ]
        )

        async def receive():
            calls.append("receive")
            return next(frames)

        async def send(message):
            sent.append(message)

        async def app(scope, receive, send):
            raise AssertionError("oversized request reached application")

        await RequestBodyLimit(app, max_bytes=5)({"type": "http"}, receive, send)
        assert calls == ["receive", "receive"]
        assert sent[0]["status"] == 413

    asyncio.run(exercise())


@pytest.mark.unit
def test_streaming_response_still_emits_all_events():
    app = FastAPI()

    @app.post("/stream")
    async def stream():
        async def events():
            yield "data: first\n\n"
            yield "data: second\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    app.add_middleware(RequestBodyLimit, max_bytes=32)
    with TestClient(app) as client:
        response = client.post("/stream", content=b"{}")
    assert response.status_code == 200
    assert response.text == "data: first\n\ndata: second\n\n"


@pytest.mark.unit
def test_first_response_frame_is_not_buffered_until_stream_completion():
    async def exercise():
        first_sent = asyncio.Event()
        finish = asyncio.Event()

        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        async def send(message):
            if message["type"] == "http.response.body":
                first_sent.set()

        async def app(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"one", "more_body": True})
            await finish.wait()
            await send({"type": "http.response.body", "body": b"two", "more_body": False})

        task = asyncio.create_task(RequestBodyLimit(app, 32)({"type": "http"}, receive, send))
        await asyncio.wait_for(first_sent.wait(), timeout=1)
        assert not task.done()
        finish.set()
        await task

    asyncio.run(exercise())


@pytest.mark.unit
def test_disconnect_and_cancellation_stop_without_dispatching():
    async def exercise():
        async def app(scope, receive, send):
            raise AssertionError("disconnected request reached application")

        async def disconnect():
            return {"type": "http.disconnect"}

        async def send(message):
            raise AssertionError("response sent after disconnect")

        await RequestBodyLimit(app, max_bytes=10)({"type": "http"}, disconnect, send)

        async def blocked():
            await asyncio.Event().wait()

        task = asyncio.create_task(
            RequestBodyLimit(app, max_bytes=10)({"type": "http"}, blocked, send)
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
