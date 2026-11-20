import asyncio
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path

import httpx
from call_budget import install
from openai import AsyncOpenAI


async def main():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ.update(
            XAI_EVAL_BUDGET_FILE=f"{tmp}/budget",
            XAI_EVAL_MAX_PROVIDER_CALLS="3",
            XAI_EVAL_MODEL_SEED="20261120",
        )
        install()
        seen = []

        async def handler(req):
            body = json.loads(req.content)
            seen.append((req.url.path, body))
            if req.url.path.endswith("/embeddings"):
                data = {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                    "model": "mock",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            else:
                data = {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "mock",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            return httpx.Response(200, json=data, request=req)

        async with AsyncOpenAI(
            api_key="test",
            max_retries=0,
            base_url="https://example.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            await client.chat.completions.create(
                model="mock", messages=[{"role": "user", "content": "synthetic"}]
            )
            await client.embeddings.create(model="mock", input="synthetic")
        assert seen[0][1]["seed"] == 20261120 and "seed" not in seen[1][1], seen
        events = [json.loads(line) for line in Path(f"{tmp}/budget").read_text().splitlines()]
        assert events[0]["request_payload"]["seed"] == 20261120 and events[2]["model_seed"] is None
        errors = []

        async def error_handler(req):
            errors.append(req)
            return httpx.Response(
                429,
                json={"error": {"message": "rate limit", "type": "rate_limit_error"}},
                request=req,
            )

        async with AsyncOpenAI(
            api_key="test",
            max_retries=0,
            base_url="https://example.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(error_handler)),
        ) as client:
            with suppress(Exception):
                await client.chat.completions.create(
                    model="mock", messages=[{"role": "user", "content": "synthetic"}]
                )
        assert len(errors) == 1
        events = [json.loads(line) for line in Path(f"{tmp}/budget").read_text().splitlines()]
        assert (
            len([e for e in events if e["phase"] == "attempt"]) == 3
            and events[-1]["status"] == "error"
        )
        async with AsyncOpenAI(
            api_key="test",
            base_url="https://example.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(error_handler)),
        ) as client:
            try:
                await client.chat.completions.create(
                    model="mock", messages=[{"role": "user", "content": "synthetic"}]
                )
                raise AssertionError("retry-enabled client was not rejected")
            except RuntimeError as exc:
                assert "SDK retries" in str(exc)
        assert len(errors) == 1
        print("mock transport seed, failure and cap accounting passed")


asyncio.run(main())
