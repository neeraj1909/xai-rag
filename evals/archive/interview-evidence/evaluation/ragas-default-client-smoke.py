"""Exercise default-owned RAGAS client with a real provider and a request cap."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import openai

from xai_rag.evaluation.ragas_eval import evaluate_response

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("ragas_live_runner", HERE / "ragas-live-runner.py")
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.MAX_ATTEMPTS = 12
transport = runner.CountedTransport()
original_client = openai.AsyncOpenAI
created = []


class RecordingClient(original_client):
    def __init__(self, **kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=transport, timeout=60)
        super().__init__(**kwargs)
        created.append(self)


async def main() -> None:
    fixture = json.loads((HERE / "generation-judge-baseline.json").read_text())
    row = fixture["rows"][0]
    openai.AsyncOpenAI = RecordingClient
    try:
        result = await asyncio.wait_for(
            evaluate_response(
                row["question"],
                row["answer"],
                [x["content"] for x in row["context"]],
                row["reference_answer"],
            ),
            timeout=180,
        )
        output = {
            "case": row["id"],
            "result": result,
            "client_count": len(created),
            "client_closed": [client.is_closed() for client in created],
            "http_attempts": transport.attempts,
            "responses": transport.responses,
            "max_actual_http_attempts": runner.MAX_ATTEMPTS,
        }
        Path(sys.argv[1]).write_text(json.dumps(output, indent=2) + "\n")
        print(result["status"], output["client_closed"], flush=True)
    finally:
        openai.AsyncOpenAI = original_client
        for client in created:
            if not client.is_closed():
                await client.close()


if __name__ == "__main__":
    asyncio.run(main())
