"""Run the project's actual RAGAS integration on saved fictional examples.

Run from the XAI-RAG checkout with its optional evaluation dependencies installed:
  PYTHONPATH=src .venv/bin/python /path/to/ragas-live-runner.py /path/to/results.json

Only use the synthetic fixture shipped with the interview materials. This sends
questions, answers, reference answers and contexts to the OpenAI API.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

from xai_rag.config import settings
from xai_rag.evaluation.ragas_eval import RagasScorers, evaluate_response

FIXTURE = Path(__file__).with_name("generation-judge-baseline.json")
MAX_ATTEMPTS = 12 if os.getenv("RAGAS_CASE_MODE") == "strong-negative" else 60


class CountedTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.inner = httpx.AsyncHTTPTransport(retries=0)
        self.attempts = 0
        self.responses: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.attempts >= MAX_ATTEMPTS:
            raise RuntimeError("evaluation request budget exhausted")
        self.attempts += 1
        response = await self.inner.handle_async_request(request)
        body = await response.aread()
        try:
            decoded = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            decoded = {}
        self.responses.append(
            {
                "status": response.status_code,
                "endpoint": request.url.path,
                "model": decoded.get("model"),
                "usage": decoded.get("usage"),
            }
        )
        return response

    async def aclose(self) -> None:
        await self.inner.aclose()


async def main(output: Path) -> None:
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    fixture_bytes = FIXTURE.read_bytes()
    fixture = json.loads(fixture_bytes)
    inputs = []
    for row in fixture["rows"][:4]:
        inputs.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "reference": row["reference_answer"],
                "contexts": [item["content"] for item in row["context"]],
                "kind": "saved_generated_answer",
            }
        )
    source = inputs[0]
    inputs.append(
        {
            **source,
            "id": "negative_control_wrong_duration",
            "answer": "Customers may return an unused product within 365 days after delivery.",
            "kind": "authored_negative_control",
        }
    )
    if os.getenv("RAGAS_CASE_MODE") == "strong-negative":
        chunk_id = fixture["rows"][0]["context"][0]["id"]
        inputs = [
            {
                **source,
                "id": "negative_control_explicit_day_365",
                "answer": f"The policy explicitly permits a return on day 365 [{chunk_id}].",
                "kind": "authored_negative_control",
            }
        ]
    transport = CountedTransport()
    http = httpx.AsyncClient(transport=transport, timeout=60.0)
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url="https://api.openai.com/v1",
        timeout=60.0,
        max_retries=0,
        http_client=http,
    )
    llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")
    scorers = RagasScorers(
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        True,
    )
    result = {
        "ragas_version": "0.4.3",
        "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "fixture": FIXTURE.name,
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "max_actual_http_attempts": MAX_ATTEMPTS,
        "requested_cases": len(inputs),
        "rows": [],
        "source": (
            "project evaluate_response with injected real RAGAS scorers and counted OpenAI client"
        ),
    }
    try:
        for case in inputs:
            before = transport.attempts
            try:
                evaluation = await asyncio.wait_for(
                    evaluate_response(
                        case["question"],
                        case["answer"],
                        case["contexts"],
                        case["reference"],
                        scorer_factory=lambda _reference: scorers,
                    ),
                    timeout=180,
                )
            except TimeoutError:
                evaluation = {"status": "timeout", "scores": {}, "error_type": "timeout"}
            result["rows"].append(
                {**case, "evaluation": evaluation, "http_attempts": transport.attempts - before}
            )
            result["http_attempts"] = transport.attempts
            result["responses"] = transport.responses
            output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            print(case["id"], evaluation["status"], flush=True)
            if transport.attempts >= MAX_ATTEMPTS or any(
                r["status"] in (401, 403, 429) for r in transport.responses
            ):
                break
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
