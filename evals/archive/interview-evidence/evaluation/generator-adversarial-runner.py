"""Run fictional adversarial cases against a frozen generator; no safety pass scoring."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

from xai_rag.config import settings
from xai_rag.generation.generator import GenerationContractError, RAGGenerator, _format_context
from xai_rag.models import RankedResult

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "generator-adversarial-cases.jsonl"
OUTPUT = HERE / "generator-adversarial-results.json"
SOURCE = Path("/tmp/xai-generator-adversarial-snapshot/xai_rag")
MAX_ATTEMPTS = 100


def source_digest():
    h = hashlib.sha256()
    for p in sorted(SOURCE.rglob("*.py")):
        h.update(str(p.relative_to(SOURCE)).encode())
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


async def main():
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "3b6f2f6c6d1d3f49c7a8e376549c538782a6fccedf42beb9017026f61e289d39"
    )
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(cases) == 20 and len({c["id"] for c in cases}) == 20
    assert all(c["label_status"] == "author_defined_not_human_validated" for c in cases)
    for c in cases:
        ids = [item["id"] for item in c["context"]]
        assert len(set(ids)) == len(ids) and set(c["acceptable_source_ids"]) <= set(ids)
    report = {
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": source_digest(),
        "generator_model_alias": "gpt-4o-mini",
        "fixture_status": "synthetic_author_defined_not_human_validated",
        "max_provider_attempts": MAX_ATTEMPTS,
        "attempts": 0,
        "transport_failure_streak": 0,
        "stop_reason": None,
        "rows": [],
    }
    with OUTPUT.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    def save():
        OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url="https://api.openai.com/v1",
        timeout=60,
        max_retries=0,
    )
    create = client.chat.completions.create
    active = None

    async def capture(**kwargs):
        if report["stop_reason"] or report["attempts"] >= MAX_ATTEMPTS:
            report["stop_reason"] = report["stop_reason"] or "provider_attempt_cap"
            save()
            raise RuntimeError("Evaluation stopped by attempt budget or provider failure")
        report["attempts"] += 1
        call = {
            "attempt": report["attempts"],
            "prompt_messages": kwargs["messages"],
            "status": "in_flight",
            "usage": None,
        }
        active["provider_calls"].append(call)
        save()
        start = time.monotonic()
        try:
            response = await create(**kwargs)
        except Exception as exc:
            call.update(
                status="provider_error",
                error_type=type(exc).__name__,
                status_code=getattr(exc, "status_code", None),
                latency_seconds=round(time.monotonic() - start, 3),
            )
            if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
                report["stop_reason"] = "authentication_or_permission_error"
            elif isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
                report["transport_failure_streak"] += 1
                if report["transport_failure_streak"] >= 2:
                    report["stop_reason"] = "repeated_transport_or_rate_limit"
            else:
                report["transport_failure_streak"] = 0
            save()
            raise
        report["transport_failure_streak"] = 0
        call.update(
            status="response",
            model=response.model,
            raw_content=response.choices[0].message.content,
            usage=response.usage.model_dump() if response.usage else None,
            latency_seconds=round(time.monotonic() - start, 3),
        )
        save()
        return response

    client.chat.completions.create = capture
    try:
        for c in cases:
            for repeat in (1, 2):
                if report["stop_reason"] or report["attempts"] >= MAX_ATTEMPTS:
                    report["stop_reason"] = report["stop_reason"] or "provider_attempt_cap"
                    save()
                    return
                chunks = [RankedResult(id=x["id"], content=x["text"]) for x in c["context"]]
                _, included = _format_context(chunks, settings.llm_context_max_chars)
                visible = [x.id for x in included]
                active = {
                    "case_id": c["id"],
                    "repeat": repeat,
                    "category": c["category"],
                    "question": c["question"],
                    "expected_behavior": c["expected_behavior"],
                    "manual_review": c["manual_review"],
                    "visible_context": [x.model_dump() for x in included],
                    "provider_calls": [],
                    "outcome": "in_progress",
                }
                report["rows"].append(active)
                save()
                start = time.monotonic()
                try:
                    outcome = await RAGGenerator(
                        client=SimpleNamespace(chat=client.chat), model="gpt-4o-mini"
                    ).generate_with_evidence(c["question"], chunks)
                    result = outcome.result
                    answer = result.answer
                    cited = re.findall(r"\[([^\[\]\n]+)\]", answer)
                    active.update(
                        outcome="generated",
                        result=result.model_dump(),
                        signals={
                            "marker_hits": [
                                m
                                for m in c["machine_signals"]["forbidden_output_markers"]
                                if m.casefold() in answer.casefold()
                            ],
                            "inline_citations": cited,
                            "non_visible_inline_citations": [x for x in cited if x not in visible],
                            "warning": (
                                "Indicators only; quotation and paraphrase require review. "
                                "No safety pass is inferred."
                            ),
                        },
                    )
                except Exception as exc:
                    active.update(
                        outcome="error",
                        error_type=type(exc).__name__,
                        status_code=getattr(exc, "status_code", None),
                    )
                    if isinstance(exc, GenerationContractError):
                        active["contract_reason"] = str(exc)
                active["latency_seconds"] = round(time.monotonic() - start, 3)
                save()
                print(
                    c["id"],
                    repeat,
                    active["outcome"],
                    "provider attempts",
                    report["attempts"],
                    flush=True,
                )
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
