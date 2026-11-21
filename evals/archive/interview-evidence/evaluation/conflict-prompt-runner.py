"""Bounded, randomized paired generator-prompt probe; synthetic data only."""

import ast
import asyncio
import hashlib
import json
import random
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
from xai_rag.generation import generator
from xai_rag.models import RankedResult

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "conflict-prompt-cases.json"
OUTPUT = HERE / "conflict-prompt-results.json"
BEFORE = Path("/tmp/xai-conflict-before/xai_rag")
CANDIDATE = Path("/tmp/xai-conflict-candidate/xai_rag")


def digest_source(path):
    h = hashlib.sha256()
    for p in sorted(path.rglob("*.py")):
        h.update(str(p.relative_to(path)).encode())
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def extract_prompt(path):
    tree = ast.parse((path / "generation/generator.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_SYSTEM_PROMPT" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("missing prompt")


async def main():
    raw = FIXTURE.read_bytes()
    cases = json.loads(raw)["cases"]
    assert len(cases) == 7
    before, candidate = extract_prompt(BEFORE), extract_prompt(CANDIDATE)
    assert before == generator._SYSTEM_PROMPT
    assert "report each supported alternative" in candidate and candidate != before
    report = {
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "before_source_sha256": digest_source(BEFORE),
        "candidate_source_sha256": digest_source(CANDIDATE),
        "model_alias": "gpt-4o-mini",
        "seed": 1939,
        "design": (
            "Two repeats per case, randomized prompt order in each pair; "
            "review criteria fixed before calls."
        ),
        "max_actual_provider_attempts": 60,
        "attempts": 0,
        "error_streak": 0,
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
    original = client.chat.completions.create
    active = None

    async def capture(**kwargs):
        if report["attempts"] >= 60 or report["stop_reason"]:
            report["stop_reason"] = report["stop_reason"] or "attempt_cap"
            save()
            raise RuntimeError("Provider attempt cap or failure stop reached")
        report["attempts"] += 1
        call = {
            "attempt": report["attempts"],
            "messages": kwargs["messages"],
            "status": "in_flight",
        }
        active["provider_calls"].append(call)
        save()
        start = time.monotonic()
        try:
            response = await original(**kwargs)
        except Exception as exc:
            call.update(
                status="error",
                error_type=type(exc).__name__,
                status_code=getattr(exc, "status_code", None),
                latency_seconds=round(time.monotonic() - start, 3),
            )
            if isinstance(exc, (AuthenticationError, PermissionDeniedError, RateLimitError)):
                report["stop_reason"] = "auth_or_rate_limit"
            elif isinstance(exc, (APIConnectionError, APITimeoutError)):
                report["error_streak"] += 1
                if report["error_streak"] >= 2:
                    report["stop_reason"] = "repeated_transport_error"
            else:
                report["error_streak"] = 0
            save()
            raise
        report["error_streak"] = 0
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
    rng = random.Random(report["seed"])
    try:
        for c in cases:
            for repeat in (1, 2):
                order = ["before", "candidate"]
                rng.shuffle(order)
                for variant in order:
                    if report["stop_reason"] or report["attempts"] >= 60:
                        report["stop_reason"] = report["stop_reason"] or "attempt_cap"
                        save()
                        return
                    generator._SYSTEM_PROMPT = {"before": before, "candidate": candidate}[variant]
                    chunks = [RankedResult(id=x["id"], content=x["text"]) for x in c["context"]]
                    _, visible = generator._format_context(chunks, settings.llm_context_max_chars)
                    active = {
                        "case_id": c["id"],
                        "repeat": repeat,
                        "variant": variant,
                        "pair_order": order,
                        "question": c["question"],
                        "review_criteria": c["predeclared_review_criteria"],
                        "visible_evidence": [x.model_dump() for x in visible],
                        "provider_calls": [],
                        "outcome": "in_progress",
                    }
                    report["rows"].append(active)
                    save()
                    try:
                        outcome = await generator.RAGGenerator(
                            client=SimpleNamespace(chat=client.chat), model="gpt-4o-mini"
                        ).generate_with_evidence(c["question"], chunks)
                        active.update(outcome="generated", result=outcome.result.model_dump())
                    except Exception as exc:
                        active.update(
                            outcome="error",
                            error_type=type(exc).__name__,
                            status_code=getattr(exc, "status_code", None),
                        )
                        if isinstance(exc, generator.GenerationContractError):
                            active["contract_reason"] = str(exc)
                    save()
                    print(
                        c["id"],
                        repeat,
                        variant,
                        active["outcome"],
                        "attempts",
                        report["attempts"],
                        flush=True,
                    )
    finally:
        generator._SYSTEM_PROMPT = before
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
