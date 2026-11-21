"""Run synthetic H05 against the frozen source; no retrieval or judge calls."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from openai import AsyncOpenAI

from xai_rag.config import settings
from xai_rag.generation.generator import GenerationContractError, RAGGenerator, _validate_citations
from xai_rag.models import RAGGenerationResult, RankedResult

BASE = Path("/home/neeraj/Code/personalized_interview_materials/projects/xai-rag/evaluation")
OUTPUT = BASE / "h05-generation-diagnostic.json"
FIXTURE = BASE / "holdout-cases.json"
SOURCE = Path("/tmp/xai-h05-snapshot/xai_rag")


def digest_source():
    result = hashlib.sha256()
    for path in sorted(SOURCE.rglob("*.py")):
        result.update(str(path.relative_to(SOURCE)).encode())
        result.update(b"\0")
        result.update(hashlib.sha256(path.read_bytes()).digest())
    return result.hexdigest()


async def main():
    fixture_bytes = FIXTURE.read_bytes()
    fixture = json.loads(fixture_bytes)
    q = next(x for x in fixture["queries"] if x["id"] == "H05")
    prior = json.loads((BASE / "holdout-stability-pass1.json").read_text())
    row = next(x for x in prior["rows"] if x["id"] == "H05")
    ranked = row["stages"]["rerank"]
    by_id = {d["id"]: d["text"] for d in fixture["documents"]}
    chunks = [
        RankedResult(id=cid, content=by_id[sid])
        for cid, sid in zip(ranked["chunk_ids"], ranked["ranking"], strict=True)
    ]
    source_hash = digest_source()
    assert source_hash == prior["source_tree_sha256"], "Frozen source hash mismatch"
    assert hashlib.sha256(fixture_bytes).hexdigest() == prior["dataset_sha256"], (
        "Fixture hash mismatch"
    )
    report = {
        "case": q,
        "source_tree_sha256": source_hash,
        "dataset_sha256": prior["dataset_sha256"],
        "context_reconstruction": (
            "H05 pass1 saved rerank IDs/order joined to unchanged fixture document contents; "
            "original failed service call did not persist its prompt-visible context, "
            "so exact ranking equivalence is unverified."
        ),
        "chunks": [c.model_dump() for c in chunks],
        "max_provider_attempts": 15,
        "rows": [],
    }
    with OUTPUT.open("x") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url="https://api.openai.com/v1",
        timeout=60,
        max_retries=0,
    )
    create = client.chat.completions.create
    calls = []

    async def capture(**kwargs):
        if len(calls) >= 15:
            raise RuntimeError("Provider attempt budget exhausted")
        call = {
            "attempt": len(calls) + 1,
            "repair_prompt": "Your previous response did not satisfy the citation contract."
            in kwargs["messages"][0]["content"],
        }
        calls.append(call)
        try:
            response = await create(**kwargs)
        except Exception as exc:
            call["error"] = {
                "type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None),
            }
            raise
        call["model"] = response.model
        call["usage"] = response.usage.model_dump() if response.usage else None
        call["raw_content"] = response.choices[0].message.content
        try:
            parsed = RAGGenerationResult.model_validate_json(call["raw_content"])
            _validate_citations(parsed, chunks)
            call["contract"] = "accepted"
        except GenerationContractError as exc:
            call["contract"] = "rejected"
            call["contract_reason"] = str(exc)
        except Exception as exc:
            call["contract"] = "validation_error"
            call["validation_error_type"] = type(exc).__name__
        return response

    client.chat.completions.create = capture
    try:
        for iteration in range(1, 6):
            before = len(calls)
            entry = {"iteration": iteration}
            try:
                result = await RAGGenerator(
                    client=SimpleNamespace(chat=client.chat), model="gpt-4o-mini"
                ).generate(q["question"], chunks)
                entry["result"] = result.model_dump()
            except Exception as exc:
                entry["error"] = {
                    "type": type(exc).__name__,
                    "status_code": getattr(exc, "status_code", None),
                }
                if isinstance(exc, GenerationContractError):
                    entry["error"]["contract_reason"] = str(exc)
            entry["provider_calls"] = calls[before:]
            report["rows"].append(entry)
            OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print(
                "iteration",
                iteration,
                "calls",
                len(calls),
                "result",
                "success" if "result" in entry else entry["error"]["type"],
                flush=True,
            )
            if len(calls) >= 15 or (
                "error" in entry and entry["error"]["type"] not in ("GenerationContractError",)
            ):
                break
    finally:
        await client.close()


asyncio.run(main())
