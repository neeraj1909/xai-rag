"""Start the real API with evaluation-only capture of generator-visible evidence.

Capture is safe only for the fictional test corpus: no API keys or headers are
written. The app's production source is not modified.
"""

from __future__ import annotations

import json
import os
import time
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from pathlib import Path

import uvicorn
from call_budget import case_context, install, purpose

from xai_rag.explainability.faithfulness import FaithfulnessChecker
from xai_rag.generation.generator import RAGGenerator
from xai_rag.generation.grounding_judge import GroundingJudge

install()
capture_path = Path(os.environ["XAI_EVAL_CAPTURE_PATH"])
if capture_path.exists():
    raise RuntimeError("Refusing to overwrite existing evidence capture")
fd = os.open(capture_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
capture = os.fdopen(fd, "a", encoding="utf-8", buffering=1)
diagnostic_path = capture_path.with_name("grounding-diagnostics.jsonl")
fd = os.open(diagnostic_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
diagnostic = os.fdopen(fd, "a", encoding="utf-8", buffering=1)
case_ids = {
    c["question"]: c["id"]
    for c in json.loads(Path(os.environ["XAI_EVAL_FIXTURE"]).read_text())["queries"]
}
query_context = ContextVar("synthetic_eval_query", default=None)
original_generate = RAGGenerator.generate_with_evidence
original_judge = GroundingJudge.check_claims
original_nli = FaithfulnessChecker.check_claims


async def captured_generate(self, query, chunks):
    query_context.set(query)
    case_context.set(case_ids.get(query))
    token = purpose.set("generation")
    try:
        outcome = await original_generate(self, query, chunks)
    finally:
        purpose.reset(token)
    capture.write(
        json.dumps(
            {
                "query": query,
                "unverified_generation": outcome.result.model_dump(mode="json"),
                "evidence": [
                    {"id": c.id, "content": c.content, "source_file": c.metadata.get("source_file")}
                    for c in outcome.evidence
                ],
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    return outcome


RAGGenerator.generate_with_evidence = captured_generate


async def captured_judge(self, *args, **kwargs):
    token = purpose.set("grounding_judge")
    try:
        accepted = await original_judge(self, *args, **kwargs)
        diagnostic.write(
            json.dumps(
                {
                    "query": query_context.get(),
                    "stage": "grounding_judge",
                    "monotonic": time.monotonic(),
                    "decision": asdict(accepted) if is_dataclass(accepted) else accepted,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        return accepted
    except Exception as exc:
        diagnostic.write(
            json.dumps(
                {
                    "query": query_context.get(),
                    "stage": "grounding_judge",
                    "error_type": type(exc).__name__,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        raise
    finally:
        purpose.reset(token)


GroundingJudge.check_claims = captured_judge


async def captured_nli(self, *args, **kwargs):
    try:
        verdicts = await original_nli(self, *args, **kwargs)
    except Exception as exc:
        diagnostic.write(
            json.dumps(
                {"query": query_context.get(), "stage": "nli", "error_type": type(exc).__name__},
                ensure_ascii=False,
            )
            + "\n"
        )
        raise
    diagnostic.write(
        json.dumps(
            {
                "query": query_context.get(),
                "stage": "nli",
                "verdicts": [v.model_dump(mode="json") for v in verdicts],
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    return verdicts


FaithfulnessChecker.check_claims = captured_nli
uvicorn.run(
    "xai_rag.api.app:app",
    host="127.0.0.1",
    port=int(os.environ["XAI_EVAL_API_PORT"]),
    log_level="warning",
    access_log=False,
)
