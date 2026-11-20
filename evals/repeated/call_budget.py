"""Cross-process provider attempt ledger for the isolated evaluation only."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from urllib.parse import urlsplit

from openai import AsyncOpenAI

purpose = ContextVar("evaluation_provider_purpose", default="unclassified")
case_context = ContextVar("evaluation_case", default=None)


def remaining():
    path = os.environ["XAI_EVAL_BUDGET_FILE"]
    cap = int(os.environ["XAI_EVAL_MAX_PROVIDER_CALLS"])
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return cap
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        with os.fdopen(os.dup(fd), "r") as reader:
            return max(0, cap - sum(json.loads(line)["phase"] == "attempt" for line in reader))
    finally:
        os.close(fd)


def install():
    path = os.environ["XAI_EVAL_BUDGET_FILE"]
    cap = int(os.environ["XAI_EVAL_MAX_PROVIDER_CALLS"])
    if cap < 1:
        raise ValueError("Provider cap must be positive")
    original = AsyncOpenAI.request

    def record(event, check_cap=False):
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if check_cap:
                os.lseek(fd, 0, os.SEEK_SET)
                with os.fdopen(os.dup(fd), "r") as reader:
                    count = sum(json.loads(line)["phase"] == "attempt" for line in reader)
                if count >= cap:
                    raise RuntimeError("Shared evaluation provider-call cap reached")
            os.write(fd, (json.dumps(event, ensure_ascii=False) + "\n").encode())
        finally:
            os.close(fd)

    async def wrapped(self, cast_to, options, **kwargs):
        # Refuse clients that could hide retries from the cross-process cap.
        if options.get_max_retries(self.max_retries) != 0:
            raise RuntimeError("SDK retries must be disabled for evaluation accounting")
        call_id = uuid.uuid4().hex
        call_purpose = purpose.get()
        payload = getattr(options, "json_data", None)
        endpoint = urlsplit(str(options.url)).path
        seeded = endpoint.rstrip("/").endswith("/chat/completions")
        if seeded:
            if not isinstance(payload, dict):
                raise RuntimeError("chat payload missing: refusing unseeded request")
            # Evaluation-only instrumentation; embeddings are not chat completions.
            payload["seed"] = int(os.environ["XAI_EVAL_MODEL_SEED"])
        requested_model = payload.get("model") if isinstance(payload, dict) else None
        request_hash = (
            hashlib.sha256(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode()
            ).hexdigest()
            if isinstance(payload, dict)
            else None
        )
        record(
            {
                "phase": "attempt",
                "id": call_id,
                "time_unix": time.time(),
                "purpose": call_purpose,
                "requested_model": requested_model,
                "endpoint": endpoint,
                "request_sha256": request_hash,
                "request_payload": payload,
                "case_id": case_context.get(),
                "run": (
                    Path(os.environ["XAI_EVAL_RUN_FILE"]).read_text().strip()
                    if os.environ.get("XAI_EVAL_RUN_FILE")
                    and os.path.exists(os.environ["XAI_EVAL_RUN_FILE"])
                    else os.environ.get("XAI_EVAL_RUN")
                ),
                "model_seed": payload.get("seed") if seeded else None,
                "seed_scope": "chat completions" if seeded else "not applicable",
            },
            check_cap=True,
        )
        try:
            result = await original(self, cast_to, options, **kwargs)
        except BaseException as exc:
            record(
                {
                    "phase": "outcome",
                    "id": call_id,
                    "purpose": call_purpose,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "status_code", None),
                }
            )
            raise
        usage = getattr(result, "usage", None)
        record(
            {
                "phase": "outcome",
                "id": call_id,
                "purpose": call_purpose,
                "status": "completed",
                "model": getattr(result, "model", None),
                "system_fingerprint": getattr(result, "system_fingerprint", None),
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
            }
        )
        return result

    AsyncOpenAI.request = wrapped
