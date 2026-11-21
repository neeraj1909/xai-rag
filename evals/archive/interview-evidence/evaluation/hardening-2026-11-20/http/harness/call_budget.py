"""Cross-process provider attempt ledger for the isolated evaluation only."""
from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from contextvars import ContextVar

from openai import AsyncOpenAI

purpose = ContextVar("evaluation_provider_purpose", default="unclassified")


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
        call_id = uuid.uuid4().hex
        call_purpose = purpose.get()
        payload = getattr(options, "json_data", None)
        requested_model = payload.get("model") if isinstance(payload, dict) else None
        record({"phase": "attempt", "id": call_id, "time_unix": time.time(),
                "purpose": call_purpose, "requested_model": requested_model,
                "endpoint": str(options.url).split("?")[0]}, check_cap=True)
        try:
            result = await original(self, cast_to, options, **kwargs)
        except BaseException as exc:
            record({"phase": "outcome", "id": call_id, "purpose": call_purpose, "status": "error",
                    "error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None)})
            raise
        usage = getattr(result, "usage", None)
        record({"phase": "outcome", "id": call_id, "purpose": call_purpose, "status": "completed",
                "model": getattr(result, "model", None),
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else None})
        return result

    AsyncOpenAI.request = wrapped
