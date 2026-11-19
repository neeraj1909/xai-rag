"""Guard against malformed keys and concurrent rate-limit admission."""

import hashlib
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException

from xai_rag.api import app as api
from xai_rag.config import settings

pytestmark = pytest.mark.unit


def test_non_ascii_api_key_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "a" * 32)
    with pytest.raises(HTTPException) as exc_info:
        api.require_api_key("é")
    assert exc_info.value.status_code == 401


def test_rate_limit_is_atomic_under_concurrency(monkeypatch):
    key = "a" * 32
    monkeypatch.setattr(settings, "api_key", key)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
    barrier = threading.Barrier(8)

    class SlowReadDeque(deque):
        def __len__(self):
            size = super().__len__()
            time.sleep(0.03)  # Enlarge the admission race while preserving actual state.
            return size

    principal = hashlib.sha256(key.encode("utf-8")).hexdigest()
    monkeypatch.setattr(api, "_RATE_LIMIT_BUCKETS", {principal: SlowReadDeque()})

    def admit():
        barrier.wait(timeout=5)
        try:
            api.require_api_key(key)
            return 200
        except HTTPException as exc:
            return exc.status_code

    with ThreadPoolExecutor(max_workers=8) as executor:
        statuses = list(executor.map(lambda _: admit(), range(8)))
    assert statuses.count(200) == 1
    assert statuses.count(429) == 7
