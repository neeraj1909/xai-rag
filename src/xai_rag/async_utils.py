"""Helpers for cancellation of asynchronous operations backed by threads."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable

_T = TypeVar("_T")


async def wait_for_thread_work(work: Awaitable[_T]) -> _T:
    """Propagate cancellation only after already-submitted thread work finishes.

    Python cannot stop a running worker thread. Shield its future and drain it
    after cancellation so callers can hold resource admission until it finishes.
    Do not use this for network calls or work that is safe to cancel directly.
    """
    future = asyncio.ensure_future(work)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not future.cancelled():
            future.exception()  # Consume errors from work abandoned by the caller.
        raise
