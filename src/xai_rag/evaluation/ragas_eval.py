"""RAGAS metrics for evaluating one generated RAG response."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from xai_rag.config import DEFAULT_OPENAI_BASE_URL, settings

logger = logging.getLogger(__name__)


class RagasMetric(Protocol):
    """Minimal async contract shared by the RAGAS collections metrics."""

    async def ascore(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class RagasScorers:
    """The RAGAS scorers used for one response evaluation."""

    faithfulness: RagasMetric
    answer_relevancy: RagasMetric
    context_precision: RagasMetric
    context_uses_reference: bool
    owned_client: Any | None = None


ScorerFactory = Callable[[str | None], RagasScorers]


async def _finish_cleanup(work: Awaitable[None]) -> None:
    """Finish cleanup despite repeated cancellation, then propagate cancellation."""
    task = asyncio.ensure_future(work)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


async def _close_client(client: Any) -> None:
    try:
        await client.close()
    except Exception as exc:
        logger.warning("RAGAS client close failed (%s)", type(exc).__name__)


async def _build_scorers(reference: str | None) -> RagasScorers:
    """Construct current RAGAS collection metrics with explicit model clients."""
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextUtilization,
        Faithfulness,
    )

    client_kwargs: dict[str, Any] = {
        "api_key": settings.openai_api_key,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": 0,
        "base_url": settings.llm_base_url or DEFAULT_OPENAI_BASE_URL,
    }

    client = AsyncOpenAI(**client_kwargs)
    try:
        llm = llm_factory(settings.ragas_llm_model, client=client)
        embeddings = OpenAIEmbeddings(client=client, model=settings.ragas_embedding_model)
        context_uses_reference = reference is not None
        return RagasScorers(
            faithfulness=Faithfulness(llm=llm),
            answer_relevancy=AnswerRelevancy(llm=llm, embeddings=embeddings),
            context_precision=(
                ContextPrecision(llm=llm) if context_uses_reference else ContextUtilization(llm=llm)
            ),
            context_uses_reference=context_uses_reference,
            owned_client=client,
        )
    except BaseException:
        await _finish_cleanup(_close_client(client))
        raise


def _score_value(result: Any) -> float:
    """Extract a finite numeric value from a RAGAS ``MetricResult``."""
    value = float(getattr(result, "value", result))
    if not math.isfinite(value):
        raise ValueError("RAGAS returned a non-finite metric value")
    return value


async def evaluate_response(
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
    *,
    scorer_factory: ScorerFactory | None = None,
) -> dict[str, Any]:
    """Return RAGAS faithfulness, answer relevancy, and context precision.

    With a reference answer, RAGAS ``ContextPrecision`` compares retrieved
    contexts with that reference. Without one, RAGAS ``ContextUtilization``
    provides its reference-free context-precision variant by comparing the
    contexts with the generated response.
    """
    try:
        scorers = (
            await _build_scorers(ground_truth)
            if scorer_factory is None
            else scorer_factory(ground_truth)
        )
    except ImportError:
        logger.warning("RAGAS dependencies are unavailable; skipping evaluation")
        return {
            "status": "unavailable",
            "scores": {},
            "error_type": "dependency_unavailable",
        }
    except Exception as exc:
        logger.error("RAGAS setup failed (%s)", type(exc).__name__)
        return {
            "status": "error",
            "scores": {},
            "error_type": type(exc).__name__,
        }

    tasks: list[asyncio.Task] = []
    group = None
    try:
        context_arguments = {
            "user_input": query,
            "retrieved_contexts": contexts,
            ("reference" if scorers.context_uses_reference else "response"): (
                ground_truth if scorers.context_uses_reference else answer
            ),
        }
        tasks.append(
            asyncio.create_task(
                scorers.faithfulness.ascore(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=contexts,
                )
            )
        )
        tasks.append(
            asyncio.create_task(scorers.answer_relevancy.ascore(user_input=query, response=answer))
        )
        tasks.append(asyncio.create_task(scorers.context_precision.ascore(**context_arguments)))
        group = asyncio.gather(*tasks)
        faithfulness, answer_relevancy, context_precision = await asyncio.shield(group)
        scores = {
            "faithfulness": _score_value(faithfulness),
            "answer_relevancy": _score_value(answer_relevancy),
            "context_precision": _score_value(context_precision),
        }
        logger.info("RAGAS evaluation completed: %s", sorted(scores))
        return {"status": "computed", "scores": scores, "error_type": None}
    except Exception as exc:
        logger.error("RAGAS evaluation failed (%s)", type(exc).__name__)
        return {
            "status": "error",
            "scores": {},
            "error_type": type(exc).__name__,
        }
    finally:

        async def cleanup() -> None:
            # Failed metrics do not stop their siblings; drain before client close.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if group is not None and group.done() and not group.cancelled():
                group.exception()
            if scorer_factory is None and (client := getattr(scorers, "owned_client", None)):
                await _close_client(client)

        await _finish_cleanup(cleanup())
