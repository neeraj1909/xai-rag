"""RAGAS evaluation for RAG pipeline quality metrics.

RAGAS is an optional dependency. Its adapter always returns an explicit
status so callers can distinguish a computed result from unavailable tooling
or an execution failure.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def evaluate_response(
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> dict[str, Any]:
    """Evaluate a RAG response using RAGAS metrics.

    Computes faithfulness, answer relevancy, and context precision (when
    ground truth is provided).

    Parameters
    ----------
    query:
        The original user question.
    answer:
        The generated answer text.
    contexts:
        List of context chunk texts that were fed to the generator.
    ground_truth:
        Optional reference answer for context precision scoring.

    Returns an envelope with ``status``, ``scores``, and ``error_type``.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )
    except ImportError:
        logger.warning(
            "RAGAS is not installed in the production runtime; skipping evaluation. "
            "Run it only in a separately audited evaluation environment."
        )
        return {
            "status": "unavailable",
            "scores": {},
            "error_type": "dependency_unavailable",
        }

    # Build the single-sample dataset that RAGAS expects.
    data: dict[str, list[Any]] = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
    }

    metrics = [faithfulness, answer_relevancy]

    if ground_truth is not None:
        data["ground_truth"] = [ground_truth]
        metrics.append(context_precision)

    dataset = Dataset.from_dict(data)

    try:
        result = ragas_evaluate(dataset=dataset, metrics=metrics)
        scores: dict[str, Any] = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in result.items()
            if k not in ("dataset",)
        }
        logger.info("RAGAS evaluation scores: %s", scores)
        return {"status": "computed", "scores": scores, "error_type": None}
    except Exception as exc:
        logger.error("RAGAS evaluation failed (%s)", type(exc).__name__)
        return {
            "status": "error",
            "scores": {},
            "error_type": type(exc).__name__,
        }
