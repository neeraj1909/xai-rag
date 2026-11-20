"""Faithfulness checker using NLI (Natural Language Inference).

Determines whether each claim in the generated answer is supported by the
retrieved evidence chunks, using a DeBERTa-based entailment model.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING

import torch

from xai_rag.async_utils import wait_for_thread_work
from xai_rag.config import settings
from xai_rag.models import Claim, ClaimVerdict, RankedResult, Verdict

if TYPE_CHECKING:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

_NLI_WORKERS = 2
_NLI_POOL = ThreadPoolExecutor(max_workers=_NLI_WORKERS, thread_name_prefix="nli")
_NLI_LOAD_LOCK = Lock()

_DEFAULT_MODEL = settings.nli_model
_DEFAULT_REVISION = settings.nli_model_revision

_REQUIRED_NLI_LABELS = frozenset({"entailment", "neutral", "contradiction"})


class NLIModelContractError(RuntimeError):
    """Raised when an NLI model does not expose the required label semantics."""


def _canonical_nli_label(label: str) -> str | None:
    normalized = label.casefold()
    for expected in _REQUIRED_NLI_LABELS:
        if expected in normalized:
            return expected
    return None


@lru_cache(maxsize=1)
def _load_nli_model(
    model_name: str,
    revision: str,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Lazily load and cache the NLI model + tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading NLI model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=False,
    )
    model.eval()
    return model, tokenizer


class FaithfulnessChecker:
    """Verifies generated claims against retrieved chunks using NLI.

    The DeBERTa model is loaded lazily on first invocation and cached.
    Inference runs in a thread-pool executor to avoid blocking the event loop.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        revision: str = _DEFAULT_REVISION,
    ) -> None:
        self._model_name = model_name
        self._revision = revision

    def _get_model_and_tokenizer(
        self,
    ) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        with _NLI_LOAD_LOCK:
            return _load_nli_model(self._model_name, self._revision)

    def _score_pair(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Run NLI inference for a single premise-hypothesis pair.

        Returns dict with keys ``entailment``, ``neutral``, ``contradiction``.
        """
        model, tokenizer = self._get_model_and_tokenizer()

        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        raw_mapping = getattr(model.config, "id2label", {})
        mapped: dict[str, float] = {}
        for index in range(len(probs)):
            raw_label = raw_mapping.get(index, raw_mapping.get(str(index), ""))
            label = _canonical_nli_label(str(raw_label))
            if label is not None:
                mapped[label] = float(probs[index])

        if set(mapped) != _REQUIRED_NLI_LABELS:
            raise NLIModelContractError(
                "NLI model must identify entailment, neutral, and contradiction labels"
            )
        return mapped

    def _check_single_claim(
        self,
        claim: Claim,
        chunks: list[RankedResult],
    ) -> ClaimVerdict:
        """Find the best supporting chunk for a claim (sync, runs in executor)."""
        best_entailment = 0.0
        best_contradiction = 0.0
        best_chunk_id: str | None = None
        contradicting_chunk_id: str | None = None

        scoped_chunks = chunks
        if claim.source_chunk_ids:
            declared_ids = set(claim.source_chunk_ids)
            scoped_chunks = [chunk for chunk in chunks if chunk.id in declared_ids]

        for chunk in scoped_chunks:
            scores = self._score_pair(premise=chunk.content, hypothesis=claim.text)
            if scores["entailment"] > best_entailment:
                best_entailment = scores["entailment"]
                best_chunk_id = chunk.id
            if scores["contradiction"] > best_contradiction:
                best_contradiction = scores["contradiction"]
                contradicting_chunk_id = chunk.id

        if (
            best_contradiction >= settings.nli_contradiction_threshold
            and best_contradiction >= best_entailment
        ):
            verdict = Verdict.NOT_SUPPORTED
            confidence = best_contradiction
        elif best_entailment >= settings.nli_entailment_threshold:
            verdict = Verdict.SUPPORTED
            confidence = best_entailment
        else:
            verdict = Verdict.NEUTRAL
            confidence = max(best_entailment, best_contradiction)

        return ClaimVerdict(
            claim=claim,
            verdict=verdict,
            nli_entailment=best_entailment,
            nli_contradiction=best_contradiction,
            supporting_chunk_id=(
                best_chunk_id if best_entailment >= settings.nli_entailment_threshold else None
            ),
            contradicting_chunk_id=(
                contradicting_chunk_id
                if best_contradiction >= settings.nli_contradiction_threshold
                else None
            ),
            confidence=confidence,
        )

    async def check_claims(
        self,
        claims: list[Claim],
        chunks: list[RankedResult],
    ) -> list[ClaimVerdict]:
        """Verify each claim against the retrieved chunks using NLI.

        Parameters
        ----------
        claims:
            Factual claims extracted from the generated answer.
        chunks:
            Evidence chunks from retrieval.

        Returns
        -------
        list[ClaimVerdict]
            One verdict per claim with entailment/contradiction scores and the
            best supporting chunk.
        """
        if not claims:
            return []

        if not chunks:
            return [
                ClaimVerdict(
                    claim=c,
                    verdict=Verdict.NOT_SUPPORTED,
                    nli_entailment=0.0,
                    nli_contradiction=0.0,
                    supporting_chunk_id=None,
                    confidence=0.0,
                )
                for c in claims
            ]

        loop = asyncio.get_running_loop()

        verdicts: list[ClaimVerdict] = []
        # Do not enqueue an unbounded number of claims into the thread pool.
        for start in range(0, len(claims), _NLI_WORKERS):
            futures = [
                loop.run_in_executor(
                    _NLI_POOL,
                    self._check_single_claim,
                    claim,
                    chunks,
                )
                for claim in claims[start : start + _NLI_WORKERS]
            ]
            # A failed claim must not release admission while sibling threads run.
            batch = await wait_for_thread_work(asyncio.gather(*futures, return_exceptions=True))
            for result in batch:
                if isinstance(result, BaseException):
                    raise result
                verdicts.append(result)

        supported = sum(1 for v in verdicts if v.verdict == Verdict.SUPPORTED)
        logger.info(
            "Faithfulness check: %d/%d claims supported",
            supported,
            len(claims),
        )

        return list(verdicts)
