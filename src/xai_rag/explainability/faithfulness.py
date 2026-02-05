"""Faithfulness checker using NLI (Natural Language Inference).

Determines whether each claim in the generated answer is supported by the
retrieved evidence chunks, using a DeBERTa-based entailment model.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from xai_rag.models import Claim, ClaimVerdict, RankedResult, Verdict

if TYPE_CHECKING:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

_NLI_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nli")
from xai_rag.config import settings

_DEFAULT_MODEL = settings.nli_model

# Label mapping for the DeBERTa MNLI model.
_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


@lru_cache(maxsize=1)
def _load_nli_model(
    model_name: str,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Lazily load and cache the NLI model + tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading NLI model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


class FaithfulnessChecker:
    """Verifies generated claims against retrieved chunks using NLI.

    The DeBERTa model is loaded lazily on first invocation and cached.
    Inference runs in a thread-pool executor to avoid blocking the event loop.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name

    def _get_model_and_tokenizer(
        self,
    ) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        return _load_nli_model(self._model_name)

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

        return {
            _LABEL_MAP[i]: float(probs[i])
            for i in range(len(probs))
        }

    def _check_single_claim(
        self,
        claim: Claim,
        chunks: list[RankedResult],
    ) -> ClaimVerdict:
        """Find the best supporting chunk for a claim (sync, runs in executor)."""
        best_entailment = 0.0
        best_contradiction = 0.0
        best_chunk_id: str | None = None
        best_scores: dict[str, float] = {}

        for chunk in chunks:
            scores = self._score_pair(premise=chunk.content, hypothesis=claim.text)
            if scores["entailment"] > best_entailment:
                best_entailment = scores["entailment"]
                best_contradiction = scores["contradiction"]
                best_chunk_id = chunk.id
                best_scores = scores

        # Determine verdict.
        if best_entailment >= 0.5:
            verdict = Verdict.SUPPORTED
        elif best_scores.get("contradiction", 0.0) >= 0.5:
            verdict = Verdict.NOT_SUPPORTED
        else:
            verdict = Verdict.NEUTRAL

        confidence = max(best_entailment, best_scores.get("contradiction", 0.0))

        return ClaimVerdict(
            claim=claim,
            verdict=verdict,
            nli_entailment=best_entailment,
            nli_contradiction=best_contradiction,
            supporting_chunk_id=best_chunk_id,
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

        # Run NLI checks concurrently in thread pool.
        futures = [
            loop.run_in_executor(
                _NLI_POOL,
                self._check_single_claim,
                claim,
                chunks,
            )
            for claim in claims
        ]

        verdicts = await asyncio.gather(*futures)

        supported = sum(1 for v in verdicts if v.verdict == Verdict.SUPPORTED)
        logger.info(
            "Faithfulness check: %d/%d claims supported",
            supported,
            len(claims),
        )

        return list(verdicts)
