"""Build an answer only from the claims checked at the release boundary."""

from __future__ import annotations

from xai_rag.models import Claim, ClaimVerdict, RankedResult, Verdict

REFUSAL = "I can't provide a complete, relevant and safe answer that I can verify."
UNAVAILABLE = "I can't verify an answer right now. Please try again later."


def claims_are_valid(claims: list[Claim], evidence: tuple[RankedResult, ...]) -> bool:
    """Reject ambiguous formatting and claims without prompt-visible evidence."""
    if not 1 <= len(claims) <= 16:
        return False
    visible = {chunk.id for chunk in evidence}
    for claim in claims:
        text = claim.text
        if (
            not text.strip()
            or len(text) > 500
            or any(ord(char) < 32 or char in "[]<>" for char in text)
            or not claim.source_chunk_ids
            or len(set(claim.source_chunk_ids)) != len(claim.source_chunk_ids)
            or any(
                source not in visible or not source or any(c in source for c in "[]\r\n")
                for source in claim.source_chunk_ids
            )
        ):
            return False
    return True


def render_supported_claims(claims: list[Claim], verdicts: list[ClaimVerdict]) -> str | None:
    """Return a cited answer only when each corresponding claim was supported.

    Support is an NLI model verdict, not a guarantee of factual correctness.
    """
    if len(claims) != len(verdicts):
        return None
    for claim, verdict in zip(claims, verdicts, strict=True):
        if (
            verdict.claim != claim
            or verdict.verdict != Verdict.SUPPORTED
            or verdict.supporting_chunk_id not in claim.source_chunk_ids
        ):
            return None
    return "\n".join(
        f"{claim.text.strip()} [{verdict.supporting_chunk_id}]"
        for claim, verdict in zip(claims, verdicts, strict=True)
    )
