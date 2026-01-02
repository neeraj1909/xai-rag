"""RAG answer generator using OpenAI structured output."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from xai_rag.models import Claim, RAGGenerationResult, RankedResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a precise, helpful assistant. Answer the user's question using ONLY \
the provided context chunks. Follow these rules strictly:

1. **Cite your sources**: When you use information from a chunk, cite it \
   inline as [chunk_id]. Every factual statement must have at least one citation.
2. **Stay faithful**: Do not add information not present in the chunks. \
   If the chunks do not contain enough information to answer, say \
   "I don't have enough information to answer this question."
3. **Structured claims**: Break your answer into distinct factual claims. \
   Each claim should reference the chunk(s) it came from.
4. **Be concise**: Provide a clear, well-organized answer without unnecessary \
   preamble.

Context chunks:
{context}
"""

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The full answer with inline [chunk_id] citations.",
                },
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "A single factual claim from the answer.",
                            },
                            "source_chunk_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Chunk IDs supporting this claim.",
                            },
                        },
                        "required": ["text", "source_chunk_ids"],
                        "additionalProperties": False,
                    },
                    "description": "Decomposed factual claims with source attribution.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All unique chunk IDs referenced in the answer.",
                },
            },
            "required": ["answer", "claims", "sources"],
            "additionalProperties": False,
        },
    },
}


def _format_context(chunks: list[RankedResult]) -> str:
    """Format chunks into a numbered context block for the LLM."""
    parts: list[str] = []
    for chunk in chunks:
        parts.append(f"[{chunk.id}]\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


class RAGGenerator:
    """Generates cited answers from retrieved chunks using an OpenAI model.

    Uses structured output (``response_format``) to enforce the response
    schema so that claims and source attribution are always machine-parseable.
    """

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._client = client or AsyncOpenAI()
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True,
    )
    async def generate(
        self,
        query: str,
        chunks: list[RankedResult],
    ) -> RAGGenerationResult:
        """Generate a cited answer for *query* using *chunks* as evidence.

        Parameters
        ----------
        query:
            The user's question.
        chunks:
            Reranked evidence chunks.

        Returns
        -------
        RAGGenerationResult
            Structured answer with inline citations, decomposed claims,
            and a source list.
        """
        if not chunks:
            return RAGGenerationResult(
                answer="I don't have enough information to answer this question.",
                claims=[],
                sources=[],
            )

        context = _format_context(chunks)
        system_message = _SYSTEM_PROMPT.format(context=context)

        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format=_RESPONSE_SCHEMA,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ],
        )

        raw = response.choices[0].message.content
        if raw is None:
            logger.warning("LLM returned empty content; treating as no-answer.")
            return RAGGenerationResult(
                answer="I don't have enough information to answer this question.",
                claims=[],
                sources=[],
            )

        parsed = json.loads(raw)

        claims = [
            Claim(
                text=c["text"],
                source_chunk_ids=c.get("source_chunk_ids", []),
            )
            for c in parsed.get("claims", [])
        ]

        result = RAGGenerationResult(
            answer=parsed["answer"],
            claims=claims,
            sources=parsed.get("sources", []),
        )

        logger.info(
            "Generated answer: %d chars, %d claims, %d sources",
            len(result.answer),
            len(result.claims),
            len(result.sources),
        )
        return result
