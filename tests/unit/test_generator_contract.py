"""Unit contracts for generated citation provenance."""

import json
from types import SimpleNamespace

import pytest

from xai_rag.generation.generator import GenerationContractError, RAGGenerator
from xai_rag.models import RankedResult

pytestmark = pytest.mark.unit


class FakeCompletions:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.payload)))]
        )


class FakeClient:
    def __init__(self, payload: dict) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(payload))


def _chunk(chunk_id: str) -> RankedResult:
    return RankedResult(id=chunk_id, content="grounded fact", rrf_score=0.1, rrf_rank=1)


@pytest.mark.asyncio
async def test_generator_rejects_unknown_structured_and_inline_citations() -> None:
    payload = {
        "answer": "An invented fact [missing].",
        "claims": [{"text": "An invented fact", "source_chunk_ids": ["missing"]}],
        "sources": ["missing"],
    }
    generator = RAGGenerator(client=FakeClient(payload))

    with pytest.raises(GenerationContractError, match="unknown chunk IDs"):
        await generator.generate("question", [_chunk("known")])


@pytest.mark.asyncio
async def test_generator_accepts_only_retrieved_citations_and_deduplicates_sources() -> None:
    payload = {
        "answer": "A grounded fact [known].",
        "claims": [{"text": "A grounded fact", "source_chunk_ids": ["known"]}],
        "sources": ["known", "known"],
    }
    generator = RAGGenerator(client=FakeClient(payload))

    result = await generator.generate("question", [_chunk("known")])

    assert result.sources == ["known"]


@pytest.mark.asyncio
async def test_generator_rejects_claims_without_inline_and_structured_sources() -> None:
    payload = {
        "answer": "An uncited claim.",
        "claims": [{"text": "An uncited claim", "source_chunk_ids": []}],
        "sources": [],
    }
    generator = RAGGenerator(client=FakeClient(payload))

    with pytest.raises(GenerationContractError, match="no source chunk IDs"):
        await generator.generate("question", [_chunk("known")])
