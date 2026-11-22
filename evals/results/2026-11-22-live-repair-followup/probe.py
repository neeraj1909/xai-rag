"""Controlled original rejection followed by actual service retry, NLI and OpenAI judge."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

from xai_rag.config import settings
from xai_rag.explainability.faithfulness import FaithfulnessChecker
from xai_rag.generation.generator import GenerationOutcome, RAGGenerator, _format_context
from xai_rag.generation.grounding_judge import GroundingJudge
from xai_rag.models import Claim, FusedResult, QueryRequest, RAGGenerationResult
from xai_rag.service import RAGService

SOURCE = Path("evals/results/2026-11-22-docker-e2e/nli-probe.txt")
RAW = SOURCE.read_bytes()
PREMISE = RAW.decode().split("Premise (exact):\n", 1)[1].strip()
CHUNK_ID = "83f1028362e75b6f85a2b65bc7326518b6e71b7a49deda9d7b36eed91ce11e21"
CLAIM = (
    "RAG couples a retrieval system with a language model so the model generates answers "
    "grounded in fetched evidence rather than relying solely on parametric memory."
)
QUERY = "What does the document explain about retrieval-augmented generation?"
events = []


class Generator:
    def __init__(self):
        self.actual = RAGGenerator()

    async def generate_with_evidence(self, query, chunks):
        _, visible = _format_context(chunks, settings.llm_context_max_chars)
        events.append({"stage": "initial_control", "claim": CLAIM})
        return GenerationOutcome(
            RAGGenerationResult(
                answer=f"{CLAIM} [{CHUNK_ID}]",
                claims=[Claim(text=CLAIM, source_chunk_ids=[CHUNK_ID])],
                sources=[CHUNK_ID],
            ),
            tuple(visible),
        )

    async def repair_with_evidence(self, query, chunks):
        outcome = await self.actual.repair_with_evidence(query, chunks)
        events.append(
            {
                "stage": "real_openai_repair",
                "result": outcome.result.model_dump(mode="json"),
                "visible_evidence_sha256": hashlib.sha256(
                    outcome.evidence[0].content.encode()
                ).hexdigest(),
            }
        )
        return outcome


class Checker:
    async def check_claims(self, claims, chunks):
        verdicts = await FaithfulnessChecker().check_claims(claims, chunks)
        events.append(
            {
                "stage": "real_nli",
                "verdicts": [x.model_dump(mode="json") for x in verdicts],
                "evidence_sha256": hashlib.sha256(chunks[0].content.encode()).hexdigest(),
            }
        )
        return verdicts


class Judge:
    async def check_claims(self, claims, evidence, nli, *, question):
        decision = await GroundingJudge().check_claims(claims, evidence, nli, question=question)
        events.append(
            {
                "stage": "real_openai_judge",
                "supported": decision.supported,
                "all_parts_addressed": decision.all_parts_addressed,
            }
        )
        return decision


class Reranker:
    async def rerank(self, query, results, top_k):
        return [x.to_ranked(reranker_score=1, reranker_rank=1) for x in results]


async def embed(query):
    return [0.0]


def vector(collection, embedding, k):
    return []


async def bm25(client, query, index, k):
    return []


def fusion(results, k):
    return [FusedResult(id=CHUNK_ID, content=PREMISE)]


async def main():
    service = RAGService(
        collection=object(),
        es_client=object(),
        embed_query_fn=embed,
        vector_search_fn=vector,
        bm25_search_fn=bm25,
        fusion_fn=fusion,
        reranker_factory=Reranker,
        generator_factory=Generator,
        faithfulness_factory=Checker,
        grounding_judge_factory=Judge,
    )
    started = time.monotonic()
    response = await service.query(
        QueryRequest(query=QUERY, top_k=1, include_explanations=False, include_faithfulness=True)
    )
    output = {
        "metadata": {
            "model": settings.llm_model,
            "judge_model": settings.grounding_judge_model,
            "nli_model": settings.nli_model,
            "nli_revision": settings.nli_model_revision,
            "source_sha256": hashlib.sha256(RAW).hexdigest(),
            "premise_sha256": hashlib.sha256(PREMISE.encode()).hexdigest(),
            "scope": "controlled initial claim and retrieval; actual service retry, "
            "NLI, OpenAI generation and judge",
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
        "events": events,
        "response": response.model_dump(mode="json"),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
