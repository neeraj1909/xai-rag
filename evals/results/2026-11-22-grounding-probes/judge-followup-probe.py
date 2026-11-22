"""Evaluate current judge on authored conditional and conflicting-source cases."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from openai import AsyncOpenAI as RealClient

from xai_rag.config import settings
from xai_rag.generation import grounding_judge
from xai_rag.models import Claim, ClaimVerdict, RankedResult, Verdict

raw = Path('evals/results/2026-11-22-docker-e2e/nli-probe.txt').read_text()
rag_premise = raw.split('Premise (exact):\n', 1)[1]
rag_claim = 'RAG couples a retrieval system with a language model so the model generates answers grounded in fetched evidence rather than relying solely on parametric memory.'
cases = [
    ('original', 'What does the document explain about retrieval-augmented generation?', rag_claim, [rag_premise], ['one'], True),
    ('conditional_fee', 'What is the fee?', 'The fee is 10 dollars.', ['If the discount is approved, the fee is 10 dollars. The discount is not approved.'], ['one'], False),
    ('rag_conflict', 'What does the document explain about retrieval-augmented generation?', rag_claim, [rag_premise, 'Contrary to the first document, RAG never uses a retrieval system.'], ['one'], False),
    ('fee_conflict_cite_support', 'What is the fee?', 'The fee is 10 dollars.', ['The fee is 10 dollars.', 'The fee is 20 dollars, not 10 dollars.'], ['one'], False),
    ('fee_conflict_cite_both', 'What is the fee?', 'The fee is 10 dollars.', ['The fee is 10 dollars.', 'The fee is 20 dollars, not 10 dollars.'], ['one', 'two'], False),
]

class CaptureClient:
    last = None
    def __init__(self, **kw):
        self.client = RealClient(**kw)
        self.chat = SimpleNamespace(completions=self)
    async def __aenter__(self):
        await self.client.__aenter__()
        return self
    async def __aexit__(self, *args):
        return await self.client.__aexit__(*args)
    async def create(self, **kw):
        response = await self.client.chat.completions.create(**kw)
        self.__class__.last = response
        return response

grounding_judge.AsyncOpenAI = CaptureClient
settings.grounding_judge_model = 'gpt-4o-mini'
settings.grounding_judge_provider = 'openai'

async def main():
    for name, question, text, chunks, cited, expected in cases:
        claim = Claim(text=text, source_chunk_ids=cited)
        evidence = [RankedResult(id='one' if i == 0 else 'two', content=value) for i, value in enumerate(chunks)]
        # Invoke the existing judge in isolation. This is not an NLI model result.
        hypothetical = ClaimVerdict(claim=claim, verdict=Verdict.SUPPORTED, supporting_chunk_id='one')
        try:
            decision = await grounding_judge.GroundingJudge().check_claims([claim], evidence, [hypothetical], question=question)
            response = CaptureClient.last
            print(json.dumps({'case':name, 'expected_release':expected, 'judge_release':decision.supported and decision.all_parts_addressed, 'judge_response':json.loads(response.choices[0].message.content), 'model':response.model, 'input_tokens':response.usage.prompt_tokens, 'output_tokens':response.usage.completion_tokens}), flush=True)
        except Exception as error:
            print(json.dumps({'case':name, 'error':type(error).__name__, 'status_code':getattr(error, 'status_code', None)}), flush=True)

asyncio.run(main())
