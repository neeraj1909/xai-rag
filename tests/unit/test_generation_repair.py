"""Controlled contracts for bounded repairs of invalid generated citations."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, AsyncOpenAI, InternalServerError

from xai_rag.generation import generator as generator_module
from xai_rag.generation.generator import GenerationContractError, RAGGenerator
from xai_rag.models import RankedResult

pytestmark = pytest.mark.unit


def _response(payload, prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class SequenceClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


_BAD = {
    "answer": "Submit a receipt; deadline unspecified.",
    "claims": [
        {"text": "Submit a receipt", "source_chunk_ids": ["c"]},
        {"text": "Deadline unspecified", "source_chunk_ids": []},
    ],
    "sources": ["c"],
}
_GOOD = {
    "answer": "Submit a receipt [c]; the deadline is not specified.",
    "claims": [{"text": "Submit a receipt", "source_chunk_ids": ["c"]}],
    "sources": ["c"],
}
_UNKNOWN = {
    "answer": "A receipt [forged]",
    "claims": [{"text": "A receipt", "source_chunk_ids": ["forged"]}],
    "sources": ["forged"],
}
_CHUNKS = [RankedResult(id="c", content="Submit a receipt with the request.")]


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(generator_module, "wait_exponential", lambda **_: lambda state: 0)


@pytest.mark.asyncio
async def test_one_contract_repair_revalidates_and_sums_usage():
    client = SequenceClient([_response(_BAD), _response(_GOOD, 20, 7)])
    outcome = await RAGGenerator(client=client).generate_with_evidence("What and when?", _CHUNKS)
    assert outcome.result.answer == _GOOD["answer"]
    assert (outcome.result.input_tokens, outcome.result.output_tokens) == (30, 12)
    assert len(client.calls) == 2
    assert len(client.calls[1]["messages"]) == 2
    assert "Deadline unspecified" not in str(client.calls[1]["messages"])
    assert "partial" in client.calls[0]["messages"][0]["content"].lower()


@pytest.mark.asyncio
async def test_two_invalid_generations_fail_closed_without_third_call():
    client = SequenceClient([_response(_BAD), _response(_BAD)])
    with pytest.raises(GenerationContractError):
        await RAGGenerator(client=client).generate("What and when?", _CHUNKS)
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_unknown_ids_are_not_silently_accepted_after_repair():
    client = SequenceClient([_response(_UNKNOWN), _response(_UNKNOWN)])
    with pytest.raises(GenerationContractError, match="unknown chunk IDs"):
        await RAGGenerator(client=client).generate("Question?", _CHUNKS)
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_attribute_style_citations_are_rejected_and_prompt_gives_exact_format():
    bad = {**_GOOD, "answer": 'Submit a receipt [chunk_id="c"]; deadline unspecified.'}
    client = SequenceClient([_response(bad), _response(bad)])
    with pytest.raises(GenerationContractError, match="unknown chunk IDs"):
        await RAGGenerator(client=client).generate("What and when?", _CHUNKS)
    for call in client.calls:
        prompt = call["messages"][0]["content"]
        assert "[abc123]" in prompt
        assert '[chunk_id="abc123"]' in prompt


@pytest.mark.asyncio
async def test_transport_retry_budget_is_shared_with_contract_repair():
    request = httpx.Request("POST", "https://example.invalid")
    client = SequenceClient(
        [_response(_BAD), APIConnectionError(request=request), APIConnectionError(request=request)]
    )
    with pytest.raises(APIConnectionError):
        await RAGGenerator(client=client).generate("Question?", _CHUNKS)
    assert len(client.calls) == 3


@pytest.mark.asyncio
async def test_transport_failures_can_exhaust_budget_before_contract_repair():
    request = httpx.Request("POST", "https://example.invalid")
    client = SequenceClient(
        [APIConnectionError(request=request), APIConnectionError(request=request), _response(_BAD)]
    )
    with pytest.raises(GenerationContractError):
        await RAGGenerator(client=client).generate("Question?", _CHUNKS)
    assert len(client.calls) == 3


@pytest.mark.asyncio
async def test_unknown_usage_is_not_reported_as_complete_total():
    first = _response(_BAD)
    first.usage = None
    client = SequenceClient([first, _response(_GOOD)])
    result = await RAGGenerator(client=client).generate("Question?", _CHUNKS)
    assert result.input_tokens is None
    assert result.output_tokens is None


@pytest.mark.asyncio
async def test_cancellation_does_not_trigger_repair_or_retry():
    client = SequenceClient([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await RAGGenerator(client=client).generate("Question?", _CHUNKS)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_valid_first_generation_is_not_repaired():
    client = SequenceClient([_response(_GOOD)])
    result = await RAGGenerator(client=client).generate("Question?", _CHUNKS)
    assert result.sources == ["c"]
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_injected_openai_client_does_not_multiply_http_retries():
    requests = []

    def server_error(request):
        requests.append(request)
        return httpx.Response(500, json={"error": {"message": "server error"}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server_error))
    client = AsyncOpenAI(api_key="test", max_retries=1, http_client=http_client)
    try:
        with pytest.raises(InternalServerError):
            await RAGGenerator(client=client).generate("Question?", _CHUNKS)
        assert len(requests) == 3
        assert client.max_retries == 1  # Do not mutate a shared injected client.
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_injected_openai_client_shares_http_budget_after_invalid_answer():
    requests = []

    def server(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "fixture",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "fixture",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(_BAD)},
                        }
                    ],
                },
            )
        return httpx.Response(500, json={"error": {"message": "server error"}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    client = AsyncOpenAI(api_key="test", max_retries=2, http_client=http_client)
    try:
        with pytest.raises(InternalServerError):
            await RAGGenerator(client=client).generate("Question?", _CHUNKS)
        assert len(requests) == 3
        assert client.max_retries == 2
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    ["The deadline is tomorrow.", "I cannot answer from the available text."],
)
async def test_uncited_zero_claim_answer_fails_closed(answer):
    payload = {"answer": answer, "claims": [], "sources": []}
    client = SequenceClient([_response(payload), _response(payload)])
    with pytest.raises(GenerationContractError):
        await RAGGenerator(client=client).generate("Question?", _CHUNKS)
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_no_claim_refusal_cannot_carry_sources():
    payload = {
        "answer": "I don't have enough information to answer this question.",
        "claims": [],
        "sources": ["c"],
    }
    client = SequenceClient([_response(payload), _response(payload)])
    with pytest.raises(GenerationContractError):
        await RAGGenerator(client=client).generate("Question?", _CHUNKS)
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_canonical_refusal_with_no_claims_is_allowed():
    answer = "I don't have enough information to answer this question."
    client = SequenceClient([_response({"answer": answer, "claims": [], "sources": []})])
    result = await RAGGenerator(client=client).generate("Question?", _CHUNKS)
    assert result.answer == answer
    assert result.claims == []
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_grounding_repair_uses_same_evidence_and_only_one_model_attempt():
    client = SequenceClient([_response(_GOOD), _response(_UNKNOWN), _response(_GOOD)])
    generator = RAGGenerator(client=client)
    first = await generator.generate_with_evidence("Question?", _CHUNKS)
    with pytest.raises(GenerationContractError):
        await generator.repair_with_evidence("Question?", _CHUNKS)
    assert len(client.calls) == 2
    assert client.calls[0]["messages"][1] == client.calls[1]["messages"][1]
    assert (
        "Your previous response could not be verified." in client.calls[1]["messages"][0]["content"]
    )
    assert first.evidence


@pytest.mark.asyncio
async def test_grounding_repair_cancellation_propagates_without_retry():
    client = SequenceClient([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await RAGGenerator(client=client).repair_with_evidence("Question?", _CHUNKS)
    assert len(client.calls) == 1
