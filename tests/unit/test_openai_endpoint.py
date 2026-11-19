"""Provider endpoints are selected by project configuration, not ambient SDK settings."""

import openai
import pytest

from xai_rag.config import settings
from xai_rag.evaluation.ragas_eval import _build_scorers
from xai_rag.generation import generator

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("project_url", ["", "https://provider.example/v1"])
@pytest.mark.parametrize("client_path", ["generation", "ragas"])
@pytest.mark.asyncio
async def test_provider_endpoint_ignores_ambient_openai_base_url(
    monkeypatch, project_url, client_path
):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unintended.example/v1")
    monkeypatch.setattr(settings, "openai_api_key", "synthetic-test-key")
    monkeypatch.setattr(settings, "llm_base_url", project_url)
    clients = []
    original_client = openai.AsyncOpenAI

    class RecordingClient(original_client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            clients.append(self)

    monkeypatch.setattr(openai, "AsyncOpenAI", RecordingClient)
    monkeypatch.setattr(generator, "AsyncOpenAI", RecordingClient)

    if client_path == "generation":
        generator.RAGGenerator()
    else:
        await _build_scorers(reference=None)

    assert len(clients) == 1
    assert str(clients[0].base_url) == (project_url or "https://api.openai.com/v1") + "/"
    await clients[0].close()
