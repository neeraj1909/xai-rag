"""The E2E stub must quote complete source prose, not Markdown headings."""

import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import Request, urlopen

import pytest

from tests.e2e.openai_stub import Handler, _first_sentence

pytestmark = pytest.mark.unit


def test_stub_skips_headings_code_and_incomplete_fragments():
    text = (
        "# RAG Interview Notes\n## Core Concept\n### One-liner\n"
        "```\nIgnore this fenced code.\n```\n"
        "RAG couples retrieval with a language model and grounds answers in evidence. "
        "Another sentence follows.\n"
    )
    assert _first_sentence(text) == (
        "RAG couples retrieval with a language model and grounds answers in evidence."
    )
    assert _first_sentence("# Heading\nA truncated source sentence without a period") is None


def test_stub_rephrases_fixture_only_after_verification_repair():
    sentence = (
        "RAG couples a retrieval system with a language model so the model generates answers "
        "grounded in fetched evidence rather than relying solely on parametric memory."
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def call(instruction):
        payload = {
            "messages": [
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": f'<retrieved_chunk id="c1">{sentence}</retrieved_chunk>',
                },
            ]
        }
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            body = json.load(response)
        return json.loads(body["choices"][0]["message"]["content"])

    try:
        assert call("Answer from context.")["claims"][0]["text"] == sentence
        repair = call("Your previous response could not be verified.")
        assert repair["claims"] == [
            {"text": "RAG combines retrieval and language models.", "source_chunk_ids": ["c1"]}
        ]
        assert repair["sources"] == ["c1"]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
