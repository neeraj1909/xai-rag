"""The E2E stub must quote complete source prose, not Markdown headings."""

import pytest

from tests.e2e.openai_stub import _first_sentence

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
