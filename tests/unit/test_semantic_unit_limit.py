"""Bound semantic chunking work before loading or running its model."""

import random
import re
import tracemalloc

import numpy as np
import pytest

from xai_rag.config import settings
from xai_rag.ingestion import chunker

pytestmark = pytest.mark.unit


def test_semantic_units_over_limit_do_not_load_model(monkeypatch):
    monkeypatch.setattr(settings, "max_semantic_units", 2, raising=False)
    monkeypatch.setattr(
        chunker, "_load_semantic_model", lambda *args: pytest.fail("model should not load")
    )

    with pytest.raises(ValueError, match="semantic.*unit.*limit"):
        chunker.chunk_text("One. Two. Three.", strategy="semantic", chunk_size=32)


def test_single_long_sentence_is_also_subject_to_unit_limit(monkeypatch):
    monkeypatch.setattr(settings, "max_semantic_units", 2, raising=False)
    monkeypatch.setattr(
        chunker, "_load_semantic_model", lambda *args: pytest.fail("model should not load")
    )

    with pytest.raises(ValueError, match="semantic.*unit.*limit"):
        chunker.chunk_text("a" * 30, strategy="semantic", chunk_size=10)


def test_exact_limit_allows_model_and_preserves_source_offsets(monkeypatch):
    monkeypatch.setattr(settings, "max_semantic_units", 3, raising=False)
    seen = []

    class Model:
        def encode(self, texts, **_kwargs):
            seen.extend(texts)
            return np.ones((len(texts), 2))

    monkeypatch.setattr(chunker, "_load_semantic_model", lambda *args: Model())
    text = "One. Two. Three."
    chunks = chunker.chunk_text(text, strategy="semantic", chunk_size=32)

    assert seen == ["One.", "Two.", "Three."]
    assert all(text[chunk.start_char : chunk.end_char] == chunk.content for chunk in chunks)


def test_sentence_spans_match_original_split_for_mixed_unicode_and_whitespace():
    rng = random.Random(793)
    alphabet = "aबé.!? \t\n\r\u2003\u00a0"
    for _ in range(500):
        text = "".join(rng.choices(alphabet, k=rng.randrange(200)))
        expected = []
        cursor = 0
        for part in re.split(r"(?<=[.!?])\s+", text):
            sentence = part.strip()
            if sentence:
                start = text.find(sentence, cursor)
                expected.append((start, start + len(sentence)))
                cursor = start + len(sentence)
        assert list(chunker._sentence_spans(text)) == expected


def test_semantic_limit_stops_early_without_materializing_sentence_list(monkeypatch):
    monkeypatch.setattr(settings, "max_semantic_units", 2)
    text = "A. " * 100_000
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="semantic.*unit.*limit"):
            chunker.chunk_text(text, strategy="semantic", chunk_size=32)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 3_000_000


@pytest.mark.parametrize("strategy", ["fixed", "parent_doc"])
def test_total_chunk_limit_rejects_oversized_output(monkeypatch, strategy):
    monkeypatch.setattr(settings, "max_chunks_per_document", 2, raising=False)
    with pytest.raises(ValueError, match="chunk.*limit"):
        chunker.chunk_text("abcde", strategy=strategy, chunk_size=1, chunk_overlap=0)


@pytest.mark.parametrize("strategy", ["fixed", "parent_doc"])
def test_total_chunk_limit_accepts_boundary(monkeypatch, strategy):
    monkeypatch.setattr(settings, "max_chunks_per_document", 2, raising=False)
    chunks = chunker.chunk_text("ab", strategy=strategy, chunk_size=1, chunk_overlap=0)
    assert len(chunks) == 2


def test_parent_children_are_capped_even_with_one_parent(monkeypatch):
    monkeypatch.setattr(settings, "max_chunks_per_document", 2)
    with pytest.raises(ValueError, match="chunk.*limit"):
        chunker.chunk_text("a" * 300, strategy="parent_doc", chunk_size=512)


def test_semantic_output_is_capped_after_grouping(monkeypatch):
    monkeypatch.setattr(settings, "max_chunks_per_document", 2)
    monkeypatch.setattr(settings, "max_semantic_units", 3)

    class Model:
        def encode(self, texts, **_kwargs):
            return np.eye(len(texts))

    monkeypatch.setattr(chunker, "_load_semantic_model", lambda *args: Model())
    with pytest.raises(ValueError, match="chunk.*limit"):
        chunker.chunk_text("One. Two. Three.", strategy="semantic", chunk_size=32)
