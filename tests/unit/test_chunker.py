"""Unit contracts for every document chunking strategy."""

import pytest

from xai_rag.ingestion.chunker import chunk_text

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("strategy", ["fixed", "semantic", "parent_doc"])
def test_empty_text_produces_no_chunks(strategy: str) -> None:
    assert chunk_text(" \n\t ", strategy=strategy, chunk_size=32, chunk_overlap=4) == []


def test_fixed_chunks_are_bounded_and_map_to_exact_source_spans() -> None:
    text = (
        "Retrieval needs representative queries. "
        "Ranking needs graded judgments.\n\n"
        "Generation needs grounded citations and refusal tests."
    )

    chunks = chunk_text(text, strategy="fixed", chunk_size=48, chunk_overlap=8)

    assert len(chunks) > 1
    assert all(0 < len(chunk.content) <= 48 for chunk in chunks)
    assert all(text[chunk.start_char : chunk.end_char] == chunk.content for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_long_single_sentence_is_bounded_for_semantic_strategy() -> None:
    text = "x" * 150

    chunks = chunk_text(text, strategy="semantic", chunk_size=40, chunk_overlap=0)

    assert [len(chunk.content) for chunk in chunks] == [40, 40, 40, 30]
    assert all(text[chunk.start_char : chunk.end_char] == chunk.content for chunk in chunks)


def test_parent_chunks_retain_child_and_parent_source_spans() -> None:
    text = "First parent sentence. " * 12

    chunks = chunk_text(text, strategy="parent_doc", chunk_size=96, chunk_overlap=0)

    assert chunks
    assert all(text[chunk.start_char : chunk.end_char] == chunk.content for chunk in chunks)
    assert all(chunk.parent_content for chunk in chunks)
    assert all(chunk.parent_start_char is not None for chunk in chunks)
    assert all(chunk.parent_end_char is not None for chunk in chunks)
    assert all(
        text[chunk.parent_start_char : chunk.parent_end_char] == chunk.parent_content
        for chunk in chunks
    )


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_fixed_chunk_configuration_is_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("content", strategy="fixed", chunk_size=chunk_size, chunk_overlap=overlap)
