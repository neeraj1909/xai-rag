"""Document chunking — three strategies for splitting text into retrievable chunks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with its position in the source document."""

    content: str
    index: int
    start_char: int
    end_char: int
    strategy: str
    parent_content: str | None = None
    parent_start_char: int | None = None
    parent_end_char: int | None = None


def chunk_text(
    text: str,
    strategy: str = "fixed",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    semantic_threshold: float = 0.75,
) -> list[TextChunk]:
    """Split text into chunks using the specified strategy.

    Strategies:
        fixed: Recursive character splitting with overlap. Tries paragraph → sentence → word → char.
        semantic: Groups sentences until cosine similarity drops below threshold (topic change).
        parent_doc: Small children (128 tok) for precise retrieval, big parents
        (512 tok) for context.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if strategy == "fixed" and chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not 0.0 <= semantic_threshold <= 1.0:
        raise ValueError("semantic_threshold must be between 0 and 1")
    if not text.strip():
        return []

    if strategy == "fixed":
        return _chunk_fixed(text, chunk_size, chunk_overlap)
    elif strategy == "semantic":
        return _chunk_semantic(text, chunk_size, semantic_threshold)
    elif strategy == "parent_doc":
        return _chunk_parent_doc(text, child_size=128, parent_size=chunk_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use: fixed, semantic, parent_doc")


def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[TextChunk]:
    """Boundary-aware character splitting with exact source offsets."""
    spans = _split_spans(text, chunk_size, overlap)
    return [
        TextChunk(
            content=text[start:end],
            index=index,
            start_char=start,
            end_char=end,
            strategy="fixed",
        )
        for index, (start, end) in enumerate(spans)
    ]


def _split_spans(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Return bounded, overlapping spans while preferring natural boundaries."""
    if not text.strip():
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        end = hard_end

        if hard_end < text_length:
            minimum_boundary = start + max(1, chunk_size // 2)
            for separator in ("\n\n", "\n", ". ", " "):
                boundary = text.rfind(separator, minimum_boundary, hard_end)
                if boundary >= minimum_boundary:
                    end = boundary + len(separator)
                    break

        if end <= start:
            end = hard_end

        if text[start:end].strip():
            spans.append((start, end))

        if end >= text_length:
            break
        start = max(end - overlap, start + 1)

    return spans


def _recursive_split(text: str, separators: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Compatibility wrapper returning the contents of bounded source spans."""
    del separators
    return [text[start:end] for start, end in _split_spans(text, chunk_size, overlap)]


@lru_cache(maxsize=1)
def _load_semantic_model(model_name: str, revision: str):
    """Load the semantic chunking model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, revision=revision, trust_remote_code=False)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Map sentence splitting results back to exact, ordered source spans."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in _split_sentences(text):
        start = text.find(sentence, cursor)
        if start < 0:
            continue
        end = start + len(sentence)
        spans.append((start, end))
        cursor = end
    return spans


def _chunk_semantic(text: str, max_size: int, threshold: float) -> list[TextChunk]:
    """Semantic chunking — detect topic boundaries using sentence embedding similarity.

    How it works:
    1. Split text into sentences.
    2. Embed each sentence with a fast model (all-MiniLM-L6-v2).
    3. Compute cosine similarity between consecutive sentences.
    4. Break at points where similarity drops below threshold.
    """
    sentence_spans = _sentence_spans(text)
    if not sentence_spans:
        return []

    units: list[tuple[int, int]] = []
    for sentence_start, sentence_end in sentence_spans:
        if sentence_end - sentence_start <= max_size:
            units.append((sentence_start, sentence_end))
            continue
        units.extend(
            (sentence_start + start, sentence_start + end)
            for start, end in _split_spans(text[sentence_start:sentence_end], max_size, 0)
        )

    if len(units) == 1:
        start, end = units[0]
        return [TextChunk(text[start:end], 0, start, end, "semantic")]

    try:
        import numpy as np

        from xai_rag.config import settings

        model = _load_semantic_model(
            settings.chunking_model,
            settings.chunking_model_revision,
        )
        unit_texts = [text[start:end] for start, end in units]
        embeddings = model.encode(unit_texts, show_progress_bar=False)

        grouped_spans: list[tuple[int, int]] = []
        group_start, group_end = units[0]

        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            next_start, next_end = units[i + 1]

            if sim < threshold or next_end - group_start > max_size:
                grouped_spans.append((group_start, group_end))
                group_start, group_end = next_start, next_end
            else:
                group_end = next_end

        grouped_spans.append((group_start, group_end))

    except ImportError:
        logger.warning("sentence-transformers unavailable, falling back to fixed chunking")
        fallback = _chunk_fixed(text, max_size, min(50, max_size - 1))
        for chunk in fallback:
            chunk.strategy = "semantic"
        return fallback

    return [
        TextChunk(text[start:end], index, start, end, "semantic")
        for index, (start, end) in enumerate(grouped_spans)
        if text[start:end].strip()
    ]


def _chunk_parent_doc(text: str, child_size: int = 128, parent_size: int = 512) -> list[TextChunk]:
    """Parent-document strategy — small children for retrieval, big parents for LLM context.

    The vector index stores small child chunks (precise matching).
    When a child matches, the FULL parent chunk is sent to the LLM (rich context).
    """
    parents = _chunk_fixed(text, parent_size, 0)
    children: list[TextChunk] = []

    for parent in parents:
        child_spans = _split_spans(parent.content, child_size, 0)
        for child_start, child_end in child_spans:
            child_text = parent.content[child_start:child_end]
            if child_text.strip():
                source_start = parent.start_char + child_start
                source_end = parent.start_char + child_end
                children.append(
                    TextChunk(
                        content=child_text,
                        index=len(children),
                        start_char=source_start,
                        end_char=source_end,
                        strategy="parent_doc",
                        parent_content=parent.content,
                        parent_start_char=parent.start_char,
                        parent_end_char=parent.end_char,
                    )
                )

    logger.info("Parent-doc: %d parents → %d children", len(parents), len(children))
    return children


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]
