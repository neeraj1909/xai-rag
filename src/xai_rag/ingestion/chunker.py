"""Document chunking — three strategies for splitting text into retrievable chunks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

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
        parent_doc: Small children (128 tok) for precise retrieval, big parents (512 tok) for context.
    """
    if strategy == "fixed":
        return _chunk_fixed(text, chunk_size, chunk_overlap)
    elif strategy == "semantic":
        return _chunk_semantic(text, chunk_size, semantic_threshold)
    elif strategy == "parent_doc":
        return _chunk_parent_doc(text, child_size=128, parent_size=chunk_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use: fixed, semantic, parent_doc")


def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[TextChunk]:
    """Recursive character splitting — tries paragraph, sentence, word, then char boundaries."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    raw_chunks = _recursive_split(text, separators, chunk_size, overlap)
    return [
        TextChunk(content=c, index=i, start_char=0, end_char=0, strategy="fixed")
        for i, c in enumerate(raw_chunks) if c.strip()
    ]


def _recursive_split(text: str, separators: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Split text trying each separator in order of preference."""
    if not text.strip():
        return []

    sep = separators[0]
    rest = separators[1:]

    if sep == "":
        return [text[i:i + chunk_size] for i in range(0, len(text), max(1, chunk_size - overlap))]

    splits = text.split(sep)
    chunks: list[str] = []
    current = ""

    for part in splits:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size and rest:
                chunks.extend(_recursive_split(part, rest, chunk_size, overlap))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    # Add overlap between consecutive chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:] if len(chunks[i - 1]) > overlap else ""
            overlapped.append(tail + chunks[i])
        return overlapped

    return chunks


def _chunk_semantic(text: str, max_size: int, threshold: float) -> list[TextChunk]:
    """Semantic chunking — detect topic boundaries using sentence embedding similarity.

    How it works:
    1. Split text into sentences.
    2. Embed each sentence with a fast model (all-MiniLM-L6-v2).
    3. Compute cosine similarity between consecutive sentences.
    4. Break at points where similarity drops below threshold.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [TextChunk(content=text.strip(), index=0, start_char=0, end_char=len(text), strategy="semantic")]

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        from xai_rag.config import settings
        model = SentenceTransformer(settings.chunking_model)
        embeddings = model.encode(sentences, show_progress_bar=False)

        chunks: list[str] = []
        current_group: list[str] = [sentences[0]]

        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            joined = " ".join(current_group + [sentences[i + 1]])

            if sim < threshold or len(joined) > max_size:
                chunks.append(" ".join(current_group))
                current_group = [sentences[i + 1]]
            else:
                current_group.append(sentences[i + 1])

        if current_group:
            chunks.append(" ".join(current_group))

    except ImportError:
        logger.warning("sentence-transformers unavailable, falling back to fixed chunking")
        return _chunk_fixed(text, max_size, 50)

    return [
        TextChunk(content=c, index=i, start_char=0, end_char=0, strategy="semantic")
        for i, c in enumerate(chunks) if c.strip()
    ]


def _chunk_parent_doc(text: str, child_size: int = 128, parent_size: int = 512) -> list[TextChunk]:
    """Parent-document strategy — small children for retrieval, big parents for LLM context.

    The vector index stores small child chunks (precise matching).
    When a child matches, the FULL parent chunk is sent to the LLM (rich context).
    """
    parents = _chunk_fixed(text, parent_size, 0)
    children: list[TextChunk] = []

    for parent in parents:
        child_texts = _recursive_split(parent.content, [". ", " ", ""], child_size, 0)
        for child_text in child_texts:
            if child_text.strip():
                children.append(TextChunk(
                    content=child_text.strip(),
                    index=len(children),
                    start_char=0,
                    end_char=0,
                    strategy="parent_doc",
                    parent_content=parent.content,
                ))

    logger.info(f"Parent-doc: {len(parents)} parents → {len(children)} children")
    return children


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]
