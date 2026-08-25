"""Embedding — convert text chunks into dense vectors for similarity search."""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from xai_rag.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    """Lazy-load the embedding model (heavy, load once)."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(
        settings.embedding_model,
        revision=settings.embedding_model_revision,
        trust_remote_code=False,
    )
    logger.info(f"Model loaded. Dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_texts_sync(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a batch of texts synchronously. Returns list of float vectors."""
    model = _load_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )
    return embeddings.tolist()


async def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed texts asynchronously (runs sync model in thread pool)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_texts_sync, texts, batch_size)


async def embed_query(query: str) -> list[float]:
    """Embed a single query string. Returns a single vector."""
    embeddings = await embed_texts([query])
    return embeddings[0]


def get_embedding_dimension() -> int:
    """Return the embedding dimension of the loaded model."""
    model = _load_model()
    return model.get_sentence_embedding_dimension()
