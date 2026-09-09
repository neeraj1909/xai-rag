"""Unit contracts for embedding-model compatibility."""

import pytest

from xai_rag.ingestion import embedder

pytestmark = pytest.mark.unit


def test_embedding_dimension_uses_current_sentence_transformers_api(monkeypatch) -> None:
    class FakeModel:
        def get_embedding_dimension(self) -> int:
            return 384

    monkeypatch.setattr(embedder, "_load_model", FakeModel)

    assert embedder.get_embedding_dimension() == 384
