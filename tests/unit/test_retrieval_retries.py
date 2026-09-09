"""Retry contracts for retrieval dependencies."""

import pytest

from xai_rag.retrieval.bm25_search import bm25_search
from xai_rag.retrieval.vector_search import vector_search

pytestmark = pytest.mark.unit


def test_vector_search_does_not_retry_permanent_input_errors() -> None:
    class BrokenCollection:
        calls = 0

        def query(self, **kwargs):
            self.calls += 1
            raise ValueError("embedding dimension is invalid")

    collection = BrokenCollection()

    with pytest.raises(ValueError, match="dimension"):
        vector_search(collection, [0.1], 1)

    assert collection.calls == 1


@pytest.mark.asyncio
async def test_bm25_search_does_not_retry_permanent_input_errors() -> None:
    class BrokenClient:
        calls = 0

        async def search(self, **kwargs):
            self.calls += 1
            raise ValueError("query is invalid")

    client = BrokenClient()

    with pytest.raises(ValueError, match="invalid"):
        await bm25_search(client, "query", "index", 1)

    assert client.calls == 1
