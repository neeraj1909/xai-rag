"""Recovery checks actual content and metadata, not merely claimed hashes."""

import pytest

from xai_rag.ingestion import store

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_expected_source_rejects_stale_content_and_metadata():
    metadata = {"source_file": "one.txt", "content_sha256": "claimed", "offset": 1}
    expected = {"id": store.record_fingerprint("intended", metadata)}

    class Chroma:
        content = "intended"
        data = metadata.copy()

        def get(self, **_kwargs):
            return {"ids": ["id"], "documents": [self.content], "metadatas": [self.data]}

    class Elastic:
        content = "intended"
        data = metadata.copy()

        async def search(self, **_kwargs):
            return {
                "timed_out": False,
                "_shards": {"failed": 0},
                "hits": {
                    "hits": [
                        {
                            "_id": "id",
                            "_source": {
                                "chunk_id": "id",
                                "source_file": "one.txt",
                                "content": self.content,
                                "metadata": self.data,
                            },
                        }
                    ]
                },
            }

    chroma, elastic = Chroma(), Elastic()
    assert await store.source_matches_expected(
        chroma, elastic, source_file="one.txt", expected=expected
    )
    chroma.content = "corrupt but same claimed hash"
    assert not await store.source_matches_expected(
        chroma, elastic, source_file="one.txt", expected=expected
    )
    chroma.content = "intended"
    elastic.content = "corrupt but same claimed hash"
    assert not await store.source_matches_expected(
        chroma, elastic, source_file="one.txt", expected=expected
    )
    elastic.content = "intended"
    elastic.data = {**metadata, "offset": 2}
    assert not await store.source_matches_expected(
        chroma, elastic, source_file="one.txt", expected=expected
    )
    assert not await store.source_matches_expected(
        chroma, elastic, source_file="one.txt", expected={"new": "fresh"}
    )


@pytest.mark.asyncio
async def test_parity_hashes_stored_documents_instead_of_trusting_metadata():
    import hashlib

    digest = hashlib.sha256(b"intended").hexdigest()
    metadata = {"source_file": "one.txt", "content_sha256": digest}

    class Chroma:
        content = "intended"

        def get(self, **_kwargs):
            return {"ids": ["id"], "documents": [self.content], "metadatas": [metadata]}

    class Elastic:
        content = "intended"

        async def search(self, **_kwargs):
            return {
                "timed_out": False,
                "_shards": {"failed": 0},
                "hits": {
                    "hits": [
                        {
                            "_id": "id",
                            "_source": {
                                "chunk_id": "id",
                                "content": self.content,
                                "metadata": metadata,
                            },
                        }
                    ]
                },
            }

    chroma, elastic = Chroma(), Elastic()
    assert (await store.check_index_parity(chroma, elastic)).consistent
    chroma.content = "corrupt"
    assert not (await store.check_index_parity(chroma, elastic)).consistent
    chroma.content = "intended"
    elastic.content = "corrupt"
    assert not (await store.check_index_parity(chroma, elastic)).consistent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"timed_out": True, "_shards": {"failed": 0}, "hits": {"hits": []}},
        {"timed_out": False, "_shards": {"failed": 1}, "hits": {"hits": []}},
        {"timed_out": False, "hits": {"hits": []}},
        {"timed_out": False, "_shards": {"failed": 0}, "hits": {}},
    ],
)
async def test_incomplete_elasticsearch_search_cannot_pass_verification(response):
    class Chroma:
        def get(self, **_kwargs):
            return {"ids": [], "documents": [], "metadatas": []}

    class Elastic:
        async def search(self, **kwargs):
            assert kwargs["allow_partial_search_results"] is False
            return response

    with pytest.raises(store.StorageWriteError, match="incomplete"):
        await store.source_matches_expected(Chroma(), Elastic(), source_file="one.txt", expected={})
    with pytest.raises(store.StorageWriteError, match="incomplete"):
        await store.check_index_parity(Chroma(), Elastic())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hits",
    [
        [{"_id": "other", "_source": {"chunk_id": "id"}}],
        [
            {"_id": "id", "_source": {"chunk_id": "id"}},
            {"_id": "id", "_source": {"chunk_id": "id"}},
        ],
    ],
)
async def test_parity_rejects_inconsistent_or_duplicate_es_ids(hits):
    class Chroma:
        def get(self, **_kwargs):
            return {"ids": [], "documents": [], "metadatas": []}

    class Elastic:
        async def search(self, **_kwargs):
            return {"timed_out": False, "_shards": {"failed": 0}, "hits": {"hits": hits}}

    with pytest.raises(store.StorageWriteError, match="invalid chunk identities"):
        await store.check_index_parity(Chroma(), Elastic())
