"""Unit contracts for repeatable dual-index writes."""

import pytest

from xai_rag.ingestion.chunker import TextChunk
from xai_rag.ingestion.parser import source_identity
from xai_rag.ingestion.store import (
    IndexCompatibilityError,
    IndexNotReadyError,
    StorageWriteError,
    compare_index_records,
    ensure_es_index,
    get_index_fingerprint,
    require_compatible_fingerprint,
    store_chunks_chromadb,
)

pytestmark = pytest.mark.unit


class FakeCollection:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get(self, **kwargs):
        return {"ids": []}

    def upsert(self, **kwargs):
        self.calls.append(kwargs)

    def delete(self, **kwargs):
        raise AssertionError("nothing should be deleted")


def test_chroma_ids_are_stable_and_provenance_is_complete() -> None:
    chunks = [
        TextChunk(
            content="evidence",
            index=0,
            start_char=12,
            end_char=20,
            strategy="fixed",
            parent_content="parent evidence",
            parent_start_char=0,
            parent_end_char=30,
        )
    ]
    collection = FakeCollection()

    first = store_chunks_chromadb(collection, chunks, [[0.1, 0.2]], "guide.md")
    second = store_chunks_chromadb(collection, chunks, [[0.1, 0.2]], "guide.md")

    assert first == second
    assert collection.calls[0]["metadatas"][0]["start_char"] == 12
    assert collection.calls[0]["metadatas"][0]["end_char"] == 20
    assert collection.calls[0]["metadatas"][0]["parent_content"] == "parent evidence"


@pytest.mark.asyncio
async def test_elasticsearch_bulk_item_failure_is_raised() -> None:
    from xai_rag.ingestion.store import store_chunks_elasticsearch

    class BrokenElasticsearch:
        class Indices:
            async def get(self, index):
                return {
                    index: {"mappings": {"_meta": {"index_fingerprint": get_index_fingerprint()}}}
                }

        class Cluster:
            async def health(self, **kwargs):
                return {"status": "green", "timed_out": False}

        indices = Indices()
        cluster = Cluster()

        async def bulk(self, **kwargs):
            return {
                "errors": True,
                "items": [{"index": {"_id": "chunk", "status": 400, "error": {"type": "mapper"}}}],
            }

    chunk = TextChunk("evidence", 0, 0, 8, "fixed")

    with pytest.raises(StorageWriteError, match="mapper"):
        await store_chunks_elasticsearch(BrokenElasticsearch(), [chunk], ["chunk"], "guide.md")


@pytest.mark.asyncio
async def test_existing_elasticsearch_index_must_have_an_active_primary() -> None:
    class RedElasticsearch:
        class Indices:
            async def get(self, index):
                return {
                    index: {"mappings": {"_meta": {"index_fingerprint": get_index_fingerprint()}}}
                }

        class Cluster:
            async def health(self, **kwargs):
                return {"status": "red", "timed_out": True}

        indices = Indices()
        cluster = Cluster()

    with pytest.raises(IndexNotReadyError, match="active primary"):
        await ensure_es_index(RedElasticsearch(), "documents")


def test_index_parity_reports_missing_and_mismatched_records() -> None:
    report = compare_index_records(
        {"same": "a", "wrong": "old", "chroma-only": "c"},
        {"same": "a", "wrong": "new", "elastic-only": "e"},
    )

    assert report.consistent is False
    assert report.missing_in_chroma == ("elastic-only",)
    assert report.missing_in_elasticsearch == ("chroma-only",)
    assert report.content_mismatches == ("wrong",)
    assert report.to_dict()["consistent"] is False


def test_index_fingerprint_fails_closed_on_missing_or_mismatch() -> None:
    require_compatible_fingerprint(get_index_fingerprint(), store_name="test")

    with pytest.raises(IndexCompatibilityError, match="Reindex required"):
        require_compatible_fingerprint(None, store_name="test")

    with pytest.raises(IndexCompatibilityError, match="Reindex required"):
        require_compatible_fingerprint("wrong", store_name="test")


def test_source_identity_preserves_nested_paths(tmp_path) -> None:
    document_root = tmp_path / "docs"
    nested = document_root / "team-a" / "guide.md"
    nested.parent.mkdir(parents=True)
    nested.write_text("guide", encoding="utf-8")

    assert source_identity(nested, document_root) == "team-a/guide.md"
