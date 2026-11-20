"""Durable service recovery uses the authorized staged source bytes."""

from types import SimpleNamespace

import pytest

from xai_rag.authorization import AccessScope
from xai_rag.ingestion.chunker import TextChunk
from xai_rag.ingestion.journal import RecoveryRequiredError
from xai_rag.ingestion.store import build_chunk_id, intended_record_fingerprint
from xai_rag.service import RAGService

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_service_replays_snapshot_and_checks_tenant_and_source(tmp_path):
    root = tmp_path / "documents"
    root.mkdir()
    document = root / "one.txt"
    document.write_text("original")
    journal = tmp_path / "journal"
    journal.mkdir(mode=0o700)
    chroma, elastic = {}, {}
    fail = True

    def chunk(text, strategy, *_args):
        return [TextChunk(text, 0, 0, len(text), strategy)]

    async def embed(texts):
        return [[0.1] for _ in texts]

    def write_chroma(_collection, chunks, _embeddings, source):
        ids = [build_chunk_id(source, c) for c in chunks]
        chroma.clear()
        chroma.update(zip(ids, (c.content for c in chunks), strict=True))
        return ids

    async def write_es(_es, chunks, ids, _source, _index):
        if fail:
            raise OSError("unknown remote write")
        elastic.clear()
        elastic.update(zip(ids, (c.content for c in chunks), strict=True))

    async def verify(_collection, _es, *, source_file, expected, index):
        assert source_file == "one.txt"
        assert index == "tenant-index"
        return all(
            {
                k: intended_record_fingerprint(TextChunk(v, 0, 0, len(v), "fixed"), source_file)
                for k, v in store.items()
            }
            == expected
            for store in (chroma, elastic)
        )

    async def parity(*_args, **_kwargs):
        return SimpleNamespace(consistent=chroma == elastic)

    def make_service(tenant="a"):
        return RAGService(
            collection=SimpleNamespace(name="tenant-collection"),
            es_client=object(),
            tenant=tenant,
            index_name="tenant-index",
            ingestion_journal_root=journal,
            parse_file_fn=lambda path: path.read_text(),
            chunk_text_fn=chunk,
            embed_texts_fn=embed,
            store_chroma_fn=write_chroma,
            store_elasticsearch_fn=write_es,
            expected_source_fn=verify,
            parity_fn=parity,
        )

    scope = AccessScope("a", ("one.txt",), True)
    result = await make_service().ingest(
        document, document_root=root, strategy="fixed", access_scope=scope
    )
    assert result.status == "failed"
    assert set(chroma.values()) == {"original"}
    document.write_text("changed on disk")
    fail = False
    restarted = make_service()
    with pytest.raises(PermissionError):
        await restarted.recover_ingestion(
            access_scope=AccessScope("a", (), True), acknowledge_quiescence=True
        )
    with pytest.raises(RecoveryRequiredError):
        await restarted.recover_ingestion(access_scope=scope)
    await restarted.recover_ingestion(access_scope=scope, acknowledge_quiescence=True)
    assert set(chroma.values()) == set(elastic.values()) == {"original"}
