"""Bounded and observable ingestion service contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from xai_rag.ingestion.chunker import TextChunk
from xai_rag.service import IngestionLimitError, RAGService, discover_document_paths

pytestmark = pytest.mark.unit


def test_discovery_rejects_per_file_and_total_size_limits(tmp_path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "one.md").write_text("12345", encoding="utf-8")
    (root / "two.md").write_text("67890", encoding="utf-8")

    with pytest.raises(IngestionLimitError, match="per-file"):
        discover_document_paths(
            root,
            root,
            max_files=10,
            max_document_bytes=4,
            max_total_bytes=100,
        )

    with pytest.raises(IngestionLimitError, match="total limit"):
        discover_document_paths(
            root,
            root,
            max_files=10,
            max_document_bytes=10,
            max_total_bytes=9,
        )


def test_dis_rejects_symlink_that_resolves_outside_root(tmp_path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    outside = tmp_path / "private.md"
    outside.write_text("secret", encoding="utf-8")
    (root / "linked.md").symlink_to(outside)

    with pytest.raises(IngestionLimitError, match="outside"):
        discover_document_paths(
            root,
            root,
            max_files=10,
            max_document_bytes=100,
            max_total_bytes=100,
        )


@pytest.mark.asyncio
async def test_ingestion_runs_through_shared_service_and_checks_parity(tmp_path) -> None:
    root = tmp_path / "docs"
    nested = root / "team" / "guide.md"
    nested.parent.mkdir(parents=True)
    nested.write_text("source text", encoding="utf-8")
    calls: list[str] = []

    def parse(path: Path) -> str:
        calls.append("parse")
        return path.read_text(encoding="utf-8")

    def chunk(text, strategy, chunk_size, chunk_overlap, semantic_threshold):
        calls.append("chunk")
        return [TextChunk(text, 0, 0, len(text), strategy)]

    async def embed(texts):
        calls.append("embed")
        return [[0.1, 0.2]]

    def store_chroma(collection, chunks, embeddings, source_file):
        calls.append(f"chroma:{source_file}")
        return ["stable-id"]

    async def store_elasticsearch(es, chunks, ids, source_file, index):
        calls.append(f"elasticsearch:{source_file}")

    async def parity(collection, es):
        calls.append("parity")
        return SimpleNamespace(consistent=True)

    service = RAGService(
        collection=object(),
        es_client=object(),
        parse_file_fn=parse,
        chunk_text_fn=chunk,
        embed_texts_fn=embed,
        store_chroma_fn=store_chroma,
        store_elasticsearch_fn=store_elasticsearch,
        parity_fn=parity,
    )

    result = await service.ingest(nested, document_root=root, strategy="fixed")

    assert result.status == "success"
    assert result.ingested_files == 1
    assert result.total_chunks == 1
    assert result.parity_consistent is True
    assert "chroma:team/guide.md" in calls
    assert "elasticsearch:team/guide.md" in calls
    assert calls[-1] == "parity"
