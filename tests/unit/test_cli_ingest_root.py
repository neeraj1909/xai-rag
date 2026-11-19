"""CLI ingestion keeps source identities relative to an explicit root."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from xai_rag.cli import main
from xai_rag.ingestion.parser import source_identity

pytestmark = pytest.mark.unit


def test_ingesting_nested_file_with_document_root_keeps_source_identity(tmp_path, monkeypatch):
    root = tmp_path / "docs"
    nested = root / "nested" / "doc.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("fictional text")
    captured = {}

    async def fake_ingest(path, strategy, chunk_size, chunk_overlap, document_root=None):
        captured["source"] = source_identity(path, document_root)

    monkeypatch.setattr("xai_rag.cli._ingest", fake_ingest)
    result = CliRunner().invoke(main, ["ingest", str(nested), "--document-root", str(root)])

    assert result.exit_code == 0, result.output
    assert captured["source"] == "nested/doc.txt"


def test_ingest_rejects_source_outside_explicit_root_before_connecting(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    source = tmp_path / "outside.txt"
    source.write_text("fictional text")
    called = []

    async def fake_ingest(*args, **kwargs):
        called.append(True)

    monkeypatch.setattr("xai_rag.cli._ingest", fake_ingest)
    result = CliRunner().invoke(main, ["ingest", str(source), "--document-root", str(root)])

    assert result.exit_code != 0
    assert "outside" in result.output.lower()
    assert not called


def test_ingest_without_explicit_root_preserves_existing_file_default(tmp_path, monkeypatch):
    source = tmp_path / "nested" / "doc.txt"
    source.parent.mkdir()
    source.write_text("fictional text")
    captured = {}

    async def fake_ingest(path, strategy, chunk_size, chunk_overlap, document_root=None):
        captured["root"] = document_root
        captured["source"] = source_identity(path, document_root)

    monkeypatch.setattr("xai_rag.cli._ingest", fake_ingest)
    result = CliRunner().invoke(main, ["ingest", str(source)])

    assert result.exit_code == 0, result.output
    assert captured == {"root": Path(source.parent), "source": "doc.txt"}


def test_cli_forwards_original_root_to_actual_service_boundary(tmp_path, monkeypatch):
    from types import SimpleNamespace

    root = tmp_path / "docs"
    source = root / "nested" / "doc.txt"
    source.parent.mkdir(parents=True)
    source.write_text("fictional text")
    seen = []

    class FakeService:
        def __init__(self, collection, es_client):
            pass

        async def ingest(self, path, *, document_root, **kwargs):
            seen.append(source_identity(path, document_root))
            return SimpleNamespace(
                documents=[],
                status="success",
                ingested_files=1,
                skipped_files=0,
                failed_files=0,
                total_chunks=1,
                parity_consistent=True,
            )

    class FakeES:
        async def close(self):
            pass

    async def get_es():
        return FakeES()

    monkeypatch.setattr("xai_rag.ingestion.store.get_chroma_client", lambda: None)
    monkeypatch.setattr("xai_rag.ingestion.store.get_chroma_collection", lambda _: object())
    monkeypatch.setattr("xai_rag.ingestion.store.get_es_client", get_es)
    monkeypatch.setattr("xai_rag.service.RAGService", FakeService)
    result = CliRunner().invoke(main, ["ingest", str(source), "--document-root", str(root)])

    assert result.exit_code == 0, result.output
    assert seen == ["nested/doc.txt"]
