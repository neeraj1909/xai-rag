"""Secure staging must not follow swapped paths or exceed actual byte limits."""

import os

import pytest

pytestmark = pytest.mark.unit


def test_snapshot_rejects_swapped_file_symlink(tmp_path):
    from xai_rag.ingestion.secure_parser import stage_document

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    document = root / "doc.txt"
    document.symlink_to(outside)
    with pytest.raises((OSError, ValueError)), stage_document(root, document, 100):
        pytest.fail("should not stage symlink")


def test_snapshot_rejects_swapped_ancestor_symlink(tmp_path):
    from xai_rag.ingestion.secure_parser import stage_document

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "doc.txt").write_text("private")
    (root / "folder").symlink_to(outside, target_is_directory=True)
    with (
        pytest.raises((OSError, ValueError)),
        stage_document(root, root / "folder" / "doc.txt", 100),
    ):
        pytest.fail("should not stage symlink")


def test_snapshot_enforces_real_bytes_and_cleans_up(tmp_path):
    from xai_rag.ingestion.secure_parser import stage_document

    root = tmp_path / "root"
    root.mkdir()
    document = root / "doc.txt"
    document.write_text("12345")
    with pytest.raises(ValueError, match="byte limit"), stage_document(root, document, 4):
        pytest.fail("oversize staged")
    with stage_document(root, document, 5) as (snapshot, size):
        assert snapshot.read_text() == "12345"
        assert size == 5
        document.write_text("changed")
        assert snapshot.read_text() == "12345"
    assert not snapshot.exists()


def test_snapshot_rejects_fifo_without_blocking(tmp_path):
    from xai_rag.ingestion.secure_parser import stage_document

    if not hasattr(os, "mkfifo"):
        pytest.skip("POSIX FIFO unavailable")
    root = tmp_path / "root"
    root.mkdir()
    os.mkfifo(root / "named.txt")
    with pytest.raises(ValueError, match="regular"), stage_document(root, root / "named.txt", 100):
        pytest.fail("should not stage FIFO")


def test_worker_parses_real_text_and_rejects_timeout(tmp_path):
    from xai_rag.ingestion.secure_parser import parse_staged_document

    path = tmp_path / "doc.txt"
    path.write_text("test document")
    assert (
        parse_staged_document(path, 1000, timeout_seconds=30, memory_bytes=1024**3)
        == "test document"
    )
    with pytest.raises(TimeoutError):
        parse_staged_document(path, 1000, timeout_seconds=0.00001, memory_bytes=1024**3)


def test_shared_budget_counts_actual_bytes_and_cleanup_on_failure(tmp_path, monkeypatch):
    import tempfile

    from xai_rag.ingestion.secure_parser import ByteBudget, stage_document

    root = tmp_path / "root"
    root.mkdir()
    (root / "one.txt").write_text("123")
    (root / "two.txt").write_text("456")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    budget = ByteBudget(5)
    with stage_document(root, root / "one.txt", 3, budget=budget):
        pass
    with (
        pytest.raises(ValueError, match="total byte limit"),
        stage_document(root, root / "two.txt", 3, budget=budget),
    ):
        pytest.fail("total byte budget exceeded")
    assert budget.used == 6
    assert not list(tmp_path.glob("xai-parse-*"))


def test_unsupported_descriptor_platform_fails_closed(tmp_path, monkeypatch):
    from xai_rag.ingestion import secure_parser

    monkeypatch.setattr(secure_parser.os, "supports_dir_fd", set())
    with (
        pytest.raises(RuntimeError, match="requires POSIX"),
        secure_parser.stage_document(tmp_path, tmp_path / "doc.txt", 100),
    ):
        pytest.fail("unsupported platform opened document")


@pytest.mark.asyncio
async def test_replacement_between_discovery_and_parse_never_reaches_parser(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from xai_rag import service as service_module

    root = tmp_path / "root"
    root.mkdir()
    document = root / "doc.txt"
    document.write_text("permitted")
    outside = tmp_path / "private.txt"
    outside.write_text("secret")
    real_discover = service_module.discover_document_paths

    def discover_then_swap(*args, **kwargs):
        found = real_discover(*args, **kwargs)
        document.unlink()
        document.symlink_to(outside)
        return found

    monkeypatch.setattr(service_module, "discover_document_paths", discover_then_swap)
    parsed = []

    async def parity(*args):
        return SimpleNamespace(consistent=True)

    service = service_module.RAGService(
        collection=object(),
        es_client=object(),
        parse_file_fn=lambda path: parsed.append(path.read_text()),
        parity_fn=parity,
    )
    result = await service.ingest(document, document_root=root)
    assert result.status == "failed"
    assert result.documents[0].source_file == "doc.txt"
    assert result.documents[0].status == "failed"
    assert not parsed


def test_worker_does_not_load_parent_project_dotenv(tmp_path, monkeypatch):
    from xai_rag.ingestion.secure_parser import parse_staged_document

    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(
        "XAI_RAG_OPENAI_API_KEY=synthetic_secret\nXAI_RAG_MAX_INGEST_FILES=not_a_number\n"
    )
    monkeypatch.chdir(project)
    path = tmp_path / "doc.txt"
    path.write_text("test document")
    assert (
        parse_staged_document(path, 1000, timeout_seconds=30, memory_bytes=1024**3)
        == "test document"
    )
