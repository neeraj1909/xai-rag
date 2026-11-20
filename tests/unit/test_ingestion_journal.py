"""Durable host-local ingestion intent and process lock."""

import asyncio
import os
import subprocess
import sys

import pytest

from xai_rag.ingestion.journal import IngestionJournal, RecoveryRequiredError

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_intent_and_exact_snapshot_survive_fresh_instance(tmp_path):
    directory = tmp_path / "state"
    journal = IngestionJournal(directory)
    async with journal.lock():
        journal.begin(b"sensitive bytes", ".txt", {"source": "one.txt"})
    other = IngestionJournal(directory)
    async with other.lock():
        pending = other.pending()
        assert pending is not None
        assert pending.snapshot_path.read_bytes() == b"sensitive bytes"
        assert pending.metadata == {"source": "one.txt"}
        assert os.stat(directory).st_mode & 0o077 == 0
        assert os.stat(pending.snapshot_path).st_mode & 0o077 == 0
        with pytest.raises(RecoveryRequiredError):
            other.begin(b"new", ".txt", {})
        other.complete()
        assert other.pending() is None
        assert not pending.snapshot_path.exists()


@pytest.mark.asyncio
async def test_corrupt_snapshot_fails_closed(tmp_path):
    journal = IngestionJournal(tmp_path / "state")
    async with journal.lock():
        pending = journal.begin(b"one", ".txt", {})
        pending.snapshot_path.write_bytes(b"two")
        with pytest.raises(RecoveryRequiredError):
            journal.pending()
        assert journal.intent.exists()


@pytest.mark.asyncio
async def test_lock_excludes_another_process(tmp_path):
    journal = IngestionJournal(tmp_path / "state")
    code = """import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(73)
sys.exit(0)
"""
    async with journal.lock():
        locked = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", code, str(journal.directory / "writer.lock")],
            check=False,
        )
        assert locked.returncode == 73
    released = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", code, str(journal.directory / "writer.lock")],
        check=False,
    )
    assert released.returncode == 0


@pytest.mark.asyncio
async def test_orphan_snapshot_removed_on_next_locked_start(tmp_path):
    journal = IngestionJournal(tmp_path)
    orphan = tmp_path / "pending.md"
    orphan.write_text("private data")
    async with journal.lock():
        assert not orphan.exists()


def test_namespace_binds_tenant_and_store_without_credentials():
    from xai_rag.ingestion.journal import index_namespace

    args = {
        "tenant": "tenant-a",
        "collection": "docs-a",
        "chroma_host": "LOCALHOST.",
        "chroma_port": 8100,
        "elastic_url": "https://user:secret@es.example:443/",
        "elastic_index": "documents-a",
    }
    base = index_namespace(**args)
    assert base == index_namespace(**{**args, "elastic_url": "https://es.example"})
    for field, value in (
        ("tenant", "tenant-b"),
        ("collection", "docs-b"),
        ("elastic_index", "documents-b"),
        ("chroma_port", 8101),
    ):
        assert base != index_namespace(**{**args, field: value})


@pytest.mark.asyncio
async def test_existing_namespace_is_synced_after_another_process_creates_it(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    namespace = root / "namespace"
    # This process intentionally creates the namespace without syncing its parent.
    subprocess.run(
        [sys.executable, "-c", "import os,sys; os.mkdir(sys.argv[1], 0o700)", str(namespace)],
        check=True,
    )
    synced = []
    real_fsync = os.fsync

    def record_fsync(fd):
        synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", record_fsync)
    async with IngestionJournal(namespace).lock():
        assert str(namespace) in synced
        assert str(root) in synced
        assert str(tmp_path) in synced


@pytest.mark.asyncio
async def test_journal_root_must_be_preprovisioned(tmp_path):
    root = tmp_path / "missing-root"
    with pytest.raises(FileNotFoundError):
        async with IngestionJournal(root / "namespace").lock():
            pass
    assert not root.exists()
