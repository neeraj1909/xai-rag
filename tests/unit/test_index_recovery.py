"""Recovery behavior across an interrupted two-store write."""

import hashlib

import pytest

from xai_rag.ingestion.journal import (
    CoordinatedIndexWriter,
    IngestionJournal,
    RecoveryRequiredError,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_interrupted_write_requires_replay_of_exact_source(tmp_path):
    chroma, elastic = {}, {}
    document = b"first version"
    metadata = {"source": "authorized/doc.txt", "strategy": "fixed", "size": 10}
    namespace = "tenant-a-index-v1"

    async def prepare(path, request):
        assert request == metadata
        content = path.read_bytes()
        expected = {"chunk": hashlib.sha256(content).hexdigest()}
        return expected, expected

    async def broken_write(expected):
        chroma.update(expected)
        raise OSError("lost connection before second write")

    async def write(expected):
        chroma.clear()
        chroma.update(expected)
        elastic.clear()
        elastic.update(expected)

    async def verify(expected):
        return chroma == elastic == expected

    first = CoordinatedIndexWriter(IngestionJournal(tmp_path), namespace)
    with pytest.raises(OSError):
        await first.write(
            document, ".txt", metadata, prepare=prepare, write=broken_write, verify=verify
        )
    assert chroma and not elastic

    # A new process constructs a new coordinator and cannot overwrite pending work.
    restarted = CoordinatedIndexWriter(IngestionJournal(tmp_path), namespace)
    with pytest.raises(RecoveryRequiredError):
        await restarted.write(
            b"different", ".txt", metadata, prepare=prepare, write=write, verify=verify
        )
    with pytest.raises(RecoveryRequiredError, match="quiescence"):
        await restarted.recover(
            authorize=lambda _: True,
            acknowledge_quiescence=False,
            prepare=prepare,
            write=write,
            verify=verify,
        )
    with pytest.raises(PermissionError):
        await restarted.recover(
            authorize=lambda _: (_ for _ in ()).throw(PermissionError()),
            acknowledge_quiescence=True,
            prepare=prepare,
            write=write,
            verify=verify,
        )

    async def async_authorize(_request):
        return True

    for denied in (lambda _: False, lambda _: None, async_authorize):
        with pytest.raises(PermissionError):
            await restarted.recover(
                authorize=denied,
                acknowledge_quiescence=True,
                prepare=prepare,
                write=write,
                verify=verify,
            )
    wrong_tenant = CoordinatedIndexWriter(IngestionJournal(tmp_path), "tenant-b-index-v1")
    with pytest.raises(RecoveryRequiredError, match="namespace"):
        await wrong_tenant.recover(
            authorize=lambda _: True,
            acknowledge_quiescence=True,
            prepare=prepare,
            write=write,
            verify=verify,
        )
    await restarted.recover(
        authorize=lambda request: request == metadata,
        acknowledge_quiescence=True,
        prepare=prepare,
        write=write,
        verify=verify,
    )
    assert chroma == elastic == {"chunk": hashlib.sha256(document).hexdigest()}
    assert restarted.journal.pending() is None


async def test_replay_rejects_changed_manifest(tmp_path):
    coordinator = CoordinatedIndexWriter(IngestionJournal(tmp_path), "one")

    async def prepare(_path, _metadata):
        return None, {"id": "old"}

    async def changed(_path, _metadata):
        return None, {"id": "new"}

    async def fail(_payload):
        raise OSError()

    async def verify(_expected):
        return True

    with pytest.raises(OSError):
        await coordinator.write(b"x", ".txt", {}, prepare=prepare, write=fail, verify=verify)
    with pytest.raises(RecoveryRequiredError, match="manifest"):
        await coordinator.recover(
            authorize=lambda _: True,
            acknowledge_quiescence=True,
            prepare=changed,
            write=fail,
            verify=verify,
        )


async def test_process_crash_between_stores_is_replayed(tmp_path):
    import os
    import subprocess
    import sys

    script = r"""
import asyncio, hashlib, os, sys
from pathlib import Path
from xai_rag.ingestion.journal import CoordinatedIndexWriter, IngestionJournal
root = Path(sys.argv[1])
async def prepare(path, request):
    expected = {"chunk": hashlib.sha256(path.read_bytes()).hexdigest()}
    return expected, expected
async def crash(expected):
    (root / "chroma").write_text(expected["chunk"])
    os._exit(29)
async def verify(expected):
    return False
asyncio.run(CoordinatedIndexWriter(IngestionJournal(root / "journal"), "tenant-a").write(
    b"original", ".txt", {"source": "source.txt"},
    prepare=prepare, write=crash, verify=verify))
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=os.environ.copy(),
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert process.returncode == 29, process.stderr.decode()
    assert (tmp_path / "chroma").exists()
    assert not (tmp_path / "elastic").exists()

    async def prepare(path, _request):
        expected = {"chunk": hashlib.sha256(path.read_bytes()).hexdigest()}
        return expected, expected

    async def write(expected):
        for name in ("chroma", "elastic"):
            (tmp_path / name).write_text(expected["chunk"])

    async def verify(expected):
        return all(
            (tmp_path / name).read_text() == expected["chunk"] for name in ("chroma", "elastic")
        )

    writer = CoordinatedIndexWriter(IngestionJournal(tmp_path / "journal"), "tenant-a")
    await writer.recover(
        authorize=lambda _: True,
        acknowledge_quiescence=True,
        prepare=prepare,
        write=write,
        verify=verify,
    )
    assert writer.journal.pending() is None
    assert (tmp_path / "elastic").read_text() == hashlib.sha256(b"original").hexdigest()
