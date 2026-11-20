"""Host-local durable write intent for a single pair of search indexes.

The caller must hold ``lock()`` for every operation, including parity checks.
This is not a distributed lock and cannot fence requests already sent to a
remote store by a process that has died.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import inspect
import json
import os
import stat
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from pathlib import Path


class RecoveryRequiredError(RuntimeError):
    """A previous write may be incomplete and needs operator-led recovery."""


def index_namespace(
    *,
    tenant: str | None,
    collection: str,
    chroma_host: str,
    chroma_port: int,
    elastic_url: str,
    elastic_index: str,
) -> str:
    """Hash the configured store identity without recording URL credentials.

    Cooperating writers must use identical canonical configuration and the same
    journal root. Aliases of one backend cannot be detected automatically.
    """
    endpoint = urlsplit(elastic_url)
    if not endpoint.hostname or not endpoint.scheme or not collection or not elastic_index:
        raise ValueError("complete index identity is required")
    identity = {
        "tenant": tenant,
        "collection": collection,
        "chroma_host": chroma_host.lower().rstrip("."),
        "chroma_port": chroma_port,
        "elastic_scheme": endpoint.scheme.lower(),
        "elastic_host": endpoint.hostname.lower().rstrip("."),
        "elastic_port": endpoint.port or (443 if endpoint.scheme == "https" else 80),
        "elastic_path": endpoint.path.rstrip("/"),
        "elastic_index": elastic_index,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class PendingWrite:
    metadata: dict[str, Any]
    snapshot_path: Path


class IngestionJournal:
    """One persistent intent and immutable source snapshot per index pair.

    A journal contains source bytes, so its parent must be service-owned and
    placed on a persistent, local filesystem. Deletion is not secure erasure.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.intent = directory / "pending.json"

    def _snapshot_path(self, suffix: str) -> Path:
        if suffix not in {".txt", ".md", ".markdown", ".pdf", ".docx"}:
            raise ValueError("unsupported snapshot suffix")
        return self.directory / f"pending{suffix}"

    def _prepare_directory(self) -> None:
        # The operator must provision the root durably before starting writers.
        # Do not auto-create an arbitrarily long, unsynced ancestor chain.
        for directory in (self.directory.parent,):
            info = directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
                raise PermissionError("journal directories must be owned by the service user")
            if stat.S_IMODE(info.st_mode) & 0o077:
                raise PermissionError("journal directories must not be accessible by other users")
        with contextlib.suppress(FileExistsError):
            self.directory.mkdir(mode=0o700)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
            raise PermissionError("journal directories must be owned by the service user")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise PermissionError("journal directories must not be accessible by other users")

        # A different process may have created the namespace but not yet
        # synced its directory entry. Sync even when both already exist.
        for directory in (self.directory, self.directory.parent, self.directory.parent.parent):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _fsync_directory(self) -> None:
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @contextlib.asynccontextmanager
    async def lock(self):
        """Serialize cooperating local processes; cancellation closes the fd."""
        self._prepare_directory()
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.directory / "writer.lock", flags, 0o600)
        try:
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.05)
            self._clean_orphans()
            yield
        finally:
            os.close(descriptor)

    def _replace_file(self, destination: Path, content: bytes) -> None:
        temp = destination.with_name(destination.name + ".tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        # A process may have died before renaming its private temporary file.
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()
        descriptor = os.open(temp, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, destination)
            self._fsync_directory()
        finally:
            with contextlib.suppress(FileNotFoundError):
                temp.unlink()

    def begin(self, source_bytes: bytes, suffix: str, metadata: dict[str, Any]) -> PendingWrite:
        """Persist the exact input bytes and intent before any remote write."""
        if self.intent.exists():
            raise RecoveryRequiredError("pending ingestion requires recovery")
        document = {
            "version": 1,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "suffix": suffix,
            "metadata": metadata,
        }
        snapshot = self._snapshot_path(suffix)
        self._replace_file(snapshot, source_bytes)
        self._replace_file(self.intent, json.dumps(document, sort_keys=True).encode())
        return PendingWrite(metadata=metadata, snapshot_path=snapshot)

    def _clean_orphans(self) -> None:
        """Remove abandoned snapshots only after acquiring the journal lock."""
        if self.intent.exists():
            return
        for suffix in (".txt", ".md", ".markdown", ".pdf", ".docx"):
            path = self._snapshot_path(suffix)
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".tmp").unlink(missing_ok=True)
        (self.directory / "pending.json.tmp").unlink(missing_ok=True)
        self._fsync_directory()

    def pending(self) -> PendingWrite | None:
        """Read and verify the durable snapshot; corruption fails closed."""
        if not self.intent.exists():
            return None
        try:
            document = json.loads(self.intent.read_text(encoding="utf-8"))
            snapshot = self._snapshot_path(document["suffix"])
            digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            if document["version"] != 1 or document["sha256"] != digest:
                raise ValueError("snapshot integrity mismatch")
            metadata = document["metadata"]
            if not isinstance(metadata, dict):
                raise ValueError("invalid metadata")
            return PendingWrite(metadata=metadata, snapshot_path=snapshot)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RecoveryRequiredError("ingestion journal is incomplete or corrupt") from exc

    def record_expected(self, expected: dict[str, str]) -> None:
        """Persist the intended manifest before any external write."""
        pending = self.pending()
        if pending is None:
            raise RecoveryRequiredError("cannot update a missing ingestion intent")
        document = json.loads(self.intent.read_text(encoding="utf-8"))
        document["metadata"]["expected"] = expected
        self._replace_file(self.intent, json.dumps(document, sort_keys=True).encode())

    def complete(self) -> None:
        """Clear only after remote writes and exact expected state are verified."""
        pending = self.pending()
        if pending is None:
            raise RecoveryRequiredError("cannot complete a missing ingestion intent")
        self.intent.unlink()
        self._fsync_directory()
        pending.snapshot_path.unlink(missing_ok=True)
        self._fsync_directory()


class CoordinatedIndexWriter:
    """Run and replay one source update under a cooperating host-local lock.

    `namespace` must uniquely describe the actual tenant and index pair, while
    `metadata` must describe the authorized original source and chunk options.
    Callers must ensure no other writer can bypass this coordinator. A replay
    requires an operator to establish that earlier remote requests have settled.
    """

    def __init__(self, journal: IngestionJournal, namespace: str) -> None:
        if not namespace:
            raise ValueError("index namespace is required")
        self.journal = journal
        self.namespace = namespace

    async def write(self, source_bytes, suffix, metadata, *, prepare, write, verify):
        async with self.journal.lock():
            if self.journal.pending() is not None:
                raise RecoveryRequiredError("pending ingestion requires recovery")
            intent = self.journal.begin(
                source_bytes, suffix, {"namespace": self.namespace, "request": metadata}
            )
            return await self._apply(intent, prepare=prepare, write=write, verify=verify)

    async def recover(self, *, authorize, acknowledge_quiescence, prepare, write, verify):
        if not acknowledge_quiescence:
            raise RecoveryRequiredError("recovery requires prior remote-write quiescence")
        async with self.journal.lock():
            intent = self.journal.pending()
            if intent is None:
                raise RecoveryRequiredError("no pending ingestion")
            if intent.metadata.get("namespace") != self.namespace:
                raise RecoveryRequiredError("pending ingestion belongs to another index namespace")
            authorization = authorize(intent.metadata["request"])
            if inspect.iscoroutine(authorization):
                authorization.close()
            if authorization is not True:
                raise PermissionError("pending ingestion is not authorized")
            return await self._apply(intent, prepare=prepare, write=write, verify=verify)

    async def _apply(self, intent, *, prepare, write, verify):
        try:
            payload, expected = await prepare(intent.snapshot_path, intent.metadata["request"])
        except Exception:
            # No remote operation has been attempted if there is no manifest.
            if "expected" not in intent.metadata:
                self.journal.complete()
            raise
        if not isinstance(expected, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in expected.items()
        ):
            raise ValueError("expected manifest must map chunk IDs to content hashes")
        recorded = intent.metadata.get("expected")
        if recorded is not None and recorded != expected:
            raise RecoveryRequiredError("replayed manifest differs from the original intent")
        self.journal.record_expected(expected)
        await write(payload)
        if not await verify(expected):
            raise RecoveryRequiredError("stored records differ from intended manifest")
        self.journal.complete()
        return payload
