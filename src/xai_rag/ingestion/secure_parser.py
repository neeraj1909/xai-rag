"""Stage documents through root-anchored descriptors before bounded parsing.

The ingest root and its ancestors must be writable only by the service owner.
The private snapshot protects against path replacement, not same-UID processes
or concurrent writes through an already-open inode.
"""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class DocumentParseError(ValueError):
    """A staged document could not be parsed safely."""


class ByteBudget:
    """Count bytes actually read across a single ingestion batch."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def consume(self, size: int) -> None:
        self.used += size
        if self.used > self.limit:
            raise ValueError("ingestion exceeds actual total byte limit")


def _open_beneath(root: Path, document: Path) -> int:
    """Open a regular file without following symlinks in any path component."""
    if (
        os.name != "posix"
        or not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"))
        or os.open not in os.supports_dir_fd
    ):
        raise RuntimeError("secure ingestion requires POSIX no-follow directory descriptors")
    try:
        relative = document.relative_to(root)
    except ValueError as exc:
        raise ValueError("document is outside the trusted root") from exc
    if not relative.parts or any(part in (".", "..") for part in relative.parts):
        raise ValueError("invalid document path")

    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    current = os.open(root, flags | os.O_DIRECTORY)
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=current)
            os.close(current)
            current = child
        fd = os.open(relative.parts[-1], flags | os.O_NONBLOCK, dir_fd=current)
    finally:
        os.close(current)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("document must be a regular file")
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def stage_document(
    root: Path, document: Path, max_bytes: int, *, budget: ByteBudget | None = None
) -> Iterator[tuple[Path, int]]:
    """Copy at most ``max_bytes`` actual bytes into a private temporary file."""
    if max_bytes < 0:
        raise ValueError("document byte limit must be nonnegative")
    fd = _open_beneath(root, document)
    try:
        with tempfile.TemporaryDirectory(prefix="xai-parse-") as temporary:
            staged = Path(temporary) / ("document" + document.suffix.lower())
            total = 0
            with staged.open("xb") as output:
                while block := os.read(fd, min(65536, max_bytes - total + 1)):
                    total += len(block)
                    if budget is not None:
                        budget.consume(len(block))
                    if total > max_bytes:
                        raise ValueError("document exceeds actual byte limit")
                    output.write(block)
            yield staged, total
    finally:
        os.close(fd)


def parse_staged_document(
    path: Path,
    max_characters: int,
    *,
    timeout_seconds: float,
    memory_bytes: int,
    max_document_bytes: int = 20_000_000,
) -> str:
    """Run the parser in a resource-bounded child; this is not an OS sandbox."""
    if os.name != "posix":
        raise RuntimeError("bounded parsing requires POSIX process controls")
    # No application credentials, proxy variables, or inherited user Python path.
    env = {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "XAI_RAG_MAX_DOCUMENT_BYTES": str(max_document_bytes),
    }
    # Separate private working directory prevents implicit loading of project .env files.
    with tempfile.TemporaryDirectory(prefix="xai-parser-cwd-") as working_directory:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "xai_rag.ingestion.parser_worker",
                str(path.resolve()),
                str(max_characters),
                str(memory_bytes),
                str(timeout_seconds),
            ],
            env=env,
            cwd=working_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(timeout=timeout_seconds)
        except BaseException as exc:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            if isinstance(exc, subprocess.TimeoutExpired):
                raise TimeoutError("document parsing exceeded its deadline") from None
            raise
    if process.returncode != 0:
        raise DocumentParseError("document parser failed")
    try:
        result = json.loads(stdout)
    except (UnicodeDecodeError, ValueError) as exc:
        raise DocumentParseError("document parser returned invalid output") from exc
    if not isinstance(result, str) or len(result) > max_characters:
        raise DocumentParseError("document parser output exceeds its limit")
    return result
