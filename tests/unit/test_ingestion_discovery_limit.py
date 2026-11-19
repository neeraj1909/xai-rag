"""Bound the number of supported documents consumed during discovery."""

from pathlib import Path

import pytest

from xai_rag.service import IngestionLimitError, discover_document_paths

pytestmark = pytest.mark.unit


def test_discovery_stops_at_first_supported_file_over_limit(tmp_path, monkeypatch):
    root = tmp_path / "docs"
    root.mkdir()
    first, second = root / "first.md", root / "second.md"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    def bounded_walk(self, pattern):
        yield first
        yield second
        pytest.fail("discovery consumed entries after the file limit was exceeded")

    monkeypatch.setattr(Path, "rglob", bounded_walk)

    with pytest.raises(IngestionLimitError, match="exceeds the 1-file limit"):
        discover_document_paths(
            root, root, max_files=1, max_document_bytes=100, max_total_bytes=100
        )


def test_discovery_sorts_supported_files_within_limit(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "z.md").write_text("last", encoding="utf-8")
    (root / "a.md").write_text("first", encoding="utf-8")
    (root / "ignored.bin").write_text("ignored", encoding="utf-8")

    result = discover_document_paths(
        root, root, max_files=2, max_document_bytes=100, max_total_bytes=100
    )

    assert result == [root / "a.md", root / "z.md"]
