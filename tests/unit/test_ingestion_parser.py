"""Guardrails for parsing compressed DOCX documents."""

from zipfile import ZipFile

import pytest
from docx import Document

from xai_rag.config import settings
from xai_rag.ingestion.parser import parse_file

pytestmark = pytest.mark.unit


def test_docx_rejects_expanded_content_above_document_limit(tmp_path, monkeypatch) -> None:
    path = tmp_path / "compressed.docx"
    document = Document()
    for _ in range(200):
        document.add_paragraph("A" * 4096)
    document.save(path)
    limit = 100_000
    with ZipFile(path) as archive:
        assert path.stat().st_size < limit < sum(entry.file_size for entry in archive.infolist())
    monkeypatch.setattr(settings, "max_document_bytes", limit)

    with pytest.raises(ValueError, match="expanded DOCX"):
        parse_file(path)


def test_docx_within_expanded_limit_is_parsed(tmp_path, monkeypatch) -> None:
    path = tmp_path / "normal.docx"
    document = Document()
    document.add_paragraph("Useful document content")
    document.save(path)
    with ZipFile(path) as archive:
        limit = sum(entry.file_size for entry in archive.infolist())
    monkeypatch.setattr(settings, "max_document_bytes", limit)

    assert parse_file(path) == "Useful document content"
