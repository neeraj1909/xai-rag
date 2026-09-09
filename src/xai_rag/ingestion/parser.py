"""Document parsing — convert PDF/DOCX/TXT/MD to plain text."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".txt", ".md", ".markdown"})


def source_identity(path: Path, document_root: Path) -> str:
    """Return a portable, collision-resistant path relative to a logical document root."""
    try:
        return path.resolve().relative_to(document_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("document path must be inside its logical document root") from exc


def parse_file(path: Path) -> str:
    """Parse a document file into plain text.

    Supports: .pdf, .docx, .txt, .md
    For PDFs, uses pymupdf4llm for markdown-aware extraction that
    preserves tables, headings, and structure.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix == ".docx":
        return _parse_docx(path)
    elif suffix in (".txt", ".md", ".markdown"):
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _parse_pdf(path: Path) -> str:
    """Parse PDF using pymupdf4llm for structure-aware extraction."""
    try:
        import pymupdf4llm

        return pymupdf4llm.to_markdown(str(path))
    except ImportError:
        # Fallback to basic pymupdf
        import pymupdf

        doc = pymupdf.open(str(path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)


def _parse_docx(path: Path) -> str:
    """Parse DOCX using python-docx."""
    from docx import Document

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def parse_directory(directory: Path, recursive: bool = True) -> list[tuple[Path, str]]:
    """Parse all supported documents in a directory.

    Returns list of (file_path, text_content) tuples.
    """
    pattern = "**/*" if recursive else "*"
    results = []

    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                text = parse_file(path)
                if text.strip():
                    results.append((path, text))
                    logger.info(f"Parsed {path.name}: {len(text)} chars")
                else:
                    logger.warning(f"Empty content after parsing: {path.name}")
            except Exception as exc:
                logger.error("Failed to parse %s (%s)", path.name, type(exc).__name__)

    logger.info(f"Parsed {len(results)} documents from {directory}")
    return results
