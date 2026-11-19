"""Sensitive source names and parser errors do not enter ingestion logs."""

import logging

import pytest

from xai_rag.ingestion import parser
from xai_rag.ingestion.chunker import TextChunk
from xai_rag.ingestion.store import delete_by_source, store_chunks_chromadb

pytestmark = pytest.mark.unit


def test_parser_logs_omit_paths_and_exception_messages(tmp_path, monkeypatch, caplog):
    directory = tmp_path / "PRIVATE_DIRECTORY_CANARY"
    directory.mkdir()
    for name in ("PRIVATE_SUCCESS.txt", "PRIVATE_EMPTY.txt", "PRIVATE_FAILURE.txt"):
        (directory / name).write_text("input")

    def parse(path):
        if "FAILURE" in path.name:
            raise ValueError("PRIVATE_EXCEPTION_CANARY")
        return "" if "EMPTY" in path.name else "content"

    monkeypatch.setattr(parser, "parse_file", parse)
    with caplog.at_level(logging.INFO):
        results = parser.parse_directory(directory)

    assert len(results) == 1
    assert results[0][0].name == "PRIVATE_SUCCESS.txt"
    assert results[0][1] == "content"
    assert "PRIVATE_" not in caplog.text
    assert "ValueError" in caplog.text
    assert "1 documents" in caplog.text


def test_store_logs_omit_source_but_preserve_provenance_and_counts(caplog):
    class Collection:
        def __init__(self):
            self.written = None
            self.deleted = None

        def get(self, **kwargs):
            return {"ids": ["old"]} if self.written else {"ids": []}

        def upsert(self, **kwargs):
            self.written = kwargs

        def delete(self, **kwargs):
            self.deleted = kwargs

    source = "PRIVATE_SOURCE_CANARY.txt"
    collection = Collection()
    chunk = TextChunk("content", 0, 0, 7, "fixed")
    with caplog.at_level(logging.INFO):
        ids = store_chunks_chromadb(collection, [chunk], [[0.1]], source)
        deleted = delete_by_source(collection, source)

    assert len(ids) == 1
    assert collection.written["metadatas"][0]["source_file"] == source
    assert collection.deleted == {"where": {"source_file": source}}
    assert deleted == 1
    assert source not in caplog.text
    assert "Stored 1 chunks" in caplog.text
    assert "Deleted 1 chunks" in caplog.text
