import asyncio
import hashlib
import unittest
from unittest.mock import AsyncMock

from index_snapshot import snapshot


def record(cid, text, source="a.txt"):
    metadata = {
        "source_file": source,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "chunk_index": int(cid),
    }
    return cid, text, metadata, [0.1, 0.2]


class FakeCollection:
    name = "test-collection"

    def __init__(self, records):
        self.records = records

    def get(self, *, where, include, limit, offset):
        rows = [r for r in self.records if r[2]["source_file"] in where["source_file"]["$in"]][
            offset : offset + limit
        ]
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [r[2] for r in rows],
            "embeddings": [r[3] for r in rows],
        }


def es_for(records):
    async def search(**kw):
        rows = sorted(
            (r for r in records if r[2]["source_file"] in kw["query"]["terms"]["source_file"]),
            key=lambda r: r[0],
        )
        if "search_after" in kw:
            rows = [r for r in rows if r[0] > kw["search_after"][0]]
        hits = [
            {
                "_id": i,
                "_source": {
                    "chunk_id": i,
                    "content": t,
                    "metadata": m,
                    "source_file": m["source_file"],
                },
                "sort": [i],
            }
            for i, t, m, v in rows[: kw["size"]]
        ]
        return {"timed_out": False, "_shards": {"failed": 0}, "hits": {"hits": hits}}

    return type("ES", (), {"search": staticmethod(search)})()


class TestSnapshot(unittest.TestCase):
    def snap(self, chroma, elastic, **kwargs):
        return asyncio.run(
            snapshot(
                FakeCollection(chroma),
                es_for(elastic),
                tenant="test",
                sources=["a.txt"],
                es_index="index",
                page_size=1,
                **kwargs,
            )
        )

    def test_stable_and_scoped(self):
        rows = [record("1", "hello"), record("2", "world"), record("3", "excluded", "private.txt")]
        first = self.snap(rows, rows)
        second = self.snap(list(reversed(rows)), rows)
        self.assertEqual(first, second)
        self.assertTrue(first["record_parity"])
        self.assertEqual(first["chroma_count"], 2)
        self.assertEqual(first["chroma_records_sha256"], first["elasticsearch_records_sha256"])

    def test_tamper_and_vector_are_distinguishable(self):
        rows = [record("1", "hello")]
        baseline = self.snap(rows, rows)
        changed = [("1", "hello", rows[0][2], [0.9, 0.2])]
        altered_vector = self.snap(changed, rows)
        self.assertTrue(altered_vector["record_parity"])
        self.assertEqual(baseline["chroma_records_sha256"], altered_vector["chroma_records_sha256"])
        self.assertNotEqual(
            baseline["chroma_vectors_float32_le_sha256"],
            altered_vector["chroma_vectors_float32_le_sha256"],
        )
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.snap([("1", "tampered", rows[0][2], rows[0][3])], rows)

    def test_incomplete_es_fails(self):
        es = type(
            "ES",
            (),
            {
                "search": AsyncMock(
                    return_value={"timed_out": True, "_shards": {"failed": 0}, "hits": {"hits": []}}
                )
            },
        )()
        with self.assertRaisesRegex(ValueError, "incomplete Elasticsearch"):
            asyncio.run(
                snapshot(FakeCollection([]), es, tenant="t", sources=["a.txt"], es_index="x")
            )

    def test_empty_scope_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "scope|sources"):
            asyncio.run(
                snapshot(FakeCollection([]), es_for([]), tenant="t", sources=[], es_index="x")
            )

    def test_duplicate_id_rejected(self):
        rows = [record("1", "first"), record("1", "second")]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.snap(rows, rows)

    def test_partial_shard_failure_rejected(self):
        es = type(
            "ES",
            (),
            {
                "search": AsyncMock(
                    return_value={
                        "timed_out": False,
                        "_shards": {"failed": 1},
                        "hits": {"hits": []},
                    }
                )
            },
        )()
        with self.assertRaisesRegex(ValueError, "incomplete Elasticsearch"):
            asyncio.run(
                snapshot(
                    FakeCollection([record("1", "hello")]),
                    es,
                    tenant="t",
                    sources=["a.txt"],
                    es_index="x",
                )
            )

    def test_es_content_difference_cannot_appear_as_parity(self):
        first = [record("1", "first")]
        second = [record("1", "other")]
        result = self.snap(first, second)
        self.assertFalse(result["record_parity"])


if __name__ == "__main__":
    unittest.main()
