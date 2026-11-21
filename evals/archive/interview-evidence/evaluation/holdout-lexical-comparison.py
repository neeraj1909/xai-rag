"""Offline diagnostic: real Elasticsearch BM25 + archived RRF over fixed rankings.

Requires a fresh disposable Elasticsearch on 127.0.0.1:19381 and PYTHONPATH
pointing to the archived XAI-RAG package. No network provider calls are made.
"""

import asyncio
import hashlib
import json
import platform
from pathlib import Path

import xai_rag
from elasticsearch import AsyncElasticsearch
from xai_rag.models import RetrievalStage, SearchResult
from xai_rag.retrieval.bm25_search import bm25_search
from xai_rag.retrieval.hybrid import rrf_fusion

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "multilingual-challenge-cases.json"
DENSE = ROOT / "multilingual-challenge-results.json"
OUT = ROOT / "holdout-lexical-results.json"
INDEX = "xai_holdout_lexical_v1"
SOURCE_HASH = "968bdafa6c1c5174f4f1385cf4f425debee1601d19bcc73f48e99807d912efb5"
DATA_HASH = "f2494b9bf4a766a374dc1fc1a6e141196ff045cf95a208d541452c94056f0887"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(directory):
    h = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        h.update(str(path.relative_to(directory)).encode())
        h.update(b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def metrics(rows):
    output = {}
    for stage in ("minilm", "bm25", "rrf"):
        output[stage] = {}
        for category in (
            "all",
            "hindi_to_hindi",
            "english_to_hindi",
            "hindi_to_english",
            "exact_code",
        ):
            selected = [
                r for r in rows if category == "all" or r["category"] == category
            ]
            valid = [r for r in selected if "error" not in r[stage]]
            output[stage][category] = {
                "attempted": len(selected),
                "scored": len(valid),
                "errors": len(selected) - len(valid),
                "hits": {
                    str(k): sum(
                        bool(
                            set(r["relevant"])
                            & {x["id"] for x in r[stage]["ranking"][:k]}
                        )
                        for r in valid
                    )
                    for k in (1, 3, 5)
                },
            }
    return output


async def main():
    if (
        sha(FIXTURE) != DATA_HASH
        or source_digest(Path(xai_rag.__file__).resolve().parent) != SOURCE_HASH
    ):
        raise ValueError(
            "Fixture or archived source digest does not match the frozen inputs"
        )
    data = json.loads(FIXTURE.read_text())
    dense = json.loads(DENSE.read_text())
    if dense["fixture_sha256"] != DATA_HASH:
        raise ValueError("Dense evidence uses a different fixture")
    dense_rows = {r["id"]: r for r in dense["models"]["multilingual_minilm"]["rows"]}
    docs = {d["id"]: d["text"] for d in data["documents"]}
    if len(docs) != len(data["documents"]) or len(dense_rows) != len(data["queries"]):
        raise ValueError("Duplicate source IDs or mismatched question count")
    es = AsyncElasticsearch("http://127.0.0.1:19381", max_retries=0, request_timeout=30)
    rows = []
    try:
        info = await es.info()
        if info["version"]["number"] != "8.17.0" or await es.indices.exists(
            index=INDEX
        ):
            raise ValueError("Expected Elasticsearch 8.17.0 with a fresh index")
        await es.indices.create(
            index=INDEX,
            settings={
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {"analyzer": {"default": {"type": "standard"}}},
            },
            mappings={
                "properties": {
                    "content": {"type": "text", "analyzer": "standard"},
                    "chunk_id": {"type": "keyword"},
                    "source_file": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": False},
                }
            },
        )
        for doc_id, text in docs.items():
            await es.index(
                index=INDEX,
                id=doc_id,
                document={
                    "content": text,
                    "chunk_id": doc_id,
                    "source_file": doc_id,
                    "metadata": {},
                },
            )
        await es.indices.refresh(index=INDEX)
        tokens = {}
        for text in ("T-24", "T-42", "R-17", "R-71", "T-24 T-42"):
            analysis = await es.indices.analyze(
                index=INDEX, analyzer="standard", text=text
            )
            tokens[text] = [t["token"] for t in analysis["tokens"]]
        for q in data["queries"]:
            old = dense_rows[q["id"]]
            if any(old[k] != q[k] for k in ("question", "category", "relevant")):
                raise ValueError("Dense query and frozen fixture differ")
            vec = [
                SearchResult(
                    id=x["id"],
                    content=docs[x["id"]],
                    score=x["cosine"],
                    stage=RetrievalStage.VECTOR,
                    rank=i,
                )
                for i, x in enumerate(old["ranking"][:5], 1)
            ]
            row = {k: q[k] for k in ("id", "category", "question", "relevant")}
            row["minilm"] = {"ranking": [{"id": x.id, "score": x.score} for x in vec]}
            try:
                bm = await bm25_search(es, q["question"], INDEX, k=5)
                row["bm25"] = {"ranking": [{"id": x.id, "score": x.score} for x in bm]}
                fused = rrf_fusion([vec, bm], k=60)
                row["rrf"] = {
                    "ranking": [
                        {
                            "id": x.id,
                            "score": x.rrf_score,
                            "vector_rank": x.vector_rank,
                            "bm25_rank": x.bm25_rank,
                        }
                        for x in fused
                    ]
                }
            except Exception as exc:  # noqa: BLE001 - preserve per-case failure
                row["bm25"] = {"error": type(exc).__name__}
                row["rrf"] = {"error": "BM25 or fusion failed"}
            rows.append(row)
        output = {
            "fixture_sha256": DATA_HASH,
            "dense_results_sha256": sha(DENSE),
            "source_tree_sha256": SOURCE_HASH,
            "python": platform.python_version(),
            "elasticsearch_version": info["version"]["number"],
            "elasticsearch_image_digest": "sha256:2f602552550869fb29b6fd5848c5118d3ef3a2e1d5d45802e3ab9088cb2de8e2",
            "method": "Archived MiniLM exact-cosine top 5; real Elasticsearch BM25 top 5 with project standard analyzer and query; archived application RRF k=60; source IDs used as chunk IDs for this diagnostic; no ANN, reranker or generator.",
            "index_name": INDEX,
            "analyzed_tokens": tokens,
            "metrics": metrics(rows),
            "rows": rows,
        }
        with OUT.open("x") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(json.dumps(output["metrics"], indent=2))
        print("TOKENIZATION", json.dumps(tokens, ensure_ascii=False))
    finally:
        await es.close()


if __name__ == "__main__":
    asyncio.run(main())
