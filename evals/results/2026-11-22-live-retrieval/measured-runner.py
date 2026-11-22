"""Run a labeled fixture through real local retrieval stores and models."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import sys
import time
from pathlib import Path

from xai_rag.evaluation.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, reciprocal_rank

STAGES = ("vector", "bm25", "rrf", "rerank")


def validate_fixture(fixture: dict) -> None:
    documents, queries = fixture["documents"], fixture["queries"]
    ids = [doc["id"] for doc in documents]
    query_ids = [query["id"] for query in queries]
    if (
        not ids
        or not query_ids
        or len(set(ids)) != len(ids)
        or len(set(query_ids)) != len(query_ids)
    ):
        raise ValueError("documents and queries must have unique, nonempty IDs")
    if any(not doc["id"] or not doc["text"] for doc in documents):
        raise ValueError("documents must have ID and text")
    for query in queries:
        relevant = query["relevant"]
        if not query["id"] or not query["question"] or len(set(relevant)) != len(relevant):
            raise ValueError("queries must have IDs, questions and unique relevance labels")
        if not set(relevant) <= set(ids):
            raise ValueError("relevance labels must refer to fixture documents")


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for stage in STAGES:
        eligible = [row for row in rows if row["relevant"]]
        scored = [row for row in eligible if "ranking" in row["stages"].get(stage, {})]
        values = []
        for row in scored:
            relevant = {item: 1.0 for item in row["relevant"]}
            ranking = row["stages"][stage]["ranking"]
            values.append(
                {
                    "hit_at_1": hit_rate_at_k(ranking, relevant, 1),
                    "hit_at_3": hit_rate_at_k(ranking, relevant, 3),
                    "recall_at_3": recall_at_k(ranking, relevant, 3),
                    "mrr_at_5": reciprocal_rank(ranking[:5], relevant),
                    "ndcg_at_3": ndcg_at_k(ranking, relevant, 3),
                }
            )
        result[stage] = {
            "eligible": len(eligible),
            "scored": len(scored),
            "failures": len(eligible) - len(scored),
            "unlabeled": len(rows) - len(eligible),
            "mean": {
                key: sum(v[key] for v in values) / len(values) if values else None
                for key in ("hit_at_1", "hit_at_3", "recall_at_3", "mrr_at_5", "ndcg_at_3")
            },
        }
    return result


def source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        digest.update(str(source.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(source.read_bytes()).digest())
    return digest.hexdigest()


async def execute(fixture: dict, output: dict, chroma_port: int, es_port: int, k: int) -> bool:
    # Override every store setting before importing the project configuration.
    name = "xai_eval_" + secrets.token_hex(12)
    os.environ.update(
        {
            "XAI_RAG_CHROMA_HOST": "127.0.0.1",
            "XAI_RAG_CHROMA_PORT": str(chroma_port),
            "XAI_RAG_CHROMA_COLLECTION": name,
            "XAI_RAG_ELASTICSEARCH_URL": f"http://127.0.0.1:{es_port}",
            "XAI_RAG_ELASTICSEARCH_INDEX": name,
            "XAI_RAG_ELASTICSEARCH_API_KEY": "",
        }
    )
    import xai_rag
    from xai_rag.config import settings
    from xai_rag.ingestion.chunker import TextChunk
    from xai_rag.ingestion.embedder import embed_texts
    from xai_rag.ingestion.store import (
        get_chroma_client,
        get_chroma_collection,
        get_es_client,
        store_chunks_chromadb,
        store_chunks_elasticsearch,
    )
    from xai_rag.retrieval.hybrid import hybrid_search
    from xai_rag.retrieval.reranker import Reranker

    output.update(
        {
            "source_tree_sha256": source_digest(Path(xai_rag.__file__).resolve().parent),
            "embedding_model": settings.embedding_model,
            "embedding_revision": settings.embedding_model_revision,
            "reranker_model": settings.reranker_model,
            "reranker_revision": settings.reranker_model_revision,
            "candidate_k": k,
            "rows": [],
        }
    )
    chroma = get_chroma_client()
    es = await get_es_client()
    created_chroma = False
    created_es = False
    failed = False
    try:
        existing = {getattr(item, "name", item) for item in chroma.list_collections()}
        if name in existing or await es.indices.exists(index=name):
            raise RuntimeError("generated evaluation index already exists")
        # Record ownership only after the collection is created; never delete preexisting data.
        collection = get_chroma_collection(chroma, name)
        created_chroma = True
        vectors = await embed_texts([doc["text"] for doc in fixture["documents"]])
        source_by_chunk = {}
        for doc, vector in zip(fixture["documents"], vectors, strict=True):
            chunk = TextChunk(
                content=doc["text"],
                index=0,
                start_char=0,
                end_char=len(doc["text"]),
                strategy="fixed",
            )
            ids = store_chunks_chromadb(
                collection, [chunk], [vector], f"evaluation/{doc['id']}.txt"
            )
            # The index may be created before a failed write; its random name was checked above.
            created_es = True
            await store_chunks_elasticsearch(es, [chunk], ids, f"evaluation/{doc['id']}.txt", name)
            source_by_chunk[ids[0]] = doc["id"]
        reranker = Reranker()
        for query in fixture["queries"]:
            row = {key: query[key] for key in ("id", "question", "relevant")}
            row["category"] = query.get("category")
            row["stages"] = {}
            output["rows"].append(row)
            started = time.monotonic()
            try:
                vector = (await embed_texts([query["question"]]))[0]
                dense, bm25, fused = await hybrid_search(
                    collection,
                    es,
                    query["question"],
                    vector,
                    k=k,
                    es_index=name,
                    rrf_k=settings.rrf_k,
                )
                for stage, hits in (("vector", dense), ("bm25", bm25), ("rrf", fused)):
                    row["stages"][stage] = {
                        "ranking": [source_by_chunk[hit.id] for hit in hits],
                        "chunk_ids": [hit.id for hit in hits],
                    }
                ranked = await reranker.rerank(query["question"], fused[:k], top_k=k)
                row["stages"]["rerank"] = {
                    "ranking": [source_by_chunk[hit.id] for hit in ranked],
                    "chunk_ids": [hit.id for hit in ranked],
                }
            except (
                Exception
            ) as exc:  # Retain individual failures, without provider messages or credentials.
                failed = True
                row["error_type"] = type(exc).__name__
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    finally:
        try:
            if created_chroma:
                chroma.delete_collection(name)
        finally:
            try:
                if created_es and await es.indices.exists(index=name):
                    await es.indices.delete(index=name)
            finally:
                await es.close()
    output["metrics"] = aggregate(output["rows"])
    return not failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chroma-port", type=int, required=True)
    parser.add_argument("--es-port", type=int, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.k < 5 or not 1 <= args.chroma_port <= 65535 or not 1 <= args.es_port <= 65535:
        parser.error("k must be >= 5 and ports must be in 1..65535")
    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    validate_fixture(fixture)
    output = {
        "fixture": str(args.fixture),
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "scope": (
            "synthetic author-labeled source passages; no parsing, chunking, "
            "generation or human evaluation"
        ),
        "planned": len(fixture["queries"]),
    }
    # Reserve output before creating stores; preserve partial evidence on failure.
    with args.output.open("x", encoding="utf-8") as stream:
        ok = False
        try:
            ok = asyncio.run(execute(fixture, output, args.chroma_port, args.es_port, args.k))
        except Exception as exc:
            output["fatal_error_type"] = type(exc).__name__
            print(f"Evaluation failed: {type(exc).__name__}", file=sys.stderr)
        finally:
            json.dump(output, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
