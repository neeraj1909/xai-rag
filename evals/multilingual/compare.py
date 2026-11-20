"""Compare exact dense rankings for a labeled source-level retrieval fixture."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

BASELINE = "BAAI/bge-small-en-v1.5"
BASELINE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
CANDIDATE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CANDIDATE_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


def rank_cases(documents, queries, similarities):
    ids = [doc["id"] for doc in documents]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("document IDs must be nonempty and unique")
    if len(queries) != len(similarities):
        raise ValueError("a similarity row is required for every query")
    rows = []
    for query, scores in zip(queries, similarities, strict=True):
        if len(scores) != len(documents):
            raise ValueError("each similarity row must score every document")
        if not set(query["relevant"]) <= set(ids):
            raise ValueError("unknown relevant document ID")
        order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
        ranking = [{"id": ids[i], "cosine": round(float(scores[i]), 8)} for i in order]
        positions = {item["id"]: index for index, item in enumerate(ranking, 1)}
        rows.append(
            {
                **query,
                "relevant_ranks": {source: positions[source] for source in query["relevant"]},
                "ranking": ranking,
            }
        )
    return rows


def summarize(rows):
    labeled = [row for row in rows if row["relevant_ranks"]]

    def fraction(predicate):
        return (
            sum(predicate(row["relevant_ranks"].values()) for row in labeled) / len(labeled)
            if labeled
            else None
        )

    return {
        "planned": len(rows),
        "labeled": len(labeled),
        "unlabeled": len(rows) - len(labeled),
        "hit_at_1": fraction(lambda ranks: min(ranks) <= 1),
        "hit_at_5": fraction(lambda ranks: min(ranks) <= 5),
        "all_at_3": fraction(lambda ranks: max(ranks) <= 3),
        "all_at_5": fraction(lambda ranks: max(ranks) <= 5),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New file; never overwritten")
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--baseline-revision", default=BASELINE_REVISION)
    parser.add_argument("--candidate", default=CANDIDATE)
    parser.add_argument("--candidate-revision", default=CANDIDATE_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    source = args.fixture.read_bytes()
    fixture = json.loads(source)
    docs, queries = fixture["documents"], fixture["queries"]
    if not docs or not queries:
        parser.error("fixture needs documents and queries")
    # Reserve before loading models, and keep partial output if a model fails.
    with args.output.open("x", encoding="utf-8") as output:
        report = {
            "fixture": str(args.fixture),
            "fixture_sha256": hashlib.sha256(source).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "method": (
                "Unprefixed inputs, normalized embeddings, exact cosine, "
                "source ID tie-break; no stores or reranking"
            ),
            "environment": {
                "python": platform.python_version(),
                **{
                    name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "sentence-transformers")
                },
                "device": "cpu",
            },
            "models": {},
        }

        def save():
            output.seek(0)
            json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
            output.truncate()
            output.flush()

        save()
        from sentence_transformers import SentenceTransformer

        for name, location, revision in (
            ("baseline", args.baseline, args.baseline_revision),
            ("candidate", args.candidate, args.candidate_revision),
        ):
            result = {"model": location, "revision": revision}
            report["models"][name] = result
            try:
                start = time.monotonic()
                model = SentenceTransformer(
                    location,
                    revision=revision,
                    device="cpu",
                    trust_remote_code=False,
                    local_files_only=args.local_files_only,
                )
                result["load_seconds"] = round(time.monotonic() - start, 3)
                start = time.monotonic()
                vectors = model.encode(
                    [doc["text"] for doc in docs],
                    batch_size=8,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                q_vectors = model.encode(
                    [query["question"] for query in queries],
                    batch_size=8,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                result["encoding_seconds"] = round(time.monotonic() - start, 3)
                result["dimension"] = model.get_sentence_embedding_dimension()
                result["rows"] = rank_cases(docs, queries, q_vectors @ vectors.T)
                result["summary"] = summarize(result["rows"])
                del model, vectors, q_vectors
                gc.collect()
            except Exception as exc:
                result["error_type"] = type(exc).__name__
                save()
                raise
            save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
