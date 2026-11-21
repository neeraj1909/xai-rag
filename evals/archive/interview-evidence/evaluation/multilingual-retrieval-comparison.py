import argparse
import gc
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer

p = argparse.ArgumentParser()
p.add_argument("--fixture", required=True)
p.add_argument("--candidate", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()
fixture = Path(args.fixture)
data = json.loads(fixture.read_text())
docs = data["documents"]
queries = data["queries"]
models = [
    ("bge", "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
    ("multilingual_minilm", args.candidate, "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"),
]
results = {}
for name, location, revision in models:
    start = time.monotonic()
    model = SentenceTransformer(
        location,
        revision=revision,
        device="cpu",
        trust_remote_code=False,
        local_files_only=True,
    )
    load = time.monotonic() - start
    start = time.monotonic()
    dv = model.encode(
        [x["text"] for x in docs],
        batch_size=8,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    qv = model.encode(
        [x["question"] for x in queries],
        batch_size=8,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    seconds = time.monotonic() - start
    sim = qv @ dv.T
    rows = []
    for q, scores in zip(queries, sim, strict=True):
        order = sorted(
            range(len(docs)), key=lambda i: (-float(scores[i]), docs[i]["id"])
        )
        ranking = [
            {"id": docs[i]["id"], "cosine": round(float(scores[i]), 8)} for i in order
        ]
        ranks = {
            source: next(i + 1 for i, x in enumerate(ranking) if x["id"] == source)
            for source in q["relevant"]
        }
        rows.append(
            {
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "relevant": q["relevant"],
                "relevant_ranks": ranks,
                "ranking": ranking,
            }
        )
    results[name] = {
        "model": location
        if name == "bge"
        else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "revision": revision,
        "dimension": model.get_sentence_embedding_dimension(),
        "load_seconds": round(load, 3),
        "encoding_seconds": round(seconds, 3),
        "rows": rows,
    }
    del model, dv, qv, sim
    gc.collect()
output = {
    "fixture": str(fixture),
    "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
    "fixture_description": data["description"],
    "method": "Exact dot-product of L2-normalized embeddings; no query prefix, matching source embed_texts; single full document per source; all 31 ranked; no reranking or generation. Ties broken by source id.",
    "environment": {
        "python": platform.python_version(),
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    },
    "models": results,
}
Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
for name, res in results.items():
    print(
        name,
        "load seconds",
        res["load_seconds"],
        "encode seconds",
        res["encoding_seconds"],
    )
    for r in res["rows"]:
        print(r["id"], r["relevant_ranks"], r["ranking"][0]["id"])
