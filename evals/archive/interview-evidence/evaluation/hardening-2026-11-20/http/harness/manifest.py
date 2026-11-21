"""Write non-secret source, model and library revisions for a synthetic run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import xai_rag
from xai_rag.config import settings

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = Path(xai_rag.__file__).parent
h = hashlib.sha256()
for path in sorted(source.rglob("*.py")):
    h.update(str(path.relative_to(source)).encode())
    h.update(b"\0")
    h.update(hashlib.sha256(path.read_bytes()).digest())
repo = source.parent.parent
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
packages = {}
for name in ("openai", "httpx", "ragas", "chromadb-client", "elasticsearch", "sentence-transformers"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
names = ("embedding_model", "embedding_model_revision", "reranker_model", "reranker_model_revision",
         "nli_model", "nli_model_revision", "nli_entailment_threshold", "nli_contradiction_threshold",
         "llm_model", "grounding_judge_model", "ragas_llm_model", "ragas_embedding_model",
         "vector_search_k", "bm25_search_k", "reranker_candidate_k", "rrf_k", "chunk_size", "chunk_overlap",
         "llm_context_max_chars", "query_timeout_seconds")
manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "git_head": commit,
            "source_tree_sha256": h.hexdigest(), "python": platform.python_version(),
            "packages": packages, "settings": {name: getattr(settings, name, None) for name in names}}
fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as out:
    json.dump(manifest, out, indent=2)
    out.write("\n")
print(f"Wrote non-secret run manifest: source sha256 {manifest['source_tree_sha256']}")
