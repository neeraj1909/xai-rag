# Dense multilingual embedding comparison

This diagnostic compares two embedding models by exact cosine similarity over a
source-level fixture. It uses normalized, unprefixed documents and questions,
matching the project's embedding approach, and records all document rankings,
source-label ranks, fixture and runner hashes, requested revisions, package
versions and timings. Ties are broken by source ID. It does not run Chroma,
BM25, RRF, the reranker, generation or the application's full retrieval path.
The separate [live retrieval evaluation](../live_retrieval/README.md) exercises
that path for the currently configured embedding model.

From the repository root, select a **new** output file:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-sync python evals/multilingual/compare.py \
  --fixture evals/archive/interview-evidence/evaluation/retrieval-challenge-cases.json \
  --output /tmp/xai-multilingual-comparison.json
uv run --no-sync pytest -q evals/multilingual/test_compare.py
```

The defaults request the pinned English BGE small and multilingual MiniLM
revisions used in the historical experiment. The first run may download models;
add `--local-files-only` to require cached files. `--baseline`, `--candidate`
and their corresponding `--*-revision` options allow other models or local
snapshot paths. For a local snapshot, its requested revision is recorded but
cannot be independently verified by this runner: select and verify the snapshot
before using it. Output is created exclusively and partial results are retained
if a model fails. Timings under shared machine load are not a benchmark.

The included fixture has 31 fictional passages and 28 author-labeled queries,
two without relevant-source labels. Unlabeled queries remain in the per-case
results but are excluded from relevance metrics; this does not test abstention.
Labels have no independent human review. This small exploratory comparison is
not a production-quality benchmark and cannot justify changing the embedding
model alone. [Historical evidence](../archive/interview-evidence/multilingual-retrieval-diagnostic.md)
remains unchanged; [the new run](../results/2026-11-22-multilingual/) uses the
same fixture but is a distinct execution.
