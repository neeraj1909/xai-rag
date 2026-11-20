# Live retrieval evaluation

This runner evaluates real vector, BM25, RRF and reranking stages over a labeled
fixture in disposable Chroma and Elasticsearch stores. It adapts the retrieval
portion of the historical holdout runner to current project APIs. The original
runner and measurements remain in the interview archive. This evaluation indexes
one supplied passage per source; it does not exercise parsing, chunking, answer
generation, guardrails or human review. Fixture labels are author-created, not a
production-quality benchmark.

The CLI requires explicit loopback service ports and an output path that does not
already exist. It creates randomized index names, rejects an existing name, and
deletes only its own indexes in `finally`; provision dedicated disposable servers
because abrupt termination can leave those indexes behind. It writes partial rows
and a fatal error type if execution fails and returns nonzero if any case fails.
Each stage reports scored and failed positive-label denominators separately from
cases with no labeled relevant source. Such cases remain in the per-case results,
but are not treated as retrieval misses.

For example, start dedicated local Chroma 1.5.9 and Elasticsearch 8.17.0
instances on loopback, discover their ports, then run from the repository root:

```bash
uv run --no-sync python evals/live_retrieval/runner.py \
  --fixture evals/archive/interview-evidence/evaluation/holdout-cases.json \
  --chroma-port <chroma-port> --es-port <elasticsearch-port> \
  --output /tmp/xai-retrieval-new.json
```

The runner uses the project-configured embedding and reranker models and pinned
revisions; their first use may download model weights. Record the resolved model
revisions and source/fixture hashes from the output. For a cached offline run,
set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Close and remove the two disposable
services after the run. Never point this evaluation at a shared service if a
randomly named test index is not permitted there.

Run the offline behavior checks with:

```bash
uv run --no-sync pytest -q evals/live_retrieval/test_live_retrieval.py
```
