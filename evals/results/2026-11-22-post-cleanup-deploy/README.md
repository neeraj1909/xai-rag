# Post-cleanup deployment validation — 22 November 2026

The local image `xai-rag:post-cleanup-20261122` was built from the project Dockerfile and locked dependencies after removal of the unused private `_split_sentences` helper. Its image ID is `sha256:f28e1c15e6e6d66257b77763737e980b484d4e63ca6498518365c6be603271f1`. The source and installed `ingestion/chunker.py` SHA-256 both equal `cef2332c4629a67ac9f8ebb21c50ba34d1663bbcc3ba1c0451095c76e2e03da5`. This checks that file, not every file in the image.

A separate Compose project, `xai-post-cleanup-20261122`, used the [override](compose-override.yml), fresh volumes, loopback ports, cached models and the new image without a source bind. The full E2E passed: **1 passed, 348 deselected, one Chroma deprecation warning, 354.09 seconds** ([pytest output](pytest.log)). It covers ingest/replay, retrieval, real NLI, repair, citations, streaming and concurrent queries; generation and judge were deterministic stubs. It does not establish live judge reliability. The project's containers and volumes were removed after the test.

Only the existing `xai-rag-local` API container was recreated using the new image. Existing Chroma/Elasticsearch containers and volumes were preserved. `/ready` returned healthy for both stores. Sample reingestion returned HTTP 200, 134 chunks and consistent parity ([result](reingest.json)).

Two real OpenAI-backed queries using `gpt-4o-mini` for generation and the separate grounding judge returned HTTP 200, `verifier_passed`, and two cited sources. The first took 112.2 seconds cold; its optional per-claim faithfulness report was not requested ([result](query.json)). The second explicitly requested that report, returned three `supported` NLI verdicts, and took 105.8 seconds ([result](query-faithfulness.json)). Neither request invoked the repair stage; the isolated E2E exercises it. One successful sample does not establish production quality or security. Generation token counters do not include judge tokens.

The prior `xai-rag:grounding-repair-final` image and the private `~/.config/xai-rag/local-compose.pre-post-cleanup-20261122.yml` override were retained for rollback. The existing `.env`, local key, and persistent stores were not changed. No credentials are included here.

The following commands show the test boundary (the E2E key is synthetic):

```sh
XAI_RAG_IMAGE=xai-rag:post-cleanup-20261122 XAI_RAG_HOST_PORT=18123 \
  docker compose -p xai-post-cleanup-20261122 -f docker-compose.yml \
  -f docker-compose.e2e.yml -f evals/results/2026-11-22-post-cleanup-deploy/compose-override.yml \
  up -d --no-build --pull never
XAI_RAG_RUN_E2E=1 XAI_RAG_E2E_BASE_URL=http://127.0.0.1:18123 \
  XAI_RAG_E2E_API_KEY=e2e-only-api-key-with-at-least-32-characters \
  uv run --no-sync pytest -m e2e -q -s
XAI_RAG_IMAGE=xai-rag:post-cleanup-20261122 XAI_RAG_HOST_PORT=18123 \
  docker compose -p xai-post-cleanup-20261122 -f docker-compose.yml \
  -f docker-compose.e2e.yml -f evals/results/2026-11-22-post-cleanup-deploy/compose-override.yml \
  down --volumes --remove-orphans
```

The `down --volumes` command applies **only** to this disposable project, never to `xai-rag-local`.
