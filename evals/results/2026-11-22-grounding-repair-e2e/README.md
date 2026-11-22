# Grounding repair Docker E2E, 22 November 2026

The isolated candidate-image end-to-end test passed: **1 passed, 347 deselected, one Chroma deprecation warning, 329.34 seconds** ([pytest output](pytest.log)). It ran actual HTTP ingestion and replay, retrieval, NLI verification, one grounded-answer repair, citations, streamed response and four concurrent queries. The unchanged safety gate and E2E assertions verify that the repaired response reached `verifier_passed`, contained the expected cited claim, included repair timings, and accounted for both generation calls. This is a synthetic local validation: generation and the grounding judge are deterministic stubs. The stub judge accepts claims and does **not** validate real judge safety, quality or availability.

The Docker image `xai-rag:grounding-repair` had ID `sha256:89263ce22320efaad9f2c46d12fbc78a5dce175131fd1268f5e86287d9b91769` and revision label `unknown`. Source was the working tree on Git HEAD `52ea91b878ebd254f22b19555835932613b63197` with uncommitted grounding-repair changes. The installed image files `service.py` and `generation/generator.py` matched the corresponding checkout hashes `5604c3f8ccfdadd2e4b18db6f3f69173152720f3946e643cb605192889c9ca38` and `41148ac8e898feeda4eee1f2abe2d21d6203762e2deaf595aa8dfe95aebd3990`; this comparison covers those two files, not every image file. Stub SHA-256 was `d900628ce8dd90407948e0317e9bafd8103425b51009f813fb99f29b7464a2f4` and E2E test SHA-256 was `642c09a81ef1251c697e4c9f76ea87960c9a99236b37f84abd7e4a97c383c192`.

The isolated compose project `xai-repair-20261122` used fresh volumes and loopback ports 18121 (Chroma), 19221 (Elasticsearch), and 18123 (API). [Override](compose-override.yml) sets those ports, two CPU threads and an offline read-only Hugging Face cache; it mounts the current deterministic LLM stub but does not mount application source. Absolute paths are machine-specific. The reranker was CI's pinned `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. The NLI model was `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`.

Commands from the project root (the test key is synthetic):

```sh
XAI_RAG_IMAGE=xai-rag:grounding-repair XAI_RAG_HOST_PORT=18123 \
  docker compose -p xai-repair-20261122 -f docker-compose.yml \
  -f docker-compose.e2e.yml -f /tmp/xai-repair-e2e-override.yml \
  up -d --no-build --pull never
XAI_RAG_RUN_E2E=1 XAI_RAG_E2E_BASE_URL=http://127.0.0.1:18123 \
  XAI_RAG_E2E_API_KEY=e2e-only-api-key-with-at-least-32-characters \
  uv run --no-sync pytest -m e2e -q -s
XAI_RAG_IMAGE=xai-rag:grounding-repair XAI_RAG_HOST_PORT=18123 \
  docker compose -p xai-repair-20261122 -f docker-compose.yml \
  -f docker-compose.e2e.yml -f /tmp/xai-repair-e2e-override.yml \
  down --volumes --remove-orphans
```

The [response](query-response.json) was captured in a separate repeat of the test query after the passing test, and is not the test's captured response. It shows `verifier_passed`, the shorter cited claim, two repair timing entries, and total generation usage of 64 input and 32 output tokens. The previous [fresh-image run](../2026-11-22-fresh-image-e2e/README.md) failed at an empty-sources assertion before this repair. The model-cache mount emitted a permission warning for an optional tree-cache file; model weights loaded and the test passed.

Cleanup returned successfully; container and volume listings filtered for this compose project were empty. The existing `xai-rag-local` API and Elasticsearch were healthy and its Chroma container remained running after cleanup. No live model API was called by this stack.
