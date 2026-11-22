# Fresh-image Docker E2E failure, 22 November 2026

This is a failed isolated end-to-end validation, not a live-provider evaluation. It repeated the unchanged `tests/e2e/test_http_workflow.py` against the freshly built original project image `xai-rag:local-validation-build` (ID `sha256:8ee8896b002cb4695be16dd44f60335b1a47d68edf361fc802fe0cb40332d2c2`, revision label `unknown`). Git HEAD was `52ea91b878ebd254f22b19555835932613b63197`, and the relevant source and E2E test files were unmodified. The image's installed `xai_rag/service.py` SHA-256 `ca44c88ff1f37b4328c097a3dc1c71cfe9213fa66fda9a3d09402406cdccefd4` matched the checkout; this single-file check is not proof that every image file matches the working tree. Application source was **not** bind-mounted into this run.

The isolated compose project `xai-eval-clean-20261122` used fresh volumes and loopback host ports 18121 (Chroma), 19221 (Elasticsearch), and 18123 (API), leaving existing `xai-rag-local` ports and services alone. The [local override](compose-override.yml) sets those ports, offline local Hugging Face cache, two CPU threads, and a read-only mount of the current deterministic LLM stub. It contains machine-specific absolute paths and needs adjustment before reuse. The reranker was the CI pin `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Embedding and NLI used the configured pinned models identified in the [earlier run](../2026-11-22-docker-e2e/README.md).

Commands from the project root (the test key is synthetic):

```sh
XAI_RAG_IMAGE=xai-rag:local-validation-build XAI_RAG_HOST_PORT=18123 \
  docker compose -p xai-eval-clean-20261122 -f docker-compose.yml \
  -f docker-compose.e2e.yml -f /tmp/xai-eval-clean-build-override.yml \
  up -d --no-build --pull never
XAI_RAG_RUN_E2E=1 XAI_RAG_E2E_BASE_URL=http://127.0.0.1:18123 \
  XAI_RAG_E2E_API_KEY=e2e-only-api-key-with-at-least-32-characters \
  uv run --no-sync pytest -m e2e -q -s
XAI_RAG_IMAGE=xai-rag:local-validation-build XAI_RAG_HOST_PORT=18123 \
  docker compose -p xai-eval-clean-20261122 -f docker-compose.yml \
  -f docker-compose.e2e.yml -f /tmp/xai-eval-clean-build-override.yml \
  down --volumes --remove-orphans
```

Readiness passed; the test ingested the sample document and replayed it with consistent index parity. The subsequent query returned HTTP 200 with empty sources and failed the unchanged assertion at `tests/e2e/test_http_workflow.py:68`: 1 failed, 331 deselected, one deprecation warning, 239.75 seconds ([pytest log](pytest-failure.log)). A separate repeat of the same query returned `insufficient_evidence` with two retrieval explanations ([response](query-response.json)); this is a later probe, not the test's captured response. The earlier run separately observed the NLI checker scoring the exact sentence from the same selected Markdown chunk as contradiction 0.8765 and entailment 0.0443; that probe was outside the API and does not directly reveal the API's internal verdict. Both image configurations failed the same assertion.

Cleanup returned successfully; filtered container and volume listings for `xai-eval-clean-20261122` were empty. The pre-existing `xai-rag-local` API and Elasticsearch remained healthy and its Chroma container remained running after cleanup. The cache mount logged a permission warning for an optional model tree-cache file, while the model weights loaded; this warning does not establish the cause of the assertion failure.
