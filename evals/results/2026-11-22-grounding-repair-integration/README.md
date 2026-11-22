# Isolated dual-store integration, 22 November 2026

The repository's real Chroma/Elasticsearch integration test passed: **1 passed, 348 deselected, 38 Chroma deprecation warnings, 16.26 seconds** ([pytest output](pytest.log)). It exercised persistence, replay/idempotency, index parity, and vector, BM25, and fused retrieval with deterministic vectors. It did not call a model or provider.

The test ran from the working tree at Git HEAD `52ea91b878ebd254f22b19555835932613b63197`, which included uncommitted grounding-repair changes. It used separate disposable compose project `xai-integration-20261122`, loopback ports 18121 and 19221, fresh volumes, and isolated collection and index `xai-integration-20261122`. Only Chroma and Elasticsearch were started; the image and API settings in the shared [override](compose-override.yml) were not used. The override contains machine-specific paths.

Commands from the project root:

```sh
XAI_RAG_IMAGE=xai-rag:grounding-repair XAI_RAG_HOST_PORT=18123 \
  docker compose -p xai-integration-20261122 -f docker-compose.yml \
  -f docker-compose.e2e.yml -f /tmp/xai-repair-e2e-override.yml \
  up -d --no-build --pull never chromadb elasticsearch
# Wait for Chroma /api/v2/heartbeat and Elasticsearch cluster health.
XAI_RAG_RUN_INTEGRATION=1 XAI_RAG_CHROMA_HOST=127.0.0.1 \
  XAI_RAG_CHROMA_PORT=18121 XAI_RAG_CHROMA_COLLECTION=xai-integration-20261122 \
  XAI_RAG_ELASTICSEARCH_URL=http://127.0.0.1:19221 \
  XAI_RAG_ELASTICSEARCH_INDEX=xai-integration-20261122 \
  XAI_RAG_ELASTICSEARCH_API_KEY='' uv run --no-sync pytest -m integration -q
XAI_RAG_IMAGE=xai-rag:grounding-repair XAI_RAG_HOST_PORT=18123 \
  docker compose -p xai-integration-20261122 -f docker-compose.yml \
  -f docker-compose.e2e.yml -f /tmp/xai-repair-e2e-override.yml \
  down --volumes --remove-orphans
```

The test deletes its configured index and collection in `finally`, so it must only run against disposable resources. Cleanup succeeded; project-filtered container and volume listings were empty. The existing `xai-rag-local` API and Elasticsearch remained healthy and its Chroma container remained running after cleanup.
