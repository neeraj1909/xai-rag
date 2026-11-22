# Isolated Docker E2E failure, 22 November 2026

This is a failed local validation attempt, not a successful deployment or a live model API evaluation. The source was Git revision `52ea91b878ebd254f22b19555835932613b63197`; `src/xai_rag` and `tests/e2e` were unmodified in the working tree. A rebuild of the current Dockerfile stalled resolving the Python base-image metadata. The fallback used existing image `xai-rag:validation` (`sha256:c3ee2bcbbd31fd207a48f0783f8f718506717aefcb2ab3bf5667bf4e058b859a`, image revision label `e7c4b2a544810657b001b06e279a0f2eeb20e328-working-tree-reviewed`) for its dependencies, with the current application and deterministic LLM stub mounted read-only. Therefore this does **not** validate a freshly built image or prove its dependencies match the current lockfile.

The isolated compose project was `xai-eval-20261122`, with fresh volumes and loopback ports 8100, 9200 and 18097. The override used local read-only Hugging Face cache mounts, offline mode, two CPU threads, and CI's reranker `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Embedding was `BAAI/bge-small-en-v1.5` revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; NLI was `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`, with entailment and contradiction thresholds 0.5. The exact local override is retained for inspection; its absolute paths are machine-specific and require adjustment before reuse. The reranker download initially failed with a proxy SSL EOF and succeeded with proxies unset only for that download process.

Commands, from the project root (the E2E key is a synthetic test value):

```sh
XAI_RAG_IMAGE=xai-rag:validation XAI_RAG_HOST_PORT=18097 \
  docker compose -p xai-eval-20261122 -f docker-compose.yml \
  -f docker-compose.e2e.yml -f /tmp/xai-eval-override.yml up -d --no-build
XAI_RAG_RUN_E2E=1 XAI_RAG_E2E_BASE_URL=http://127.0.0.1:18097 \
  XAI_RAG_E2E_API_KEY=e2e-only-api-key-with-at-least-32-characters \
  uv run --no-sync pytest -m e2e -q -s
XAI_RAG_IMAGE=xai-rag:validation XAI_RAG_HOST_PORT=18097 \
  docker compose -p xai-eval-20261122 -f docker-compose.yml \
  -f docker-compose.e2e.yml -f /tmp/xai-eval-override.yml down --volumes --remove-orphans
```

The stack's readiness check passed. The test ingested and replayed `sample_docs/rag-interview-notes.md` with matching chunk counts and reported index parity. The subsequent HTTP query returned 200 but `sources` was empty, failing `tests/e2e/test_http_workflow.py:68` (1 failed, 331 deselected; [pytest log](pytest-failure.log)). The [captured response](query-response.json) reports `grounding_status=insufficient_evidence` and includes two retrieval explanations. It was recorded in a separate repeat of the same query after the failed test, not captured from the assertion itself.

A separate offline host invocation of the configured `FaithfulnessChecker._score_pair` used the full source text of selected Elasticsearch chunk `83f1028362e75b6f85a2b65bc7326518b6e71b7a49deda9d7b36eed91ce11e21` and its exact quoted sentence, both derived from the ingested sample document. It returned contradiction 0.8765 and entailment 0.0443 ([probe](nli-probe.txt)); this is consistent with the verifier withholding the response. This independent probe does not directly capture the verdict from inside the API request. An initial run with cached `BAAI/bge-reranker-v2-m3` also failed at the same assertion; the CI-reranker rerun above establishes that changing to the pinned test reranker did not remove the failure.

Cleanup completed successfully. Subsequent `docker ps -a` and `docker volume ls` filtered by the compose project returned no resources. Other projects were not stopped or removed. No live API calls were used by this E2E stack.
