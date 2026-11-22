# Final grounding image Docker E2E, 22 November 2026

The full isolated E2E test passed against final image `xai-rag:grounding-repair-final`: **1 passed, 348 deselected, one Chroma deprecation warning, 320.84 seconds** ([pytest output](pytest.log)). It exercised HTTP ingestion/replay, retrieval, real NLI verification, grounded-answer repair, citation and token accounting, streaming, and four concurrent queries. Generation and the grounding judge used deterministic stubs; this is not evidence of live judge safety, quality or availability.

Image ID was `sha256:b482d389cbdcd908124d7404b7a1bc6a78d87977141b1bade01ee798242f8013`; its revision label was `unknown`. The checkout was Git HEAD `52ea91b878ebd254f22b19555835932613b63197` with uncommitted changes. Image and checkout hashes matched for `service.py` (`5604c3f8ccfdadd2e4b18db6f3f69173152720f3946e643cb605192889c9ca38`), `generation/generator.py` (`41148ac8e898feeda4eee1f2abe2d21d6203762e2deaf595aa8dfe95aebd3990`), and `explainability/faithfulness.py` (`fff00707c5e72a487ad90e9d6bfb595e5c1501467b86a4f94b2894549feab1cc`); these three checks do not verify every image file. The mounted stub SHA-256 was `d900628ce8dd90407948e0317e9bafd8103425b51009f813fb99f29b7464a2f4` and test SHA-256 was `642c09a81ef1251c697e4c9f76ea87960c9a99236b37f84abd7e4a97c383c192`.

Compose project `xai-final-20261122` used fresh volumes and loopback host ports 18121, 19221 and 18123. Its [override](compose-override.yml) mounted the current deterministic stub and offline Hugging Face cache read-only and limited CPU threads; it did not mount application source. The override contains machine-specific paths. NLI used `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`; reranking used `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`.

Commands, from the project root (the E2E key is synthetic):

```sh
XAI_RAG_IMAGE=xai-rag:grounding-repair-final XAI_RAG_HOST_PORT=18123 \
  docker compose -p xai-final-20261122 -f docker-compose.yml \
  -f docker-compose.e2e.yml -f /tmp/xai-repair-e2e-override.yml \
  up -d --no-build --pull never
XAI_RAG_RUN_E2E=1 XAI_RAG_E2E_BASE_URL=http://127.0.0.1:18123 \
  XAI_RAG_E2E_API_KEY=e2e-only-api-key-with-at-least-32-characters \
  uv run --no-sync pytest -m e2e -q -s
XAI_RAG_IMAGE=xai-rag:grounding-repair-final XAI_RAG_HOST_PORT=18123 \
  docker compose -p xai-final-20261122 -f docker-compose.yml \
  -f docker-compose.e2e.yml -f /tmp/xai-repair-e2e-override.yml \
  down --volumes --remove-orphans
```

Cleanup succeeded; no containers or volumes remained for this project. Existing `xai-rag-local` API and Elasticsearch remained healthy and its Chroma container remained running. This run follows the [candidate-image passing run](../2026-11-22-grounding-repair-e2e/README.md) and includes the later NLI cold-load lock.
