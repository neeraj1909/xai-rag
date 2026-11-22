# Local OpenAI-backed deployment — 22 November 2026

The original freshly built application image is running locally in the Compose
project `xai-rag-local`. Its API, Chroma and Elasticsearch ports are bound to
loopback. `/ready` reported both stores healthy and `/docs` returned HTTP 200.
The user-provided OpenAI configuration was used for real `gpt-4o-mini`
generation and the separate `gpt-4o-mini` grounding judge; no LLM stub or source
bind was used. The image ID is
`sha256:8ee8896b002cb4695be16dd44f60335b1a47d68edf361fc802fe0cb40332d2c2`.

The fixed-strategy sample ingestion returned HTTP 200, status `success`, 134
chunks and `parity_consistent: true`. A second ingestion of the same fixture
returned the same 134 chunks and consistent parity. A single live query returned HTTP 200,
`verifier_passed` and two cited source IDs with two retrieval explanations;
[the sanitized observed result](live-query.json) retains the exact query,
answer, sources and reported 97,199 ms cold-query latency. The generation usage
was 733 input and 412 output tokens; the application does not include judge usage
in those counters. This is one sampled live query, not a statistically meaningful
quality assessment or proof that the separate deterministic E2E failure is fixed.
The evidence fixture and response are project sample content.

The Compose project uses the persistent volumes `xai-rag-local_chromadata`,
`xai-rag-local_esdata` and `xai-rag-local_model_cache`. The last volume is
present but the model hub subdirectory is overlaid by the machine-local
read-only cache configured in `$HOME/.config/xai-rag/local-compose.yml`.
The override sets two OMP/MKL threads, offline cached models, `gpt-4o-mini`
generation and judging, a 300-second query timeout and local docs. The local
API authentication key is stored in `$HOME/.config/xai-rag/local-api-key` with
mode 0600; it differs from the OpenAI key in the existing `.env`, which was not
changed. The credential values are intentionally absent here.

Recreate the same project from the repository root, reusing the named volumes:

```bash
XAI_RAG_API_KEY="$(<"$HOME/.config/xai-rag/local-api-key")" \
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    XAI_RAG_IMAGE=xai-rag:local-validation-build XAI_RAG_HOST_PORT=8000 \
    docker compose -p xai-rag-local -f docker-compose.yml \
    -f "$HOME/.config/xai-rag/local-compose.yml" up -d --no-build
```

Compose reads the existing `.env` for the real provider key. `stop`/`start`
retain the current containers and configuration, while `down` without
`--volumes` preserves indexed data and cache volumes. Do not use `down --volumes`
if those data should be kept. No global proxy settings were changed.

The historical [Docker E2E failure](../2026-11-22-docker-e2e/README.md) remains
open: its deterministic generated text was withheld by the grounding pipeline.
The successful live response is a different model-produced answer and must not
be presented as demonstrating that this failure or safety limitations are fixed.
