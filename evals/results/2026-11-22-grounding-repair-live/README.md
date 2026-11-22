# Repaired local deployment — 22 November 2026

The existing `xai-rag-local` project was updated by recreating **only** its API
container with the separately built `xai-rag:grounding-repair` image, ID
`sha256:89263ce22320efaad9f2c46d12fbc78a5dce175131fd1268f5e86287d9b91769`.
Its installed Python tree SHA-256 is
`cb27effc4504eb90961e3946ca1c969a54e2f39b8b78c5caa0e79dda19133d9c`.
The Chroma and Elasticsearch containers and named data volumes were retained;
the previous `xai-rag:local-validation-build` image remains available for
rollback. No application source was mounted into the container.

After the update, `/ready` reported both stores healthy. Reingesting the
existing sample document returned HTTP 200, status `success`, 134 chunks and
consistent Chroma/Elasticsearch parity ([recorded result](reingest.json)),
matching the previous deployment's 134 chunks. The API then sent a real
`gpt-4o-mini` generation and separate `gpt-4o-mini` grounding-judge request. A
single sample query returned HTTP 200, `verifier_passed`, two cited sources and
two supported local NLI verdicts in approximately 74.3 seconds
([sanitized result](query.json)). Generation usage was 733 input and 414 output
tokens; the API's counters exclude the judge. This is one sample, not an
independent quality or safety evaluation. The captured response subset did not
include stage-latency keys, so it does **not** establish whether this particular
query triggered answer regeneration. The separate deterministic E2E exercises
the retry path; do not infer the live-query path from the final answer alone.

The API and `/docs` are available on loopback at `http://127.0.0.1:8000`. The
machine-local `$HOME/.config/xai-rag/local-compose.yml` override selects the
repair image and supplies the read-only host Hugging Face cache, offline mode,
two OMP/MKL threads, both model names, 300-second query timeout and local docs.
The separate local API key is in `$HOME/.config/xai-rag/local-api-key` (mode
0600). The existing `.env` supplies the OpenAI key and was not changed; no
credential values are included in this evidence.

From the repository root, recreate the project with the current configuration
and persistent named volumes using:

```bash
XAI_RAG_API_KEY="$(<"$HOME/.config/xai-rag/local-api-key")" \
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy XAI_RAG_HOST_PORT=8000 \
    docker compose -p xai-rag-local -f docker-compose.yml \
    -f "$HOME/.config/xai-rag/local-compose.yml" up -d --no-build
```

`docker compose -p xai-rag-local stop` and `start` keep the existing containers
and volumes. Avoid `down --volumes` unless deletion of the persistent data is
intended. The previous machine-local override was preserved as
`$HOME/.config/xai-rag/local-compose.pre-grounding-repair.yml`, and the old
image remains tagged `xai-rag:local-validation-build`; rollback requires the
previous override and the same private API key. The old
[deployment evidence](../2026-11-22-local-deployment/README.md) remains separate.

The system still relies on fallible local NLI and a model judge. This successful
sample does not prove the absence of false acceptance, prompt injection, or
other grounding failures; consult the associated controlled probes and full E2E
records for their actual coverage and limitations.
