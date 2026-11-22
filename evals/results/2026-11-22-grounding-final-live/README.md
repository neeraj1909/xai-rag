# Final local deployment — 22 November 2026

The existing `xai-rag-local` project was updated by recreating **only** its API
container with the E2E-tested `xai-rag:grounding-repair-final` image, ID
`sha256:b482d389cbdcd908124d7404b7a1bc6a78d87977141b1bade01ee798242f8013`.
Its installed Python tree SHA-256 is
`769a60ea6346231a988c1e04a31920ff5ab61346755921589970e605998c84d8`.
The Chroma and Elasticsearch containers and persistent named volumes were
retained; the previous `xai-rag:grounding-repair` image remains available for
rollback. No application source was mounted into the container.

After the update, `/ready` reported both stores healthy. Reingesting the
existing sample document returned HTTP 200, status `success`, 134 chunks and
consistent Chroma/Elasticsearch parity ([recorded result](reingest.json)),
matching the earlier deployment. A real `gpt-4o-mini` generation and separate
`gpt-4o-mini` grounding-judge request for a sample query returned HTTP 200,
`verifier_passed`, two cited sources and two supported local NLI verdicts in
approximately 77.7 seconds ([sanitized result](query.json)). Generation usage
was 733 input and 415 output tokens; the API's counters exclude the judge.

The result's stage timings contain `generate`, `faithfulness` and
`grounding_judge` but no `generation_repair` or `faithfulness_repair`, so this
live query did not exercise the grounding-triggered regeneration path. The
separate deterministic E2E exercises that path; this one sample is not an
independent assessment of quality or safety.

The API and Swagger are available on loopback at `http://127.0.0.1:8000` and
`http://127.0.0.1:8000/docs`. The machine-local
`$HOME/.config/xai-rag/local-compose.yml` override selects the final image and
supplies the read-only host Hugging Face cache, offline mode, two OMP/MKL
threads, both model names, a 300-second query timeout and local docs. The local
API key is in `$HOME/.config/xai-rag/local-api-key` (mode 0600). The existing
`.env` supplies the OpenAI key and was not changed. Neither credential value is
included in this evidence.

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
and volumes. Avoid `down --volumes` if the persistent data should be retained.
The immediately previous override is saved at
`$HOME/.config/xai-rag/local-compose.pre-grounding-final.yml`; the prior image
is tagged `xai-rag:grounding-repair`. The earlier
[repair deployment](../2026-11-22-grounding-repair-live/README.md) is retained
as a separate run.

The system still relies on fallible local NLI and a model judge. This successful
sample does not prove the absence of false acceptance, prompt injection, or
other grounding failures; consult the controlled probes and E2E records for
their actual coverage and limitations.
