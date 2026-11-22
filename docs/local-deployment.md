# Local deployment with OpenAI

**Local status (22 November 2026):** the `xai-rag-local` Compose project is
running at <http://127.0.0.1:8000>, with local API documentation at
<http://127.0.0.1:8000/docs>. Ingestion and replay retained 134 chunks with
index parity. After the final repair image was deployed, a real OpenAI-backed
query returned `verifier_passed`, two supported claims and two cited sources;
it took about 78 seconds cold and did not invoke the retry. A separate isolated
E2E test of that image passed in 321 seconds with real NLI and deterministic
generator and judge stubs; it exercised the retry path. The NLI
model's original false contradiction remains
reproducible; the candidate retries generation once after a valid NLI rejection
and requires the new claims to pass the same verifier and judge. Neither test
establishes general reliability or judge quality.

The [sanitized live result](../evals/results/2026-11-22-grounding-final-live/)
records the deployment probe and its scope.

This guide runs the API, Chroma and Elasticsearch on this machine. Compose binds
their published ports to loopback; Elasticsearch and Chroma are not configured
for an internet-facing deployment. The API uses the real OpenAI endpoint for
generation and for a separate grounding judge. These calls use API credits; do
not send documents you do not want the configured provider to receive.

On an ordinary, well-formed NLI rejection the current application can make one
additional generation request using the same evidence. Any retry still needs
valid citations, an NLI pass and an independent judge pass; it may be refused,
and provider errors still result in an unavailable response. The retry can add
latency and API usage. It does not correct the NLI model's false verdict or
guarantee an answer.

## Existing local project

The running project uses the image `xai-rag:grounding-repair-final` and the
private Compose override `$HOME/.config/xai-rag/local-compose.yml`. The override
selects `gpt-4o-mini` for both generation and judging, enables local `/docs`,
sets a 300-second query deadline, uses two CPU threads and mounts the existing
Hugging Face model cache read-only in offline mode. The local API key is stored
in `$HOME/.config/xai-rag/local-api-key` with mode `0600`. The existing `.env`
supplies the OpenAI API key; it was not changed during deployment. These
machine-specific files are required to reproduce the verified configuration.

To inspect and access the running stack from the repository root:

```bash
docker compose -p xai-rag-local ps
curl --fail-with-body http://127.0.0.1:8000/ready
```

The API and both stores are bound to loopback. `/ready` checks the stores; it
does not independently test the provider or guarantee a useful answer.

To recreate the containers while retaining their named volumes, use the same
override and local API key. The override explicitly selects the image. The shell substitution reads the key from
the private file without printing it; the proxy variables are removed only for
this Compose process.

```bash
XAI_RAG_API_KEY="$(<"$HOME/.config/xai-rag/local-api-key")" \
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      -u http_proxy -u https_proxy -u all_proxy \
      XAI_RAG_HOST_PORT=8000 \
      docker compose -p xai-rag-local \
      -f docker-compose.yml -f "$HOME/.config/xai-rag/local-compose.yml" \
      up -d --no-build
```

The prebuilt image must be rebuilt after application source changes; a rebuild
and full E2E test remain necessary to validate any future repair. Do not run a
plain `up --build` for this project and assume the private override, image and
local API key are preserved.

The private override selects the image used at recreation. The final-image
deployment and its local stores were healthy during the recorded live check.

## Prepare configuration for a separate installation

Run commands from the repository root. Install Docker with Compose and make
sure ports 8000, 8100 and 9200 are available. Allow sufficient free disk space
for Elasticsearch's normal disk watermarks, images, data and downloaded model
weights. The API has a default 8 GB memory limit and the local models can be
slow to load on CPU. First use may download pinned Hugging Face models.

If you do not already have `.env`, create it from `.env.example`; **do not
overwrite an existing file**. Edit `.env` privately and ensure it contains:

```dotenv
XAI_RAG_LLM_PROVIDER=openai
XAI_RAG_LLM_MODEL=gpt-4o-mini
XAI_RAG_GROUNDING_JUDGE_PROVIDER=openai
XAI_RAG_GROUNDING_JUDGE_MODEL=gpt-4o-mini
XAI_RAG_OPENAI_API_KEY=<your-openai-api-key>
XAI_RAG_API_KEY=<a-different-random-key-at-least-32-characters>
```

Generate a local API key with `openssl rand -hex 32`; keep it distinct from
your OpenAI credential. Select models your account can access and that support
the project's structured response format. The OpenAI key and explicit judge
model are both required: the judge fails closed when unconfigured. The example
above uses the model that passed a small direct generator-and-judge probe; that
probe did not establish the complete application workflow or judge accuracy.
Keep credentials out of shell history, logs, screenshots and commits. Compose
reads `.env` for interpolation, and exported shell variables take precedence.
`docker compose config` prints resolved values, so use `--quiet` when checking
configuration with secrets present.

## Build and start a separate installation

```bash
docker compose -p xai-rag-example config --quiet
docker compose -p xai-rag-example up -d --build
docker compose -p xai-rag-example ps
curl --fail-with-body http://127.0.0.1:8000/ready
```

These commands use the same default host ports as the existing project, so
stop it first or arrange non-conflicting service port mappings. They describe a
separate setup and have not been validated as a live workflow. If
`XAI_RAG_HOST_PORT` in your configuration differs from 8000, use that port
for the API commands. `/ready` checks the local search dependencies; it does not
test OpenAI authentication or generation. The API can be ready while its first
model-backed query is still slow or while the configured provider is unavailable.

The inherited proxy settings on this machine previously timed out on a direct
OpenAI transport probe. To test the Docker build route without changing any
system-wide settings, run just that command with proxy variables removed:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  docker compose -p xai-rag-example build
```

This affects only the launched command and its descendants. Container runtime
proxy configuration is separate; inspect your Docker configuration if a running
API still cannot reach its provider. Do not assume an authentication error is a
proxy problem without checking the endpoint and key separately.

## Ingest and query

The Compose file mounts the repository's `sample_docs/` read-only at
`/app/sample_docs` in the container. Add the files you want indexed there; the
example uses the existing interview notes. The API key used below is your
**local** `XAI_RAG_API_KEY`, not the OpenAI key. For the existing deployment,
read it from the private file without printing it:

```bash
XAI_RAG_LOCAL_KEY="$(<"$HOME/.config/xai-rag/local-api-key")"

curl --fail-with-body -X POST \
  -H "X-API-Key: $XAI_RAG_LOCAL_KEY" \
  --get --data-urlencode 'source_path=/app/sample_docs/rag-interview-notes.md' \
  --data-urlencode 'strategy=fixed' \
  http://127.0.0.1:8000/ingest

curl --fail-with-body -X POST http://127.0.0.1:8000/query \
  -H "X-API-Key: $XAI_RAG_LOCAL_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does the document explain about retrieval-augmented generation?","top_k":2,"include_explanations":true}'

unset XAI_RAG_LOCAL_KEY
```

The response should be examined, not judged by HTTP status alone: check
`grounding_status`, `answer`, `sources` and retrieval explanations. The API can
return HTTP 200 while withholding an answer. Both the local NLI check and the
configured judge must pass; `insufficient_evidence` or
`verification_unavailable` is not a verified answer. Scores and model judgments
are fallible and are not a guarantee of truth or safety.

## Operate and stop

```bash
docker compose -p xai-rag-local logs --tail=100 api
docker compose -p xai-rag-local stop
docker compose -p xai-rag-local start
docker compose -p xai-rag-local down
```

`stop` and `down` without `--volumes` preserve the project's named Chroma,
Elasticsearch and model-cache volumes. `down` removes containers and the
project network; recreate the existing project using the complete command
above so it also restores the private override and keys. Do not run
`down --volumes` unless you intend to delete that project's indexed
data and model cache. Treat application logs and volume contents as potentially
sensitive. Back up data and verify recovery before relying on this local setup.

For automated HTTP checks, the separate `docker-compose.e2e.yml` overlay uses
deterministic generator and judge stubs and disposable test stores. A pass there
tests the local application boundary but does not validate real OpenAI access.
Conversely, a successful direct API probe does not validate ingestion, retrieval,
grounding or persistence in the deployed stack.
