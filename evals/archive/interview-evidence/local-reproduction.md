# XAI-RAG: run and test locally

This walkthrough is intended for the hardening version of **the current checkout**, against disposable, loopback-only Chroma and Elasticsearch containers. It was originally executed on an earlier source snapshot; the updated two-gate walkthrough with a live provider has not been independently re-run. A separate later [stub-provider E2E](evaluation/hardening-2026-11-20/final-e2e/README.md) passed on a simple source using an overlay image, while the original Markdown source still produced a false refusal. It ingests one fictional text file, asks a real OpenAI model a question, and shows how to interpret the result. Keep the commands in one terminal so the temporary names and process ID remain available. This is a local demonstration, not a production deployment.

A separate [current-source live OpenAI smoke](evaluation/hardening-2026-11-20/current-live-smoke/README.md) passed 15 checks on one fictional, scoped museum-hours scenario, using real isolated stores. It did not execute the exact manual walkthrough below, and the earlier Markdown false refusal remains an observed limitation.

## What you need

- Python **3.11–3.14**, [`uv`](https://docs.astral.sh/uv/), Docker Engine and permission to run containers; allow time and disk space for the pinned images and model downloads.
- A checkout of [XAI-RAG](https://github.com/neeraj1909/xai-rag). Commands below run from its root, not from the interview-materials repository. This guide describes the current source; a previous GitHub revision may lack later changes. Check `git rev-parse HEAD` and `git status --short` before comparing results.
- A working OpenAI API key. The generation request and mandatory judge request are live and billable. Optional RAGAS evaluation makes additional billable calls. Never commit `.env` or paste keys into issue reports. A separate, locally generated `XAI_RAG_API_KEY` protects your local API; it is **not** the OpenAI credential.
- Free loopback ports **18177** (Chroma), **19377** (Elasticsearch), and **18077** (API). Check them before starting with `ss -ltn`. Do not stop an unrelated service to make a port available.

The first load may download the embedding, reranking and mandatory NLI models. A cached-model, offline test is possible only after those exact model revisions are available. The images and models are pinned by the project; changing the embedding model requires a new compatible index, not reusing an old one.

## 1. Prepare the checkout and a fictional document

```bash
cd /path/to/xai-rag
uv sync --locked --extra dev
# Keep an existing .env intact. If none exists, copy the template once.
test -e .env || cp .env.example .env
```

In `.env`, set `XAI_RAG_OPENAI_API_KEY` using your editor. Leave the existing value alone if already configured. The commands below set other settings **only in this terminal**; the shell environment takes precedence over `.env`. They generate a temporary API key and a new document directory. The OpenAI key is read by the application from `.env`; do not print it.

```bash
export XAI_RAG_API_KEY="$(.venv/bin/python -c 'import secrets; print(secrets.token_hex(32))')"
export XAI_RAG_LLM_BASE_URL=https://api.openai.com/v1
# Set explicitly even if an existing .env predates the hardening change.
# This sends bounded evidence in a second, billable provider request.
export XAI_RAG_GROUNDING_JUDGE_PROVIDER=openai
export XAI_RAG_GROUNDING_JUDGE_MODEL=gpt-6-luna
export XAI_RAG_CHROMA_HOST=127.0.0.1
export XAI_RAG_CHROMA_PORT=18177
export XAI_RAG_ELASTICSEARCH_URL=http://127.0.0.1:19377
export XAI_RAG_ELASTICSEARCH_API_KEY=''
export XAI_RAG_INGEST_ROOT="$(mktemp -d)"
export XAI_RAG_OTEL_ENDPOINT=''
export XAI_RAG_EXPOSE_DOCS=true
export XAI_RAG_ALLOW_REQUEST_EVALUATION=false
XAI_REPRO_SUFFIX="$(date +%s)-$$"
XAI_REPRO_CHROMA="xai-repro-chroma-${XAI_REPRO_SUFFIX}"
XAI_REPRO_ES="xai-repro-es-${XAI_REPRO_SUFFIX}"
printf '%s\n' 'Customers may return an unused product within 30 days after delivery.' > "$XAI_RAG_INGEST_ROOT/returns.txt"
```

Use the official endpoint only with a credential intended for it. A compatible custom provider can instead be configured through `XAI_RAG_LLM_BASE_URL`. The current code does not inherit the generic SDK `OPENAI_BASE_URL` when the project setting is empty. HTTP proxy settings are a separate concern.

## 2. Start isolated search services and the API

These containers use new names, do not mount the project's existing data volumes, and remove their writable container layers on stop. The images match the repository's Compose file. Elasticsearch security is disabled **only for this loopback-bound local example**.

```bash
docker run --rm -d --name "$XAI_REPRO_CHROMA" -p 127.0.0.1:18177:8000 \
  -e ANONYMIZED_TELEMETRY=false \
  chromadb/chroma:1.5.9@sha256:1e0b73a187a28757c572acba508c46f48c9e8b0acaf5c20e6d95cdedce1acdf6

docker run --rm -d --name "$XAI_REPRO_ES" -p 127.0.0.1:19377:9200 \
  -e discovery.type=single-node -e xpack.security.enabled=false \
  -e 'ES_JAVA_OPTS=-Xms512m -Xmx512m' \
  docker.elastic.co/elasticsearch/elasticsearch:8.17.0@sha256:2f602552550869fb29b6fd5848c5118d3ef3a2e1d5d45802e3ab9088cb2de8e2
```

Wait until Elasticsearch responds and has an available primary; first startup can take a minute:

```bash
curl --noproxy '*' -fsS 'http://127.0.0.1:19377/_cluster/health?wait_for_status=yellow&timeout=5s'
```

If the first request fails while Elasticsearch starts, wait and retry. A red cluster is not ready even if an HTTP response arrives. Then run the API in the background and retain its PID:

```bash
.venv/bin/uvicorn xai_rag.api.app:app --host 127.0.0.1 --port 18077 --no-access-log &
XAI_REPRO_APP_PID=$!
```

The app connects to both stores and creates its required index on startup. In this terminal, poll readiness until it succeeds:

```bash
curl --noproxy '*' -fsS http://127.0.0.1:18077/ready
```

Expected: HTTP 200 and `status: healthy`, with `chromadb` and `elasticsearch` marked `ok`. If startup fails, inspect terminal errors without sharing secrets or document content. Do not use `docker compose down -v` for this walkthrough: it could remove unrelated project volumes.

If the index is red after the API starts creating it, inspect the cause in this **disposable** cluster:

```bash
curl --noproxy '*' -fsS -X POST http://127.0.0.1:19377/_cluster/allocation/explain \
  -H 'Content-Type: application/json' -d '{}'
```

On the recorded walkthrough the disk-allocation decider rejected a shard: the host was 93% full, above Elasticsearch's 90% high watermark. Freeing disk or increasing the local disk allocation is the appropriate general fix. For this throwaway, loopback-only cluster **only**, the following transient setting allowed the demonstration to continue; its effect disappears when this disposable container is removed. Do not use it for a real or shared cluster:

```bash
curl --noproxy '*' -fsS -X PUT http://127.0.0.1:19377/_cluster/settings \
  -H 'Content-Type: application/json' \
  -d '{"transient":{"cluster.routing.allocation.disk.threshold_enabled":false}}'
```

Retry `/ready`. If startup has already exited after its retries, restart the API with the same command and save its new PID. This workaround removes a disk-safety check and is not a production configuration.

## 3. Ingest and ask a question

The following uses the local API key from the environment without placing it in the command line. It prints only the fictional result. Use the `.venv` created by `uv sync`:

```bash
.venv/bin/python - <<'PY'
import os
import httpx

base = "http://127.0.0.1:18077"
headers = {"X-API-Key": os.environ["XAI_RAG_API_KEY"]}
with httpx.Client(headers=headers, timeout=180.0, trust_env=False) as client:
    ingested = client.post(
        f"{base}/ingest",
        params={"source_path": os.environ["XAI_RAG_INGEST_ROOT"], "strategy": "fixed"},
    )
    ingested.raise_for_status()
    result = ingested.json()
    print("Ingestion:", result["status"], "chunks:", result["total_chunks"],
          "index parity:", result["parity_consistent"])
    if result["status"] != "success" or result["parity_consistent"] is not True:
        raise SystemExit("Ingestion or parity check failed; inspect the response before querying.")
    response = client.post(f"{base}/query", json={
        "query": "How long after delivery can an unused product be returned?",
        "top_k": 5,
        "include_explanations": True,
        "include_faithfulness": False,
    })
    response.raise_for_status()
    answer = response.json()
    print("Grounding status:", answer["grounding_status"])
    if answer["grounding_status"] != "verifier_passed":
        raise SystemExit("Answer was withheld; inspect grounding status and provider configuration.")
    print("Answer:", answer["answer"])
    print("Source IDs:", answer["sources"])
    print("Structured claims:", answer["claims"])
PY
```

A successful response should have `grounding_status` equal to `verifier_passed`, mention the **30-day** window and identify a source. The result remains a model-based check, not proof of factual correctness. Wording, scores, latency and model usage can vary. If an answer looks plausible but lacks correct evidence, do not count it as a pass. `parity_consistent` checks index consistency, not semantic correctness or successful completion of a separately failed write.

**Checked locally, 19 November 2026:** The documented fictional walkthrough ran against the then-current uncommitted source, the specified isolated ports, cached pinned models and a live OpenAI request. It ingested one chunk with reported parity, returned the 30-day answer with one source and one structured claim, and then stopped both containers and the app. The run needed the test-only Elasticsearch setting above. The API process also had proxy variables removed **only for that process** because the inherited proxy was failing; do that only when direct network access is permitted. It did not rerun the optional RAGAS command or prove production readiness.

After the later chunking-guardrail change, the same walkthrough passed again on the then-current uncommitted working tree: one document and one chunk ingested, parity true, and a live 30-day answer with one source and one structured claim. The same disposable-only disk-watermark override and process-local proxy bypass were used, and the three test ports were free after cleanup. The SHA-256 over sorted `src/xai_rag` Python paths and contents (each path and content followed by a NUL byte) was `9e4f7cd45018add35371d729a02f46b615e79d3df211aeaf80614d04c6df2f5f` both before and after this run. This fingerprint identifies the working tree checked at that time; it is not a separately archived source snapshot.

For example, when a proxy is confirmed to be the cause and direct access is allowed, replace the API startup line with this process-local command; it does not change the shell's proxy configuration:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  .venv/bin/uvicorn xai_rag.api.app:app --host 127.0.0.1 --port 18077 --no-access-log &
XAI_REPRO_APP_PID=$!
```

Under the hood, ingestion writes matching chunk IDs to Chroma and Elasticsearch. At query time Chroma ranks embedding similarity; Elasticsearch uses BM25 to rank lexical matches. Reciprocal rank fusion merges the candidate rankings, then the cross-encoder reranks a bounded pool. The generator sees a bounded selection and returns structured claims and source IDs. Elasticsearch is the search engine; **BM25 is its keyword-ranking algorithm**. A valid citation ID does not establish that the cited text entails the claim. For the reasoning and a worked ranking example, see the [master guide](interview-mega-guide.md).

The API also has `/query/stream`: its HTTP status can be 200 even if a later SSE `error` event reports failure. In the hardening version, answer text is held until NLI and the configured LLM judge pass; `include_faithfulness` controls the additional NLI report, not whether the mandatory release checks run. `verification_unavailable` indicates checking failed or could not run; `insufficient_evidence` indicates the answer was withheld for insufficient support. The separately billable judge receives the visible evidence. The browser docs at `http://127.0.0.1:18077/docs` are enabled only because this example explicitly set `XAI_RAG_EXPOSE_DOCS=true`.

## 4. Run tests and optional RAGAS evaluation

The deterministic test suite does not require a provider call. With exact models already cached, these settings also prohibit Hugging Face downloads:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/pytest -m unit -q -p no:cacheprovider --no-cov
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/pytest -q -p no:cacheprovider --no-cov \
  tests/test_api_error_privacy.py tests/test_openapi_validation.py
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```

An offline test can fail when required weights were never cached. A passing unit suite is not a live end-to-end test. See [HTTP evidence](http-e2e-evidence.md) for the separate earlier real-service, live-model run, including its exact snapshot and failures. The live [RAGAS evaluation](ragas-live-evaluation.md) used fictional data and reports the rubric, scores, negative controls, request counts and source snapshots.

To rerun the saved RAGAS fixture, first install the **optional** evaluation extra (`uv sync --locked --extra dev --extra evaluation`); the default Docker image does not include it. RAGAS 0.4.3 is pinned in this checkout and the project documents an upstream advisory in a separate multimodal helper; the recorded run used text metrics. From the XAI-RAG checkout, replace the path with your interview-materials checkout:

```bash
XAI_NOTES_DIR=/path/to/personalized_interview_materials
XAI_RAGAS_OUTPUT_DIR="$(mktemp -d)"
RAGAS_DO_NOT_TRACK=true PYTHONPATH=src .venv/bin/python \
  "$XAI_NOTES_DIR/projects/xai-rag/evaluation/ragas-live-runner.py" \
  "$XAI_RAGAS_OUTPUT_DIR/results.json"
```

This runner sends the included **fictional** question, answer, reference and context to the provider and incurs billable requests. It caps actual attempts; outcomes can vary. The fresh output directory avoids overwriting saved evidence. This command runs the current code; to reproduce a historical snapshot, use the source patches and hashes cited in the report, not an assumed current version. Human review was **not** performed in the recorded study.

### Repeated seeded experiment

The [reproducibility interview Q&A](reproducibility-interview-qa.md) explains the three recorded repeats, source and index provenance, per-run scores, and the separate scoring-only recovery. The [XAI-RAG evaluation documentation](https://github.com/neeraj1909/xai-rag/blob/main/evals/README.md#repeated-runs-and-reproducibility) links the preserved manifest, measured harness and outputs. Recomputing an aggregate from saved scores is distinct from running a new live evaluation, which makes billable model calls and may produce different results. Review the archived harness and paths before adapting it; retain the original evidence unchanged.

### Historical runners and saved results

The saved reports include the original scripts as evidence of how those particular experiments were performed. Some are **machine-specific historical harnesses**, not portable commands. In particular, `evaluation/http-e2e-runner.py` hardcodes the original checkout and output paths and overwrites its results; `soak-runner.py` and its two companion scripts use original paths and can overwrite the soak records or snapshot; `h05-paired-diagnostic.py` updates a shared budget-state file; and `h05-generation-diagnostic.py` assumes an original temporary source path. `judge-fixture-run.py` also expects the original `.env` location, although it refuses to overwrite its results.

Do not run those scripts in place against the saved evidence. Copying a script alone does not fix its absolute paths: an unchanged copy can still read or overwrite the original locations. For a portable current-code end-to-end check, use the isolated walkthrough in sections 1–3 above. For the standard RAGAS fixture, use the fresh-output command in section 4. To repeat one of the historical experiments exactly, first make an independent copy of the evaluation fixtures and source snapshot, review and adapt every input, output, budget-state and credential path in that copy, and confirm its run conditions and hashes; retain the original evidence unchanged. Any changes to a copied harness must be recorded as part of the new experiment. The archived source snapshots and published results describe the original run, not a guarantee that an adapted harness will produce identical scores.

## 5. Cleanup and common problems

In the same terminal, stop only the app and containers created by this walkthrough:

```bash
kill "$XAI_REPRO_APP_PID"
docker stop "$XAI_REPRO_CHROMA" "$XAI_REPRO_ES"
```

The containers' `--rm` removes their temporary layers. The `mktemp` document directory remains for inspection; it contains only the fictional file. To remove it, check its path and contents first. Clear the temporary API key from your shell with `unset XAI_RAG_API_KEY`.

- **401 from `/query`:** Check the local `XAI_RAG_API_KEY` and `X-API-Key` match. An upstream OpenAI 401 is a different credential/provider problem. Confirm the project key and endpoint without printing either key.
- **`/ready` is 503 or startup fails:** Check both containers, port mappings and Elasticsearch cluster health. Local Docker disk allocation watermarks can prevent a primary shard from starting; inspect actual cluster state and free disk before changing settings. The [historical E2E report](http-e2e-evidence.md) documents a test-only workaround, not a production recommendation.
- **502/503 during a search outage:** The current query path reports known Elasticsearch connection failures as HTTP 503; it does not silently switch to vector-only search. Streaming clients must inspect SSE events. The ingestion endpoint reports all-failed batches as 500 and partial batches as 207.
- **Model download or offline cache failure:** Allow a first download of pinned revisions or configure an approved local cache; do not claim an offline test succeeded when weights are missing.
- **`verification_unavailable`:** Verify that a supported `XAI_RAG_GROUNDING_JUDGE_MODEL` is configured, the provider credential and endpoint work, the NLI model is cached or downloadable, and the judge request did not time out. Do not disable the gate or treat this as a successful quality score.
- **No useful answer or a citation-contract error:** Inspect ingestion status, per-store retrieval, fusion, reranking and prompt-visible sources in that order. A model output may fail the structured citation contract even after a bounded repair attempt. See [evaluation evidence](evidence-index.md) for actual examples and limitations.
