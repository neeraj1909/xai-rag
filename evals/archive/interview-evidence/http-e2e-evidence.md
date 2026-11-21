# XAI-RAG: isolated HTTP end-to-end evidence

The final run passed 14 checks against an actual Uvicorn/FastAPI process, two disposable search services, locally cached embedding and reranking models, and a live OpenAI endpoint. This is a small synthetic smoke test, not a production qualification or a measured evaluation of answer quality across a representative dataset. The run took place on 19 November 2026, 18:09–18:10 UTC.

## What ran

- Application source: then-uncommitted snapshot based on Git HEAD `f4c374d51d090beba61d34aba81fcc23b8ef700a`, including the then-untracked `src/xai_rag/async_utils.py`. The aggregate SHA-256 of the sorted Python-file manifest is `548c3051350a2b89e9ef315fca0984bd41482f0108a33fdb6798aacab2c9a97b`. The full per-file manifest is in the results JSON. A source archive was created after verifying that every Python file still matched that manifest; archive SHA-256: `c521058fd21ba2811b1bfa1c0a92ea903bfc1473d7590e100d130229551252e0`.
- Search services: pinned Chroma 1.5.9 and Elasticsearch 8.17.0 Docker images, published only on loopback ports 18159 and 19300. The API ran directly from the snapshot on port 18059, rather than from a potentially stale Docker image. A second API process on 18060 used a deliberately invalid test credential to check the SSE failure path.
- Models: pinned `BAAI/bge-small-en-v1.5` embeddings and `BAAI/bge-reranker-v2-m3` reranker, loaded from local cache with offline model loading enabled. Generation used `gpt-4o-mini` through the official OpenAI API. Faithfulness/NLI and RAGAS were not enabled in these requests.
- Data: two newly generated, synthetic text documents, one containing a refund policy and the other containing an instruction-injection marker. No existing user documents were ingested. The project credential was read into the API process environment; neither it nor the generated local API key was recorded in the results.

## Observed behavior

| Check | Result |
|---|---|
| `/live` and `/ready` | HTTP 200; readiness reported both stores healthy. |
| Missing or wrong API key | HTTP 401 for both. |
| Whitespace-only and more than 4,000-character query | HTTP 422 for both. |
| Request-scoped external evaluation disabled | HTTP 400 for both regular and streaming endpoints. |
| Ingesting `/etc/passwd` outside the configured root | HTTP 400. |
| Synthetic ingestion | HTTP 200; 2 documents, 2 chunks, no failed files, and reported store parity. |
| Grounded REST question | HTTP 200; answer stated the 30-day deadline, supplied a source ID, and returned retrieval explanations. |
| SSE question | HTTP 200; `retrieval`, `generation`, then `done`; the answer stated four business hours. |
| Untrusted instruction | The injected instruction appeared in retrieved evidence, but the answer did not reproduce its marker and answered the refund question. This is one case, not evidence of general prompt-injection resistance. |
| SSE upstream authentication failure | With a deliberately invalid upstream credential, the HTTP stream emitted `retrieval` then `error`, without `done`. A client must inspect stream events, since the initial HTTP status was 200. |

The initial assertions ran during the test; the SSE answer and presence of the injected instruction in retrieved evidence were additionally checked against the saved responses afterward. The two successful REST responses report **1,102 input and 338 output tokens** combined. SSE does not expose token usage in its events, and failed calls or repair attempts may not be included; these figures are not a complete billable-usage total.

## Earlier failures and limits

The first local cluster had an unassigned Elasticsearch primary and the application failed to start with a red index and an Elasticsearch 408. This was consistent with the host's limited free disk and Elasticsearch allocation watermarks, but no allocation-explain response was captured, so the exact cause is unconfirmed. The subsequent disposable cluster disabled the disk allocation threshold and started successfully. This setting was limited to the throwaway local cluster; production needs adequate disk capacity and allocation monitoring.

Two subsequent runs reached generation but received upstream authentication errors with different credential and endpoint combinations. They inherited a provider base URL that did not match the credential used. The final run used the project's credential with the explicit official OpenAI endpoint and succeeded. These errors do not establish that the project credential was invalid. All attempts, including the failures, are summarized in the JSON results.

This run does not verify multi-user authorization, TLS, load behavior, recovery, representative prompt injection coverage, complete citation correctness, NLI/RAGAS quality, or human evaluation. Container services were stopped and the temporary source snapshot and documents removed after each attempt. The separately saved source archive contains source files only.

## Reproduce

From the interview-materials repository, with Docker available, the source checkout at `/home/neeraj/Documents/xai-rag`, its `.venv` and cached pinned models present, and a valid project-local `.env` credential:

```bash
/home/neeraj/Documents/xai-rag/.venv/bin/python projects/xai-rag/evaluation/http-e2e-runner.py
```

The runner creates fresh temporary documents, copies the then-current source tree, starts and stops the pinned services, binds only to loopback, and makes live billable OpenAI requests. It uses the direct official endpoint and process-local proxy overrides. Its temporary Elasticsearch instance disables the disk-allocation threshold. Ports 18159, 19300, 18059, and 18060 must be free. It overwrites `http-e2e-results.json`; preserve the existing evidence before rerunning. The historical attempts in the saved JSON were merged after execution and are not automatically reproduced by one run of the runner.
