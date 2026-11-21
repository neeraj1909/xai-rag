# Readiness keepalive transport diagnostic

## Question and result

Can a five-second HTTP client request schedule encounter connection failures near Uvicorn's five-second keepalive boundary, and does disabling connection reuse change the outcome? In a separate local experiment, it did:

| Client arm | Requests | Transport failures | Breakdown |
| --- | ---: | ---: | --- |
| HTTPX pooled connection | 241 | 33 | 23 `RemoteProtocolError`, 10 `ReadError` |
| `Connection: close` | 241 | 0 | None observed |

The run started 19 November 2026 at 19:09:07 UTC and finished at 19:29:12 UTC. The two arms ran concurrently against the same loopback-only FastAPI stub with a constant `200` response and no XAI-RAG dependencies. Each scheduled one request every five seconds. HTTPX used one allowed connection per arm, a 60-second client keepalive expiry, and a three-second request timeout. Uvicorn used a five-second keepalive timeout. Proxy use was disabled. Environment: Python 3.14.2, HTTPX 0.28.1, Uvicorn 0.52.4, FastAPI 0.141.1.

This demonstrates a **reproducible transport failure mode under this particular configuration**, not the cause of the XAI-RAG soak's two readiness errors. `Connection: close` avoided failures in this comparison but incurs connection setup costs; the result does not establish general performance, TLS behavior, or production reliability.

## Relationship to the XAI-RAG soak

The [soak runner](soak-runner.py) also scheduled readiness every five seconds and launched Uvicorn without a keepalive override. Unlike this stub, it ran the real XAI-RAG API, Chroma and Elasticsearch, concurrent query and ingestion work, and planned dependency faults. Its single HTTPX client was shared across the other traffic and used HTTPX defaults: 100 maximum connections, 20 maximum keepalive connections, and five-second client keepalive expiry in the inspected HTTPX 0.28.1 environment. This comparison deliberately used a **60-second** client keepalive expiry and a single connection per arm, so its error frequency must not be applied to the soak. The [soak event records](soak-intervals.jsonl) include the original `RemoteProtocolError` observations; they do not identify their underlying cause.

## Evidence and reproduction

- [Sanitized per-request events](readiness-keepalive-events.jsonl) contain timestamps, arm, request number, status or exception type, and latency. They contain no request bodies, tokens, or credentials.
- [Runner](readiness-keepalive-diagnostic.py) uses an ephemeral local port and shuts down the server after both arms finish. The original port was verified closed after completion.
- Run from the XAI-RAG checkout with the inspected environment: `.venv/bin/python /path/to/personalized_interview_materials/projects/xai-rag/evaluation/readiness-keepalive-diagnostic.py`. A fresh run writes to `/tmp/xai-ready-diagnostic-rerun/events.jsonl`, preserving the recorded evidence. Expect about 20 minutes. It does not call a provider or start Docker.
- A shortened runner smoke check verified both arms completed and shut down; it is not an additional keepalive comparison or a substitute for the 20-minute run.

A stronger causal investigation would reproduce the original soak's actual connection-pool settings and shared traffic, log connection reuse and FIN/RST timing, and compare controlled settings. Do not infer that a failed client-side health request means a search dependency was unhealthy.
