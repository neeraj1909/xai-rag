# XAI-RAG bounded synthetic load baseline

Source `c375903fb46327744d076438cb90f7b02711a94f`, archive SHA-256 `9ca5a2fa5fa028db3f8c5d1b0792c3cb21f78108056f53ef98b99c0127715949`, fixture SHA-256 `e88a18b2e1fda89b799998be193409834cbed53fccb190720e1387bf4ea35b93`. 2026-11-20T09:43:23.320428+00:00 to 2026-11-20T10:06:52.672203+00:00 UTC. One run, shared host.

100 synthetic documents; 100 ingested with store parity. Local Chroma and Elasticsearch, cached embedding and reranking models; deterministic loopback LLM stub, no paid calls. Five warmups excluded. Closed-loop requests, 30 scheduled and attempted per level, 150-second client timeout, app concurrency cap 4, test-only rate limit 1000/min.

| Client concurrency | Success / attempted | p50 s | p95 s | max s | Successes/s | synthetic hit@5 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30/30 | 6.829 | 19.176 | 27.861 | 0.105 | 30/30 |
| 2 | 30/30 | 11.222 | 18.643 | 18.682 | 0.167 | 30/30 |
| 4 | 30/30 | 25.295 | 44.622 | 45.604 | 0.151 | 30/30 |
| 8 | 30/30 | 55.066 | 70.173 | 71.55 | 0.127 | 30/30 |

All 120 completed with HTTP 200; no measured errors. Hit@5 is a simple synthetic ID check and does not establish answer quality. Latencies are client-side, success-only; throughput is successes divided by elapsed level time. P95 on 30 samples is unstable. Concurrency 8 includes queueing at the four-query cap. The level-to-level variation includes shared-host interference. These results are not a production capacity estimate or an LLM quality evaluation.

Resource watchdog polled every 5 seconds: minimum available RAM 3 GiB, free swap 1 GiB, free disk 10 GiB; API RSS cap 4 GiB and 30-minute wall limit. Observed min available RAM 4.80 GiB, min free swap 2.62 GiB, min free disk 13.30 GiB, max API RSS 2.05 GiB. Container OOM/restarts were not independently monitored. No separate rate-limit test was run. Both owned containers removed with exit 0; all four leased ports were verified free after completion.

Artifacts: `runner.py`, `events.jsonl` (per request and resources), `summary.json`, `api.log`. Requests were originally planned at 30 per level; the coordinator requested 10 during startup, but the already-running Python process had loaded the 30-request loop. The original cap remained in force and all levels completed.
