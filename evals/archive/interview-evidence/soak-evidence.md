# XAI-RAG isolated stability run

The 75-minute measurement completed on 19 November 2026, **18:47:25–20:02:25 UTC**, against a frozen local source snapshot and disposable search services. It exercised real embedding, hybrid retrieval, fusion and reranking with synthetic data. Generation used a local deterministic stub: no paid model calls were made, and this run does not measure LLM answer quality. It is a low-rate stability check on one machine, not a throughput, capacity or production-readiness test.

## Results

- **Readiness:** 892/901 probes returned HTTP 200. Of the nine other samples, seven occurred during the two intentional service pauses (six HTTP 503 and one transport error). Two outside those windows were `RemoteProtocolError` at 18:54:05.076 UTC and 19:50:55.081 UTC. The next samples were healthy. The preserved Uvicorn log has no error or traceback corresponding clearly to either event; its access lines lack timestamps. The cause of those two errors is **not established**.
- **Retrieval:** 14/15 scheduled queries passed. The sole failure returned HTTP 500 during the Elasticsearch pause. The overlapping Chroma-pause query succeeded after recovery.
- **Re-ingestion/parity:** 3/4 scheduled checks passed. The failed check overlapped the Elasticsearch pause: HTTP 500, `ConnectionTimeout`, zero ingested files and zero chunks. Its response also reported `parity_consistent: true`; that field did not establish successful ingestion. A separately recorded fresh query and re-ingestion after recovery both returned HTTP 200; the latter ingested one file and one chunk with parity true. The minute-60 scheduled parity check also passed.
- **Recovery:** Chroma was paused for approximately 20.08 seconds starting 19:07:25.146 UTC. During the pause readiness saw three 503s and one transport error; a probe returned 200 immediately after unpause. Elasticsearch was paused for approximately 20.15 seconds starting 19:32:25.186 UTC. Readiness saw three 503s, then 200 at recovery. These are single controlled interruptions, not exhaustive failure testing.
- **Resources:** Across 150 samples, minimum host available memory was 5.69 GiB, minimum free swap 1.41 GiB, minimum free disk 13.58 GiB, and maximum API RSS 2.25 GiB. Host swap use changed during this shared-machine run; this measurement does not attribute it to the application. The stop watchdog did not fire.

## Query timing

The 15 scheduled samples include one cold-model query, one query overlapping each fault, and one HTTP 500. Median latency was **1.198 seconds**; maximum was **21.434 seconds** during the Chroma pause. The sample is too small and the workload too light to infer capacity.

| Approximate start, minutes | HTTP | Duration, seconds | Context |
|---:|---:|---:|---|
| 0.00 | 200 | 6.626 | cold model load |
| 5.00 | 200 | 0.929 |  |
| 10.00 | 200 | 1.260 |  |
| 15.00 | 200 | 1.445 |  |
| 20.00 | 200 | 21.434 | Chroma pause overlap |
| 25.00 | 200 | 1.087 |  |
| 30.00 | 200 | 0.946 |  |
| 35.00 | 200 | 1.026 |  |
| 40.00 | 200 | 1.167 |  |
| 45.00 | 500 | 10.736 | Elasticsearch pause overlap |
| 50.00 | 200 | 0.953 |  |
| 55.00 | 200 | 1.198 |  |
| 60.00 | 200 | 1.010 |  |
| 65.00 | 200 | 3.321 |  |
| 70.00 | 200 | 1.290 |  |

## Protocol and evidence

The frozen source was based on Git HEAD `f4c374d51d090beba61d34aba81fcc23b8ef700a` and included then-uncommitted changes and untracked Python modules. The aggregate SHA-256 of the sorted Python-file manifest is `728ca85ab1b11a87248619012648031090cc97d1924a5f127b980120cc8a0e03`; the full per-file manifest is in `evaluation/soak-summary.json`. The saved source archive SHA-256 is `9280365ea4b29c35d82e2dbe6f1919833d2ade5399ab6ae194654859ea40c7e2`.

The runner ingested one newly created synthetic policy file, then sampled readiness every 5 seconds, ran a query every 5 minutes, repeated ingestion and parity every 15 minutes, and sampled resources every 30 seconds. It started Uvicorn directly from the snapshot, alongside disposable pinned Chroma 1.5.9 and Elasticsearch 8.17.0 images. All services bound to loopback ports 18169, 19310, 18069 and 18070. Embedding and reranking used the pinned cached models. The API process had a synthetic credential and a loopback-only generation endpoint; inherited OpenAI settings were removed.

The Elasticsearch pause was a protocol amendment and was logged independently. Consequently the runner's `fault_window` flag marks only the Chroma pause; classify Elasticsearch samples using the external pause/unpause timestamps in the interval log. The saved runner applies the container CPU limits at startup; in this recorded run those same limits were applied with `docker update` immediately after container creation. Containers also had memory limits, model threads were bounded, and the watchdog stopped the experiment on sustained host memory or disk pressure or API process exit. Elasticsearch's disk-allocation threshold was disabled only for this throwaway cluster because the host had limited free space; this is not a production configuration.

HTTPX 0.28.1 used its default limits: 100 connections, 20 keep-alive connections and a 5.0-second keep-alive expiry. The async client had a 120-second default timeout, overridden to 6 seconds for readiness and 90 seconds for query and re-ingestion requests. Uvicorn 0.52.4 used its default 5-second server keep-alive timeout. Polling remained unchanged throughout. These settings may help investigate the transport errors but do not prove their cause. A separate transport diagnostic uses different client settings and is not part of this soak.

The append-only `evaluation/soak-intervals.jsonl` contains every sample and both fault markers. `evaluation/soak-summary.json` contains the source manifest, environment bounds, timing and terminal cleanup. The saved Uvicorn log through the final minutes is `evaluation/soak-api-near-end.log`. The Elasticsearch recovery query and re-ingestion are separately recorded as `recovery_check` entries. The independently written fault markers and runner entries can be slightly out of timestamp order.

The runner reported successful container stops; an independent check found neither disposable container present and no listener on any of the four test ports. The temporary source copy and fixture were removed. The archived source and synthetic evidence remain for inspection.

## Reproduce

With Docker, the local XAI-RAG checkout and its `.venv`, and the pinned model weights already cached, run from this repository:

```bash
/home/neeraj/Documents/xai-rag/.venv/bin/python projects/xai-rag/evaluation/soak-runner.py
```

While the runner is active, start `evaluation/soak-es-fault.py` in a second process to schedule the minute-45 Elasticsearch pause, and run `evaluation/soak-recovery-check.py` after that pause for the fresh recovery checks. These scripts require the original local paths and free test ports. They overwrite the recorded summary and interval log; preserve the current evidence before rerunning. The stubbed provider avoids paid calls; this workflow does not verify a real provider's availability.
