# Local connection-boundary diagnostic

This experiment compares a shared default `httpx` connection pool with a separate client configured to send `Connection: close`. Both make loopback requests to a small FastAPI/Uvicorn **stub**, not the XAI-RAG app, search stores, OpenAI, or the original soak workload. It used 5-second readiness intervals, httpx's default 5-second keepalive expiry and the server's 5-second keepalive timeout. Nine short synthetic query calls were interleaved with 49 readiness calls for each arm.

| Client arm | Readiness transport errors | Synthetic-query transport errors |
| --- | ---: | ---: |
| Reused default pool | 8/49 | 3/9 |
| `Connection: close` | 0/49 | 0/9 |

Failures in the reused-pool arm were `ReadError` or `RemoteProtocolError`. A **separate, later** run kept the same probe pattern and client defaults but set Uvicorn keepalive to 15 seconds. Both arms then had 0/49 readiness and 0/9 synthetic-query errors. These short runs were sequential, not a randomized controlled production comparison; the later run is in `keepalive-15/`. This artificial test shows that pooled transport errors can occur at this timing boundary; it does **not** prove the cause of the historical app soak's two errors, measure current app capacity or establish an SLO. Client timing, retry policy, deployment network and connection reuse require separate verification. The Docker healthcheck uses a fresh `urllib` request rather than this shared httpx pool, so this particular pooling comparison does not apply to it. Increasing server keepalive may retain more idle sockets/file descriptors; reverse proxies and other deployment runtimes have their own independent timeouts. The development Dockerfile now passes `--timeout-keep-alive 15` to Uvicorn. That setting applies to the Docker CMD; it is a local mitigation for that runtime, not evidence that every deployment is fixed.

`events.jsonl` and `summary.json` preserve each attempt and counts. `experiment.py` is the original machine-specific script; it hardcodes `/tmp/xai-transport-boundary` as its output directory and overwrites those paths. Review and adapt an independent copy before any rerun. The selected files contain only synthetic local requests and were checked for common token patterns and configured secret values without printing them; this is a best-effort scan, not a complete privacy audit.
