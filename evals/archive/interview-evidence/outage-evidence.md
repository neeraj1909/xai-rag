# Elasticsearch outage: real-store REST check

On 19 November 2026, a fresh copy of the then-uncommitted source was tested with actual isolated Chroma 1.5.9 and Elasticsearch 8.17.0 containers and a local Uvicorn API. The test ingested one newly generated synthetic policy document. Embedding and reranking used the project's cached pinned models; generation used a deterministic loopback stub with a synthetic key. No real provider credentials or paid generation calls were used.

| Step | Observation |
|---|---|
| Initial ingestion | HTTP 200; one file and one chunk, reported consistent across stores. |
| Baseline query | HTTP 200 with a cited synthetic refund deadline. |
| Pause disposable Elasticsearch; query | HTTP **503** after **10.071 seconds**, with exactly `{"detail":"Search dependency unavailable"}`. The local generation stub's call count remained **1 before and 1 after** the failed query. The app log reported `ConnectionTimeout`. |
| Unpause Elasticsearch; wait for readiness; query again | HTTP **200** after **1.162 seconds**, with the cited synthetic deadline. |
| Cleanup | Both disposable containers stopped successfully. A separate check found no containers with the test prefix and no listeners on test ports 18189, 19330, 18089 or 18090. |

The app shutdown log included a Python `resource_tracker` warning that one loky semaphore appeared to require cleanup. This is an observed shutdown warning; the experiment does not establish whether any persistent resource leak exists.

The then-uncommitted snapshot was based on Git HEAD `f4c374d51d090beba61d34aba81fcc23b8ef700a`, including source changes and a helper that were uncommitted at the time. Its aggregate sorted Python-manifest SHA-256 is `82cdecf60f719914c6490b727f81736f606e8de3feb802e537e6a0fc113f76d7`. The source archive SHA-256 is `8daa7668136e179a8aadb41e4c7d97f2853f028a0e3dd294174ff127823867e4`; all Python files in the archive were verified against the manifest in `evaluation/outage-summary.json`. The separate `evaluation/outage-intervals.jsonl` contains the test events. These artifacts contain synthetic data and no keys.

This is one timeout caused by pausing a single local Elasticsearch container. It does not demonstrate classification of every possible search failure, streaming behavior, production deployment recovery, or real-provider answer quality. The isolated Elasticsearch instance had its disk-allocation threshold disabled for this experiment only. Services had CPU and memory limits, and the runner checked available disk and memory before starting.

To repeat from the interview-materials repository, with Docker, the XAI-RAG local `.venv`, and pinned model weights already cached:

```bash
/home/neeraj/Documents/xai-rag/.venv/bin/python projects/xai-rag/evaluation/outage-runner.py
```

The runner uses loopback-only test ports, creates and removes fresh containers and synthetic fixtures, and overwrites the `evaluation/outage-*` results. Preserve the current results before rerunning.
