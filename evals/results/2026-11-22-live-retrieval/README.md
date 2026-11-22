# Live retrieval holdout — 22 November 2026

The current project code ran the 40-question historical fictional holdout through
real Chroma vector search, Elasticsearch BM25, production RRF and the configured
cross-encoder. The run used one indexed fixture passage per source and did not
measure ingestion parsing/chunking, generation or guardrails. The labels are
synthetic and not independently reviewed. This run is a new measurement, not a
reproduction of the original 19 November measurement.

The runner creates randomly named indexes and deletes them on normal completion.
Each service ran in a separate disposable Docker container with its ports bound to
loopback, with no persistent volumes. Elasticsearch was single-node, without
security, configured with a 256 MB heap, and its disk watermark disabled only
for this disposable instance. The explicit settings do not represent production
recommendations.

The run command from the repository root was:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
uv run --no-sync python evals/live_retrieval/runner.py \
  --fixture evals/archive/interview-evidence/evaluation/holdout-cases.json \
  --chroma-port 32768 --es-port 32769 \
  --output /tmp/xai-rag-live-retrieval-20261122.json
```

Ports were assigned dynamically to the disposable containers for this run and
will differ in another run. Its output was copied without modification to
`holdout.json` in this directory after the run. The fixture SHA-256 was
`b666492a41e218d6fefa0fcca0fa6ab26277bfb7dfba34413ffadb672afdc8cf`;
the runner SHA-256 was
`5518fab4bb4a7470c467fce603d7dfb0a11824abeab956fe602bedd2e864e5c4`;
`uv.lock` SHA-256 was
`be504e18331197618409fb18de62f2d4b4c6f0ac02de24b65adb0acae84d834c`.
The tracked source revision was `52ea91b878ebd254f22b19555835932613b63197`,
with an uncommitted working tree; the result records the actual package source
tree digest. Python was 3.14.2, PyTorch 2.13.0+cpu, sentence-transformers
6.0.0, transformers 5.15.1, chromadb-client 1.5.9, and elasticsearch 8.19.3.
The Chroma image ID was
`sha256:1e0b73a187a28757c572acba508c46f48c9e8b0acaf5c20e6d95cdedce1acdf6`;
the Elasticsearch image ID was
`sha256:2f602552550869fb29b6fd5848c5118d3ef3a2e1d5d45802e3ab9088cb2de8e2`.
The result includes the configured model IDs and revisions, per-case rankings,
errors, source hashes and stage denominators.

The configured RRF constant was 60 and the candidate count was 5. Models were
`BAAI/bge-small-en-v1.5` at `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
and `BAAI/bge-reranker-v2-m3` at `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.

The run completed all 40 queries with no recorded case errors. Each stage scored
37/37 queries with labeled relevant sources; the remaining three had no labeled
relevant source and were excluded from retrieval-quality averages.

| Stage | Hit@1 | Hit@3 | Recall@3 | MRR@5 | NDCG@3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector | 0.9730 | 1.0000 | 1.0000 | 0.9865 | 0.9900 |
| BM25 | 0.8649 | 1.0000 | 1.0000 | 0.9324 | 0.9501 |
| RRF | 0.9730 | 1.0000 | 1.0000 | 0.9865 | 0.9900 |
| Rerank | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The JSON SHA-256 is
`415ea9904777cfff6f64d54f2662f3ed299c2243c1e9b3b5e9f8711535df18c3`.
`measured-runner.py` retains the exact runner used for this result. The active
runner was subsequently extended to record its own and the lockfile's SHA-256,
the configured embedding dimension, and the RRF constant in future outputs.
The measured output is not retroactively modified. Before the containers were
stopped, their collection and index lists were empty after runner cleanup.
