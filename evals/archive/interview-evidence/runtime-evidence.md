# XAI-RAG: local runtime evidence

These checks exercised the local [`f4c374d51d090beba61d34aba81fcc23b8ef700a`](https://github.com/neeraj1909/xai-rag/tree/f4c374d51d090beba61d34aba81fcc23b8ef700a) checkout on 19 November 2026. They are reproducible engineering checks, not production benchmarks or evidence of model-answer quality. The checkout already contained an untracked `Dockerfile.cpu`; it was preserved.

## What ran against real databases

Two disposable containers ran on loopback-only ports with independent data. Commands below were run from the repository root:

```bash
docker run -d --rm --name xai-interview-check-chroma-20261119 \
  -p 127.0.0.1:18109:8000 -e ANONYMIZED_TELEMETRY=false \
  chromadb/chroma:1.5.9
docker run -d --rm --name xai-interview-check-es-20261119 \
  -p 127.0.0.1:19250:9200 -e discovery.type=single-node \
  -e xpack.security.enabled=false -e ES_JAVA_OPTS='-Xms384m -Xmx384m' \
  -e cluster.routing.allocation.disk.threshold_enabled=false \
  docker.elastic.co/elasticsearch/elasticsearch:8.17.0
```

Once Elasticsearch's local cluster reported green, the actual repository integration test ran:

```bash
XAI_RAG_RUN_INTEGRATION=1 \
XAI_RAG_CHROMA_HOST=127.0.0.1 XAI_RAG_CHROMA_PORT=18109 \
XAI_RAG_CHROMA_COLLECTION=xai_interview_test_20261119 \
XAI_RAG_ELASTICSEARCH_URL=http://127.0.0.1:19250 \
XAI_RAG_ELASTICSEARCH_INDEX=xai_interview_test_20261119 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/pytest -m integration -q -p no:cacheprovider --no-cov
```

**Observed:** one test passed, 75 were deselected, and there were 38 deprecation warnings. The test used two synthetic texts and hand-built deterministic vectors. It wrote them to the real database processes, checked matching chunk IDs and 2/2 parity, asserted that vector retrieval, BM25 retrieval and fusion placed the expected chunk first, then re-ingested one chunk and verified 1/1 parity and stale-chunk removal. This establishes the tested storage/retrieval boundary for this fixture, not semantic quality on real questions or a general distributed consistency guarantee. [Test source](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/tests/integration/test_dual_store.py).

Both containers were stopped and removed with:

```bash
docker stop -t 10 xai-interview-check-chroma-20261119 xai-interview-check-es-20261119
docker ps -a --filter name=xai-interview-check --format '{{.Names}}'
```

The final container listing was empty. The security-disabled Elasticsearch and disk-threshold override were for this isolated disposable local test; they are not production settings.

## Offline evaluation and unit tests

```bash
.venv/bin/xai-rag evaluate evals/smoke/evaluator-contract.jsonl \
  --k 1 --k 2 --gates evals/gates.toml \
  --output /tmp/xai-interview-evaluator-20261119.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/pytest -m unit -q -p no:cacheprovider --no-cov
```

**Observed evaluator:** all six blocking and six advisory gates passed for the synthetic fixture. This invocation explicitly requested **both `@1` and `@2` cutoffs**; any metrics at those cutoffs describe the supplied fixture observations, not retrieval newly performed by the evaluator. The fixture has two query records and one ingestion record; a metric's actual denominator can be smaller when evidence is absent. No model or provider call occurred. See the [fixture and evaluation contract](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/evals/README.md).

**Observed unit suite:** 73 passed, one failed, and two were deselected. `test_long_single_sentence_is_bounded_for_semantic_strategy` attempted to load the uncached `sentence-transformers/all-MiniLM-L6-v2` revision with offline mode enabled. The failure establishes a missing local model dependency for this run; it does not establish whether the semantic-chunking behavior is correct or incorrect.

## What these checks do not prove

There was **no full API-to-answer run**, real embedding inference, cross-encoder reranking, NLI inference, or live LLM generation in this check. The available prebuilt API image was older than the inspected source checkout, and the needed model weights were not locally cached. No paid model API was called. A later end-to-end run should use an image built from the exact reviewed commit, isolated Chroma and Elasticsearch data, pinned locally available model weights, and either a clearly identified local stub or an explicitly chosen model provider. Evaluate generated answers on a separate human-reviewed, representative dataset before quoting quality or latency figures.
