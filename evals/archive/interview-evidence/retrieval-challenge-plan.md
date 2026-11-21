# XAI-RAG: additional retrieval challenge plan

This is a **proposed run**, not an evaluation result. The companion [challenge fixture](evaluation/retrieval-challenge-cases.json) was authored before its own retrieval run. It contains 31 fictional passages and 28 questions: 26 have labeled relevant source IDs; C08 and C27 have empty relevance labels. Its SHA-256 is `9d630857de2beca9ca9d270598276b0c9e0898918649a96864151f2e541e59eb`.

The questions and relevance labels were authored together by an AI agent and have **not been independently human-reviewed**. The author knew the results of earlier synthetic experiments; this is a fresh exploratory slice, not a blinded or sealed holdout. Gold labels refer to source IDs, not exact supporting spans. No production generalization, statistical superiority or independent human evaluation can be inferred.

## Why this slice is different

Near-collision IDs and strings, semantically related distractors, negation, old versus new versions, partial answers, Hindi questions and documents, an English question over Hindi evidence, and five questions with two necessary sources make the set more discriminating than the original small smoke fixture. It is still hand-authored, nonrepresentative and susceptible to correlated label and wording bias. A difficult answer can require reasoning beyond retrieval; a retrieval hit is not proof of a correct, grounded answer.

**The five multi-source cases are C06, C11, C15, C22 and C28.** Their gold lists are intended to describe the sources required for the expected complete answer, not a graded list of every semantically related passage. Confirm these labels and adjudicate plausible alternatives with a qualified human before using them for a release decision.

## Reproduce a retrieval-only run

These instructions use the existing [holdout runner](evaluation/holdout-runner.py) with its `--fixture` option. It writes to fresh local stores and performs real embedding, Chroma and BM25 retrieval, RRF and reranking; it does not call an LLM unless `--generate` is explicitly supplied. Use an isolated XAI-RAG working tree and record its commit **and working-tree source digest** from the output. Do not run against an existing user's stores or reuse an already populated index.

From the XAI-RAG source checkout, substitute the actual absolute path for `INTERVIEW_REPO`. These disposable containers use loopback ports and intentionally disable Elasticsearch security only for this test; they are not a production configuration. Check that the chosen ports and container names are unused first.

```bash
INTERVIEW_REPO=/absolute/path/to/personalized_interview_materials
sha256sum "$INTERVIEW_REPO/projects/xai-rag/evaluation/retrieval-challenge-cases.json"

docker run -d --rm --name xai-challenge-chroma-20261119 \
  -p 127.0.0.1:18139:8000 -e ANONYMIZED_TELEMETRY=false chromadb/chroma:1.5.9
docker run -d --rm --name xai-challenge-es-20261119 \
  -p 127.0.0.1:19280:9200 -e discovery.type=single-node \
  -e xpack.security.enabled=false -e ES_JAVA_OPTS='-Xms384m -Xmx384m' \
  -e cluster.routing.allocation.disk.threshold_enabled=false \
  docker.elastic.co/elasticsearch/elasticsearch:8.17.0
# Wait for the Chroma heartbeat and Elasticsearch cluster readiness before running.

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  XAI_RAG_CHROMA_HOST=127.0.0.1 XAI_RAG_CHROMA_PORT=18139 \
  XAI_RAG_CHROMA_COLLECTION=xai_holdout_challenge_20261119 \
  XAI_RAG_ELASTICSEARCH_URL=http://127.0.0.1:19280 \
  XAI_RAG_ELASTICSEARCH_INDEX=xai_holdout_challenge_20261119 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=3 \
  .venv/bin/python "$INTERVIEW_REPO/projects/xai-rag/evaluation/holdout-runner.py" \
    --execute --fixture "$INTERVIEW_REPO/projects/xai-rag/evaluation/retrieval-challenge-cases.json" \
    --output /tmp/xai-retrieval-challenge-20261119.json \
    --source-label 'record-the-source-commit-and-working-tree-state' --k 5

# Once the result is safely stored and reviewed, remove only these test containers.
docker stop -t 10 xai-challenge-chroma-20261119 xai-challenge-es-20261119
```

Make cleanup happen even if a run fails. Record missing models, initialization failures, each query/stage error, model revisions, fixture hash, source hash, store versions, k, elapsed time, and temporary resource cleanup. Do not treat a failed run or timeout as a successful result. The runner validates IDs and records raw per-question rankings; its `--self-test` validates metric calculations on its own built-in fixture, not this fixture. Do not modify the fixture after results are seen and still call it the same run; if labels change, preserve and hash both versions and explain why.

## Metrics and denominators

Calculate each stage—vector, BM25, RRF and reranker—on the **same 26 relevance-labeled questions** and keep stage errors explicit. Report successful-query quality and an independently named operational view with failures as zero, rather than quietly dropping failed cases. The two empty-label cases C08 and C27 are excluded from retrieval-relevance denominators and evaluated separately for appropriate abstention if answer generation is run.

- **Hit@k:** one if at least one gold source appears in the first k ranked positions, otherwise zero. This is useful for a single-source question but can hide missing evidence on a multi-source question.
- **Recall@k:** the number of distinct gold sources found in the top k divided by the number of gold sources for that question. Use the runner's definition, which counts a duplicate source only on first occurrence; preserve ranking positions of repeated chunks. State whether the reported number is a macro-average over questions.
- **All-required@k, the key multi-source measure:** for each of the five two-source cases, one only if **both** required source IDs occur in the first k ranking positions; otherwise zero. Report the numerator and denominator separately for each retrieval stage and k, alongside the case IDs and raw rankings. Also report attempted, scored and failed case counts; if a stage fails on any of the five, show both the successful-only fraction and the explicitly labeled failures-as-zero operational fraction. Never count an error as a successful absence or silently drop it.
- **Worked example:** gold `{A, B}`, ranking `[A, X, Y]`, k = 3. Hit@3 = 1, Recall@3 = 1/2, and All-required@3 = 0. For ranking `[A, X, B]`, all three metrics equal 1. Thus Hit@k can look perfect while a multi-source answer still lacks necessary evidence.
- **MRR and NDCG:** MRR rewards the first relevant result and therefore also does not ensure both required sources appear. NDCG@3 discounts the rank of each gold source, but its score depends on this fixture's binary labels and duplicate-source treatment. Report the runner's exact definitions and examples, not a generic number without methodology.

Report each case's expected behavior and per-stage rankings. Hindi, cross-language, historical, exact-ID and negation subsets are each tiny: list numerator/denominator and errors instead of treating their scores as statistically reliable. If desired, separately repeat the *same frozen fixture* on fresh stores for stability; repeated runs on the same questions do not increase dataset size or make the results independent samples. A corrected label or modified corpus requires a newly named, hashed run.

## Answers, judges and human review

Retrieval alone does not establish correctness, faithfulness, relevance, refusal safety, source permissions, latency at representative load, or prompt-injection resistance. If a later bounded model run is made, preserve the actual prompt-visible evidence, answer, citations, model and judge versions, token usage, rubric, failed attempts and per-case scores. Model-judge scores are not human scores. At present the fixture's expected answers and relevance labels are **authored references**, not an independent human assessment of model outputs. Conduct and document independent human adjudication of labels and answers before making a human-evaluation claim.
