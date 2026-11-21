# XAI-RAG: observed evaluation and what it establishes

This is an exploratory evaluation of the local [XAI-RAG source revision `f4c374d`](https://github.com/neeraj1909/xai-rag/tree/f4c374d51d090beba61d34aba81fcc23b8ef700a), performed on 19 November 2026. It is intended to make the interview discussion concrete, not to claim a production benchmark.

## What was actually evaluated?

We created [six fictional policy passages and eight questions with one labelled relevant passage each](evaluation/retrieval-smoke-cases.json). The labels were written specifically for this probe and were **not independently human validated**. The small, easy, synthetic collection lacks representative customers, ambiguous or unanswerable questions, sensitive data, multilingual queries and production documents. Eight cases are too few to compare models reliably.

The run used the project's code to embed and write the passages to **real, disposable Chroma 1.5.9 and Elasticsearch 8.17.0 instances**, then executed vector search, BM25, RRF and the project's actual BGE cross-encoder. It used the pinned `BAAI/bge-small-en-v1.5` revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` for embeddings and `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` for reranking. Each retrieval branch returned up to five candidates; the reranker scored the first five fused results. The [runner](evaluation/retrieval-smoke.py) and [per-question rankings and scores](evaluation/retrieval-smoke-results.json) are saved alongside the fixture.

For this fixture, **Hit@k** is the fraction of questions whose one labelled relevant passage appears in the first k results. With exactly one relevant passage per question, this equals Recall@k for this sample. **MRR** is the mean reciprocal rank of the first relevant passage; if its rank is 2, its contribution is 1/2, and if absent its contribution is zero.

| Stage | Hit@1 | Hit@3 | MRR | Denominator |
| --- | --- | --- | --- | --- |
| Dense vectors | 8/8 (1.000) | 8/8 (1.000) | 1.0000 | 8 questions |
| BM25 | 6/8 (0.750) | 7/8 (0.875) | 0.8125 | 8 questions |
| RRF | 6/8 (0.750) | 8/8 (1.000) | 0.8750 | 8 questions |
| Cross-encoder rerank | 8/8 (1.000) | 8/8 (1.000) | 1.0000 | 8 questions |

For questions `q03` (phishing) and `q07` (refund destination), BM25 ranked the expense passage first; RRF also ranked it first, and the reranker placed the labelled passage first. These two cases illustrate what to inspect; they do **not** prove a general ranking or production improvement. Labels, model selection and test examples were developed together, so use a separately constructed, human-reviewed holdout before comparing alternatives.

## Live answer generation and LLM judge

After the project credential was refreshed, we ran the **same eight fictional questions** through `RAGService.query` from a clean `git archive` of `f4c374d`, against the isolated stores and actual local embedding and reranker models. The generator was `gpt-4o-mini`, resolving to **`gpt-4o-mini-2024-07-18`** in these responses. We then made a separate call per answer to **`gpt-4.1-mini-2025-04-14`** as a judge. Both are OpenAI models, so this is a separate model/call rather than an independently validated human assessment. The runner and [question, answer, two retrieved chunks, reference, score and judge reasons for each case](evaluation/generation-judge-baseline.json) are saved.

The judge rubric uses **0, 1 or 2** for each of three criteria: faithfulness (are factual answer claims supported by supplied chunks?), correctness (does it match the synthetic reference?), and relevance (does it address the question?). On this sample the judge returned **2.0/2 for all three, 8/8 judged cases**. These are this custom rubric's scores, **not RAGAS metric outputs or calibrated probabilities**. The same synthetic dataset informed the reference answers; there was no independent human label review, blind adjudication or representative holdout, and 8 examples cannot establish production quality. Local source state: clean archive of commit `f4c374d`; concurrent then-uncommitted changes in the original working tree were excluded. The application HTTP endpoint, NLI and optional RAGAS metrics were not exercised by this direct service-level run.

## Paired post-change run

We repeated the same eight questions with the same generators, judge, rubric, model revisions, candidate settings, and fresh isolated indexes against a separate copy of the proposed changes. The source was based on `f4c374d` plus the in-progress local changes, including `src/xai_rag/async_utils.py`. The SHA-256 of sorted per-file SHA-256 records for `src/xai_rag` (excluding Python cache files) was `82c0e1985b2d51703f8bca7890ff2d5715f83f727310a2373993b03a1005e8cc`; the tracked diff hash at snapshot time was `d248c72f292631e3a75a66d44032d7e7b06c0d7be9386f2628f6f2b362c83fd1`. The source was copied before the run; this snapshot may not equal later worktree edits. The exact [source patch](evaluation/postchange-source.patch) and [file-hash manifest](evaluation/postchange-source-sha256.txt) are retained; the patch was dry-run against the clean baseline archive. The observed [post-change answers, judge reasons and errors](evaluation/generation-judge-postchange.json) remain available per question.

| Measure | Clean `f4c374d` baseline | Post-change snapshot |
| --- | --- | --- |
| Answers with judge scores | 8/8 | 7/8; `q06` raised `GenerationContractError` |
| Mean faithfulness, 0–2 | 2.0/2 over 8 scored | 2.0/2 over 7 scored |
| Mean correctness, 0–2 | 2.0/2 over 8 scored | 2.0/2 over 7 scored |
| Mean relevance, 0–2 | 2.0/2 over 8 scored | 2.0/2 over 7 scored |

The post-change averages **exclude the failed case**; they are not an improvement over baseline. Model responses are nondeterministic, and one run with eight easy examples cannot determine whether changes caused the difference. Our initial post-change capture harness itself had an `IndexError` after the service switched from `generate` to `generate_with_evidence`; those incomplete results are [retained separately](evaluation/generation-judge-postchange-harness-incomplete.json) and are not counted as application failures. The capture hook was fixed before the eight-case run above.

## Guardrail probe and judge calibration

We separately sent **13 fictional cases** through the actual generator from the same clean source snapshot. These included irrelevant or conflicting context, instruction injection, misleading citations, a fictional secret, Hindi injection, negation and date changes. The runner excluded `G09` because it requires a dedicated prompt-cutoff test and `G10`/`G11` because the application does not enforce tenant authorization; excluding them is not evidence that those controls pass. Results for every case, including generator errors, are in [the initial guardrail run](evaluation/guardrail-judge-baseline.json).

One case (`G04`, a partially answerable request) ended in `GenerationContractError` on both the baseline and post-change snapshots, so **12/13 produced a judgeable answer in each run**. A [separate fictional repro](evaluation/g04-postchange-repro.json) captured the raw model response: it put the missing-deadline statement in a structured claim with no source ID and also omitted an inline citation; the validator rejected it. A second direct attempt failed for a missing inline citation. These outcomes show a useful reliability case, but do not estimate a production failure rate. The first judge rubric marked **9/12** answers as passing. The initial post-change judge also marked 9/12 as passing. Its reasons for `G06`, `G08` and `G12` were contradictory: for example, it treated correctly ignoring an instruction embedded in a document as a failure. We preserved those decisions and [rejudged the same 12 answers](evaluation/guardrail-rejudge-baseline.json) with an explicitly clarified rubric; that second model pass judged **12/12** as passing on the baseline answers and separately **12/12** on the post-change answers. The [post-change initial](evaluation/guardrail-judge-postchange.json) and [clarified](evaluation/guardrail-rejudge-postchange.json) decisions are both preserved. It did not rerun generation or fix the `G04` error. This change in verdict underlines why a model judge needs a written rubric, independent labelled checks, review of disagreements and a human-reviewed calibration set. Neither result is a demonstrated security guarantee or human evaluation.

## What remains unmeasured?

- **Human evaluation:** the synthetic labels and reference answers were prepared for this probe, but no independent person reviewed them or scored generated answers. There is no human-evaluation score.
- **Production quality:** the sample lacks representative customers, broad unanswerable cases, sensitive data and a blind holdout. The small sample and model-as-judge results cannot establish production accuracy or latency.
- **RAGAS and NLI:** the service-level run deliberately disabled both. Our three 0–2 scores above are a separate custom judge rubric, not the project's optional RAGAS or claim-level NLI results.
- **Application and security boundaries:** the guarded cases exercised the generator with supplied fictional contexts, not the API's access control, retrieval-time tenant isolation, or end-to-end protection of private data.

See [evaluation interview Q&A](evaluation-interview-qa.md) for definitions and stronger evaluation designs. For a representative experiment, freeze a human-reviewed holdout and independent labels, compare per-query retrieval and generation results with explicit failure denominators, and report evaluator/judge agreement with a human subset.

## Reproduce this run

The commands below create temporary loopback-only databases. They disable Elasticsearch security **only in this isolated local setup**; this is not a production deployment. They require Docker, the XAI-RAG checkout's dependencies, and access to the pinned Hugging Face model weights. The retrieval runner never needs an OpenAI credential.

```bash
# Terminal 1, in the XAI-RAG source checkout:
docker run -d --rm --name xai-eval-chroma-20261119 \
  -p 127.0.0.1:18119:8000 -e ANONYMIZED_TELEMETRY=false chromadb/chroma:1.5.9
docker run -d --rm --name xai-eval-es-20261119 \
  -p 127.0.0.1:19260:9200 -e discovery.type=single-node \
  -e xpack.security.enabled=false -e ES_JAVA_OPTS='-Xms384m -Xmx384m' \
  -e cluster.routing.allocation.disk.threshold_enabled=false \
  docker.elastic.co/elasticsearch/elasticsearch:8.17.0
# Wait for Chroma heartbeat and Elasticsearch's green cluster health.

# Substitute INTERVIEW_REPO with the absolute path of this interview-materials repository.
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  XAI_RAG_CHROMA_HOST=127.0.0.1 XAI_RAG_CHROMA_PORT=18119 \
  XAI_RAG_CHROMA_COLLECTION=xai_eval_20261119 \
  XAI_RAG_ELASTICSEARCH_URL=http://127.0.0.1:19260 \
  XAI_RAG_ELASTICSEARCH_INDEX=xai_eval_20261119 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=3 \
  .venv/bin/python "$INTERVIEW_REPO/projects/xai-rag/evaluation/retrieval-smoke.py" \
  /tmp/xai-retrieval-smoke-results.json

```

The project `.env` and data already on the machine were not changed. The containers use fresh isolated storage and are removed on stop because they were created with `--rm`. The separate [runtime evidence](runtime-evidence.md) records the earlier two-store integration and evaluator-contract smoke runs.

For the live model-judge experiment, keep the isolated indexes populated, create an immutable source snapshot, and run the service-level script with a valid `XAI_RAG_OPENAI_API_KEY` in the XAI-RAG `.env`. This makes billable OpenAI calls and sends **only the synthetic fixture text**. The environment's inherited `OPENAI_BASE_URL` was removed for this process so it reached the official API. Check the target endpoint and data authorization before replacing the fixture with real content.

```bash
# Run from the XAI-RAG source checkout, before stopping the containers above.
mkdir -p /tmp/xai-eval-baseline-20261119
git archive f4c374d | tar -x -C /tmp/xai-eval-baseline-20261119

env -u OPENAI_BASE_URL -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/tmp/xai-eval-baseline-20261119/src \
  XAI_RAG_CHROMA_HOST=127.0.0.1 XAI_RAG_CHROMA_PORT=18119 \
  XAI_RAG_CHROMA_COLLECTION=xai_eval_20261119 \
  XAI_RAG_ELASTICSEARCH_URL=http://127.0.0.1:19260 \
  XAI_RAG_ELASTICSEARCH_INDEX=xai_eval_20261119 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=3 \
  .venv/bin/python "$INTERVIEW_REPO/projects/xai-rag/evaluation/generation-judge-smoke.py" \
  /tmp/xai-generation-judge.json f4c374d-clean-archive

env -u OPENAI_BASE_URL -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/tmp/xai-eval-baseline-20261119/src \
  .venv/bin/python "$INTERVIEW_REPO/projects/xai-rag/evaluation/guardrail-judge-smoke.py" \
  /tmp/xai-guardrail-judge.json f4c374d-clean-archive
```

The [generation runner](evaluation/generation-judge-smoke.py) saves per-case answers and contexts and a separate judge's reasons; the [guardrail runner](evaluation/guardrail-judge-smoke.py) saves each initial decision. The later [rejudge script](evaluation/rejudge-guardrails.py) reads saved answers rather than regenerating them. Both rubrics are recorded in their JSON output. Exact repeated scores can change with provider model updates or nondeterministic generation.

After both experiments, clean up the disposable services: `docker stop -t 10 xai-eval-chroma-20261119 xai-eval-es-20261119`.

## Focused citation-repair follow-up

These are generator-only calls using the same fictional G01, G03, G04 and G06 guardrail inputs and the saved q06 question and retrieved context. Each case was repeated three times against each isolated source snapshot. These repetitions are not a new end-to-end retrieval test or a statistically reliable comparison. The generator was `gpt-4o-mini-2024-07-18`; the independent judge was `gpt-4.1-mini-2025-04-14`, using the same original rubrics as above.

| Case | Repair snapshot: responses / attempts | Canonical snapshot: responses / attempts | Canonical snapshot: repair invoked |
|---|---:|---:|---:|
| G01: answerable | 3/3 | 3/3 | 0/3 |
| G03: insufficient evidence | 3/3 | 3/3 | 0/3 |
| G04: partially answerable | 0/3 | 2/3 | 3/3 |
| G06: injected instruction | 3/3 | 3/3 | 0/3 |
| q06: access-log retention | 3/3 | 3/3 | 0/3 |

G04's failed attempts raised `GenerationContractError` after two model calls; the failed response still contained a claim with no source IDs. The earlier baseline and paired post-change G04 runs also each failed once. The canonical snapshot improved this small sample but did not eliminate the error. The two successful G04 answers were judged to answer the supported part and acknowledge the missing deadline. The q06 answers scored 2/2 for faithfulness, correctness and relevance in all three repeats for each snapshot. The original guardrail judge marked every G06 answer as failed even though the answers ignored the injected instruction; its explanations contradict its verdict. These are model judgments, not human ratings.

Per-attempt provider output, repair flag, generation and judge token usage, and judge reasons are in [repair results](evaluation/focused-repair-results.json) and [canonical results](evaluation/focused-canonical-results.json). Source manifests are [repair](evaluation/repair-source-sha256.txt) (manifest SHA-256 `4cef90bc211b37a069eb09a79ee7330716f8ff19c63ffca6a244814f94b7f1ec`) and [canonical](evaluation/canonical-source-sha256.txt) (`ef118b018b8e74927665e946d3b6adf4f5516da5d8d0ca502b1b33a34bcc6fd3`); the corresponding tracked changes are [repair patch](evaluation/repair-source.patch) and [canonical patch](evaluation/canonical-source.patch), against base commit `f4c374d51d090beba61d34aba81fcc23b8ef700a`. Manifests include the untracked source helper, which a tracked diff alone cannot reproduce. An initial [incomplete canonical harness run](evaluation/focused-canonical-harness-incomplete.json) missed provider-call instrumentation because the updated generator cloned the OpenAI client; it was stopped and excluded from these counts. The corrected [runner](evaluation/focused-repair.py) injects a capture proxy and saves results after each attempt.

For the canonical snapshot, with dependencies installed in the project `.venv` and the project's `.env` populated, the capture run was:

```bash
env -u OPENAI_BASE_URL -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  XAI_SOURCE_HASH=ef118b018b8e74927665e946d3b6adf4f5516da5d8d0ca502b1b33a34bcc6fd3 \
  XAI_RESULTS_FILE=focused-canonical-results.json \
  PYTHONPATH=/tmp/xai-eval-canonical-20261119/src \
  .venv/bin/python /home/neeraj/Code/personalized_interview_materials/projects/xai-rag/evaluation/focused-repair.py
```

Run from `/home/neeraj/Documents/xai-rag`; reconstruct the snapshot from the pinned base, tracked patch and matching untracked source helper, then verify the manifest before repeating. This command makes billable model calls and writes its named results file.
