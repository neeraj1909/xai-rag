# XAI-RAG evaluation datasets

The files under `smoke/` validate the evaluator and CI wiring. They are deliberately synthetic and **must not be presented as production-quality evidence**.

## Dataset contract

Each JSONL file starts with one `manifest` record, followed by `query` and/or `ingestion` records. The manifest pins:

- dataset and corpus revisions;
- the pipeline configuration revision;
- every model revision used to capture observations;
- whether labels are synthetic, expert-reviewed, or derived from production feedback.

Query records keep graded relevance judgments separate from observed vector, BM25, fused, and reranked IDs. They also retain answer citations, claim-level NLI verdicts, refusals, errors, and per-stage latency. Ingestion records cover parse outcome, chunk distributions, source-span/provenance validity, and Chroma/Elasticsearch parity.

## Run locally

```bash
uv run xai-rag evaluate evals/smoke/evaluator-contract.jsonl \
  --gates evals/gates.toml \
  --output artifacts/evaluation-smoke.json
```

A blocking metric that is missing is a failure, not a pass. Advisory gates communicate proposed targets without implying approval while the product team is still collecting representative evidence.

## Repeated runs and reproducibility

A seed controls only the component to which it is applied. A case-order seed
can reproduce a shuffle; it does not seed generation, the grounding judge,
RAGAS, or an embedding model. For model requests, record whether the selected
endpoint actually received and accepted a seed. OpenAI describes the Chat
Completions seed as a deprecated, best-effort parameter and does not guarantee
identical results. Record the returned model and backend fingerprint when
available; an absent fingerprint is not evidence of an unchanged backend.
[OpenAI API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/)

For each experiment, record the source revision and any working-tree changes,
dataset and corpus hashes, index-content fingerprint and its scope, model and
dependency versions, prompt and parameter hashes, seed values and where they
were applied. Preserve per-case results, refusals, provider errors and unscored
cases. A logical index fingerprint is not necessarily an atomic storage
snapshot: document its coverage, such as whether vectors, search settings or
index-internal structures are included, and prevent writes during capture.

Report each run's planned and scored denominators before the mean and sample
standard deviation across run-level scores. The same fixed cases repeated three
times are not 120 independent users, and run-to-run variation does not measure
uncertainty about a production population. Inspect answerable and no-answer
cases separately: a correct refusal can receive a low answer-quality score.
Model-judged scores and author-created labels need independent human review
before they can support claims about production quality.

### Recorded three-run experiment

The [20 November 2026 archive](results/2026-11-20-seeded-repeated/) contains
three evaluations of the same 40 fictional, author-labelled cases over one
ingested 25-document corpus. The archive includes the manifest, HTTP results,
per-case RAGAS scores, provider-attempt ledger, index snapshots and the measured
harness. The source revision was `398be05288d0b822a1bc113629756ee253bc4495`;
the manifest also records the Python source-tree digest and a tracked
working-tree diff (documentation was modified). The fixture, dependency lock,
model revisions, retrieval settings and runner file hashes are recorded there.
Provider request captures preserve the actual prompts and parameters. Container
image IDs record what ran, but the tags used to launch future runs are mutable.

The current portable runner was separately checked in a live four-question
smoke test over 25 fictional documents: 4/4 HTTP responses and 4/4 RAGAS scores,
matching pre/post index fingerprints, and successful cleanup of its disposable
containers. Its 55 provider attempts comprised 47 seeded chat requests and
eight unseeded embedding requests. This validates the portable runner's basic
path; it is not the archived three-run measurement.

The order seed `20261120` fixes the case shuffle. Evaluation-only transport
instrumentation sent model seed `20261120` on AsyncOpenAI chat completions,
including generation, grounding and RAGAS judge requests; embeddings were not
seeded. It does not change the normal application configuration. The endpoint
accepted the parameter, but a separate three-call probe with that seed produced
two different serialized outputs. A seed improves experimental control, not
provider determinism. Repeating a single seed measures variation at that seed,
not sensitivity across different seeds.

Before and after each run, logical index fingerprints were captured and
compared while the fixture was quiescent. These capture document content,
metadata and vector representations, not every internal search-index structure
or an atomic storage snapshot. The archive records their exact scope and hashes.

RAGAS measures each scored case on a 0–1 scale (higher is better):

- **Faithfulness:** whether answer claims are supported by retrieved context;
  it does not independently establish that the context is true.
- **Answer relevancy:** how closely the response addresses the question;
  it does not by itself establish factual correctness.
- **Reference-based context precision:** whether retrieved passages relevant
  to the reference answer rank ahead of irrelevant passages; it depends on
  the reference and judge.

The [README results](../README.md#repeated-seeded-evaluation) show each run's
40-case mean and sample SD across the three run means. All 120 HTTP requests
returned 200; this says nothing by itself about answer quality. The original
scoring pass produced 40, 40 and 39 scores because run 3 case H08 encountered a
connection error. A separately budgeted scoring-only retry used the same
captured H08 response; the recovered aggregate has 40 scores per run. The
original, recovery input and output, and recovered aggregate are retained
separately. Original and retry attempts numbered 1,656 and 12 respectively.
Never substitute zero for an unscored case or silently discard a failure.

These cases have no independent human review, and the judge can share errors
with the generation model. Three repeats of fixed cases do not establish
production quality or a confidence interval. Use human-reviewed, representative
cases and calibrate the judge before making release claims. The separate
[earlier evaluation](results/2026-11-20-synthetic-http/) used an older working
tree; its scores should not be treated as a paired comparison.

## Build a real production set

1. Sample the actual query distribution, including rare intents, ambiguous queries, no-answer cases, safety cases, fresh/stale content, long documents, and known prior failures.
2. Freeze a corpus snapshot and pipeline/model revisions.
3. Have domain reviewers assign graded relevant chunk IDs and expected answer/refusal behavior. Use independent adjudication for disagreements.
4. Capture raw per-stage observations; never store only aggregate averages.
5. Calibrate NLI or LLM judges against a human-labeled subset, then monitor disagreement and drift.
6. Add low-quality production traces and user feedback back into the offline set after privacy review.
7. Obtain named owner approval for business thresholds before moving advisory gates to blocking.

At minimum, stratify reports by intent, document type, language, recency, answerability, and risk tier. Always report sample sizes alongside scores.
