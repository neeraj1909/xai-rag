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

## Build a real production set

1. Sample the actual query distribution, including rare intents, ambiguous queries, no-answer cases, safety cases, fresh/stale content, long documents, and known prior failures.
2. Freeze a corpus snapshot and pipeline/model revisions.
3. Have domain reviewers assign graded relevant chunk IDs and expected answer/refusal behavior. Use independent adjudication for disagreements.
4. Capture raw per-stage observations; never store only aggregate averages.
5. Calibrate NLI or LLM judges against a human-labeled subset, then monitor disagreement and drift.
6. Add low-quality production traces and user feedback back into the offline set after privacy review.
7. Obtain named owner approval for business thresholds before moving advisory gates to blocking.

At minimum, stratify reports by intent, document type, language, recency, answerability, and risk tier. Always report sample sizes alongside scores.
