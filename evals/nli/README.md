# Offline NLI stress diagnostic

Run from the repository root with the configured NLI model revision already in
the Hugging Face cache:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-sync python evals/nli/nli_stress.py \
  --output /tmp/xai-rag-nli-stress.json
uv run --no-sync pytest -q evals/nli/test_runner.py
```

Choose a new output filename for each run. The runner forces Hugging Face
and Transformers offline mode before importing the checker, records the
configured model, revision, thresholds, source and fixture hashes, and
retains per-case errors. A missing cached model produces unscored cases and a
nonzero exit. Agreement is reported with both planned and scored denominators;
a mismatch is diagnostic, not a release gate. The default fixture is a copy of
`evals/archive/interview-evidence/evaluation/nli-stress-cases.json`; the original
runner and historical results remain in that archive.

The 42 synthetic pairs were authored with their labels and have not been
independently reviewed. They are an exploratory stress set, not a representative
holdout or production-quality estimate. This checks the current single-chunk
NLI checker, not retrieval, answer generation, safety, or the historical
measurement. A different model, threshold, or source revision is a different
experiment; compare configuration and hashes before comparing results.
