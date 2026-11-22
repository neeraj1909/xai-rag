# Offline NLI stress diagnostic — 22 November 2026

The active `evals/nli/nli_stress.py` runner processed all 42 cases from
`evals/nli/stress-cases.json` against the current `FaithfulnessChecker` with
Hugging Face offline mode enforced. The report retains every case, model and
revision, thresholds, fixture/runner/checker SHA-256 hashes, and package
versions. The configured cached model was
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` at revision
`6f5cf0a2b59cabb106aca4c287eed12e357e90eb`; both thresholds were 0.5.

Run from the repository root:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python evals/nli/nli_stress.py \
  --output /tmp/xai-rag-nli-final-source-full-20261122.json
```

All 42 cases were scored and 37 matched their authored labels. Five did not:
N12 was labelled neutral and observed supported; N31, N32, N34 and N35 were
observed neutral instead of their supported or not-supported labels. This is a
fresh diagnostic, not a replacement for the historical report. The fixture is
synthetic and its labels are not independently reviewed, so agreement is not a
production accuracy estimate. Individual results and the full configuration are
in `stress-report.json`; neither retrieval nor generation is evaluated.
