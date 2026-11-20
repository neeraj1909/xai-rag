# Convert captured HTTP evidence

`to_dataset.py` adapts the earlier synthetic HTTP harness's saved HTTP and
optional RAGAS observations into the current `xai-rag evaluate` JSONL contract.
It reads files only and makes no model or network calls. It creates a new output
file exclusively; it will not overwrite an existing file.

For example, from the repository root (choose a new output path):

```bash
base=evals/archive/interview-evidence/evaluation/hardening-2026-11-20/http
uv run python evals/conversion/to_dataset.py \
  --input "$base/run/holdout-http.json" \
  --fixture "$base/fixtures/holdout-cases.json" \
  --ragas "$base/run/holdout-ragas.json" \
  --config-revision c375903fb46327744d076438cb90f7b02711a94f-dirty \
  --output /tmp/xai-converted-evidence.jsonl
uv run xai-rag evaluate /tmp/xai-converted-evidence.jsonl \
  --output /tmp/xai-converted-report.json
uv run pytest -q evals/conversion/test_convert.py
```

This converts an existing measurement; it does not rerun the system or judge.
The fixture bytes must match the recorded SHA-256. Select the corresponding
RAGAS file for the same HTTP run: case IDs alone cannot prove matching
provenance. The original partial RAGAS run above contains 15 computed scores,
one error and 24 unscored cases. The separately captured extension has a
combined RAGAS file; do not silently mix those scoring rounds.

The adapter retains every planned case, including unattempted cases, missing
prompt evidence, RAGAS status and errors. It records original expected behavior
and document labels as metadata; document IDs are not chunk relevance labels,
so the evaluator cannot calculate retrieval quality from them. Verification
status alone cannot resolve semantic abstention for a verified response, so such
cases have an unknown actual refusal. The synthetic, author-labelled fixture
has no independent human review and does not establish production quality.
The input run's revision and dirty working tree are documented in the
[historical evidence](../archive/interview-evidence/evaluation/hardening-2026-11-20/http/README.md).
