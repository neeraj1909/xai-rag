# Active evaluation workflow validation

The commands below ran on **2026-11-22** against checkout
`bebe3bd44412cbd4784beb922203336a14df8988`.

## Focused regression: expected-label leakage

```sh
uv run --no-sync python evals/results/2026-11-22-eval-validation-followup/reproduce_converter_regression.py
```

The script copies the converter and its existing test to a temporary directory.
It changes only the verified-answer classification in the copy from unknown to
the fixture's expected refusal label. The paired cases have the same response
but opposite expected labels. The mutation makes the focused assertion fail:
`[False, True]` instead of `[None, None]` (exit 1). Restoring the unmodified
current converter makes the same test pass (exit 0). See
[`red-mutated.log`](red-mutated.log), [`green-current.log`](green-current.log),
and [`regression-summary.log`](regression-summary.log). The current converter
SHA-256 is `9fe8677fba8cc12241f7b599d125d1e00652a93bca233004369fe3c5abd8e830`.

This is a **mutation-based reproduction** of the failure mode, not retained
pre-implementation test output. It demonstrates that the existing regression
test detects the specific expected-label leakage. It does not establish model
quality, prove all possible classifications, or retrospectively prove when the
original check was written. No project or historical source was mutated.

## Six workflow focused checks

```sh
uv run --no-sync pytest -q evals/retrieval evals/conversion evals/nli \
  evals/live_retrieval evals/multilingual evals/judge
```

Result: **15 passed** in 6.82 seconds; see
[`focused-suite.log`](focused-suite.log). These checks cover ranking replay,
captured-response conversion, NLI accounting, retrieval metrics and fixture
validation, multilingual ranking, and judge-runner accounting. They use
controlled inputs; they do not validate live model or service quality.
