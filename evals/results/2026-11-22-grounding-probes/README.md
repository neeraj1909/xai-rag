# Grounding diagnosis probes — 22 November 2026

These disposable CPU probes use synthetic, author-labelled examples and the
[failing E2E premise](../2026-11-22-docker-e2e/nli-probe.txt). They are
exploratory diagnostics, not independent human review, a representative
benchmark, a security guarantee, or an implementation recommendation. No
production code was changed by these probes. Scores are model outputs, not
calibrated probabilities of truth.

`model-comparison.py` compared the existing pinned English DeBERTa and pinned
multilingual mDeBERTa on the same five premise/claim pairs, using offline cached
models and two OpenMP/MKL threads. Results are in `model-comparison.jsonl`.
Both models rejected the actual E2E pair despite accepting the identical
sentence alone; both rejected a statement explicitly quoted as false. The
multilingual model is therefore not a fix for this particular case.

`markdown-probe.py` compared the original pair with heading markers stripped,
whole heading lines removed, and an explicitly false quotation. Outputs are in
`markdown-results.jsonl`. Removing only markers did not fix the false
contradiction. Removing heading lines did, but discards context and cannot be
recommended from this observation. A first concurrent attempt exited with
signal 4 before producing results; the saved output comes from a sequential
one-thread rerun.

`window-probe.py` tested a naive line/sentence splitter and one- and adjacent
two-sentence windows, including heading text as separate windows. It scores
all windows with the current NLI model and compares the maximum entailment and
maximum contradiction. The first output corresponds to full-premise baseline;
`windows` uses the current precedence rule (contradiction must reach 0.5 and
be at least as high as entailment). A stricter **diagnostic** alternative is:
reject if maximum contradiction is at least 0.5, otherwise accept if maximum
entailment is at least 0.5. These are not optimized or validated thresholds.
The 0.5 thresholds were taken from the current settings, not tuned to this set.

Seven of ten selected priority cases completed in about five minutes of shared
CPU time. `N20`, `N07` and `N08` were not completed in that initial run; the process was
stopped at the requested bound. All three were completed in the resumed run below. `window-results-partial.jsonl` retains the seven complete
cases and per-window scores. `N19` comes from the existing NLI stress fixture;
the rest are the E2E pair and newly constructed negative controls. The expected
labels are author hypotheses, including the inferred meaning of headings.

| Case | Expected | Full premise | Current aggregation | Strict contradiction veto |
| --- | --- | --- | --- | --- |
| E2E original | supported | not supported | supported | supported |
| Quoted/refuted assertion | not supported | not supported | not supported | not supported |
| Incorrect-statement heading with corrected fee | not supported | supported | not supported | not supported |
| Prohibited-actions heading | not supported | supported | supported | not supported |
| Explicitly negated fee | not supported | not supported | not supported | not supported |
| Two sources with conflicting fee | not supported | not supported | not supported | not supported |
| N19, two unrelated package prices | supported | supported | supported | supported |

The stricter diagnostic fits these seven examples, but this tiny sample cannot
establish safety. Taking the maximum contradiction over noisy fragments can
falsely veto a supported answer, and splitting can remove negation, attribution,
temporal qualification or table structure. Conversely, the current comparison
rule accepted the prohibited-actions example even with a window contradiction
score of 0.9946 because maximum entailment was 0.9956. The baseline also falsely
accepted both heading examples. A candidate must be tested on a separate,
representative, independently labelled set with false-accept and false-reject
rates, including cross-source conflicts and malformed/long documents, before
any change to the release boundary.

The research scripts read repository files relative to its root, depend on the
current project environment and cached models, and are snapshots of a bounded
experiment. Running them again can take several minutes and the window script
will continue through the three unfinished cases. Historical results are not
modified.

## Independent judge probe

`judge-probe.py` exercised the existing `GroundingJudge` directly with seven
synthetic cases, using a fabricated NLI-supported verdict and citation to reach
its interface. It therefore did not run the normal NLI gate and is not an
end-to-end application measurement. The requested model was `gpt-4o-mini`;
the API returned `gpt-4o-mini-2024-07-18`. One call per case was made with API
default sampling controls and process-local proxy bypass. `judge-results.json`
contains manually transcribed responses; consult the script for exact inputs.
No scoring conclusion is reliable from this small authored sample.

The judge released the original supported case and rejected the negative
controls involving an explicitly negated claim, an explicit refutation,
injection, irrelevant information, and an incomplete multi-part answer. It
**also released a claim when a second supplied source directly contradicted
it**, contrary to the expected decision. The judge alone cannot be treated as a
reliable conflict veto on this evidence.

The `nli-probe.py` / `nli-results.jsonl` files and the `nli-markdown-probe.py` /
`nli-markdown-results.jsonl` files are byte-identical to the corresponding
`model-comparison` and `markdown` filenames retained above; they reflect the
same measurements, not additional replications.

## Resumed window probe and extra counterexamples

The separate `window-resumed-probe.py` completed the original three unfinished
cases (`N20`, `N07`, `N08`), then added three author-constructed temporal,
conditional, and attribution counterexamples. It used the same pinned English
NLI model, 0.5 thresholds and naive window algorithm, CPU with two OpenMP/MKL
threads, and cached offline weights. All six cases completed in about 2 minutes
40 seconds of shared wall time. The original seven-case result and source script
remain unchanged; raw scores and every window for this run are in
`window-resumed-results.jsonl`.

| Case | Expected handling | Full premise | Existing windows | Strict contradiction veto |
| --- | --- | --- | --- | --- |
| N20: mismatched package price | reject | reject | reject | reject |
| N07: paraphrase of a prohibition | accept | accept | accept | accept |
| N08: opposite of a prohibition | reject | reject | reject | reject |
| Previous/current fee headings | reject | reject | reject | reject |
| Denied discount, conditional fee | withhold as unestablished | **accept** | **accept** | **accept** |
| Rumor explicitly denied | reject | reject | reject | reject |

The conditional example's original machine-readable `expected` field says
`not_supported`, meaning the unconditional claim should not be released. More
precisely, the premises do not determine whether the fee is $10 for some other
reason, so its NLI label should be **neutral**, not necessarily contradiction.
The source output is retained unchanged to preserve the actual run. The model
nonetheless marked it supported: under windows, maximum entailment was 0.9839
and maximum contradiction 0.2113. This is a false accept under all three tested
rules. It directly disproves treating the window/veto candidate as generally
safe merely because it handled the first seven examples. Additional independent,
representative review and an evaluated treatment of conditional statements are
needed before changing the release policy.

## Same-input repair and evidence-boundary check

`repair-same-input-probe.py` evaluated the original 286-character E2E premise
against the exact initial claim, the deterministic E2E stub's shorter repair,
and a contradictory claim, with the final source's `FaithfulnessChecker`. It
also evaluated that repair with a deliberately truncated 70-character premise,
an incorrect chunk ID, and no evidence. The [raw output](repair-same-input-results.jsonl)
records individual model scores, citation decisions and visible lengths. It used
CPU/offline mode, pinned `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` revision
`6f5cf0a2b59cabb106aca4c287eed12e357e90eb`, thresholds 0.5, checker SHA-256
`fff00707c5e72a487ad90e9d6bfb595e5c1501467b86a4f94b2894549feab1cc`,
PyTorch 2.13.0+cpu and Transformers 5.15.1. NLI label expectations are
synthetic, author-authored hypotheses rather than independent annotation.

The original claim was rejected (entailment 0.0443, contradiction 0.8765),
while the stub's repair was supported (0.9541 / 0.0067). A direct negation was
rejected (0.0002 / 0.9995). The truncated premise gave a neutral verdict and no
rendered answer; a mismatched source ID and absent evidence were rejected by
citation validation without running NLI. These are local scoring and citation
checks: they do **not** run generation, the independent judge, HTTP, or the
repair loop, and a rendered candidate is not itself a released verified answer.
The separate [final-image E2E](../2026-11-22-grounding-final-e2e/README.md)
exercised the loop with deterministic generation/judge stubs and real NLI.

Focused `tests/unit/test_service.py`, `test_generation_repair.py`,
`test_grounding_judge.py`, `test_faithfulness.py`, and
`test_nli_normalized_evidence.py` passed 97 tests, with one Chroma deprecation
warning. The judge's prior cross-source conflict miss, conditional NLI false
accept and limited coverage still apply. This small diagnostic does not prove
that all verifier, prompt-injection, truncation or repair failure modes are
covered, and the real-provider sample did not trigger repair.
