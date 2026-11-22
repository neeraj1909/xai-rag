# Experimental per-source adjudicator probes — 22 November 2026

These are disposable, authored synthetic examples evaluating a possible judge
change, not production code or a release recommendation. The existing NLI gate
remains in force. Neither the small sample nor the author labels are independent
human review, and model judgments are fallible.

## Method

The baseline invokes the project's actual `GroundingJudge` with a fabricated
NLI-supported verdict and supporting source `s0` solely to reach its interface.
It therefore bypasses the actual NLI gate and does **not** demonstrate an
end-to-end application release. Baseline passes all visible evidence to the
judge but specifies only one supporting source ID. The experimental prompt asks
for an entailment/contradiction/unknown decision for each supplied source and
computes release in code only if (a) cited `s0` is entailed, (b) no supplied
source contradicts the claim, and (c) relevance, safety and completeness are
true. Missing/duplicate/wrong source IDs or malformed responses are errors,
not successful verdicts. The implementation used in the probe is **not**
shipped and does not add this gate to the application.

Every row saves the full synthetic question, claim, source texts, author
expectation, structured response, actual returned model name, token usage and
wall-clock time. Calls requested `gpt-4o-mini` or `gpt-4.1-mini`; the
per-source classifier explicitly requests temperature zero, while the existing
judge uses its actual provider defaults. The application uses these judge calls
independently; no assumptions of independent model errors or guaranteed
correctness should be inferred. Source and case labels are synthetic.

The first 16-case selection contains the original E2E premise, the prior
cross-source conflict, conditional, quoted-refutation, injection, relevance,
question completeness and heading examples, plus additional authored cases.
A deliberately unqualified answer to a weekday-only library schedule is
labelled withheld; this is a conservative author interpretation, not an
independent annotation. `probe.py` and `probe-gpt41.py` record the baseline and
the first per-source prompt on the same inputs. `probe-clarified*.py` add an
explicit instruction to consider headings and temporal/conditional scope; the
prompt was selected after inspecting the earlier outcomes, so these 16 cases
are **development**, not a blind or generalization test.

`probe-holdout-gpt41.py` adds 18 later, newly written examples after prompt
selection. These are an additional **author-created** case set, not an
independently labelled representative holdout. They include unsupported numbers,
non-cited conflicts, wrong citation, conditional cases, scope, prompt injection,
known unknowns and true controls. All case definitions and exact prompts are
in the scripts. Only one call per method/case was made; stochastic variability
was not measured.

All actual direct OpenAI calls used the existing project key and explicitly
selected the default OpenAI base URL with process-local proxy variables removed.
No key values are stored. During initial setup, the generic SDK's inherited
`OPENAI_BASE_URL` pointed to a different endpoint despite proxy removal. The
first candidate calls returned 401 while the app's explicitly addressed judge
succeeded; `initial-endpoint-mismatch.jsonl` retains those failures and baseline
results. The corrected `probe.py` names the actual endpoint; it is not the exact
script used for those initial candidate calls.

## Results

| Run | Existing judge | Per-source classifier | Limitation |
| --- | --- | --- | --- |
| First 16 cases, `gpt-4o-mini` | 13/16 agreed | 12/16 agreed | Per-source corrected RAG conflict, but accepted irrelevant and incomplete cases. |
| First 16 cases, `gpt-4.1-mini` | 12/16 agreed | 14/16 agreed | Both accepted prohibited heading and weekday scope cases. |
| Clarified prompt, `gpt-4o-mini` | Not rerun | 12/16 agreed | Same four disagreements as its first per-source run. |
| Clarified prompt, `gpt-4.1-mini` | Not rerun | 15/16 agreed | Accepted weekday-only evidence as an unqualified rule; same development cases used to motivate clarification. |
| Later 18 cases, `gpt-4.1-mini` | 15/18 agreed | 17/17 returned decisions agreed; **1 error** | Invalid source-ID response for wrong-citation case; fail closed. |

The baseline missed an uncited-source conflict, a stated unknown cause, and an
ambiguous prohibited heading in the later set. The experimental classifier
made one schema-valid response that failed its source-ID consistency check,
which remains a verification error even though withholding the answer would
match the authored release expectation. We do not silently count errors as
scores. The model's success on the two conflicted examples does not prove it
will detect all contradictory sources, and it has already accepted an
unqualified weekday-only answer on the development cases.

The case count, response detail, errors and usage are in `results.jsonl`,
`gpt41-results.jsonl`, `clarified-results.jsonl`,
`clarified-gpt41-results.jsonl` and `holdout-gpt41-results.jsonl`. No calculated cost estimate is needed for evaluating the decisions; each
successful call records exact per-call usage. Re-run only with a new output path:
the scripts intentionally open outputs exclusively.

## What this does and does not support

A per-source classification can give the model a better chance to examine
conflicts than a single overall decision; these samples expose both benefits and
new mistakes. Combining it with the existing judge rather than replacing that
judge may catch more errors but adds latency/cost and does not guarantee error
independence. It cannot simply overrule the current NLI contradiction: the
same model can fail on qualifiers or source interpretations. Before proposing
any change, evaluate a frozen independently reviewed case set, cases with 16
claims and many cited/uncited sources, repeated sampling and failure/timeout
handling, exact citation provenance, and an honest status for uncertain claims.
No measured result here makes the existing deterministic E2E pass through the
real application boundary.
