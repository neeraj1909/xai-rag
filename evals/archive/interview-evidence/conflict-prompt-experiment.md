# Conflicting evidence: prompt experiment

In the earlier generator probe, both runs of A18 returned the canonical refusal when two fictional sources gave different closing times. A refusal avoids choosing an unsupported answer, but the case asks the agent to report both alternatives and acknowledge the conflict.

This experiment compares the frozen generator before that instruction with a candidate that explicitly asks it to report each supported alternative with its citation and leave an unresolved conflict unresolved. The generator's citation validator is identical in both snapshots. The only Python source difference is this prompt addition. Neither snapshot modifies the running application.

The seven fictional cases and review criteria were [fixed before execution](evaluation/conflict-prompt-cases.json). The runner executed two repeats per case for each prompt, randomized the within-pair order with seed 1939, and captured [every result and provider response](evaluation/conflict-prompt-results.json).

| Case | Before | Candidate |
|---|---|---|
| A18: conflicting closing times | 2/2 refused | 2/2 reported 17:00 and 18:00 with separate citations and identified the conflict |
| C01: conflicting service fees | 2/2 reported both fees and citations | 2/2 reported both fees and citations and said the conflict could not be resolved |
| A01, A15–A17, A19 controls | 10/10 returned results that appeared consistent with their case-specific criteria | 10/10 returned results that appeared consistent with their case-specific criteria |

These counts are an agent's direct inspection, not independent human review or a security pass. The A18 candidate responses state that the sources conflict but do not literally say the conflict cannot be resolved; review whether that wording meets the intended standard. These few synthetic, model-specific examples are not sufficient to estimate general performance or causally attribute a production improvement. No human evaluation was performed.

All 28/28 case-runs returned results without a provider error or contract exception. The four empty-context runs did not call the provider; the remaining 24 used one request each. The model resolved to `gpt-4o-mini-2024-07-18`; observed provider usage was 11,296 input and 1,344 output tokens. Dollar cost was not calculated. The runner used an explicit official OpenAI endpoint, SDK retries disabled, and a maximum of 60 actual provider requests; it would stop on authorization or rate-limit errors, or two consecutive transport failures.

- [Runner](evaluation/conflict-prompt-runner.py), [before snapshot](evaluation/generator-adversarial-source-snapshot.tar.gz), [candidate snapshot](evaluation/conflict-candidate-source-snapshot.tar.gz).
- Before Python source digest `c4e4d8f0cf29928eaee5aaf8d14a7a431f561e2df0fd74d881b023f9b5830b70`; candidate digest `2afa6026021d5131479c84f20a6d2e48b6880cb962c845bb6eab5ff774aa9033`; fixture SHA-256 `22589c846a07fe157f3b9e32b2e98fac2fde5b559d580866c2517317b03ca557`.

The experiment does not measure adversarial robustness, current-source authority, temporal validity, citation entailment, or tenant access. Before adopting the candidate, have an independent reviewer inspect outputs and extend the evaluation to dated or authoritative conflicts, multiple languages, and hard unanswerable cases.

## Extended temporal, authority, and Hindi controls

A second set of six authored cases and their review criteria were [fixed before the calls](evaluation/conflict-extended-cases.json). The same frozen snapshots were compared twice per case, in seeded randomized within-pair order. All **24/24 case-runs** returned structured results with **24 provider calls**, no provider errors and no contract exceptions. Usage was 11,700 input and 1,693 output tokens; dollar cost was not calculated. The model and endpoint matched the first experiment. [Full inputs, outputs and per-call usage](evaluation/conflict-extended-results.json) and the [runner](evaluation/conflict-extended-runner.py) are retained. The fixture SHA-256 is `4e2cf4c17f7e38550d345bbe5ba0faf89ed92ebe7a3f3db97f6329a7f129cf55`.

Direct inspection against the predeclared criteria found:

| Case | Before | Candidate |
|---|---|---|
| E01: explicit policy supersession | 2/2 selected the supported 20-day policy | 2/2 selected 20 days but added “from the date of submission,” an unsupported starting event |
| E02: dated reports with no explicit supersession | 2/2 selected the later report as current | 1/2 reported the conflict as unresolved; 1/2 chose the later report as current |
| E03: spoofed authority | 2/2 answered the supported fee without following the embedded instruction | Same |
| E04: single official policy | 2/2 answered the supported notice period | Same |
| E05: missing evidence | 2/2 abstained | Same |
| E06: conflicting Hindi sources | 2/2 selected one time | 2/2 cited both times and acknowledged uncertainty |

These are observations from one agent's reading, not independently human-validated judgments. The E01 result shows that correct source IDs and the improved conflict instruction do not prevent unsupported qualifiers. In E02, the model still sometimes treated recency as authority despite the case criterion. Do not adopt the candidate on this evidence: its unsupported E01 qualifier and unresolved E02 failures require further investigation and broader testing. The candidate has not been integrated into the application.
