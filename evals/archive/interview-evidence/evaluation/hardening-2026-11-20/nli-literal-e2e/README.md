# Literal claim rejected by the NLI gate

In a local final-validation E2E attempt using an OpenAI-compatible **stub**, the generator produced a claim that appears verbatim in its cited Markdown evidence. The captured NLI gate nevertheless returned `not_supported`; the API responded `insufficient_evidence` and withheld the answer. `selected-context.json` preserves only the matching claim, its cited evidence, NLI result and refusal—not the full application log or unrelated source chunks.

A separate one-example probe reported:

| Premise presented to the NLI | Entailment | Contradiction |
| --- | ---: | ---: |
| Original Markdown chunk | 0.0443 | 0.8765 |
| Sentence alone | 0.9932 | 0.0013 |
| Plain heading plus sentence | 0.9902 | 0.0022 |

Exact raw scores are in `nli-literal-probe.json`. This comparison is consistent with sensitivity to context or formatting on this example; it does **not** identify the root cause, establish an error rate, or prove that stripping Markdown is a safe general repair. The actual full E2E answer was a false refusal for a literal evidence match. Preserve it in quality denominators and investigate on a broader independently labelled set. Do not bypass the NLI or judge to make the demo appear successful.

The selected artifacts were checked against configured secret values and common key patterns without printing them. Original API logs and policy/credentials are not copied. This check is a best-effort privacy scan, not a guarantee that arbitrary data is safe to publish.
