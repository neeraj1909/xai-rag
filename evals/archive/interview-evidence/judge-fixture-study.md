# XAI-RAG judge sensitivity: authored fictional cases

This experiment asks whether an LLM judge follows a deliberately explicit citation-aware rubric on examples with known intended outcomes. It is **not human validation**: an AI assistant authored both the 16 fictional cases and their expected labels, so the labels can contain errors and are not independent expert ground truth. The cases were written down before the judge ran. They are a small development set, not a representative test of judge quality, production answers, or robustness to unseen attacks.

## Method

- Dataset: [16 complete fictional cases and author-expected labels](evaluation/judge-fixture-cases.json), including supported controls, mixed support, numeric and negation contradictions, incorrect and nonexistent citation IDs, correct abstention and false refusal, Hindi text, and a source containing an injected instruction.
- Judge: `gpt-4.1-mini-2025-04-14`, requested as `gpt-4.1-mini`; 16 calls completed out of 16 attempted, no retries. The judge saw each case's question, evidence, reference answer and candidate answer. It did **not** see the expected labels.
- Rubric: faithfulness `2` if every factual claim is supported by its cited evidence; `1` for mixed support; `0` if no factual claim is supported; `null` for abstention with no factual claim. Correctness is `2` for fully correct, `1` for partly correct and `0` for incorrect or a false refusal. The full system prompt and JSON schema appear in the [raw results](evaluation/judge-fixture-results.json). Correctness can be high even if the citation is wrong.
- Repeatability: [runner](evaluation/judge-fixture-run.py), fixture SHA-256 `ad995dbfba6239085cf65b1ceec8637197de74bc89beb259e470d9eb06aa0626`, `temperature=0`, `max_tokens=250`, strict JSON output. The runner uses only fictional case data and reads the project credential without saving or printing it. Provider behavior can still change between calls or model revisions.

## Results

| Measure | Observed | Denominator and interpretation |
| --- | --- | --- |
| Exact faithfulness-score agreement with authored labels | 12/14 | Only cases with a factual claim; two abstentions excluded. |
| Correctness-score agreement with authored labels | 16/16 | Includes both abstentions. This set is easy and intentionally constructed. |
| Unsupported or mixed answers incorrectly scored fully faithful | 0/9 | The nine cases with authored faithfulness `0` or `1`; no estimate beyond these cases. |
| Fully supported answers scored below fully faithful | 0/5 | The five cases with authored faithfulness `2`. |
| Abstentions assigned `null` for faithfulness | 1/2 | The judge assigned `2` to one correct abstention, despite the rubric requesting `null`. |

The judge disagreed on three faithfulness outcomes. In **J02** and **J12** it returned `0` for an answer containing one supported and one unsupported factual assertion; the expected mixed-support score was `1`. Its reasons discussed the incorrect assertion but did not credit the supported portion. In **J08** it returned `2` for a correct abstention, although this rubric explicitly requested `null` when no factual claim was made. Wrong-source and nonexistent-source examples J05, J06 and J07 matched the authored labels on this run. Read the individual responses before drawing conclusions: a score is not a substitute for the accompanying rationale.

The 16 successful judge calls used **5,316 input tokens and 1,199 output tokens**, 6,515 total. The actual account charge was not retrieved and is therefore **not reported as a dollar amount**; billing, model pricing and cached-token treatment depend on the provider and account. No generator or human-review calls were part of this experiment.

## What this means for a production evaluation

A judge can miss rubric requirements even when it returns valid structured JSON. These cases expose **claim decomposition** and **abstention/null handling** as failure modes: segment multi-claim answers, distinguish unsupported claims from refusals, and run deterministic source-ID checks before semantic judging. A future evaluation needs a separate, unseen and independently human-annotated dataset, reviewer agreement, and explicit analysis of false-safe and false-unsafe outcomes. Do not tune on these same 16 cases and then call the result a held-out score.

To reproduce, move the existing result file aside and run, from the interview-materials repository:

```bash
/home/neeraj/Documents/xai-rag/.venv/bin/python projects/xai-rag/evaluation/judge-fixture-run.py
```

This makes **16 billable API attempts** and refuses to overwrite the existing result file. The script clears inherited proxy variables only in its own process; it does not change system configuration. Store the new results separately and compare model version, prompt, fixture digest and denominators before comparing scores.
