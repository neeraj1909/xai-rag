# Exploratory NLI stress study

This probe ran 42 fictional premise–claim pairs, balanced across 14 authored labels each for supported, contradiction, and insufficient evidence. The fixture was written and hashed before inference, but the author knew earlier diagnostic outcomes. It is neither independently human reviewed nor an external or representative holdout. The labels are provisional; agreement on this set is not a production accuracy estimate.

The actual `FaithfulnessChecker._check_single_claim` evaluated one declared evidence chunk per claim with the configured [DeBERTa model and pinned revision](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli/tree/6f5cf0a2b59cabb106aca4c287eed12e357e90eb), 512-token truncation, and entailment/contradiction thresholds of 0.5. `not_supported` means the checker found contradiction above threshold; `neutral` means neither threshold won. The model ran offline from cache, without provider calls. This probes the checker in isolation, not retrieval, claim extraction, answer generation, HTML formatting, multi-chunk conflict resolution, or a production deployment.

## Confusion matrix

Rows are AI-authored expected labels; columns are checker verdicts. “Contradiction” corresponds to `not_supported`.

| Expected → observed | Supported | Contradiction | Neutral |
| --- | ---: | ---: | ---: |
| Supported | 12 | 0 | 2 |
| Contradiction | 0 | 12 | 2 |
| Insufficient evidence | 1 | 0 | 13 |

Agreement is 37/42 on these authored examples. That single number is descriptive only: each slice has just six templated cases and individual observations dominate it.

| Slice | Matches / 6 | Mismatches |
| --- | ---: | --- |
| basic | 6/6 | None |
| negation | 5/6 | N12 |
| numeric | 6/6 | None |
| distractor | 6/6 | None |
| hindi_to_hindi | 6/6 | None |
| cross_language | 2/6 | N31, N32, N34, N35 |
| quantifier | 6/6 | None |

N12 is a false support under the authored label: “No visitor may enter the lab without a badge” does not establish that *all* visitors have badges, but the checker outputs supported (entailment 0.9043). In N31 and N34, cross-language translations of supported claims were neutral; in N32 and N35, opposite cross-language claims were also neutral. Both cross-language insufficient-evidence cases were neutral. These few examples show concrete failure modes, not the prevalence of failures.

## Case-level outcomes

| ID | Slice | Expected | Observed | Entailment | Contradiction |
| --- | --- | --- | --- | ---: | ---: |
| N01 | basic | supported | supported | 0.9980 | 0.0003 |
| N02 | basic | not_supported | not_supported | 0.0002 | 0.9995 |
| N03 | basic | neutral | neutral | 0.0007 | 0.0007 |
| N04 | basic | supported | supported | 0.9961 | 0.0007 |
| N05 | basic | not_supported | not_supported | 0.0002 | 0.9990 |
| N06 | basic | neutral | neutral | 0.0005 | 0.0053 |
| N07 | negation | supported | supported | 0.9844 | 0.0011 |
| N08 | negation | not_supported | not_supported | 0.0004 | 0.9985 |
| N09 | negation | neutral | neutral | 0.0004 | 0.0045 |
| N10 | negation | supported | supported | 0.9951 | 0.0010 |
| N11 | negation | not_supported | not_supported | 0.0039 | 0.9941 |
| N12 | negation | neutral | supported | 0.9043 | 0.0048 |
| N13 | numeric | supported | supported | 0.9961 | 0.0007 |
| N14 | numeric | not_supported | not_supported | 0.0020 | 0.9956 |
| N15 | numeric | neutral | neutral | 0.0006 | 0.0006 |
| N16 | numeric | supported | supported | 0.9956 | 0.0006 |
| N17 | numeric | not_supported | not_supported | 0.0003 | 0.9990 |
| N18 | numeric | neutral | neutral | 0.0003 | 0.0004 |
| N19 | distractor | supported | supported | 0.9873 | 0.0005 |
| N20 | distractor | not_supported | not_supported | 0.0007 | 0.9980 |
| N21 | distractor | neutral | neutral | 0.0002 | 0.0007 |
| N22 | distractor | supported | supported | 0.9966 | 0.0007 |
| N23 | distractor | not_supported | not_supported | 0.0009 | 0.9976 |
| N24 | distractor | neutral | neutral | 0.0005 | 0.0006 |
| N25 | hindi_to_hindi | supported | supported | 0.9902 | 0.0018 |
| N26 | hindi_to_hindi | not_supported | not_supported | 0.0295 | 0.9546 |
| N27 | hindi_to_hindi | neutral | neutral | 0.1859 | 0.3000 |
| N28 | hindi_to_hindi | supported | supported | 0.7925 | 0.0420 |
| N29 | hindi_to_hindi | not_supported | not_supported | 0.0006 | 0.9980 |
| N30 | hindi_to_hindi | neutral | neutral | 0.1956 | 0.2030 |
| N31 | cross_language | supported | neutral | 0.0032 | 0.0493 |
| N32 | cross_language | not_supported | neutral | 0.0038 | 0.0025 |
| N33 | cross_language | neutral | neutral | 0.0007 | 0.0022 |
| N34 | cross_language | supported | neutral | 0.4316 | 0.0446 |
| N35 | cross_language | not_supported | neutral | 0.0639 | 0.0146 |
| N36 | cross_language | neutral | neutral | 0.0352 | 0.0128 |
| N37 | quantifier | supported | supported | 0.9941 | 0.0012 |
| N38 | quantifier | not_supported | not_supported | 0.0001 | 0.9995 |
| N39 | quantifier | neutral | neutral | 0.0007 | 0.0017 |
| N40 | quantifier | supported | supported | 0.9863 | 0.0022 |
| N41 | quantifier | not_supported | not_supported | 0.0004 | 0.9985 |
| N42 | quantifier | neutral | neutral | 0.0002 | 0.0004 |

The [fixture](evaluation/nli-stress-cases.json) contains each complete premise and claim; the [raw results](evaluation/nli-stress-results.json) also retain the scores, verdicts, package versions and per-case elapsed times. Neutral score in the JSON is derived as `1 − entailment − contradiction` for these single-chunk softmax outputs; rounding and floating-point precision apply. Scores are not calibrated probabilities that a claim is true.

## Boundaries and label ambiguity

A single evidence chunk cannot test the real application's selection and aggregation of multiple chunks. The inputs are short: truncation at 512 tokens is configured but not exercised as a stress case. These cases do not test other Indic languages, Hinglish, adversarial instructions, encoded evidence, or the actual generation prompt. We did not validate unsupported-language behavior beyond these authored examples. Some labels assume ordinary single-valued facts: for example, the red-versus-blue case treats the vehicle's color as exclusive and the workshop case assumes one start time. An independent annotator should resolve such ambiguity before making benchmark claims. Neutral should be treated as uncertainty, not proof that the claim is false.

## Provenance and reproduction

- Fixture SHA-256: `a9f7d226a95e2fd367393f445f55dd89d01f544f56073c2b73f24ce41dd8650f`.
- Checked then-uncommitted `src/xai_rag/explainability/faithfulness.py` SHA-256: `d26495e4e23ccfaebcebc8aa5f6922c47fa52b430b6cccca65175bdd20302c63`; `src/xai_rag/config.py` SHA-256: `030d3740af42a7261d9b4590a7d9a86c89c5d529bdbd3610e03351611fd385da`. These identify the files tested at that time; they do not establish an exact match to later commits. The checker hash was unchanged before and after the run.
- Model revision: `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`. Both thresholds were verified at runtime. The runner requires the same model and revision and exits if configuration differs.

From the interview-materials repository, with that exact project source and model revision cached:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH=/home/neeraj/Documents/xai-rag/src \
  /home/neeraj/Documents/xai-rag/.venv/bin/python projects/xai-rag/evaluation/nli-stress-runner.py \
  --fixture projects/xai-rag/evaluation/nli-stress-cases.json \
  --output /tmp/nli-stress-results.json
```

The timing was observed under shared CPU load and is not a latency benchmark. Next, obtain independently reviewed premise–claim labels on representative retrieved contexts, explicitly test long and multiple chunks and cross-language slices, and decide a conservative policy for neutral and language coverage before relying on these decisions in a user-facing workflow.
