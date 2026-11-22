# Same-corpus NLI comparison

Run on **2026-11-22 Asia/Kolkata**, using local CPU and cached pinned model checkpoints. The fixture contains 53 synthetic, author-labelled cases, with expectations fixed before these runs. Labels have not been independently human reviewed. The cases target known failure modes; agreement is not a production accuracy estimate. All four model reports have the identical `fixture_sha256` and give a decision for every case.

| NLI checkpoint | Revision | Exact authored-label agreement | Previously recorded cases reused | New case checks |
| --- | --- | ---: | ---: | ---: |
| MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | `6f5cf0a2b59cabb106aca4c287eed12e357e90eb` | 44/53 | 48 | 5 |
| cross-encoder/nli-deberta-v3-xsmall | `a150876415327c80daeff35ca6f68f5ed8cf5c24` | 41/53 | 47 | 6 |
| cross-encoder/nli-deberta-v3-small | `fa2804872c3b4bd748f38c0185cc85775361e735` | 41/53 | 47 | 6 |
| cross-encoder/nli-MiniLM2-L6-H768 | `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d` | 37/53 | 47 | 6 |

The 53 cases include 42 prior stress pairs and 11 original-evidence and adversarial/citation controls. On the original premise, all four withheld the original supported claim, and all supported the **prewritten** shorter repair. All four falsely supported the prohibited-heading case. The base, small and MiniLM models also supported the injection claim; xsmall withheld it. The contradictory-source case was withheld by base and small but supported by xsmall and MiniLM. The conditional case was supported by base and MiniLM; small and xsmall withheld it but labelled it contradiction when the authored expectation was neutral. Wrong and missing source IDs failed citation validation before NLI. See each report for every decision and probability.

These are isolated model and citation checks. They do **not** run generation, retrieval, the independent grounding judge, or the full deployed application. The prewritten repair is not evidence that a live generator will produce it. A real provider repair probe is a separate experiment. A released NLI-only case is a risk signal, not proof of an application bypass, and successful detection in one case does not prove safety. Removing either verifier is not supported by these small tests. Prompt-injection, conditional statements and cross-source conflicts require broader independently reviewed cases.

## Provenance and reproducibility

`cases.json` retains every premise, claim, expected outcome and citation. `base.json`, `xsmall.json`, `small.json` and `minilm.json` retain per-case raw scores and observed decisions, plus each row's provenance. The prior base reports only recorded entailment and contradiction, and prior alternative reports recorded three probabilities; **prior logits and absent probabilities are unavailable**, and are not reconstructed. Newly inferred pairs include logits and all three probabilities. The thresholds are 0.5 for entailment and contradiction, with the project's contradiction priority and cited-source scope. Expected `neutral` and `not_supported` are treated as different for the exact-label tally even though both withhold.

Reuse avoids duplicate inference only after validating the exact inputs and recorded model revisions. For base, the saved 42-case report's fixture hash, model, thresholds, and source checker hash are verified. AST comparisons verify that `_score_pair` and `_check_single_claim` are unchanged between the prior and current checker; only the cold-load lock changed. Six original-input cases come from the separately saved current-checker probe, whose checker and premise hashes are verified. Both base sources record Torch 2.13.0+cpu and Transformers 5.15.1. Historical baseline reports did not retain all model-file hashes or logits, so bit-for-bit reproducibility cannot be established.

For alternatives, the original fixture, prior runner and original premise hashes are checked against the recorded SHA manifest, as are every local pinned checkpoint file and model label mapping. Forty-two stress cases and five separately identified probes have their exact source texts and claims checked. Other cases are run fresh. Previous and new inferences occurred in separate processes, may differ in numerical precision or runtime conditions, and are not a latency benchmark. The reports and their exact source hashes remain intact in [the prior alternative-model evidence](../2026-11-22-grounding-probes/model-alternatives/README.md), [the original NLI run](../2026-11-22-nli/README.md), and [the same-input probe](../2026-11-22-grounding-probes/README.md).

The initial full rerun was intentionally stopped after memory contention and before its raw report was written. Its partial decisions are kept in `interrupted-base-decisions.jsonl`; they are **not** counted in the final reports. The final base run took approximately 141 seconds; the three alternatives reported 3, 6 and 3 seconds respectively after cached-file verification. These times exclude prior reused inference and are not comparable performance measurements.

From the repository root, with the recorded checkpoints cached, rerun with new output paths (the fixed base script writes `base.json` in its directory and should be run only in a disposable copy to preserve this evidence):

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python evals/results/2026-11-22-same-corpus/compare_reuse.py --name xsmall --model cross-encoder/nli-deberta-v3-xsmall --revision a150876415327c80daeff35ca6f68f5ed8cf5c24 --output /tmp/xai-corpus-xsmall.json
.venv/bin/python evals/results/2026-11-22-same-corpus/compare_reuse.py --name small --model cross-encoder/nli-deberta-v3-small --revision fa2804872c3b4bd748f38c0185cc85775361e735 --output /tmp/xai-corpus-small.json
.venv/bin/python evals/results/2026-11-22-same-corpus/compare_reuse.py --name minilm --model cross-encoder/nli-MiniLM2-L6-H768 --revision b95119ce93d3e065de6214e38cd4a97b0f2f2c6d --output /tmp/xai-corpus-minilm.json
```

`compare.py` was the original all-new-inference runner. `base_reuse.py` and `compare_reuse.py` produced the final reports. Each records its own SHA-256, as well as the SHA-256 of the fixture and reused reports. These are preserved as measured artifacts, not maintained application modules.
