# Local alternative NLI checkpoint comparison

Three pinned, public Hugging Face checkpoints were evaluated offline against the original Markdown premise and claim, six adversarial probes, and all 42 authored stress cases. These examples are small and author-labelled, not independently adjudicated; the extra heading labels represent intended contextual interpretation. The results do not establish general model quality. The six cross-language stress cases are particularly limited, and these English NLI checkpoints should not be assumed to support other languages.

| Model (official card) | Revision | Authored label agreement | Original E2E | Incorrect heading | Prohibited heading | Conditional | Cross-source |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| [DeBERTa v3 xsmall](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall) | `a150876415327c80daeff35ca6f68f5ed8cf5c24` | 34/42 | withheld | released | released | withheld, labelled contradiction | released |
| [DeBERTa v3 small](https://huggingface.co/cross-encoder/nli-deberta-v3-small) | `fa2804872c3b4bd748f38c0185cc85775361e735` | 34/42 | withheld | withheld | released | withheld, labelled contradiction | withheld |
| [MiniLM2 L6 H768](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768) | `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d` | 31/42 | withheld | released | released | released | released |

The original checker previously agreed on 37/42 (see [historical diagnostic](../../../archive/interview-evidence/nli-stress-diagnostic.md); that run used its production setup and does not constitute a controlled latency or precision comparison). On the original E2E premise its entailment was 0.044 and contradiction 0.876, and the three alternatives gave entailment/contradiction respectively 0.116/0.720, 0.010/0.946 and 0.416/0.395. All models correctly withheld the explicit false quotation and simple negated hypothesis. The small checkpoint is the only tested alternative to withhold this particular cross-source contradiction, but it still accepts the prohibited-heading probe. No tested replacement fixes the original failure while maintaining the available regression evidence.

## Method and reproducibility

`compare.py` uses `evals/nli/stress-cases.json` and the exact premise from `evals/results/2026-11-22-docker-e2e/nli-probe.txt`. It uses the recorded hypothesis, the model's own named label mapping, CPU float model inference, 512-token truncation, and the current release decision: maximum contradiction at least 0.5 and at least maximum entailment withholds as contradiction, otherwise maximum entailment at least 0.5 releases, otherwise neutral. All token counts and individual scores are in the JSON reports; cross-source combines two separate pair predictions. Distinguish withholding from the precise neutral versus contradiction classification. Thresholds are not tuned on these cases.

The immutable revisions and SHA-256 of each downloaded cache file, script, fixture and E2E premise are in `sha256.json`. Use an existing compatible `.venv`, with the pinned checkpoint downloaded into the Hugging Face cache, then from the repository root run each model sequentially (replace the three arguments with the values in the table and a new output path):

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python evals/results/2026-11-22-grounding-probes/model-alternatives/compare.py --model cross-encoder/nli-deberta-v3-small --revision fa2804872c3b4bd748f38c0185cc85775361e735 --output /tmp/nli-small-repeat.json
```

The xsmall run used two CPU threads; small and MiniLM used one. The first small attempt exited with SIGILL at two threads; retry at one thread completed. The result timings include model loading, reflect shared host contention and differing thread counts, and must not be used as a performance comparison. All downloads and tests were local; no serving configuration or production code was modified.
