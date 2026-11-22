# Controlled retry with real OpenAI repair and judge

Executed **2026-11-22 UTC**.

On the exact original failing chunk and claim, the real application service
rejected the controlled initial claim using the pinned real NLI model, invoked
its actual retry, received a response from `gpt-4o-mini`, checked both resulting
claims using real NLI, and submitted them to a separate `gpt-4o-mini` judge.
The judge approved the claims and question coverage; the service returned
`verifier_passed` with one cited source. See [result.json](result.json) for the
claim text, scores, evidence hashes, timing, usage and final response.

- Original NLI verdict: `not_supported`; entailment 0.04428, contradiction 0.87646.
- Repair claims: both `supported`; entailments 0.85498 and 0.96045.
- The evidence hash was the same across initial NLI, retry and final NLI.
- Repair consumed 591 input and 262 output tokens. Response-level token totals
  are null because the scripted initial generation had unknown usage; judge
  tokens are not included. Elapsed service time was 250.98 seconds on a
  resource-contended host, not a representative performance measurement.

This is a **service-level, controlled-input experiment**: retrieval, reranking,
and the first generation were supplied by deterministic test substitutes so
the historical rejection would trigger. The original failing claim came from
the existing deterministic E2E fixture; the repair generation, NLI and judge
were actual model calls through the unchanged service. It is not a full HTTP or
live-retrieval test, and a single model-generated result does not establish
reliability, judge safety, or behavior on representative data.

Source: HEAD `bebe3bd44412cbd4784beb922203336a14df8988`. Exact source hashes:

| File | SHA-256 |
| --- | --- |
| `probe.py` | `b6df9afe9be0b221facf41e8000450d5b699aff7744e552e48fe4dedf8fb1a6a` |
| `src/xai_rag/service.py` | `5604c3f8ccfdadd2e4b18db6f3f69173152720f3946e643cb605192889c9ca38` |
| `src/xai_rag/generation/generator.py` | `41148ac8e898feeda4eee1f2abe2d21d6203762e2deaf595aa8dfe95aebd3990` |
| `src/xai_rag/generation/grounding_judge.py` | `12936c09cdd4309b42cbccec1d2c0ca7d92726419b06534735b769a4cd067399` |
| `src/xai_rag/explainability/faithfulness.py` | `fff00707c5e72a487ad90e9d6bfb595e5c1501467b86a4f94b2894549feab1cc` |

NLI: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, revision
`6f5cf0a2b59cabb106aca4c287eed12e357e90eb`. Generator and judge model aliases
may change. The `.env` supplied the existing OpenAI key and was not modified;
no credentials are included in these artifacts. The following command makes
billable OpenAI calls and requires the locally cached NLI model:

```sh
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  XAI_RAG_GROUNDING_JUDGE_MODEL=gpt-4o-mini \
  XAI_RAG_GROUNDING_JUDGE_TIMEOUT_SECONDS=60 \
  XAI_RAG_LLM_MODEL=gpt-4o-mini \
  timeout 360 uv run --no-sync python \
  evals/results/2026-11-22-live-repair-followup/probe.py > /tmp/xai-rag-live-repair.json
```

`uv run --no-sync ruff check evals/results/2026-11-22-live-repair-followup/probe.py`
passed. No deployed service or persistent store was changed.
