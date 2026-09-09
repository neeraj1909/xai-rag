# Model inventory

This file records the externally supplied model artifacts used by the default
configuration. A pinned revision makes an artifact reproducible; it does not
make its quality, security, or license automatically suitable for a particular
deployment. Re-verify this inventory and rerun the evaluation suite before any
model or revision change.

| Pipeline role | Model | Immutable revision | Declared license | Operational note |
|---|---|---|---|---|
| Query/document embedding | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | MIT | 384 dimensions; a change requires a full vector reindex. |
| Cross-encoder reranking | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | Apache-2.0 | Roughly 2.3 GB of model files; benchmark warm CPU latency and memory on the target. |
| Claim/evidence NLI | [`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) | `6f5cf0a2b59cabb106aca4c287eed12e357e90eb` | MIT | Automated verdicts require calibration against human labels. |
| Semantic chunking | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Apache-2.0 | The model card states that inputs beyond 256 word pieces are truncated. |

The hosted generation default, `gpt-4o-mini`, is a provider alias rather than
an immutable local artifact. Production owners must approve the provider's
terms, data handling, region, retention, and model-change policy. Record the
resolved provider model version in evaluation and telemetry whenever it is
available. No repository test should spend provider credits by default.

Evaluation fixtures and `sample_docs/` are repository-owned synthetic/example
content. They validate mechanics only and are not a representative production
quality benchmark.
