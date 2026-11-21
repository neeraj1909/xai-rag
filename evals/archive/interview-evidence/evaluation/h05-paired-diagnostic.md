# H05 citation-format diagnostic

This generator-only experiment used a fictional, reconstructed H05 context and `gpt-4o-mini-2024-07-18`. The failed original service call did not persist its prompt-visible context. We rebuilt it from the saved rerank order and IDs, joining each ID to the same hash-verified fixture text; exact equivalence with that original call is unverified. The source and fixture hashes are in each result file. Both snapshots use an explicitly injected client with SDK retries disabled, the same model alias, generation settings and supplied evidence.

| Snapshot | Generations accepted | Provider responses | Observed input tokens | Observed output tokens |
|---|---:|---:|---:|---:|
| Before citation instruction | 0/5 | 10 | 8,530 | 3,223 |
| After citation instruction | 5/5 | 5 | 4,245 | 1,828 |

All ten before responses used an inline attribute-style citation and were rejected for an unknown citation ID. All five after responses passed the existing strict validator and answered that boiling is unnecessary. The after prompt gives the example `[abc123]` and explicitly rejects `[chunk_id="abc123"]` and `[id="abc123"]`. The contract was not weakened. Each failure was given one semantic repair opportunity, within the shared transport/repair limit.

The 15 calls used half the 30-attempt cap; all returned responses. Token counts are provider-reported, not dollar costs. A separate earlier diagnostic encountered connection errors whose billing status is unknown. The order was not randomized, the sample contains one synthetic case, and the two archived source trees have differences in modules unrelated to this direct generator call. The generator also changed its default client setup, but this experiment injected the same explicit client for both. These observations support further testing; they do not establish a general failure rate or a causal production improvement. The accepted results were not human-rated, and valid IDs alone do not establish claim support.

- Before: [raw results](h05-paired-before.json), [frozen source](holdout-source-snapshot.tar.gz), source digest `968bdafa6c1c5174f4f1385cf4f425debee1601d19bcc73f48e99807d912efb5`.
- After: [raw results](h05-paired-after.json), [frozen source](h05-post-source-snapshot.tar.gz), source digest `f62c14c4949f0c8ef87d09ccbfcf41562276902d6262eb3b691157bb12b138ed`.
- [Harness](h05-paired-diagnostic.py) records each fictional provider response, usage, contract result and sanitized errors. [Budget state](h05-paired-budget.json) records attempts across both phases. The harness checks source and fixture hashes and stops after repeated provider failures. It requires the XAI-RAG environment and makes billable provider calls; it refuses to overwrite a result file.
