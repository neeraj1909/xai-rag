# Multilingual lexical and fusion diagnostic

On 20 November 2026, we tested the [same fictional 20-document, 40-question multilingual fixture](evaluation/multilingual-challenge-cases.json) against real Elasticsearch BM25 and the archived application's reciprocal-rank fusion (RRF). The precomputed multilingual MiniLM exact-cosine rankings were reused without changing the frozen fixture. This exploratory dataset was AI-authored with its labels after an earlier experiment; it is neither independently reviewed nor a representative production benchmark. Every question has one labelled relevant source. No model, index configuration, prompt, or production code was changed.

For each question, MiniLM's first five sources and Elasticsearch's first five matching sources were combined with the archived app's RRF implementation, `k=60`. Elasticsearch used one shard, zero replicas, and the app's standard analyzer on the `content` text field; its archived `bm25_search` function supplied the actual `multi_match` query. Rankings use source IDs directly as the test chunk IDs. All 40 requests succeeded for all three stages. The denominator for each slice is ten questions; the overall denominator is 40. Hit@k means the labelled source appears within the first k positions.

| Slice | MiniLM @1/@3/@5 | BM25 @1/@3/@5 | RRF @1/@3/@5 |
| --- | --- | --- | --- |
| Hindi question → Hindi document | 10/10, 10/10, 10/10 | 10/10, 10/10, 10/10 | 10/10, 10/10, 10/10 |
| English question → Hindi document | 4/10, 6/10, 10/10 | 0/10, 0/10, 0/10 | 0/10, 3/10, 6/10 |
| Hindi question → English document | 2/10, 7/10, 10/10 | 0/10, 0/10, 0/10 | 0/10, 4/10, 10/10 |
| Exact-code slice | 8/10, 10/10, 10/10 | 6/10, 10/10, 10/10 | 9/10, 10/10, 10/10 |
| All questions | 24/40, 33/40, 40/40 | 16/40, 20/40, 20/40 | 19/40, 27/40, 36/40 |

On M38, MiniLM ranked the wrong T-24 passage first for a T-42 query; BM25 ranked the T-42 passage first. RRF put T-42 first, but the T-24 and T-42 sources had **exactly equal fused scores**: the archived implementation breaks ties by ID, and this diagnostic used human-readable source IDs. The real application uses content/provenance-derived chunk IDs, so a tie can resolve differently. M40 still ranked the correct T-42 passage second after fusion. The result is evidence to test exact-code handling, not a demonstrated general fix.

The same RRF combination substantially worsened the two cross-language slices here. For M11, MiniLM ranked the correct Hindi Falcon passage first; English-language BM25 results instead boosted two English documents above it in the fused list. A lexical branch can be useful for identifiers without being safe to apply unconditionally across language pairs. Consider an exact-identifier field and language-aware fusion as **candidates for independently evaluated changes**, not changes already implemented or proven by this sample.

The actual Elasticsearch `_analyze` response split `T-24` into `t` and `24`, `T-42` into `t` and `42`, `R-17` into `r` and `17`, and `R-71` into `r` and `71`. The numeric tokens differ, but the letter token is shared and other query terms contribute to BM25. Tokenization therefore does not enforce exact whole-code matching. No security, abstention, generated-answer correctness, or latency-at-scale claims follow from these results.

## Evidence and reproduction

The [runner](evaluation/holdout-lexical-comparison.py) uses archived app `bm25_search` and `rrf_fusion`; the [raw output](evaluation/holdout-lexical-results.json) includes every question, scored rankings, analyzer output and failure denominators. MiniLM's model revision was `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. No embeddings were recomputed, so the dense results are exact cosine over all 20 sources rather than a Chroma ANN query. No reranker or generator was used. Results can differ with actual chunk boundaries, chunk IDs, candidate budgets, model/index versions, document distributions, or Elasticsearch tie ordering.

- Fixture SHA-256: `f2494b9bf4a766a374dc1fc1a6e141196ff045cf95a208d541452c94056f0887`.
- Saved dense ranking SHA-256: `d7507aa34f4acad91b47745d38c851e42893cc082a4dc3fe5d7027b869bb18f7`.
- Archived Python source tree SHA-256: `968bdafa6c1c5174f4f1385cf4f425debee1601d19bcc73f48e99807d912efb5`; [archive](evaluation/holdout-source-snapshot.tar.gz) SHA-256: `43375aa1c302b7ad92ed7e7b4ed1c67d82256684bbf47b72b4c6d43404f01106`.
- Elasticsearch image: `docker.elastic.co/elasticsearch/elasticsearch@sha256:2f602552550869fb29b6fd5848c5118d3ef3a2e1d5d45802e3ab9088cb2de8e2`, version 8.17.0.

To reproduce, unpack the archive into a temporary directory and expose only a disposable Elasticsearch on `127.0.0.1:19381`; the runner refuses an existing index. With the project virtual environment and `PYTHONPATH` set to that temporary directory, run `python projects/xai-rag/evaluation/holdout-lexical-comparison.py` from this repository. The runner requires a new output path: move or preserve the existing output before rerunning. It does not call an embedding model or provider. Stop and remove the disposable Elasticsearch after the run. The original experiment's container was removed after recording the results.
