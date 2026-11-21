# XAI-RAG: architecture and retrieval interview questions

> **Version note:** This document describes the earlier `c375903` or explicitly cited historical source. Statements here that call NLI optional, say document authorization is absent, or describe recovery as only a proposal do not describe the later hardening work. Read the [current hardening questions and answers](hardening-interview-qa.md) first, and verify the source revision before an interview. Historical measurements remain tied to their stated snapshots.

**Revision boundary:** The historical baseline is `f4c374d`; later safeguards are included in [published source `c375903`](https://github.com/neeraj1909/xai-rag/tree/c375903fb46327744d076438cb90f7b02711a94f). Source links below may point to the baseline; see [guardrail implementation](guardrails-implementation.md) and [runtime evidence](runtime-evidence.md) for implementation and verification limits. Historical experiment snapshots are not automatically identical to the published revision.

This chapter covers the inspected [XAI-RAG revision `f4c374d`](https://github.com/neeraj1909/xai-rag/tree/f4c374d51d090beba61d34aba81fcc23b8ef700a). The answers describe repository behavior; use “I built” only for work you actually did. The numerical example is illustrative, not a reported run.

## The complete path

```text
INGEST: document → parse → chunk → embed → ChromaDB (vectors + chunk)
                                  └────→ Elasticsearch (text + same IDs)
                                         └→ report per-file status and parity

QUERY:  question → embed → dense search ─┐
                └──────→ BM25 search ───┴→ RRF → cross-encoder rerank
                                                   ↓
                    optional retrieval explanation → structured answer
                                                   ↓
                                optional claim-level NLI diagnostic
```

### 1. Question: What problem does XAI-RAG solve and what is its architecture?

**Answer:** It answers a question using retrieved document chunks and exposes evidence about retrieval and claim support. Ingestion parses documents, splits text and indexes chunks in ChromaDB and Elasticsearch. Query processing embeds the question, retrieves dense and lexical candidates, combines their ranks with reciprocal rank fusion (RRF), reranks a small pool and generates a structured answer with source IDs. Optional explanations and NLI diagnostics help inspect the pipeline. They do not establish that the answer is true. The orchestration is ordinary Python `asyncio`, not a LangGraph state machine. [Query service][service].

**Follow-up:** What is the difference between explainability and correctness? Retrieval ranks and claim verdicts explain particular pipeline decisions; correctness still needs independent source checking and evaluation.

### 2. Question: What happens when a document is ingested?

**Answer:** The service checks that paths are inside the configured document root and applies file-count and size limits. For each supported file it parses text, chunks it, embeds the chunks, writes vectors and metadata to ChromaDB, then indexes the same chunk IDs and text in Elasticsearch. It reports each document as successful, skipped or failed, and compares the stores after the batch. Revision `c375903` serializes local ingestion batches with a shared per-event-loop lock covering discovery through parity checking; separate processes or event loops are not coordinated. The stores are written separately: a successful first write and failed second write can leave drift. [Ingestion service][service], [storage implementation][store].

**Follow-up:** Why share IDs? If both stores return the same chunk, fusion can combine their rank contributions, and a citation can point back to one source record. Shared IDs alone do not ensure cross-store atomicity.

### 3. Question: How does this code split text and what are the trade-offs?

**Answer:** `fixed` uses character-bounded spans and overlap, preferring natural boundaries; its defaults are 512 characters and 50 characters of overlap. `semantic` groups adjacent sentence spans according to embedding similarity and a maximum character length. `parent_doc` indexes small child spans and retains the larger parent text as possible generation context. Small chunks can match precisely but lose context; large chunks can dilute retrieval and increase prompt cost. Compare strategies on the same labeled questions and inspect source offsets. The parent-doc implementation uses character-based spans even though its docstring mentions tokens. [Chunker][chunker], [settings][config].

**Follow-up:** What happens if the semantic chunking dependency cannot be imported? Its `ImportError` branch falls back to fixed splitting and labels the resulting chunks `semantic`; other model-loading errors are not guaranteed to trigger that fallback. Monitor actual behavior, not the label alone.

### 4. Question: What does the embedding model do, and why does its version matter?

**Answer:** The configured `BAAI/bge-small-en-v1.5` model converts text to normalized dense vectors; the query uses the same embedding function. The configured embedding dimension is 384, and the model revision is pinned. Existing indexes carry a fingerprint including model, revision, dimension and cosine distance; an incompatible or missing fingerprint can cause an index compatibility error. Changing the embedding model requires an explicit reindex and a comparison on the target corpus, not merely changing the query model. [Embedder][embedder], [index fingerprint][store].

**Follow-up:** Are semantically similar chunks guaranteed to answer the question? No. Similarity is a retrieval signal; a relevant-looking chunk can be incomplete, stale or unsupported.

### 5. Question: Why use Elasticsearch BM25 and ChromaDB together?

**Answer:** ChromaDB compares a query vector with indexed chunk vectors; Elasticsearch applies BM25 to indexed text. Exact names, codes and rare terms can favor lexical search, while paraphrases can favor dense search. Combining them gives each a chance to supply candidates, at the cost of operating two stores and keeping their records aligned. Whether hybrid retrieval improves this application is an empirical question; compare dense, BM25 and hybrid on the same labeled questions. [Vector search][vector], [BM25 search][bm25].

**Follow-up:** Is Elasticsearch used as the vector database in this implementation? No. In this path ChromaDB serves dense retrieval and Elasticsearch serves lexical retrieval.

### 6. Question: Are the two searches sequential?

**Answer:** The service starts the vector and BM25 operations together with `asyncio.gather`; the synchronous vector call runs in `asyncio.to_thread`, while Elasticsearch uses an async client. Fusion starts after both return. This can reduce waiting when both dependencies have spare capacity, but thread offload does not guarantee speed or throughput. The current query path does not implement a guaranteed single-retriever fallback; one failing branch can fail the request. [Query service][service].

**Follow-up:** What would a fallback need? A clearly marked degraded response, the same access filters, metrics and a separate quality evaluation for the surviving retriever. That is a proposed change.

### 7. Question: Why use reciprocal rank fusion instead of adding scores?

**Answer:** BM25 scores and vector similarities have different meanings and scales. RRF instead adds a rank-based contribution for each list containing a chunk: `score(d) = Σ 1/(k + rank(d))`. This avoids treating unlike raw scores as directly comparable, but throws away score magnitude. The code deduplicates IDs within each list, uses the provided stage rank when present, and breaks equal fused scores by ID. `k=60` is the rank-smoothing constant; it is not the number of candidates to retrieve. [RRF implementation][hybrid].

### 8. Question: Can you calculate RRF for a concrete question?

**Answer:** Suppose the question is “What must I submit for reimbursement?” and chunk B says to submit a receipt. Illustratively, dense search returns `[A, B, C]` and BM25 returns `[B, C, A]`. With `k=60`, B receives `1/(60+2) + 1/(60+1) = 0.032522`; A receives `1/(60+1) + 1/(60+3) = 0.032266`. B therefore ranks above A at **fusion**. This does not mean 3.25% confidence, and it does not guarantee B remains first after reranking. A chunk missing from one list contributes zero from that list. [RRF implementation][hybrid].

**Follow-up:** What does a larger smoothing constant do? It dampens differences between nearby ranks, giving relative weight to appearing in multiple lists. Tune it using labeled queries rather than treating 60 as universally optimal.

### 9. Question: What is the difference between retrieval, fusion and reranking?

**Answer:** First-stage retrieval finds candidates from the full indexes. RRF merges and deduplicates the two candidate lists. A cross-encoder then evaluates each question–candidate pair in a bounded fused pool and orders the final selection. The reranker can use query and text together, but costs work per pair; it cannot recover a relevant chunk missing from both initial lists. [Reranker][reranker], [query service][service].

**Follow-up:** What are the different `k` values? Configured vector and BM25 limits default to 100 each; the fusion constant defaults to 60; the reranker candidate bound is `max(request.top_k, reranker_candidate_k)` with configured default 5; final `top_k` is requested separately. They control different stages. [Settings][config], [query service][service].

### 10. Question: What exactly reaches the generator?

**Answer:** After reranking, selected chunks are formatted with source IDs into a bounded context. For a parent-document chunk, the code can use `context_content` rather than just the child text. The generator requests a structured answer, separate claims and source IDs. The configured context limit can truncate later content, so the fact that a chunk ranked highly does not guarantee every part of it reached the model. [Generator][generator].

**Follow-up:** How can the context limit affect answers? A supporting sentence cut off by the limit may make a previously relevant chunk insufficient; log the appropriate non-sensitive context-selection metadata and test long inputs. This logging recommendation is a proposed production measure.

### 11. Question: How does citation validation work, and what does it prove?

**Answer:** The generator requires each structured claim to name at least one source. If the structured claims list is nonempty, it also requires at least one bracketed ID somewhere in the answer; it does not require an inline citation for each claim or detect factual assertions omitted from the claims list. At baseline `f4c374d`, IDs were checked against the full ranked list, including omitted chunks. In `c375903`, inline and structured IDs must belong to chunks with nonblank text actually included in the bounded prompt; a visible prefix makes its ID eligible, even if the supporting sentence was cut off. These checks catch some malformed or unknown-source outputs, but do not prove that a cited passage supports a claim. Claim coverage and source support need separate evaluation. [Citation validation][generator].

### 12. Question: What happens with no retrieved evidence versus irrelevant evidence?

**Answer:** With no selected chunks, the generator returns a fixed insufficient-information response without calling the generation model. If nonempty chunks are irrelevant, the prompt tells the model not to guess, but that is not a deterministic guarantee. Test those two cases independently; a green empty-context test cannot prove the irrelevant-context case is safe. [Generator][generator].

### Follow-up: What happens when generation fails or a citation is invalid?

**Answer:** In `c375903`, transport retries and at most one semantic repair share a limit of three attempts; the default OpenAI client disables its own retries. An injected custom client can have its own retry behavior, so that bound does not cover arbitrary client internals. Invalid output can still end in an error; retries are not proof of a supported answer. See [guardrail implementation](guardrails-implementation.md) for the exact failure policy and tests.

### 13. Question: What does the claim-level NLI model check?

**Answer:** If requested and claims exist, the service runs the configured DeBERTa NLI checker against cited evidence and provides claim-level verdicts and scores. In `c375903`, the generator carries the actual prompt-visible evidence per request, including parent text and truncation, to the service. The prompt contains escaped text; the service decodes it once for NLI, while optional RAGAS receives the literal displayed text. This aligns the evidence boundary, but does not prove that either model interprets it correctly. A verdict is a diagnostic about model-estimated support in supplied context, not a truth guarantee, calibrated confidence, or a source-authority check. In this implementation an unsupported verdict does not automatically block answer release. [Faithfulness checker][nli], [query service][service].

**Follow-up:** What needs testing? Compare verdicts with human-labeled claim–evidence pairs, particularly numbers, negation, mixed support and domain-specific language. Report per-class errors; a neutral verdict need not mean contradiction.

### 14. Question: How does streaming affect the safety story?

**Answer:** The shared query service emits a `retrieval` event after reranking, then a `generation` event containing the answer, followed by `faithfulness` when that optional check runs, and finally `done`. The client can therefore see the answer before the NLI result. If the product requires checks before publication, change the release contract—buffer the answer until the chosen gate completes—and test timeouts and the user-visible latency cost. A later warning cannot retroactively prevent an earlier answer from being displayed. [Service event order][service], [API stream][api].

### 15. Question: What if a document is updated or the two indexes disagree?

**Answer:** The chunk ID is a hash of source identity, chunk position, strategy and content; a change can create new IDs. The Chroma write upserts current IDs and deletes stale records for that source; Elasticsearch similarly indexes current IDs and removes stale records after its bulk write. A parity check compares IDs and content hashes, but the two-store sequence is not transactional and parity reporting does not automatically repair drift. A production design could use versioned ingestion, a durable operation record and idempotent reconciliation, tested by crashing between writes. [Storage and parity][store].

**Follow-up:** Does an unchanged content-derived ID establish that the document is authorized for a user? No. Identity, versioning and access control solve different problems.

### 16. Question: How would you debug a wrong answer using this pipeline?

**Answer:** Trace one case across boundaries: verify that parsing preserved the source sentence and offsets; check whether either retriever returned its ID; inspect its fused and reranked position; verify the text actually included in the bounded context; then compare each generated claim with its cited source. If the chunk never appears, investigate ingestion, embedding, vocabulary and indexes. If it appears but is dropped, inspect fusion and reranking. If it reaches the model but the claim is wrong, inspect generation and evidence-checking failures. Keep retrieved IDs, versions and stage timing alongside the case without unnecessarily storing raw sensitive text. [Query service][service], [retrieval explanations][explainer].

### 17. Question: What are the main performance and quality trade-offs?

**Answer:** Search limits affect recall and work; the reranker candidate limit affects how much of the fused list gets the more expensive pairwise scoring; `top_k` and context length affect prompt size and possibly evidence coverage; NLI adds a post-generation diagnostic cost. Measure p50/p95 by stage, throughput and failures alongside retrieval and answer quality on a fixed workload. A faster setting is not an improvement if it drops required evidence, and a larger candidate pool is not automatically more accurate. [Settings][config], [query service][service].

### 18. Question: What should I say if asked whether this project is production-ready?

**Answer:** Explain the concrete implemented pieces: bounded ingestion, two indexes, configurable retrieval and reranking, citation-ID validation, API controls and diagnostic instrumentation. Then identify the target workload and remaining design work: authorization before both searches, durable index reconciliation, measured quality and failure behavior, secure dependency access, shared limits for replicas, and tested recovery. These are design proposals unless implemented and demonstrated for the deployment. Do not turn an offline test or this illustrative example into a claim of live production performance. [API][api], [production notes](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/docs/production-readiness.md).

[service]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/service.py
[store]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/ingestion/store.py
[chunker]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/ingestion/chunker.py
[embedder]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/ingestion/embedder.py
[config]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/config.py
[vector]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/retrieval/vector_search.py
[bm25]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/retrieval/bm25_search.py
[hybrid]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/retrieval/hybrid.py
[reranker]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/retrieval/reranker.py
[generator]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/generation/generator.py
[nli]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/explainability/faithfulness.py
[explainer]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/explainability/retrieval_explainer.py
[api]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/api/app.py
