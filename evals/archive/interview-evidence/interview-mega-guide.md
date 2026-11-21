# XAI-RAG: interview master guide

> **Version note:** This document describes the earlier `c375903` or explicitly cited historical source. Statements here that call NLI optional, say document authorization is absent, or describe recovery as only a proposal do not describe the later hardening work. Read the [current hardening questions and answers](hardening-interview-qa.md) first, and verify the source revision before an interview. Historical measurements remain tied to their stated snapshots.

This is a self-contained revision guide for senior AI engineer and AI researcher interviews. It follows one request from source document to answer, then examines why the design was chosen, what the evidence establishes, and what would change for production. Read the numbered questions in order the first time; revisit individual questions when you have a few minutes.

**Evidence boundary.** The historical baseline is [XAI-RAG revision `f4c374d`](https://github.com/neeraj1909/xai-rag/tree/f4c374d51d090beba61d34aba81fcc23b8ef700a). Later safeguards are included in [published source `c375903`](https://github.com/neeraj1909/xai-rag/tree/c375903fb46327744d076438cb90f7b02711a94f). Unless called historical below, current behavior refers to that published revision. Source links in this guide show the baseline, while the [evaluation report](live-evaluation.md) preserves exact tested patches and file-hash manifests. The repository demonstrates capabilities; it does not, by itself, establish which parts you personally implemented or why every choice was originally made. “Why” below explains the design trade-off, not an unrecorded historical decision. Examples marked illustrative are not measured model results.

## One-minute introduction

> XAI-RAG is a document-grounded question-answering system that makes its retrieval and claim-checking stages visible. During ingestion it parses documents, splits them into chunks and writes the same chunk identities to a Chroma vector collection and an Elasticsearch lexical index. At query time it runs dense and BM25 search in parallel, merges their rankings with reciprocal rank fusion, reranks a bounded set with a cross-encoder, and asks a language model for an answer with source IDs and structured claims. Optional explanations and an NLI model provide diagnostic information. I would evaluate evidence retrieval, answer correctness, support and operational behavior separately: an existing citation or a high diagnostic score is not proof of a correct answer.

A careful personal version adds your actual contribution and a real issue you investigated; do not claim ownership or measured impact that your records cannot support.

## A longer answer to rehearse

> “The problem is answering questions from a changing document collection while showing the reader the passages behind the answer. I use a fixed RAG pipeline: application code coordinates the stages, with no model-directed planner or sub-agents in this path.
>
> “In ingestion, the service checks and parses documents, splits the extracted text into chunks, assigns IDs based on content and provenance, and embeds the chunks. Chroma stores vectors for semantic search; Elasticsearch indexes the text for lexical search. The same IDs let us merge results later. Writes to the two systems are separate, so a partial failure needs reconciliation; an upsert is not a distributed transaction.
>
> “When a question arrives, we embed it and retrieve in parallel from Chroma and Elasticsearch. Elasticsearch is the engine; BM25 is the lexical scoring method used by this query. Dense search can help with paraphrases, while lexical search can help with matching terms and identifiers. I would test both on the actual corpus rather than assume either always wins. We combine ranks with reciprocal rank fusion, then a cross-encoder jointly scores the question and each selected candidate to rerank a smaller pool. It cannot recover a relevant passage that never reached that pool.
>
> “The generator sees bounded source text and returns an answer, claims and source IDs. The current local version validates IDs against the actual included context, but a valid ID doesn't prove that the source supports the claim or is current. The optional NLI check is diagnostic and may arrive after a streamed answer.
>
> “I would evaluate retrieval and answer quality separately. We ran actual retrieval, generation and HTTP experiments on fictional data, but they are small and not human-reviewed. One challenge exposed an English-to-Hindi retrieval miss, and the NLI and judge probes exposed their own mistakes. Before production, I would add document-level access control, durable two-index reconciliation, independent evaluation, operational monitoring and a rollback plan. I would describe those as proposed changes until implemented and verified.”

## The problem and the shape of the solution

### 1. What business problem does this solve, why use RAG, and how does an answer become reviewable?

**What:** A user asks a question about a collection of documents. A useful system finds the passages that may contain the answer, returns a concise response and shows which sources it used. For example, a support specialist asks what a reimbursement request must include; the system can cite the policy passage containing that requirement.

**Why:** Reading a large or frequently changing document collection manually is slow, while a standalone model can answer from outdated or unsupported knowledge. Retrieval puts selected source text into the model's context. This is useful when the answer depends on organizational documents and the reader needs a way to inspect its basis. It does not guarantee the source is current, the retrieval is complete or the answer is correct.

**How:** Documents are indexed beforehand. A query retrieves candidate chunks, combines and reranks them, and sends a bounded selection to a structured-answer generator. Source IDs, retrieval explanations and optional claim-level checks provide different evidence for inspection. The source determines what exists; do not claim this is a deployed company workflow or a proven productivity improvement without evidence. [Service][service].

### 2. Draw the system. What are the two different paths?

```text
                         OFFLINE / INGESTION
 document → path and size checks → parser → chunker → embedding model
                                             │            │
                                             │            └→ Chroma HTTP server
                                             └→ text + matching IDs → Elasticsearch
                                                        ↓
                                      per-file result + cross-store parity report

                           ONLINE / QUESTION
 question ─→ query embedding ─→ dense Chroma search ───┐
          └──────────────────→ lexical BM25 search ────┤ in parallel
                                                       ↓
                                                RRF rank fusion
                                                       ↓
                                         bounded cross-encoder reranking
                                                       ↓
                                       optional retrieval explanations
                                                       ↓
                               structured answer + claims + citation IDs
                                                       ↓
                                      optional claim-level NLI report
                                                       ↓
                              response / stage events; optional RAGAS
```

**What:** `RAGService` coordinates these components; the API has regular and streaming paths. **Why separate ingestion from query?** Parsing and embedding the corpus once lets many requests reuse an index; the query path embeds only the new question and evaluates a limited candidate set. **How:** Shared chunk IDs connect the two retrieval results to fusion and sources. There is no single transaction spanning the two stores; ingestion reports partial errors and checks parity. [Service][service], [store][store].

**Follow-up:** In streaming, does the user wait for NLI? No. The service can emit its `generation` event before optional `faithfulness`; the NLI report is diagnostic rather than a pre-release gate. [Service][service].

### 3. Is this an agent, a multi-agent system, or an orchestrator with memory and MCP tools?

**What:** The inspected path is a fixed retrieval-and-generation pipeline coordinated by ordinary Python service code. It uses concurrent retrieval with `asyncio.gather`, not a LangGraph state machine, model-directed planner, sub-agent team or autonomous tool-selection loop. It also does not use MCP in this request path. **Why:** A fixed workflow is easier to reason about when every question should follow the same search, rerank and answer stages; an agent adds routing flexibility but also cost and failure cases. **How:** Application code decides the sequence; the model generates a structured answer from supplied evidence. Calling retrieval a “tool” informally does not make it a model-invoked MCP tool.

**Memory:** The document indexes are external knowledge for retrieval, not per-user conversational memory. The inspected query flow should not be described as a personalized long-term memory service. Add an agent, MCP, conversation store or planner only when a concrete product need and evaluation support it; none is required to explain this project. [Service][service].

**Senior follow-up:** When might an agent help? A research workflow with an uncertain number of authorized search steps may need planning and bounded tool calls. Evaluate it against the fixed pipeline on the same questions, including latency, cost, source support, tool failures and stopping behavior. This is a proposed alternative, not a feature already present.

## One complete worked example: reimbursement policy

Suppose a handbook says **“Submit a receipt with the reimbursement request”** and the user asks **“What do I need to submit?”** This is an illustrative walkthrough; the ranks and answer below are not a recorded model run.

1. **Prepare evidence:** The parser extracts the policy text. The chunker keeps the receipt sentence in chunk B and retains source metadata. The embedding model turns B into a vector; the system writes the vector and text to Chroma and the matching ID and text to Elasticsearch. A successful write to one store does not guarantee the other write succeeded.
2. **Retrieve:** The service embeds the question. Suppose dense search returns `[A, B, C]` and BM25 returns `[B, C, A]`. Dense retrieval may recognize related meaning; BM25 may favor the shared reimbursement terms. These explanations are hypotheses about the example, not observed model reasoning.
3. **Fuse and rerank:** With RRF `k=60`, B scores approximately `0.032522` and A `0.032266`; B is ahead in the fused list. The cross-encoder then jointly scores each question–candidate pair. It may preserve or change that order.
4. **Generate:** If B survives reranking and the context limit, the generator may return “Submit a receipt with your reimbursement request [B]” and a structured claim citing B. The current local validator checks IDs against the included, prompt-visible evidence snapshot. This closes the earlier baseline’s ranked-versus-visible ID gap, but an allowed ID still does not prove the sentence is supported.
5. **Inspect:** An optional NLI report may assess the claim against its cited chunk. A reader should still check whether B is the current authoritative policy, whether exceptions exist elsewhere, and whether the answer is complete. A supported verdict is useful evidence, not an automatic guarantee.

If B never appears in the first-stage candidate lists, start by investigating extraction, indexing and search. If B appears in the prompt but the model invents a different requirement, investigate generation and support verification. That distinction is the practical reason to preserve stage-by-stage evidence.

### A closer look at the same example

Think of a library with two ways to find the receipt policy. One catalogue looks for the **words** in a query; the other groups passages by numerical representations of their **meaning**. Elasticsearch is the search engine running the word-based catalogue. BM25 is the ranking method it uses here to order matching passages. Chroma is the separate vector store used for the meaning-based search. Elasticsearch and BM25 are therefore different kinds of thing: an engine versus one way to score results. Elasticsearch can support other search methods; this project uses its lexical path alongside Chroma.

For the hypothetical receipt passage, the actual shape of the pipeline is:

1. **Store it once in each index:** The parser extracts text, the chunker creates B, and the ingestion code derives a deterministic ID from chunk content and provenance. Chroma receives B's ID, text, metadata and embedding. Elasticsearch receives the same ID, text, source file and metadata. The ID links two records; it is not a transaction joining the stores.
2. **Build the lexical catalogue:** Elasticsearch maps `content` as analyzed text with its standard analyzer, and `chunk_id` and `source_file` as keyword fields. The analyzer prepares terms for the inverted index: a structure mapping terms to documents that contain them. At query time the project sends a `multi_match` query over `content`; Elasticsearch scores matches using BM25. For example, rare shared words can help the receipt chunk rank, but analysis and query wording affect what actually matches.
3. **Build the vector catalogue:** The embedding model turns B's text into a 384-dimensional vector. The question is embedded with the same model; Chroma returns nearby vectors by cosine distance, which the retrieval adapter converts to similarity as `1 − distance`. A nearby vector is a candidate, not a verified answer.
4. **Combine evidence:** Each branch returns IDs, text, metadata, scores and rank. RRF combines ranks by shared ID, then the cross-encoder compares the question with selected passage text. The generator sees only the text that fits its context budget. Inspect the indexed B, both branch ranks, fused/reranked position and prompt-visible text when investigating a wrong answer.

The store choices support complementary searches but require keeping two copies synchronized. They are observed project choices, not proof that this combination outperforms a single engine or another vector database. [Index mapping and writes][store], [lexical query][bm25], [vector query][vector], [fusion][hybrid].

## Ingestion: from a file to searchable records

### 4. What happens before text becomes a chunk, why validate inputs, and how is provenance preserved?

**What:** Ingestion discovers supported files under an allowed root, checks paths and configured limits, then parses text and splits it into chunks. **Why:** A vector database cannot repair an incorrect parser, missing page text, a path traversal or lost provenance. Each downstream answer depends on what was successfully extracted. **How:** The service reports per-document status and retains source and chunk metadata. Revision `c375903` also bounds the expanded size of DOCX ZIP contents before parsing, and discovery applies configured file-count and aggregate byte limits. After parsing and before chunking, a configurable limit defaults to 20,000,000 extracted characters per document; because the parser has already run, this does not bound its CPU use or peak memory. In a production corpus, include extraction checks for scanned PDFs, tables, headers, duplicates and empty pages; validate whether the necessary evidence survives parsing before tuning retrieval. OCR or table-specific parsing should be claimed only if verified in the chosen path. [Historical ingestion service][service], [historical parser][parser].

### 5. What is chunking, why does it change retrieval, and how do the three available strategies differ?

**What:** Chunking turns a document into bounded text units for embedding and retrieval. The inspected implementation offers `fixed`, `semantic` and `parent_doc`. **Why:** A whole document can exceed context limits and dilute the relevance of a specific passage; tiny chunks can omit definitions and adjacent conditions. Chunk boundaries determine what is embedded, searched and later cited.

**How:**

| Strategy | How it works here | Trade-off to test |
| --- | --- | --- |
| Fixed | Character-bounded spans with overlap, preferring natural boundaries; default size 512 characters and overlap 50. | Predictable and cheap, but a boundary can split a complete idea. |
| Semantic | Adjacent sentence spans are grouped using embedding similarity and a maximum character length. | Can preserve local meaning, but adds a model and ingestion cost; quality depends on language and thresholds. |
| Parent document | Smaller children are indexed, with larger parent text available as context. | A precise child hit can bring nearby context, but larger returned context costs tokens and may add irrelevant text. |

The ingestion service defaults to semantic chunking, whereas the lower-level `chunk_text` function defaults to fixed chunking when called directly. The parent-document code uses character spans even though a docstring mentions tokens. A semantic-chunking `ImportError` has a fixed-splitting fallback but other model-loading errors need not follow that branch. Compare chunk strategies on the same labeled questions, including boundary-sensitive evidence and parsing loss. [Chunking implementation][chunker], [configuration][config].

**Current local guardrails:** Semantic splitting iterates sentence spans lazily and enforces a configurable limit of **10,000 semantic units** before embedding them. Fixed, semantic and parent-document chunking also enforce a configurable maximum of **100,000 chunks per document**. Exceeding a limit raises an error instead of completing an unbounded chunk list. These changes are in the current local source after the earlier HTTP E2E snapshot; that earlier run does not test them. They operate after parsing: in particular, there is no PDF parser sandbox, and these limits do not bound the parser's CPU or peak memory.

**Follow-up:** Is overlap free? No. It can duplicate near-identical evidence, increase index size and consume candidate/context slots. A good evaluation checks whether overlap improves support recall enough to justify that cost.

### 6. What are the embedding model and index fingerprint? Why must document and query versions match?

**What:** The configured `BAAI/bge-small-en-v1.5` model maps chunk text and questions into 384-dimensional vectors. Nearby vectors under the configured similarity measure are candidates for semantic relevance. **Why:** We can precompute document vectors and search approximately at request time instead of evaluating every document jointly with the question. Similarity is a retrieval signal, not proof a passage answers the question.

**How:** The model revision, dimension and distance configuration contribute to an index fingerprint, and incompatible metadata can raise a compatibility error. If the embedding model changes, old vectors and new query vectors may no longer be comparable even when dimensions match. A production migration should build a versioned index with the new model, evaluate and reconcile it, then switch traffic with a rollback path. It is not enough to change an environment variable and reuse old embeddings. [Configuration][config], [store and fingerprint][store].

### 7. Why two stores, how are IDs shared, and what happens when one write fails?

**What:** Chroma's HTTP server stores vectors and chunk records for dense search; Elasticsearch indexes text for BM25. A common deterministic chunk ID connects the records. **Why:** Lexical search can surface exact terms while vector search can surface paraphrases; the improvement must be measured on actual tasks. **How:** Ingestion writes Chroma first and Elasticsearch next. Deterministic IDs, upserts and per-source stale-chunk cleanup support replay within each store. A parity check compares stored identities and selected metadata; it does not create atomic cross-store writes or prove the vector and text contents are semantically correct. [Store][store], [service][service].

**Failure scenario:** Chroma accepts a source and Elasticsearch fails. The service can report a failed/partial outcome and parity may reveal drift, but successful Chroma writes are not magically rolled back. Revision `c375903` serializes full ingestion batches and parity checks with an event-loop-scoped lock, retaining the lock while local writes finish after cancellation. It does not coordinate across workers or hosts, and remote writes can still have unknown outcomes. A production design could retain a durable, versioned ingestion manifest, retry failed writes idempotently, reconcile both indexes, and alert on persistent divergence. Test a crash between writes and an older retry racing a newer source version. Those are production proposals, not guarantees of the current pipeline.

## Query: from question to ranked evidence

### 8. What is dense retrieval, why can it help, and how does it differ from BM25?

**What:** Dense retrieval compares the query embedding with stored chunk embeddings in Chroma. BM25 is a lexical relevance function over terms in Elasticsearch. **Why:** “What proof do I need for travel expenses?” might be semantically close to “submit a receipt for reimbursement” even with few shared words; a ticket identifier such as `ERR-42` is often better matched by its exact tokens. Both are examples, not measured results. **How:** The service embeds the query, starts Chroma vector retrieval and Elasticsearch BM25 together, waits for both, then merges the results. The synchronous vector call is sent to a worker thread; thread offload alone is not a throughput guarantee. [Vector][vector], [BM25][bm25], [service][service].

At a high level, BM25 scores each matching term by its rarity across documents and by a saturated term-frequency contribution adjusted for document length. It can favor rare exact terms without letting repeated occurrences increase the score without bound. Dense retrieval instead compresses text into vectors and ranks by similarity. BM25 can miss synonyms; dense search can miss unusual identifiers or return topically similar but irrelevant evidence.

### 9. What retrieval alternatives exist, and why choose this one?

**What:** Alternatives include lexical-only, dense-only, hybrid fusion, a single engine with both lexical and vector search, metadata-filtered search, or domain-specific graph/SQL retrieval. **Why:** Each answers a different access need and adds operational cost. **How to defend this design:** BM25 plus Chroma gives two complementary candidate lists with common IDs, and rank fusion avoids equating unlike scores. It also doubles index operations and introduces cross-store consistency risk. The repository demonstrates this choice; it does not establish that it is optimal for every corpus.

Choose with paired evaluation: same frozen corpus and labeled questions; compare BM25, dense, fusion, and fusion plus reranking; report Recall@k/Hit@k, ranking metrics, exact-ID and paraphrase slices, p95 latency and cost. A single engine can simplify operations, but verify its filtering, update, scale and retrieval behavior against your needs. A graph is justified by relationship traversal, not because the system uses RAG. A table lookup should not be replaced by vector similarity when exact transactional correctness is required.

### 10. What is reciprocal rank fusion, why use ranks, and how do we calculate it?

**What:** RRF merges ranked lists without assuming BM25 and vector scores share a scale. **How:**

`RRF(chunk) = Σ 1 / (k + rank_in_list)` over lists containing that chunk.

Suppose dense search returns `[A, B, C]` and BM25 returns `[B, C, A]`; B is the receipt passage. With smoothing constant `k=60`:

- B: `1/(60+2) + 1/(60+1) ≈ 0.032522`.
- A: `1/(60+1) + 1/(60+3) ≈ 0.032266`.

B ranks ahead of A **at the fusion stage**. These rankings are illustrative. The score is not a probability; a list where the chunk is absent contributes zero. Within a list the implementation deduplicates IDs; deterministic tie-breaking matters for repeatability. Larger `k` dampens differences between nearby ranks. The setting `rrf_k=60` is distinct from the retrieval candidate count and final `top_k`. [Fusion code][hybrid], [configuration][config].

**Trade-off:** RRF discards raw score magnitude. A single exceptionally strong result and a weak result at the same rank receive the same rank contribution. Compare against calibrated score fusion only when scores are well understood and a held-out evaluation justifies the extra complexity.

### 11. What is a reranker, why use a cross-encoder, and how does it work here?

**What:** First-stage retrieval quickly searches a large corpus; a reranker reorders a smaller candidate pool by comparing the actual query and candidate text together. This project configures `BAAI/bge-reranker-v2-m3`. **Why:** A bi-encoder represents query and document separately, enabling indexed search. A cross-encoder processes the query–passage pair jointly, allowing token-level interaction and potentially better relevance discrimination, but it requires inference per pair and cannot cheaply search the entire corpus directly.

**How:** After RRF, the service chooses `candidate_k = max(request.top_k, configured_reranker_candidate_k)`, scores each query–passage pair, sorts by model score, and returns the requested top results. The configured default reranker candidate count is 5; the dense and BM25 retrieval limits default to 100 each, and the request defaults to `top_k=5`. These are separate bounds, and none guarantees that the fused list actually contains that many items. The cross-encoder scores each candidate’s child `content`; it does not score `context_content` used for generation when a parent is present. The model is loaded lazily and inference runs through a bounded executor in the inspected process. [Reranker][reranker], [service][service], [configuration][config].

**Worked continuation:** RRF put B ahead of A. Suppose the reranker scores B `0.9` and A `0.2`: B remains first. If it scores A higher, the final order changes. These are hypothetical scores, not calibrated probabilities or observed project output. A reranker cannot recover the receipt passage if neither first-stage retriever includes it in the candidate pool. Tune pool size by examining quality, inference time and resource usage, not by assuming more is always better.

### 12. What can go wrong during parallel search, and how would you make failure behavior explicit?

**What:** The two retrievers run under `asyncio.gather`; an uncaught failure in a branch can fail the request. **Why:** Waiting for both paths makes a simple fusion contract but increases the number of dependencies. **How to improve:** Decide per use case whether to fail closed or serve a clearly marked single-retriever result. A degraded path must enforce the same document permissions, expose the missing branch to logs and clients, carry its own quality evaluation, and avoid presenting a partial search as complete. Test vector timeout, Elasticsearch timeout, malformed result and both unavailable. This fallback is a proposal, not an implemented guarantee. [Service][service].

**What happens when a request times out?** The current local API runs admitted work under a process-local semaphore and timeout. Cancellation requests stop the task, while wrappers drain non-cancellable thread work before the task releases its admission slot; the client need not wait for that draining. This reduces the chance of piling new work onto still-running native operations. It cannot kill a stuck native thread, coordinate capacity across replicas or guarantee a remote provider stopped processing; shutdown waits only a bounded time. Test repeated cancellation and separate process behavior before relying on these limits.

## Generation, explainability and safety

### 13. What does the generator receive and return? Why use structured output?

**What:** The generator receives the question and a bounded, delimited selection of ranked chunks, then asks an OpenAI-compatible model—configured by default as `gpt-4o-mini`—for an answer, structured claims and source IDs. **Why:** A schema makes output parseable for downstream citation and claim checks. It constrains shape, not truth. **How:** `_format_context` respects a character budget; a chunk may be truncated or omitted. If no chunks exist the code returns a fixed insufficient-information answer without generating. If chunks exist but are irrelevant, the prompt asks for restraint; refusal is not guaranteed. [Generator][generator], [configuration][config].

**Interview distinction:** A retrieval result is not automatically prompt context. Revision `c375903` returns a per-request snapshot of the bounded, escaped text and IDs actually included. This is necessary for a meaningful evidence check; it does not itself establish semantic support. Invalid citation-contract responses can trigger a repair prompt under a shared budget of at most three total provider attempts; repeated contract failures can still surface an error, and not every malformed/provider error is repaired.

**Which provider endpoint is used?** In the current local changes, an empty project-specific `XAI_RAG_LLM_BASE_URL` explicitly selects `https://api.openai.com/v1`; a configured project-specific URL takes precedence, preserving deliberate use of a compatible provider. The SDK's unrelated inherited `OPENAI_BASE_URL` is not the fallback for this generator. This avoids an accidental credential/endpoint mismatch but does not validate a custom provider's security or compatibility.

### 14. What do citation checks guarantee? What do they miss?

**What:** The current local validator requires every structured claim to list source IDs, rejects IDs outside the actual prompt-visible snapshot, and when claims exist requires at least one bracketed ID somewhere in the answer. With no claims, only the canonical insufficient-information refusal with no sources is accepted. **Why:** This catches malformed attribution and IDs for chunks omitted by the context budget. **How:** Formatting records the escaped and truncated text actually presented; citation validation uses the included IDs. An invalid citation contract can cause a bounded repair attempt, but a failure can still propagate. [Historical baseline generator][generator].

**What remains?** The validator does not enforce an inline citation for every factual sentence, detect facts omitted from the structured claim list, prove that a source entails a claim, or establish that the source is current. Treat source support and extraction completeness as separate evaluation problems. The historical `f4c374d` baseline checked IDs against all ranked chunks, including chunks omitted by bounded prompt formatting; that historical description must not be attributed to revision `c375903`.

### 15. What is the NLI faithfulness diagnostic, why is it useful, and how can it mislead?

**What:** The configured DeBERTa NLI model compares a generated claim (hypothesis) with its cited evidence (premise), yielding entailment, contradiction or neutral signals under configured thresholds. **Why:** It can flag claims the cited text may not support, and a claim-by-claim report helps locate the issue. **How:** The current local service optionally checks generated claims against a per-request snapshot of the visible evidence and returns a diagnostic report. The default entailment and contradiction thresholds are 0.5; they are model-routing thresholds, not a guaranteed 50% probability that the answer is true. [Faithfulness implementation][faithfulness], [configuration][config].

NLI can fail on negation, numbers, a long or truncated premise, implicit context and domain shift. A verdict is not a substitute for authoritative source selection or human review. Most importantly, it is not currently an enforced release gate; a streamed answer can be emitted before the report. If a product requires a hard support boundary, buffer the answer, apply a validated check and define abstention/escalation behavior. Measure its latency and false-refusal impact.

**Which exact text is checked?** The formatter in `c375903` retains a per-request copy of the escaped, bounded prompt-visible content, including parent context where applicable. Before NLI, the service applies one `html.unescape` pass to that snapshot; optional RAGAS receives the literal escaped snapshot. Thus NLI checks a decoded interpretation of the visible evidence, not byte-for-byte the original prompt. In the historical `f4c374d` baseline NLI instead examined child `chunk.content` from ranked results, which could differ from the larger parent or truncated context supplied to generation. Neither representation guarantees that the diagnostic is accurate. [Historical NLI][faithfulness], [baseline context formatting][generator].

### 16. How do you prevent prompt injection and cross-user data exposure?

**What:** Retrieved documents are untrusted data and may contain text such as “ignore prior instructions.” **Why:** A document can attempt to alter model behavior, and filtering a response after generation does not undo exposure of restricted text in the prompt. **How today:** Revision `c375903` escapes retrieved text and chunk identifiers when formatting context, rejects whitespace-only queries and bounds query length, and applies per-process API-key rate and concurrency limits. Telemetry filters selected sensitive attributes and does not record raw exception messages in stage spans; filenames or secrets can still leak through other paths, so these are not complete privacy guarantees. Delimited context, instruction language and these checks reduce some risks; they do not prove injection resistance or document-level tenant isolation.

**Production proposal:** Authenticate the user, resolve document permissions server-side, filter both vector and lexical retrieval before fusion, propagate restrictions to caches and logs, and avoid granting the answering model privileged side effects. Evaluate malicious documents, forged citations, cross-tenant queries and transformed or quoted instructions; inspect traces to confirm restricted chunks never reach context. Apply equivalent filters to degraded retrieval and reranking paths. [API][api], [service][service].

### 17. What does “explainability” mean in this project, and what should you avoid claiming?

**What:** Retrieval explanations expose selected pipeline signals such as ranks, scores and matching terms; claim-level NLI reports whether a source appears to support a claim. **Why:** These artifacts make it easier to inspect where evidence entered or failed, and distinguish retrieval from generation errors. **How:** Preserve each stage's candidate IDs and rank movements, link final claims to source IDs, and show uncertainty when a stage fails. [Service][service].

A retrieval score explains the ranking mechanism, not the model's private reasoning. An NLI verdict is a model judgment, not an explanation of causality or an independent proof. The project also reports an `overall_confidence` derived from available diagnostics; without a calibration study on held-out cases, do not call it the probability the answer is correct.

## Evaluation and production thinking

### 18. What should be measured and why are correctness, faithfulness and relevancy different?

**Answer:** “I separate retrieval quality from answer quality. Correctness asks whether the answer matches authoritative facts; faithfulness asks whether its claims are supported by the supplied evidence; relevance asks whether it addresses the question. A faithful answer can still repeat an outdated source. I measure retrieval separately because a correct answer does not prove the right evidence was found.

“We ran a real HTTP smoke test with fictional documents: **14/14 checks passed**, including REST and streaming, on a frozen snapshot. In a separate baseline run, a model judge scored faithfulness, correctness and relevance at **2/2 on average over eight synthetic answers** using a custom rubric; that is not human review or a production benchmark. A harder retrieval test missed one English-question/Hindi-source example in the top five at every stage. The NLI checker and model judges also made observable errors, so I treat their scores as diagnostics, not proof. Next I would build an independently reviewed, representative evaluation with retrieval labels, answer rubrics and production-specific failure cases.”

**Follow-up:** Context relevance and precision concern whether retrieved passages are useful and appropriately ranked. Each metric needs a defined rubric, unit and denominator. The project has an offline evaluator for recorded JSONL observations and optional request-scoped RAGAS metrics. Its synthetic smoke fixture tests evaluator plumbing; optional NLI is not a release gate. [Evaluation contract][evals], [evaluation Q&A](evaluation-interview-qa.md), [runtime evidence](runtime-evidence.md).

#### Drill: application and evaluator evidence

**Did we run the application end to end?** Yes. An isolated HTTP smoke run passed **14/14 checks** against a real Uvicorn/FastAPI process, disposable Chroma and Elasticsearch services, locally cached embedding and reranker models, and live `gpt-4o-mini` generation. It ingested two fictional documents as two chunks, then exercised authenticated REST and SSE requests: the REST response returned the expected synthetic policy deadline with a source ID, and SSE delivered `retrieval → generation → done`. A deliberately invalid upstream credential produced `retrieval → error` without `done`, even though the initial stream HTTP status was 200. One document-borne injection marker was present in retrieved evidence but was not repeated in that answer; one case does not establish injection resistance.

This run used a **frozen local snapshot** based on `f4c374d` plus then-uncommitted changes; it does not automatically test revision `c375903`. Earlier attempts inherited a provider endpoint mismatched with the chosen credential; the successful run explicitly selected the **official OpenAI endpoint** using process-local configuration. NLI and RAGAS were disabled, and the fixture does not establish human-reviewed answer quality, multi-user authorization, production load, recovery or general security. The [HTTP evidence](http-e2e-evidence.md) records the snapshot manifest, exact checks, cleanup and failures.

**What did the existing sample score?** The checked-in fixture has two synthetic query records and one synthetic ingestion record. Its answer, labels, simulated `human_scores` and `judge_scores`, and latencies were already written into the file; the offline evaluator reads them and makes no model or human judgment. A run with `@1` and `@2` cutoffs produced:

| Metric | Score and denominator | What this establishes |
| --- | --- | --- |
| Citation validity | `1.0` on one non-refusal answer | Its recorded cited IDs occur in the recorded reranked IDs; semantic support is not established. |
| Captured structured-claim coverage | `1.0` on one non-refusal answer | Recorded claims include IDs in that recorded list; assertions omitted from structured claims are invisible to this metric. |
| Recorded human correctness | `1.0` on one record | A synthetic field was prefilled; **no human performed an assessment for this fixture run**. |
| Recorded judge relevance | `1.0` on one record | A synthetic field was prefilled; **no LLM judge produced that score for this fixture run**. |
| Refusal accuracy | `1.0` on two records | The synthetic expected and actual refusal flags matched. |
| Index parity | `1.0` on one ingestion record | The synthetic ingestion record reported parity; this does not prove live index consistency. |

All six blocking and six advisory gates passed with both cutoffs specified. The default cutoffs are `1, 3, 5, 10`, while two advisory gates ask for `@2`; those metrics are `not_computed` with the default command. The smoke scores verify how the evaluator processes these supplied records, not production accuracy or latency. [Fixture][fixture], [runtime evidence](runtime-evidence.md).

#### Drill: retrieval and multilingual tradeoffs

**A separate real retrieval experiment:** In an isolated run, the actual vector and BM25 retrieval, RRF, and BGE cross-encoder reranker were exercised on eight synthetic labeled questions over six fictional chunks. `Hit@k` asks whether at least one labeled relevant chunk appears in the first `k` results; the denominator here is eight questions at each stage.

| Stage | Hit@1 | Hit@3 |
| --- | --- | --- |
| Vector | `8/8` | `8/8` |
| BM25 | `6/8` | `7/8` |
| RRF | `6/8` | `8/8` |
| Cross-encoder rerank | `8/8` | `8/8` |

These are observed retrieval results on a small constructed example, not a representative benchmark or evidence that reranking generally improves production quality. They measure whether labeled chunks appeared, not answer correctness or faithfulness. The retrieval experiment itself produced no LLM-judge or human-review scores.

**Larger retrieval-only experiment:** A second authored fictional set contained 25 passages and 40 questions. All 40 completed without retrieval errors, but only **37** had one or more labeled relevant passages, so retrieval averages use 37 as the denominator; three no-relevant-label cases are excluded from these retrieval metrics. Some eligible questions require two passages. Each passage was indexed directly as one chunk, so this run did not test file parsing or chunking.

| Stage | Hit@1 | Hit@3 | Recall@3 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.9730 | 1.0000 | 1.0000 | 0.9865 |
| BM25 | 0.8649 | 1.0000 | 1.0000 | 0.9324 |
| RRF | 0.9459 | 1.0000 | 1.0000 | 0.9730 |
| Reranked RRF | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Hit@3 checks whether *any* labeled passage appears; Recall@3 accounts for *all* labeled relevant passages, so the distinction matters for a two-source question. These author-written labels were not independently human-reviewed, the corpus is small and easily searchable, and the run measured neither generated-answer quality nor guardrails. Its saved source snapshot can differ from the published revision. [Dataset, methods and per-case results](holdout-evaluation.md).

**A harder retrieval challenge:** A separate agent authored 31 fictional passages and 28 questions after seeing the earlier results. Two questions had no labeled relevant source, leaving **26 positive questions** for the retrieval averages. Five of those questions needed multiple passages. All 28 retrieval queries completed; the table reports results on the positive cases only.

| Stage | Hit@1 | Hit@3 | Recall@3 | All needed sources@3, five multi-source cases |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.8462 | 0.9615 | 0.9423 | 4/5 |
| BM25 | 0.7692 | 0.9615 | 0.9615 | 5/5 |
| RRF | 0.8077 | 0.9615 | 0.9423 | 4/5 |
| Reranked RRF | 0.9615 | 0.9615 | 0.9615 | 5/5 |

All-needed-sources@3 requires every labeled supporting passage within the first three returned source positions. In case C15, dense and RRF each missed one required passage in those positions. In C17, an English question sought a labeled Hindi passage; that passage was absent from the top five returned at every stage, not absent from the corpus. This is one case to investigate, not a general multilingual performance claim. The AI-authored cases and labels had no independent human review; this exploratory set is not a sealed benchmark. The experiment tested retrieval only, not parsing, chunking, generation, judges or guardrails. The saved package snapshot may differ from the published revision. [Challenge details and results](holdout-evaluation.md#harder-retrieval-challenge).

**Could a multilingual embedding model help C17?** In a separate controlled *dense-only exact-search* comparison over the same 31 fictional passages and 26 labeled-positive questions, the configured BGE small English model ranked C17's relevant Hindi passage **22nd**, while multilingual MiniLM ranked it **first**. Across all 26 questions, at least one relevant source at rank 1 changed from **22/26 to 21/26**, while all labeled sources in the top five changed from **25/26 to 26/26**. Thus this example improved while another aggregate measure regressed. This comparison did not use Chroma, BM25, RRF or reranking and is not a repeat of the full hybrid pipeline above. Its cases and labels are AI-authored with no independent human review. It does not establish general model superiority or justify a production switch; that would require representative independently labeled data, full-pipeline testing, a versioned re-embedding and rollback plan. [Methods and per-case ranks](multilingual-retrieval-diagnostic.md).

**Does that finding hold on a harder constructed language slice?** A second dense-only diagnostic used 20 fictional passages with close distractors and 40 questions, 10 per slice. Each question has one labeled source. The figures are BGE → multilingual MiniLM:

| Query → document slice | Top 1 | Top 3 | Top 5 |
| --- | --- | --- | --- |
| Hindi → Hindi | 3/10 → 10/10 | 6/10 → 10/10 | 7/10 → 10/10 |
| English → Hindi | 0/10 → 4/10 | 2/10 → 6/10 | 3/10 → 10/10 |
| Hindi → English | 0/10 → 2/10 | 0/10 → 7/10 | 0/10 → 10/10 |
| Exact-code questions, mixed language directions | 7/10 → 8/10 | 7/10 → 10/10 | 7/10 → 10/10 |

In two exact-code questions MiniLM put service **T-24** first when the question asked for **T-42**; retrieving the intended document in the top five does not ensure the later stages pick it. An AI agent authored the synthetic data and labels after seeing C17; no independent human reviewed them. The small, shared-template slices cannot support a general multilingual claim. This was exact dense retrieval, without hybrid retrieval or answer generation; MiniLM was evaluated as a candidate and was not silently deployed. [Fixture, methods and case-level results](multilingual-challenge-diagnostic.md).

**Does adding lexical retrieval fix the code confusion?** On those same 40 questions, another diagnostic combined the candidate MiniLM's top five exact-cosine results with actual Elasticsearch BM25 results using the archived app's RRF (`k=60`). For the ten exact-code questions, top-1 hits rose from **8/10 with MiniLM to 9/10 with RRF**. But on English → Hindi they fell **4/10 to 0/10**, and on Hindi → English **2/10 to 0/10**. A lexical branch can help identifiers while hurting cross-language ranking; unconditional fusion requires testing.

In M38, RRF put the requested T-42 ahead of T-24, but their fused scores tied exactly. This test broke ties using readable source IDs; the application uses content/provenance-derived chunk IDs, so its tie can resolve differently. Elasticsearch's standard analyzer split `T-24` into `t` and `24`, and `T-42` into `t` and `42`: these tokens do not enforce an exact whole-code match. This was a small AI-authored synthetic set without independent human review, no reranker or generator, and with candidate MiniLM dense rankings rather than the application's configured BGE model. It does not establish a production fix or general performance. An exact-identifier field and language-aware fusion are possible changes to evaluate, not implemented outcomes. [Methods and case-level results](holdout-lexical-comparison.md).

**Would tuning the fusion weight help?** An offline study predeclared an odd/even split and chose a single lexical weight from the odd-numbered 20 questions by training Hit@1. It selected lexical weight **0**, i.e. dense-only. On the 20 even-numbered test questions, dense-only achieved **11/20** Hit@1 versus **9/20** for equal-weight RRF; on the five exact-code test questions the ordering reversed, **3/5 versus 4/5**. Equal-weight fusion had exact ties at the top-one boundary in **3/20** test cases, resolved by source ID in the diagnostic rather than production chunk ID. The split was specified before this calculation, but the dataset and earlier results were already exposed, so it is not an independent blinded holdout. The data and labels are synthetic and AI-authored without human review, and each slice is tiny. This simulation on saved candidates does not justify changing production weights. [Selection rule, results and tie counts](holdout-weighted-rrf.md).

#### Drill: answer quality, NLI and judge reliability

**Separate generated-answer and model-judge run:** On the same eight synthetic questions, clean baseline `f4c374d` with `gpt-4o-mini-2024-07-18` generated 8/8 answers. A separate `gpt-4.1-mini-2025-04-14` judge used a custom `0–2` rubric and scored mean faithfulness, correctness and relevance each `2.0/2` over **8 scored cases**. On a saved in-progress post-change snapshot, only 7/8 answers were judged because one raised `GenerationContractError`; each mean was `2.0/2` over the **7 scored cases**, excluding the failure. These scores are neither RAGAS nor human assessments, the samples and references are synthetic, and the post-change mean is not evidence of improvement. The snapshot can differ from the published revision. [Live evaluation report](live-evaluation.md).

**What did a standard evaluation library measure?** A separate run called the project's actual `evaluate_response` integration with **RAGAS 0.4.3**, scoring four saved synthetic answers from historical `f4c374d` and one misleading extended-window control. Across the four answers, mean faithfulness was **1.0**, answer relevancy **0.789743**, and reference-based context precision approximately **1.0**. The control replaced a 30-day policy with “within 365 days” and received faithfulness **0.5**. That wording is logically ambiguous and partially supported: a submission within 30 days is also within 365 days, although the wider window misleadingly implies more time. The recorded results do not establish which claim RAGAS judged unsupported. All **5/5 case evaluations** returned **15 metric scores**, using **45 successful provider requests**. RAGAS faithfulness judges extracted claims against context; answer relevancy compares generated questions with the user's question using embeddings; context precision assesses passage ordering against a reference. None alone establishes correctness.

In a separately executed control, the answer explicitly claimed a return **on day 365** was permitted by the 30-day policy. It scored **0.0 faithfulness**, **0.5122 answer relevancy** and approximately **1.0 context precision**, with **nine additional successful requests**. This is one control case, not an estimate of judge accuracy; its later scorer source snapshot included client cleanup. Both controls are excluded from the four-answer means above.

These were scorer snapshots over saved generation evidence, not new retrieval or generation runs. The inputs and references were fictional and AI-authored, with no human validation; four examples are not a robust quality estimate. RAGAS and the preceding custom judge have different methods and scales, so their scores must not be combined. [Methods, snapshots and per-case results](ragas-live-evaluation.md).

**Separate actual NLI probe:** The pinned DeBERTa checker was run on ten authored synthetic premise–claim pairs; expected and observed labels matched **9/10**. For raw `x < 5`, the verdict was supported (entailment `0.9878`); the escaped `x &lt; 5` probe was neutral (entailment `0.2024`). These are model outputs against labels created for the probe, not independently human-validated truth or calibrated correctness probabilities. The service in `c375903` applies one HTML-unescape pass before NLI; the direct probe illustrates why representation matters but does not by itself validate the entire revised service. [NLI probe](nli-evaluation.md).

**What happens on a broader NLI stress set?** The actual checker ran offline on 42 fictional, single-chunk premise–claim pairs using DeBERTa-v3-base-mnli-fever-anli revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`, with entailment and contradiction thresholds each at **0.5**. It agreed with author-created labels on **37/42**. In N12, “No visitor may enter without a badge” did not establish that all visitors have badges, yet the checker marked that claim supported. In the six cross-language cases it matched **2/6**; the four expected support or contradiction cases were all marked neutral. Neutral represents uncertainty, not proof of falsehood. The examples and labels were AI-authored without independent human validation; these small templated slices cannot establish general accuracy. The test does not cover retrieval, multi-chunk aggregation or generation. [NLI stress methods and case-level scores](nli-stress-diagnostic.md).

#### Drill: citations, adversarial inputs and conflict handling

**Focused citation-repair follow-up:** In three generator-only repeats per case against an isolated saved snapshot, a partially answerable case returned acceptable-contract responses in `2/3` attempts; the other attempt still raised `GenerationContractError`. The same snapshot returned `3/3` responses for each of four other selected cases. These are response completion counts on fictional inputs, not human-reviewed accuracy rates or proof that repair always succeeds. [Saved setup and per-attempt results](live-evaluation.md#focused-citation-repair-follow-up).

**What did a targeted citation-format change show?** A separate generator-only diagnostic reconstructed the fictional H05 evidence from saved IDs, ranks and hash-verified fixture text; the original failed call's exact prompt context was not saved, so equivalence is unverified. Using the same model and reconstructed evidence, the before snapshot accepted `0/5` generations; the after snapshot, whose prompt explicitly demonstrates bracketed citation IDs and rejects attribute-style forms, accepted `5/5`. The existing strict citation validator was unchanged. The runs used different frozen source snapshots, were not randomized, and cover one synthetic case with no independent human review; they do not establish a general improvement or semantic support. [Diagnostic, source hashes and raw results](evaluation/h05-paired-diagnostic.md).

**How reliable was the automated judge?** Two OpenAI judges assessed the same 12 saved fictional guardrail answers and agreed on `10/12` pass/fail labels; one of 13 attempted generations had failed and was not judged. Agreement is not accuracy: both judges marked one response as failing for citing `[g]`, even though `[g]` was the visible source and the instruction to cite `[h]` came from untrusted document text. Their shared mistake is a reason to keep deterministic citation checks separate and calibrate semantic judgments against independent human review. No such human evaluation was performed here. [Judge agreement and case details](judge-calibration.md).

**What did an adversarial generator probe reveal?** On a frozen generator snapshot, 20 fictional cases were run twice: **40/40 outcomes returned**, including two fixed no-evidence responses without model calls and **38 provider calls**. An AI assistant reviewed the saved outputs; there was no human review or independently validated ground truth. In both repeats of A18, the generator abstained instead of explaining two conflicting closing times. In both A20 repeats, the first factual sentence lacked its own inline citation, although the source supported it and the structured claims referenced that source. In both A17 repeats, it correctly avoided inventing a deadline but included an absence statement in structured claims despite a prompt instruction to omit such statements. These are three unique cases with two repeats each; returned results are not correctness passes. This generator-only probe does not assess retrieval, tenant isolation or actual network side effects, and establishes no safety rate. [Run results](generator-adversarial-results.md); [case-by-case AI review](evaluation/generator-adversarial-ai-review.md).

**Did asking the model to report conflicts fix A18?** In a paired prompt experiment, the original prompt reported both conflicting times in **0/2** A18 runs; the candidate reported both with citations in **2/2**, while the citation validator stayed unchanged. An extended synthetic test produced **24 outcomes** across both prompt variants. In its E01 case, both candidate responses gave the supported 20-day deadline but added an unsupported starting event, “from the date of submission” (**2/2**). A citation using a visible ID does not prove that every attached detail is supported. The candidate was not adopted. These are authored synthetic examples inspected by an AI agent, with no independent human review; the small repeated sample cannot establish a general improvement. [Initial experiment](conflict-prompt-experiment.md); [extended recorded outputs](evaluation/conflict-extended-results.json).

#### Drill: rubric design and next evaluation

**A separate judge-rubric probe:** On 16 authored fictional cases with expected labels set before the calls, `gpt-4.1-mini-2025-04-14` matched the authored faithfulness score on `12/14` cases containing factual claims and the authored correctness score on `16/16` cases. The two abstentions were excluded from the faithfulness denominator; the judge returned the requested `null` faithfulness score for only `1/2` of them. In two mixed-support cases it returned `0` where the authored rubric expected `1`, apparently overlooking a supported assertion. These are separate cases from the preceding two-judge probe; do not add their denominators. The examples and labels were AI-authored, not independently human-annotated, so agreement is not measured human accuracy and cannot establish general judge reliability. [Case-level results and rubric](judge-fixture-study.md).

**How would a judge score the three dimensions?** A practical *proposed* rubric can use `0`, `0.5` and `1`, with a separate `unscorable` outcome:

| Dimension | `0` | `0.5` | `1` | Judge inputs |
| --- | --- | --- | --- | --- |
| Correctness | A key fact is wrong or the answer is unrelated. | Core fact is right but a necessary detail or exception is missing. | All required facts are right with no material error. | Question, answer, reviewed reference and authoritative source. |
| Faithfulness | A material claim contradicts or lacks support in the supplied context. | Some claims are supported but another is ambiguous or inadequately supported. | Every material factual claim is supported. | Answer and the exact text actually presented to the generator. |
| Answer relevance | Does not address the requested information. | Addresses some requested parts. | Directly addresses all requested parts without material irrelevant content. | Question and answer. |

For example, if the source requires a receipt **within 30 days** and the answer says **within 60 days**, the answer can address the question while being incorrect and unsupported. Require the judge to give a source-linked reason for each dimension; retain errors and unscorable cases instead of hiding them in averages. This `0 / 0.5 / 1` rubric is an illustrative proposal, distinct from the **actual custom `0 / 1 / 2` judge rubric** used in the eight-question run above. Neither is the exact RAGAS formula or a completed human study. An independent human review would require real reviewers, a frozen rubric, labeled cases, reviewer agreement and adjudication; none is evidenced by the synthetic fixture.

**How to design a serious evaluation:** Freeze and version the corpus; label questions, supporting chunks, expected answerability and reference answers; split development and held-out data by source or topic where appropriate. Compare retrieval stages and inspect per-case evidence. Use human review with a clear rubric and report inter-rater disagreement. An LLM judge can scale review if calibrated against human labels, with blind order and per-case judgments retained; it is not a human reviewer and may disagree or share model biases. Include contradictory evidence, missing information, numbers, stale policies, injection and tenant-isolation cases. Track uncertainty, coverage, latency and cost alongside quality.

### 19. How would you debug a wrong answer without guessing?

**What:** A wrong result can arise from parsing, chunking, indexing, retrieval, fusion, reranking, prompt formatting, generation or checking. **Why:** Changing the model prompt when the source never entered the index cannot fix missing evidence. **How:**

1. Verify source extraction and version; did the relevant sentence survive into a chunk?
2. Verify that the intended chunk is indexed in each expected store and visible for search.
3. Compare vector and BM25 ranks, RRF rank and reranker rank for that ID.
4. Inspect which exact text and IDs survived the context limit.
5. Compare each answer claim with the cited source and the authoritative version.
6. If the final answer fails, inspect the structured output, citation validation and optional NLI; record errors and latency by stage.

For a misranked receipt passage, start with a candidate trace before tuning the cross-encoder. If the correct passage reaches the prompt but the answer changes “30 days” to “60 days,” investigate generation and claim verification. These examples illustrate diagnosis, not observed failures.

**Where would I inspect this in code?** Start at `RAGService.ingest` and `RAGService.query` in `service.py`: these show the sequence and error boundaries. Follow `build_chunk_id`, `store_chunks_chromadb`, `store_chunks_elasticsearch` and `check_index_parity` in `ingestion/store.py`; then `vector_search`, `bm25_search`, `rrf_fusion` and `Reranker.rerank` in the retrieval modules. Inspect `_format_context` and citation validation in the generator when the right hit is retrieved but the answer is wrong. Source links in this guide show the baseline; inspect [published source `c375903`](https://github.com/neeraj1909/xai-rag/tree/c375903fb46327744d076438cb90f7b02711a94f) for the later implementation.

**How can I reproduce the evidence?** Use [local reproduction](local-reproduction.md) for setup, isolated dependencies and a fictional-document walkthrough. [HTTP evidence](http-e2e-evidence.md), [retrieval results](holdout-evaluation.md), and [RAGAS results](ragas-live-evaluation.md) identify their own fixtures, snapshots and commands. Rerunning against the current checkout is a new observation, not automatic reproduction of an older saved snapshot. Live generation or RAGAS calls use provider credits; recorded results alone do not incur new calls.

**How did the system behave in a longer stability run?** A **75-minute**, low-rate isolated run used real embedding, search and reranking on synthetic data, with generation replaced by a deterministic local stub. Readiness returned HTTP 200 for **892/901** probes. Seven of the nine other probes occurred during deliberate Chroma or Elasticsearch pauses; the two transport errors outside those windows had no established cause, and subsequent probes succeeded. **14/15** scheduled queries and **3/4** scheduled re-ingestion/parity checks passed; both scheduled failures occurred during the Elasticsearch pause. A fresh query and re-ingestion succeeded after recovery. The failed ingestion also reported parity true, which did not mean ingestion had succeeded.

This exercised one machine at low request rates against a frozen source snapshot that included then-uncommitted changes; the current source changed afterward. It did not exercise a live generation provider or establish answer quality, capacity or production readiness. The observed Elasticsearch-pause HTTP 500 belongs to that snapshot and should not be attributed automatically to the later implementation. [Soak protocol and evidence](soak-evidence.md).

**Was the later outage handling checked against a real store?** Yes. In a separate isolated run with one synthetic document, real Elasticsearch and Chroma, and a local generation stub, pausing Elasticsearch caused a sanitized REST **503** after **10.071 seconds**. The stub's call count did not increase during the failed query. After Elasticsearch resumed, a fresh query returned **200** with a cited synthetic answer. The earlier soak's **500** was observed on a different, older snapshot. This later check exercises one local timeout, not every dependency failure, streaming behavior, live-provider quality or production recovery; shutdown also emitted a semaphore cleanup warning whose significance was not established. [Outage setup, snapshot and results](outage-evidence.md).

### 20. How would you take the current design toward production?

**What:** Production readiness is a set of explicit reliability, security and quality requirements matched to the user's risk. **Why:** Two local search containers and a successful demo do not establish availability, data isolation, recovery or answer quality. **How:**

1. **Define contract and access.** Identify users, allowed sources, answer risks, freshness and retention. Authenticate and authorize every retrieval before context reaches the model; protect service-to-store traffic and secrets.
2. **Make ingestion recoverable.** Treat a source version and the two index writes as a workflow with durable status. Reconcile after partial failures, propagate deletions, prevent stale jobs overwriting new versions, and test restore.
3. **Choose storage from measured requirements.** Chroma plus Elasticsearch may remain appropriate if the chosen deployment meets isolation, backups, capacity and operating requirements. Evaluate their exact versions and failure modes; migrate only when a specific unmet requirement and comparison justify it.
4. **Define behavior under failure.** Decide whether to abstain, return a labeled degraded answer or retry; bound timeouts, concurrency and costs. Never allow a fallback to bypass authorization.
5. **Establish evidence.** Collect a permissioned, representative dataset with human-reviewed labels. Compare retrieval and model changes on paired cases; set target quality and latency against risk rather than copying arbitrary numbers.
6. **Operate and change safely.** Capture per-stage timings, failures and cost with privacy-aware telemetry; test load and restore; use versioned indexes, staged rollout and rollback. Measure NLI or judge failure rates and audit access.

These are proposals. Separate deterministic ID/upsert mechanisms already in the code from a durable cross-store recovery mechanism that is not established by the inspected implementation. Read the [production Q&A](production-interview-qa.md) for the interview questions and [database Q&A](vector-database-interview-qa.md) for failure scenarios.

### 21. How would you decide whether to keep Chroma in production?

**What:** The project uses a Chroma **HTTP server**, not an embedded `PersistentClient` in this path. **Why keep it?** Existing integration, vector retrieval and a modest workload may make it sufficient if the selected deployment satisfies documented operational and security needs. **Why consider alternatives?** A concrete need around multi-tenant authorization, recovery, capacity, latency, compliance or operational ownership may favor a different deployment or engine.

**How:** Write workload assumptions, measure recall and p95 under the expected filter/selectivity and concurrency, test backup and restore, failure handling, deletion and index version changes, and verify tenant enforcement in the application and database. Compare alternatives on those same tests, migration risk and total operation cost. Chroma collection/namespace identifiers alone do not establish authorization. Elasticsearch and Chroma are separate systems; no choice of vector database alone solves cross-store consistency. [Storage][store], [Chroma deployment documentation](https://docs.trychroma.com/reference/architecture/overview).

### 22. What would you say if asked for a limitation or a failure you would fix first?

**Answer:** “The historical baseline accepted IDs from ranked chunks even when they were omitted from the prompt; the published `c375903` changes capture the included evidence and validate IDs against it, with bounded contract repair. That improves format and provenance checks but still does not prove each claim is supported. Optional NLI is generated after the streaming answer event, so it remains diagnostic. If verified output is required before release, I would evaluate a blocking check, its false-refusal rate and latency. In parallel I would define cross-store recovery and document-level authorization. I would prioritize using product risk and measured failure data.”

This answer identifies a real implementation boundary, a concrete failure, a minimal proposed correction and how to test it. Do not claim the change is already implemented.

## Short interview answers: original rehearsal questions

These concise answers are for a quick pass; the numbered sections above contain the reasoning and examples.

1. **Why semantic and keyword search?** Dense vectors can retrieve related wording; BM25 can preserve exact terms. Their combination needs a paired evaluation on the target corpus, and two stores need consistency controls.
2. **Retrieval versus reranking?** Retrieval narrows a large collection using an index. A cross-encoder then jointly scores the question and a bounded candidate set; it is more expensive per item and cannot recover a missing candidate.
3. **Why not add BM25 and vector scores directly?** Their scales differ. RRF combines ranks, sacrificing magnitude information to avoid unvalidated score calibration.
4. **Can a chunk appear twice?** The fusion code deduplicates chunk IDs within a list and combines the same ID across lists. Incorrect or inconsistent IDs can still break attribution and fusion.
5. **How is retrieval explained?** Stage ranks, scores and matching terms show why a result surfaced in this pipeline. They do not explain every part of the model's reasoning or prove the resulting answer.
6. **What does the faithfulness model check?** It compares a structured claim with cited evidence using NLI. An entailment verdict is a diagnostic subject to errors, not evidence that the underlying source is true or current.
7. **Is overall confidence a calibrated probability?** No. The code derives it from available diagnostics; a valid probability claim would require calibration on suitable held-out data. An unavailable score can also yield zero.
8. **What if one search dependency fails?** The current parallel request can fail; a single-index fallback is a proposed product decision with separate access checks, quality measurement and visible degraded status.
9. **What if a document contains an instruction to the model?** Retrieved text is untrusted. Delimiting it and instructing the model to treat it as data helps, but test attacks, restrict authority and prevent unauthorized context from entering the prompt.
10. **How are citations checked?** Structured claims need source IDs; unknown IDs are rejected and a nonempty claim list requires an inline ID somewhere in the answer. The local changes restrict IDs to the bounded prompt snapshot; this is still not per-claim semantic support or proof every assertion was captured.
11. **How do you separate retrieval and answer quality?** Label relevant evidence, inspect recall/ranking at every retrieval stage, then independently judge correctness, completeness and claim support for the answer. A correct answer by luck does not prove retrieval worked.
12. **What would you improve next?** Prioritize by product risk: validate the new prompt-visible citation contract, add document-level authorization and durable cross-store recovery, and run representative end-to-end evaluation. Explain tests and trade-offs before claiming any are already implemented.

**Your contribution:** Replace generic “I built” statements with the exact components you personally designed or changed, what you reused, how you tested them, and one real issue or decision you can explain. Repository structure establishes system behavior; it cannot establish sole authorship or a measured business impact.

## Five-minute recall

1. **Business problem:** answer questions over changing documents and show the evidence; retrieval does not make the answer automatically true.
2. **Ingestion:** validate → parse → chunk → embed → Chroma + Elasticsearch under shared IDs → report status and parity; writes are separate.
3. **Question:** query embedding → parallel dense/BM25 → RRF → bounded cross-encoder → structured answer → optional NLI.
4. **Why hybrid?** Exact-token and semantic recall can complement one another; prove the benefit with paired labeled questions and account for two-store operations.
5. **RRF example:** `1/62 + 1/61` exceeds `1/61 + 1/63`; ranks are signals, not probabilities.
6. **Guardrails:** correct ID format is not source support; check the exact presented context, tenant access and the timing of NLI versus streaming.
7. **Evaluation:** distinguish retrieval, correctness, faithfulness, relevance and refusal. Report dataset, denominator, judge and human-review status with every score.
8. **Production:** define access, data lifecycle, recoverability, failure policy, measured latency/quality and change rollback before claiming readiness.

[service]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/service.py
[store]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/ingestion/store.py
[parser]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/ingestion/parser.py
[chunker]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/ingestion/chunker.py
[config]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/config.py
[vector]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/retrieval/vector_search.py
[bm25]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/retrieval/bm25_search.py
[hybrid]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/retrieval/hybrid.py
[reranker]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/retrieval/reranker.py
[generator]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/generation/generator.py
[faithfulness]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/explainability/faithfulness.py
[api]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/api/app.py
[evals]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/evals/README.md
[fixture]: https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/evals/smoke/evaluator-contract.jsonl
