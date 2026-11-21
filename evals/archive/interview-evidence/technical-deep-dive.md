# XAI-RAG: architecture, retrieval math and production interview answers

> **Version note:** This document describes the earlier `c375903` or explicitly cited historical source. Statements here that call NLI optional, say document authorization is absent, or describe recovery as only a proposal do not describe the later hardening work. Read the [current hardening questions and answers](hardening-interview-qa.md) first, and verify the source revision before an interview. Historical measurements remain tied to their stated snapshots.

**Revision boundary:** The historical baseline is `f4c374d`; later safeguards are included in [published source `c375903`](https://github.com/neeraj1909/xai-rag/tree/c375903fb46327744d076438cb90f7b02711a94f). Source links below may point to the baseline; see [guardrail implementation](guardrails-implementation.md) and [runtime evidence](runtime-evidence.md) for implementation and verification limits. Historical experiment snapshots are not automatically identical to the published revision.

Read with the [project overview](README.md). This chapter describes inspected source revision [`f4c374d`](https://github.com/neeraj1909/xai-rag/tree/f4c374d51d090beba61d34aba81fcc23b8ef700a). Defaults are configuration, not proof that a model was run or a particular result achieved. Rehearse first-person contributions only where you can substantiate them.

## System boundaries and the two pipelines

```text
INGESTION (asynchronous to user questions)
 allowed file → parse → chunk (fixed / semantic / parent-doc)
                        ├─ embedding → Chroma vector + chunk metadata
                        └─ same chunk IDs + text → Elasticsearch BM25
                                         ↓
                             report failures and index parity

QUERY (ordinary async service orchestration; no LangGraph state machine)
 user question ─ embedding ─→ vector search ─┐
               └────────────→ BM25 search ─┴→ rank fusion (RRF)
                                                  ↓
                                  bounded cross-encoder reranking
                                                  ↓
                         optional rank explanation → answer + source IDs
                                                  ↓
                          optional NLI diagnostic / optional RAGAS
```

Chroma and Elasticsearch are separate stores. Matching chunk IDs make fusion and source attribution possible, but do not create an atomic write across the stores. For streaming, the generation event can reach the client before optional faithfulness diagnostics. The NLI output is diagnostic, not a blocking acceptance gate. [Service](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/service.py), [generator](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/generation/generator.py).

### Component and model choices

| Component | Inspected default / role | Interview trade-off |
| --- | --- | --- |
| Embedding | `BAAI/bge-small-en-v1.5`, configured dimension 384; model revision pinned. | Fast reusable corpus vectors support broad retrieval; semantic similarity is not proof of relevance. Changing embedding family/revision calls for index compatibility planning. |
| Lexical index | Elasticsearch BM25 over text and IDs. | Exact terminology and identifiers can be strong; vocabulary mismatch can lose semantically relevant passages. |
| Fusion | RRF `k=60`, two configured retrieval lists. | Combines ranks without comparing raw score scales; loses score-magnitude information. |
| Reranker | `BAAI/bge-reranker-v2-m3`, configured candidate pool default 5. | Joint question–chunk scoring can improve ordering, but costs inference per candidate and cannot recover a chunk missing from the pool. |
| Generator | OpenAI-compatible structured-output path; default configured `gpt-4o-mini`. | Produces answer, citations and claims under schema constraints; valid structure is not grounding. Provider calls may incur cost. |
| Claim diagnostic | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`; configured entailment and contradiction thresholds 0.5. | Compares claim and cited source but can fail on numbers, negation and domain shift; not a gate or calibrated probability of truth. |
| Semantic chunking | `all-MiniLM-L6-v2` when semantic chunking is selected. | Extra embedding work at ingestion; chunk coherence versus a fixed-size baseline is an empirical question. |

Settings can be overridden; cited values are defaults in [configuration](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/config.py). The documented default search limits are 100 vector and 100 BM25 candidates, followed by a configured bounded reranker pool; do not conflate search `k`, RRF smoothing `k`, candidate-pool size and final response `top_k`.

## Worked retrieval and generation case

Suppose a synthetic employee manual has: A = reimbursement deadline; B = “Submit a receipt with the reimbursement request”; C = finance review. The question is “What do I submit?” Assume the vector index returns `[A, B, C]`, while BM25 returns `[B, C, A]`; these are **illustrative**, not observed scores.

At `k=60`, `RRF(B) = 1/(60+2) + 1/(60+1) ≈ 0.03252`, while `RRF(A) = 1/61 + 1/63 ≈ 0.03227`. Thus B precedes A at fusion. A document absent from one list contributes zero from that list; duplicate IDs within a list contribute once. Ties are resolved deterministically by ID in the inspected implementation. Reranking may then change the order: fusion rank is not the final reranker rank. [Fusion code](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/retrieval/hybrid.py).

If B is selected, the structured generator might return an answer citing B and a claim with B as its source. At baseline the validator checked IDs against the ranked list. In `c375903`, it rejects IDs without nonblank evidence actually included in the bounded prompt, but B's visible ID alone does not prove the answer is supported. In `c375903`, NLI receives request-local prompt-visible evidence, decoded once from the prompt’s escaped representation; RAGAS receives the literal displayed text. NLI can still be wrong. Empty retrieval triggers a fixed insufficient-information answer. Irrelevant but nonempty retrieval is a different case: a refusal is requested, not guaranteed.

**Follow-up: Why does `1/61` matter less when `k` is large?** The constant dampens rank differences: two nearby ranks produce similar contributions, while appearing in both lists can be important. It is a ranking heuristic, not an answer confidence. Tune it only against labelled queries and check exact-term and semantic slices separately.

## Dataset and evaluation design

The project ships a synthetic [evaluator contract fixture](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/evals/README.md). Its manifest pins dataset/corpus/configuration/model revisions and records the label source; query rows can contain relevance labels and observed IDs per stage, citations, NLI, refusals, errors and stage latency. Ingestion rows capture parse, provenance and index parity observations. The fixture tests evaluator/CI behavior; it is **not a representative benchmark**. Missing evidence is `not_computed`, and missing blocking metrics fail rather than silently pass. Optional RAGAS evaluates configured metrics and may involve paid calls; no score is assumed here.

| Evaluation layer | Concrete question | Example measurement |
| --- | --- | --- |
| Parsing and chunking | Did the necessary sentence and source offsets survive? | Human-checked parse coverage, empty chunks, chunk boundaries and source/ID integrity. |
| First-stage retrieval | Is a labelled supporting chunk in either top-k list? | Recall@k or Hit@k per vector, BM25 and union; include no-answer cases separately. |
| Fusion and reranking | Is relevant evidence ranked higher? | MRR/NDCG on the same queries; paired per-query wins and regressions. |
| Generation | Is the answer correct and complete? | Human judgement against the source; valid citation IDs and actual support are separate metrics. |
| NLI diagnostic | Are verdicts reliable in this domain? | Label claim–source pairs and inspect class confusion, especially negation, numbers and conflicting dates. |
| Reliability | Is behavior acceptable under failures and load? | p50/p95 stage and end-to-end latency, dependency failures, timeout, cost and degraded-result behavior at a stated concurrency. |

**Numerical drill:** Three answerable queries each have at least one annotated relevant chunk. If a relevant chunk appears in the first five for two of them, Hit@5 is `2/3`, not `2/5`. For Recall@5, each question's count of retrieved relevant chunks is divided by *all labelled relevant chunks for that question*, then averaged using the specified convention. The two metrics differ when a query has multiple relevant chunks. These numbers illustrate metric calculation, not this project's observed results.

**Ablation protocol:** freeze the corpus and document versions, model/configuration and question set; compare BM25 only, dense only, RRF, and RRF plus reranking. Label relevant chunk IDs and expected answerability independently; keep source-level separation where a document split matters. Report sample sizes, per-query differences, latency, errors, and results sliced by exact ID, paraphrase, long context, no answer, stale content and injection attempts. Calibrate any model judge against human review. Tune on a development split, then evaluate once on a separately frozen test set.

## Guardrails and production decisions

**What if the indexes disagree?** A successful Chroma write followed by an Elasticsearch failure can create drift; parity reporting surfaces it but does not roll back or repair it. Revision `c375903` serializes batches within one event loop, including parity checks; different event loops or processes can still race, and cancellation between writes can leave drift. For production, propose stable content/version IDs, an ingestion ledger, idempotent upserts, reconciliation and deletion propagation. Exercise a crash between writes and verify recovery; do not call this a distributed transaction.

**Where should access control apply?** The existing API key protects an endpoint, not each user's document permissions. A multi-tenant design must derive identity server-side and apply equivalent authorized-document filters in both search paths before fusion and model context creation. Test cross-user access and cache keys; post-generation filtering is too late to prevent context exposure.

**What if BM25 is unavailable?** The reviewed retrieval uses `asyncio.gather` and has no guaranteed single-retriever fallback. A failure can fail the query. A proposed fallback should mark the degraded mode, respect the same authorization rules, expose telemetry and be evaluated on the same questions; silently returning a weaker answer changes the product contract.

**How does source injection differ from a direct user instruction?** Retrieved text is untrusted content. The prompt tells the model to treat it as data, but instructions alone are not a complete boundary. Limit tool access, maintain source separation, test adversarial documents and ensure restricted data cannot enter retrieval. Citation-ID validation does not detect every injected claim.

**What would you change before production?** Define tenant identity, access, retention, encryption and restore behavior; measure per-stage latency and output quality; decide error and degraded-mode behavior; secure stores and secrets; add shared rate limiting where multiple replicas need it; then set thresholds based on reviewed task risk. The current local Compose setup and a green smoke test do not establish those controls.

**How would you debug a wrong answer?** Inspect whether source text was parsed, whether either index returned its ID, whether fusion/reranking retained it, whether the selected context reached generation, and whether each cited passage actually supports the claim. Compare scores and latency at each boundary. Correct retrieval with a wrong claim points to generation or claim checking; absent evidence points earlier in the pipeline.

## Oral rehearsal

Start with the two-pipeline diagram, calculate one RRF ranking, then explain why a valid citation and an NLI verdict provide different kinds of evidence. Give one concrete index-drift failure, one tenant-isolation control and one evaluation ablation. Use the [original rehearsal questions](interview-mega-guide.md#short-interview-answers-original-rehearsal-questions) for short spoken answers.
