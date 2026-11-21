# XAI-RAG: production system design interview questions and answers

> **Version note:** This document describes the earlier `c375903` or explicitly cited historical source. Statements here that call NLI optional, say document authorization is absent, or describe recovery as only a proposal do not describe the later hardening work. Read the [current hardening questions and answers](hardening-interview-qa.md) first, and verify the source revision before an interview. Historical measurements remain tied to their stated snapshots.

Read this alongside the [project overview](README.md). The linked [commit `f4c374d`](https://github.com/neeraj1909/xai-rag/tree/f4c374d51d090beba61d34aba81fcc23b8ef700a) is the historical baseline. The newer controls are included in [published source `c375903`](https://github.com/neeraj1909/xai-rag/tree/c375903fb46327744d076438cb90f7b02711a94f); links pinned to `f4c374d` show the historical baseline and do not establish those changes. “I would” identifies a proposal, not a deployed feature. The repository's [production-readiness report](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/docs/production-readiness.md) records a historical local test run, not an independently verified deployment or a current benchmark.

## Start with a real production scenario

An employee asks: “What receipt do I need for a reimbursement?” Their company has two versions of an expense policy, and another company's private policy is also indexed. The system must find evidence the employee can access, answer from the current policy, cite the right passage, and handle a failed search dependency without leaking private data or silently inventing an answer. The following questions explain how the inspected project handles parts of this flow and what must change for that scenario.

## Deployment and architecture

### Q1. How would you describe the system you would put into production?

**Answer:** I would start with the existing modular service: an API accepts a question; a shared application service runs dense Chroma and lexical Elasticsearch retrieval, fuses and reranks candidates, and sends bounded evidence to the generator. The API returns structured answers and citations, with optional explanation and NLI diagnostics. A separate ingestion path parses and indexes source files. For production, the next boundary is identity and policy before retrieval, followed by reliable indexing, operations, evaluation and recovery. I would not split every stage into its own network service without measured scaling or ownership needs; each extra hop creates another timeout, deployment and failure boundary. [Service](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/service.py), [API](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/api/app.py).

### Q2. What would you change first, and in what order?

**Answer:** First define the product and data boundary: permitted users, documents, allowed failure modes, retention, and whether an unsupported answer can cause harm. Then:

1. Establish authentication, server-side document authorization in **both** retrievers, and secure network/store connections. Add tests proving another tenant's chunks cannot enter prompts or citations.
2. Make ingest state recoverable across the two stores: source-of-truth versions, a durable job ledger, idempotent retry, drift reconciliation and deletion propagation.
3. Create a reviewed, versioned golden set, measure retrieval/answer/citation/refusal quality and agree release criteria with the owner. Include stale, restricted and unanswerable cases.
4. Instrument and run target-environment load, timeout, failure, restore and rollout drills. Tune timeouts and capacity only from those observations.

This order is a proposed starting point, not proof the existing service already has these controls; a high-risk domain may require stricter human review before any release.

### Q3. Could I deploy the existing Docker Compose setup unchanged?

**Answer:** I would use it for local development and reproduction, then change the deployment boundary. The checked Compose file binds ports to loopback and explicitly describes Chroma as local-only; Elasticsearch disables security for the local stack. Production needs private network access, service authentication and TLS, a secrets manager, persistent storage with tested backups, controlled model artifacts and capacity monitoring. The checked Compose and API settings are the evidence; passing an endpoint smoke test does not prove multi-tenant security or disaster recovery. [Compose](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/docker-compose.yml), [readiness report](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/docs/production-readiness.md).

### Q4. Do we have to replace ChromaDB and Elasticsearch to make the system production-ready?

**Answer:** Not automatically. I would evaluate whether a secured, supported deployment of each store meets our corpus size, throughput, tenant isolation, filters, recovery, upgrade and operational requirements. The immediate issue in the repository is that its Compose configuration is local-only and the whole system lacks document-level authorization and cross-store recovery. Test the actual workload and failure modes before deciding to keep, operate differently, or migrate a store. Keeping two stores still incurs consistency and operational costs; replacing them does not itself make citations, access control or recovery correct.

### Q5. What would the API contract expose?

**Answer:** The current API provides `/query`, `/query/stream`, `/ingest`, `/live` and `/ready`; protected endpoints use an `X-API-Key`. For production, I would define a versioned request with a bounded question and retrieval options, and a response with answer, source IDs, document versions, request ID and an explicit success, refusal, degraded or error status. I would derive user and tenant identity from trusted authentication, not from a model or an unchecked request field. Client-visible errors should be useful without exposing stack traces or secrets. Preserve compatibility during upgrades and test the HTTP and streaming contracts. [API code](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/api/app.py).

### Q6. What is the difference between liveness and readiness?

**Answer:** `/live` answers whether the process is alive. `/ready` checks the configured Chroma heartbeat and Elasticsearch connection plus index health; the API returns 503 if a required dependency is not ready. Readiness can stop a load balancer sending traffic to an unready instance, while liveness should not restart healthy processes simply because a downstream service is briefly unavailable. The report describes an Elasticsearch primary-shard/disk-watermark failure found in an earlier local test; it should not be interpreted as evidence that the target deployment has exercised that failure. [Health implementation](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/api/app.py).

## Identity, data and ingestion

### Q7. The API already has an API key. Why is that insufficient for several customers?

**Answer:** The implementation checks one configured key with constant-time comparison and applies a per-process rate bucket. That authenticates access to the service, but it does not identify an individual user or decide which documents that user may read. I would authenticate callers through a trusted gateway or service, map their principal to tenant and document permissions, and push equivalent authorized filters into both Chroma and Elasticsearch before fusion. Test direct and indirect leakage through snippets, citations, logs and cached responses; never trust an ID proposed by the model. [API authentication](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/api/app.py).

### Q8. What exactly happens when an administrator re-ingests a document today?

**Answer:** The ingestion service checks the configured path, count and size bounds, parses and chunks each document, embeds the chunks, writes to Chroma, then writes the same IDs and text to Elasticsearch. The store computes deterministic IDs from source/chunk provenance and content and uses upsert; for an updated source it removes stale IDs in each respective store. The service reports per-document outcomes and checks parity after processing. Revision `c375903` serializes complete ingestion batches within one event loop, including discovery, both writes and parity checks. Replaying the same document can converge in the tested path, but this lock is not shared across loops or processes; it does not make separate store writes atomic or protect against a crash or cancellation between them. [Ingestion service](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/service.py), [store](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/ingestion/store.py).

### Q9. What happens if the process crashes after the Chroma write and before the Elasticsearch write?

**Answer:** The indexes may contain different versions or IDs. The existing parity check can reveal drift if it runs; it cannot roll back both systems as one transaction or guarantee it runs after a process crash. I would persist an ingestion job with source version, target index generation and per-store progress, make each step retryable, and reconcile the authoritative source against both stores. A visible document should be promoted only when the intended version is ready in both paths. Inject a process failure at every boundary and verify eventual consistency, status and user-facing behavior.

### Q10. How would you prevent an old document from replacing a newer version during concurrent ingestion?

**Answer:** The `c375903` ingestion lock serializes batches in one event loop, but stable IDs, upserts and a process-local lock do not order writers across workers or machines: a delayed older job could still perform stale cleanup after a newer job starts. I would give each document a monotonically ordered version and use a per-document lease or optimistic compare-and-swap in a durable metadata store. Write into a versioned generation, verify both indexes, then atomically switch the active version in that metadata store. The stores still have independent writes; the active-version pointer controls which generation queries should see. Test an intentionally delayed older worker, duplicate delivery, lease expiry and delete/update races. This is a proposal, not an existing guarantee.

### Q11. How would you support document deletion?

**Answer:** The store contains per-source deletion helpers, but a production deletion contract needs more than a helper in one store. I would define who may request deletion, write a durable tombstone or deletion job, exclude the document from both retrieval paths immediately according to the agreed policy, delete every indexed version and invalidate affected caches, then track and retry incomplete deletion. Include logs, backups and retention rules in the policy and verify restoration does not silently reintroduce deleted content. A shared transaction across these systems should not be assumed. [Store helpers](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/ingestion/store.py).

## Reliability, latency and cost

### Q12. What happens if Elasticsearch fails while Chroma is healthy?

**Answer:** In the inspected query path the dense and lexical work is combined using `asyncio.gather`; there is no documented single-store fallback, so a branch failure can fail the request. I would first choose an explicit contract: return an unavailable error, or allow a labelled dense-only degraded mode. If the product permits fallback, it must apply the same authorization policy, record which branch failed, set a bounded recovery policy and be evaluated on the same labelled questions. A degraded response is not equivalent to a hybrid result and should never be silently presented as one. [Query orchestration](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/service.py).

### Q13. What happens when many users ask questions at once?

**Answer:** Each process currently uses a semaphore and a total queue-plus-execution deadline for queries, and a per-process rate limit. Revision `c375903` also protects rate-bucket updates with a thread lock. These controls protect one process, but four replicas would have four separate semaphores and buckets; they do not impose one organization-wide budget. I would decide per-user/tenant quotas at a shared gateway, bound queue depth, reject overload predictably, and size model workers and store connections from concurrency tests. Measure p95/p99 and errors under load, not just average latency. [Limits](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/api/app.py), [defaults](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/config.py).

### Q14. What happens if the client disconnects during a streaming answer?

**Answer:** The SSE generator has a `finally` block that cancels and awaits its pipeline task when the stream ends. Cancellation of a Python task does not automatically guarantee that every external request or thread-backed operation has stopped, nor does it undo a response already sent. I would exercise real disconnects at retrieval and generation, verify provider and worker resource cleanup, and count orphaned work and costs. Define whether partial output is allowed and make the terminal status visible to clients. [Streaming implementation](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/api/app.py).

### Q15. Where would you set timeouts and retries?

**Answer:** The query has a total timeout including semaphore wait, and the configuration separately provides an LLM timeout. In `c375903` a timed-out request can return promptly, but native work already running in a thread cannot be killed; admission remains occupied until that work drains. A stuck native call needs process-level supervision. Generator contract repair is limited to one semantic retry and three total provider attempts; SDK retries are disabled for the actual AsyncOpenAI client. I would allocate a budget among queueing, retrieval, reranking and generation, leave space to return an error, and instrument each stage. Retry only failures considered transient, with capped attempts and backoff inside the remaining deadline; retries can multiply cost and duplicate writes if the effect is not idempotent. Dependency-specific timeouts, retry budgets and client-visible semantics should be measured and tested rather than copied as universal values. [Configuration](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/config.py).

### Q16. Where would you cache, and how would you avoid returning another user's answer?

**Answer:** The code already caches the loaded embedding model in process; that is different from a shared query or answer cache. If measurements justify a result cache, I would bind entries to authorized principal or permission scope, corpus and active index generation, embedding/reranker/generator and prompt versions, retrieval settings and normalized query. Authorize every lookup, define an expiry, and invalidate on document, permission and model changes. For sensitive or rapidly changing material I might avoid answer caching altogether. Cache-hit rate is not useful if stale or unauthorized answers are served. [Embedding model cache](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/ingestion/embedder.py).

### Q17. How would you reduce cost without hiding a quality regression?

**Answer:** Break cost down by model, request, tokens, indexed documents, reindex frequency and compute. Reuse loaded models, measure whether a smaller candidate pool or fewer reranker pairs reduces latency, and compare against the same reviewed queries and relevant-document labels. Reducing context or reranking may miss the supporting receipt instruction even if the request is faster. Set a cost-per-successful-answer budget, track empty and incorrect answers separately, and keep a rollback to the previous configuration. No savings or performance gain is claimed without a measured experiment.

### Q18. What are SLI, SLO and error budget for this project?

**Answer:** An SLI is the measurement, such as the proportion of authorized queries that return a terminal response within a chosen time or the proportion of newly approved documents searchable within a defined freshness window. An SLO is an owner-approved target over a defined period; the error budget is the allowed fraction that misses it. I would track latency, error/timeout, retrieval freshness and an independently reviewed answer-support indicator, each with eligibility rules and denominators. The repository records one four-query local smoke result; it is not an SLO or a capacity estimate. Set targets only after product and operations owners assess business risk.

### Q19. What should telemetry contain without exposing company policy text?

**Answer:** Record a request ID, model/index/prompt version, selected pipeline mode, stage duration, outcome/error class, counts, token usage and policy-safe resource identifiers. In historical revision `c375903`, tracing recorded hashed query and source attributes rather than raw content. In the later hardening implementation, query-derived hashes were removed; an ingestion span can still include a SHA-256 hash of the source path. Candidate file paths can be hashed and compared against that attribute, so source hashes are not anonymization and trace access must be controlled. Automatic span exception capture is disabled and selected parser/store log paths sanitize source names and exception messages, but that does not prove every library or proxy log is safe. Set sampling, retention, encryption and access policy; monitor errors, p95/p99, index drift, ingestion failures, timeouts, model latency and cost. Preserve a privacy-approved way to investigate a wrong answer without logging unrestricted documents. [Tracing](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/observability/tracing.py).

## Releases and interview follow-ups

### Q20. How would you deploy an embedding-model change safely?

**Answer:** First pin the new model revision and dimension, compare relevance and latency on a frozen labelled set, and build a new index generation rather than mixing incompatible vectors. The store already computes and checks a model/index fingerprint; that is a compatibility check, not a complete migration system. Run reindexing from the authoritative documents, verify dual-store IDs, ACLs, sample retrieval and deletion behavior, then shift a small authorized traffic slice and compare quality and errors. Retain the old generation and configuration for rollback; decide how to handle source updates while both generations exist. [Fingerprint implementation](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/ingestion/store.py).

### Q21. How would you evaluate a change before a production rollout?

**Answer:** Freeze corpus versions and a held-out, reviewed query set with relevant chunk IDs, answerability, expected answers and permission labels. Compare retrieval recall and ranking, citation validity **and support**, answer correctness, refusal, injection and cross-tenant leakage, alongside cost, errors and latency. Check slices such as exact IDs, paraphrases, stale documents and no-answer cases; use per-query comparisons to see regressions. The deterministic fixture tests evaluator behavior; a separate real-model run evaluated eight fictional cases, and a later then-uncommitted snapshot generated seven of eight. Neither establishes real-corpus quality. See [live evaluation](live-evaluation.md) for denominators and judge limitations. If criteria fail or evidence is missing, hold the release and investigate the stage that failed. [Evaluator](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/evals/README.md).

### Q22. Does a cited answer become safe because NLI says it is supported?

**Answer:** No. In `c375903` a valid citation ID points to a chunk actually included in the bounded prompt, not merely to any reranked item; NLI is a diagnostic estimate of the relationship between a claim and cited text. It can miss numerical errors, negation, stale policy or adversarial content. In the existing streaming flow, generation is emitted before optional faithfulness diagnostics, and the NLI result is not a blocking gate. A proposed pre-release check or human review would need risk-based thresholds, labelled calibration data and a clear rule for neutral/uncertain outcomes. [Generation and diagnostic flow](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/service.py).

### Q23. How would you roll back a bad release?

**Answer:** Treat application, prompt/model, schema, index generation and configuration as related versions. Keep a known-good release and compatible index generation; canary a small traffic slice, compare quality and operations against pre-agreed gates, then stop or roll back on breach. For an authorization incident, disable affected access rather than waiting for a quality metric. Test rollback with updates arriving during the migration, and record which version served each request. A previous container image is not a complete rollback if it cannot read the current index or permission format.

### Q24. An interviewer asks, “What is the hardest production trade-off here?” What would you say?

**Answer:** I would choose consistency and authorization across hybrid retrieval. Chroma and Elasticsearch can find complementary evidence, but every source update, deletion and access filter must behave consistently across both. A parity check can surface drift, while durable versioning and reconciliation handle recovery. I would demonstrate the trade-off with the employee-policy scenario: an old or unauthorized chunk must not reach the generator even if it ranks first. Then I would explain the test that detects it, the operational signal and the decision when a store is unavailable. This is more informative than naming a vector database without describing its failure boundaries.

## Practice path

Explain Q1 and Q2 in two minutes. Then rehearse the policy scenario through Q7–Q12, including a crash between writes. Finish with Q18–Q23: define the metrics, release gate and rollback. State what the checked project already does and what you would add; do not describe a proposed control as a deployed feature.
