# XAI-RAG: hardening and production interview questions

This is the companion to the [master guide](interview-mega-guide.md) for the hardening work after published revision `c375903`. Read it first when asked about the newer implementation. The safety-and-coverage judge was integrated into the development checkout on 20 November 2026; check the source commit and final test report before presenting it as a published release. The adaptive experiments below identify their own exact source snapshots. Historical experiments in the other documents retain their original versions and must not be attributed to this newer code.

## One-minute answer

> “XAI-RAG answers questions over documents by combining Chroma vector retrieval and Elasticsearch BM25, fusing the rankings and reranking a bounded candidate set. It gives the model a limited set of source passages and returns claims with citations. In the hardening work, we added an opt-in server-owned document policy before retrieval, a bounded parser, a configurable recovery journal and checks before answer release. A local NLI model checks support and a separate LLM judge screens claim support, relevance and safety plus coverage of the full question; if checking fails, the service withholds the answer. These are risk controls, not proof of truth. We tested failure cases and measured latency on synthetic workloads; human evaluation, broader security testing and production capacity validation are still necessary.”

Explain your own contribution accurately; a codebase feature is not proof you personally built it. The system is a fixed Python workflow, not a LangGraph agent, MCP client, planner, or multi-agent system.

## Follow one request

```text
Operator provisions trusted root + (optional) principal/source policy
    → open allowed source without following path symlinks
    → stage bounded private bytes → resource-limited parser → chunks
    → optional durable intent/snapshot → Chroma write → Elasticsearch write
    → compare records / complete intent

Caller authenticates → principal + permitted source IDs
    → Chroma dense search AND Elasticsearch BM25, each filtered before top-k
    → reciprocal rank fusion → cross-encoder rerank → bounded visible evidence
    → structured candidate claims + source IDs → NLI support check
    → separate structured LLM judge over question and visible evidence
       checks support, relevance, safety and whether all question parts are addressed
    → only accepted complete claims rendered with citations, or a fixed refusal/status
```

**What:** Both stores contribute retrieval; the second model judges the candidate rather than retrieving documents. **Why:** Keeping a restricted chunk out before ranking and model context is stronger than removing it from the final answer. Valid citation IDs alone do not establish support. **How:** For example, an employee allowed `policy/refunds.txt` asks, “How long do I have to return an unused item?” If the visible passage says 30 days, the generator can propose that claim with its source ID. The local NLI must support it, and the judge must accept its support, relevance and safety and say the question is fully addressed before the answer is released. If a source says 30 days but the claim says 60, the intended outcome is refusal; model-based checks can still make mistakes. This example is illustrative, not a recorded result.

## Grounding and prompt safety

### 1. What exactly changed at answer release, and why?

**What:** The newer path buffers generated text and renders accepted structured claims, rather than immediately showing the generator's free-form prose. Each claim must cite prompt-visible evidence, pass local NLI, and pass a separately configured LLM judge checking support, relevance and safety. The judge must also find that all substantive parts of the question are addressed. **Why:** Earlier versions could stream an answer before the optional NLI report; a valid ID only proves the evidence was available, not that it supports every statement. **How:** Unsupported, irrelevant, unsafe, incomplete or mixed claims are withheld together; missing judge configuration, timeout, malformed output or provider failure produces `verification_unavailable`, and insufficient support produces `insufficient_evidence`. `verifier_passed` is a check result, not a factual guarantee.

### 2. Does streaming now mean the answer appears token by token?

**What:** No: retrieval/stage events may arrive, but answer text is held until the release checks finish. **Why:** Streaming unchecked text and later retracting it would already expose it. **How:** A client must handle stage events, a delayed generation event, and SSE error events; HTTP 200 alone does not prove the stream completed. This design trades time-to-first-answer for the opportunity to withhold unsupported output.

### 3. Why two checks? What did they miss?

**What:** NLI compares a claim with its evidence; the LLM judge separately assesses support, relevance and safety with visible evidence and checks whether the whole question is addressed. **Why:** They have different failure patterns, so an AND gate can block some errors either one alone misses. **How:** In one 50-case fictional, author-labelled probe, `gpt-6-luna` had 0 binary false accepts and 0 false rejects; on the original 42 cases, NLI had one false accept and two false rejects, so the AND gate had zero false accepts and two false rejects. One NLI false accept involved an unjustified universal claim despite a 0.9043 entailment score. The LLM disagreed on the type of two rejections; labels need independent adjudication. The judge was an alias without a documented dated snapshot and can drift. This small set is neither human review nor an accuracy guarantee. Local SDK-mock tests checked malformed, truncated, empty and failed responses; those are not live provider outages. [Probe](evaluation-hardening-evidence.md).

### Follow-up: Can NLI recognize whether a document instruction is safe?

No. NLI estimates whether a claim is supported by a premise; it is not a safety, relevance or authorization classifier. In a small authored diagnostic it supported three instruction-like claims, which can be textually entailed while unsafe or irrelevant to release. A separate paired prompt experiment also found release errors; the earlier question-aware candidate was rejected. A later adaptive 25-case safety/coverage probe matched its authored labels, but is not a blind or human-reviewed evaluation. The revised safety-and-coverage judge is now integrated in the development checkout. A 10-case REST/SSE candidate run refused a partially answerable case and a same-scope conflict; this avoids releasing that incomplete answer but leaves useful partial answers unresolved. It did not force the earlier suspect claim into generation. [Results and limits](evaluation-hardening-evidence.md#bilingual-nli-pair-diagnostic-and-adaptive-judge-studies).

### Follow-up: Can an exact quotation still be rejected?

Yes. In a local stub-provider E2E check, the candidate claim matched the cited Markdown text verbatim, yet NLI rated the original chunk 0.0443 entailment and 0.8765 contradiction; the API withheld the answer. On the same claim, a sentence-only premise scored 0.9932 entailment. That single example suggests sensitivity to context or formatting but does not prove its cause or justify bypassing a guardrail. Record the false refusal and evaluate fixes across independently labelled data. [Captured evidence](evaluation/hardening-2026-11-20/nli-literal-e2e/README.md).

### 4. Why not call this a verified or truthful answer?

Both checks can be wrong, and evidence can be incorrect, stale, poisoned or missing. Claims can omit important qualifications, and an incomplete retrieval set can miss contradictions. The NLI and judge make probabilistic decisions; a source-support decision is not external fact checking. Evaluate false acceptance, false refusal, completeness and current-source correctness separately against independently labelled data, and define escalation for high-impact use.

### 5. What do the confidence and token fields mean?

`overall_confidence` can now be `null` when an answer was not released; clients previously expecting a float must be updated. `overall_confidence_source` identifies whether an available value came from NLI entailment or RAGAS faithfulness. Neither is a calibrated probability that the answer is true. The API's generator `input_tokens` and `output_tokens` do not include the extra mandatory grounding-judge call; calculate full cost from a provider-wide usage ledger. Verify this against the deployed version because response contracts can change.

### 6. How do you handle instructions inside a retrieved document?

**What:** Retrieved text is untrusted data, not application instructions. **Why:** A malicious page can tell the model to reveal secrets, ignore the task or invent an answer. **How:** The generator receives evidence in a lower-trust user message, and claim checks operate on bounded visible evidence. Test benign quotations as well as attacks, including multilingual, encoded and conflicting-source cases. No delimiter, prompt, NLI model or second LLM makes prompt injection impossible. Keep stores private, minimize data sent to providers, enforce authorization outside the model and inspect attack and false-refusal rates independently.

### 7. What is the cost of the second judge?

It adds another sequential billable provider request, latency and disclosure of the bounded evidence to that provider. Set `XAI_RAG_GROUNDING_JUDGE_MODEL` explicitly; if absent, the service fails closed instead of silently falling back to unchecked answers. Review provider terms and data handling before using sensitive documents. A fictional 51-call probe reported a 1.203-second median request latency and estimated $0.00204 in token charges under its stated pricing and concurrency; that is not full application latency or a cost forecast.

## Identity and ingestion

### 8. How are tenants and document permissions enforced?

**What:** Optional server-controlled policy maps each API credential digest to a tenant, role and exact relative source paths. **Why:** A shared API key alone authenticates access to an application, not a user's right to every document. **How:** The server derives scope from the credential; each tenant gets separate indexes and both Chroma and Elasticsearch receive source filters before top-k. Unexpected sources are rejected before fusion and generation. A reader cannot ingest; an ingester can work only with granted sources. Missing or invalid configured policy fails closed. Direct database access must be restricted because it bypasses application enforcement.

**Important default:** Without an explicit policy, the legacy single-key mode is still available and does **not** provide tenant or document isolation. Policy changes require application replicas to restart; this is not a distributed identity provider or instant revocation service. Existing shared records are not automatically migrated: approve and re-ingest sources into the tenant namespace.

### 9. Why filter before top-k in both stores?

Filtering after retrieval can expose document text to logs, rerankers or the provider and can also waste the candidate budget on forbidden records. Pushing equivalent source filters to both stores prevents restricted records from becoming candidates on the supported path. Test same-name documents across tenants, empty grants, lexical and dense paths, parent context, errors and streaming; ensure caches and future fallback paths apply the same scope. Results depend on the underlying stores enforcing their native filters correctly.

### 10. Can an ingester use a different directory containing an allowed filename?

No. The scoped CLI is bound to the configured ingest root and rejects a different `--document-root` or outside path; a matching basename alone does not grant access. Lexical permission checks occur before discovering unauthorized file metadata. Directory requests select paths explicitly listed in the principal's grants rather than enumerating everything. Secure opening then avoids following symlinks, while scoped parity checks avoid disclosing unrelated sources' index health.

### 11. Is the parser a security sandbox?

No. On POSIX the code opens path components without following symlinks, stages bounded bytes into a private temporary file and parses in a subprocess with resource limits. These controls reduce path-race and resource-exhaustion exposure. They do not prevent a malicious same-UID process from changing an open inode or exploiting hardlinks; the root, ancestors and source inodes must be controlled by trusted operators. The subprocess is not a filesystem or network sandbox. For hostile uploads, use separate OS identities and a hardened sandbox with controlled filesystem, network, CPU and memory.

### 12. What protects the HTTP and ingestion boundaries?

A request body cap applies before JSON parsing and authentication, including chunked requests, and returns 413 for oversized input. It does not replace gateway limits on headers, slow clients, connection counts or proxy buffering. Ingestion is fail-fast when the local batch slot is occupied; cancellation keeps the lock while underlying writes finish. The slot is local to an event loop, is shared across tenants and provides no fairness, cross-process coordination or whole-batch deadline. Distributed admission and quotas remain deployment work.

## Two stores and recovery

### 13. Chroma succeeded but Elasticsearch failed. Is the write atomic?

No. Two stores do not share a transaction here. With the **optional** host-local journal configured, the service saves exact staged source bytes and an intent before writes. A pending intent blocks new journaled writes for the same namespace until recovery. An operator stops earlier writers and establishes that already-sent remote requests have completed, then invokes `xai-rag recover-ingestion --acknowledge-quiescence` using the correct scope. Replay rechecks permission, source bytes and expected records, rewrites both stores and verifies actual stored text and metadata against the intended manifest before clearing intent.

### 14. What was actually tested, and what is still not guaranteed?

A synthetic test against disposable real Chroma and Elasticsearch forced process exit after the Chroma write, changed the original input file, then recovered the saved original bytes in a fresh process. It verified one intended record in each store. Earlier attempts are retained: one failed on a test-branch interface mismatch and one on an Elasticsearch timeout. The successful attempt used a 60-second test-client timeout and a disk-watermark override on **that disposable Elasticsearch node only**. After the journal fsync fix, the scenario passed again on the then-final development source with a pre-provisioned private journal; file hashes were unchanged during the run. Both passing runs used the same disposable Elasticsearch exceptions above. Neither simulates power loss or proves delayed remote writes are fenced. [Evidence](evaluation-hardening-evidence.md#crash-and-replay-real-disposable-stores).

The journal needs a private, persistent, provisioned directory shared by cooperating writers using the same canonical endpoints. It is not a distributed lock or transaction; it cannot fence a late store request, prevent direct bypass writers, ensure atomic reads, verify embedding vector values, detect a missing marker, or coordinate hosts. Pending snapshots contain document content and require retention and backup controls. Empty journal configuration disables recovery. See the source repository's `docs/ingestion-recovery.md` before operating it.

### 15. Is parity the same as recovering the intended version?

No. If both stores contain the same old version, parity can pass while both are wrong. Current parity checks actual text and retrieval metadata, not just a stored hash label; intended-state recovery additionally compares each store against the saved manifest. Neither proves vector correctness, source truth or freshness. Test incomplete Elasticsearch responses, source deletion and concurrent updates as separate cases.

## Evaluation and operations

### 16. What should an evaluation measure separately?

- **Retrieval recall@k:** fraction of labelled relevant evidence retrieved in the top *k*; hit@k asks whether at least one relevant item was retrieved.
- **Faithfulness:** whether answer claims are supported by supplied context. A faithful answer may repeat an incorrect source.
- **Correctness:** whether an answer agrees with a trusted reference, including important qualifications; a correct guess can still lack retrieved support.
- **Answer relevancy:** whether the answer addresses the question. It can be relevant and wrong.
- **Citation validity and citation support:** whether a cited ID was actually visible to the model, and whether the source supports the associated claim. These are different checks.
- **Operational and safety outcomes:** retrieval failures, unsupported release, unsafe compliance, false refusal, timeout, latency and cost, with the total attempted cases retained in denominators.

In a current-development HTTP run on 40 fictional, author-labelled questions, combined RAGAS means were 0.875 faithfulness, 0.6355 answer relevancy and 0.9597 reference context precision. The initial cap left 25 cases incomplete, and a separate extension scored them; five of 27 directly answerable cases were refused with relevant evidence visible. The scores are not human evaluation or production estimates; the generator and judge shared a model family. [Exact evidence and failures](evaluation-hardening-evidence.md#current-application-http-and-ragas-evaluations). The project uses RAGAS for selected metrics and can use LLM judges; neither is a substitute for a defined rubric and independent human labels. Distinguish scores produced on synthetic data from human-reviewed evaluation and record unavailable metrics rather than inventing a zero or dropping failures. The evaluator now distinguishes prompt-visible evidence from merely reranked candidates and rejects non-finite scores.

### 17. What did the load test establish?

On one shared host, published baseline `c375903` used 100 synthetic documents, real Chroma and Elasticsearch, real cached embedding/reranking models and a stubbed generator. All 120 measured requests returned HTTP 200, 30 per concurrency level. At concurrency 1/2/4/8, p50 was 6.829/11.222/25.295/55.066 seconds and p95 was 19.176/18.643/44.622/70.173 seconds. At level 8, requests queued behind an application cap of four concurrent queries. The p95 estimate uses only 30 samples per level; shared-host interference and the stubbed generator limit interpretation. These are not the new two-gate system's latencies, a production capacity estimate, a quality score or an SLO. [Full conditions and evidence](evaluation-hardening-evidence.md).

### Follow-up: What did the transport investigation establish?

A loopback **stub** with both the default httpx pool expiry and Uvicorn keepalive set to five seconds showed 8/49 readiness and 3/9 synthetic query transport errors with reused connections, versus none in an otherwise parallel `Connection: close` arm. A separate later stub run with 15-second server keepalive saw no errors in either arm, but these short runs were sequential. It is a timing hypothesis, not proof of the earlier app soak's cause or an SLO. The Docker CMD now sets Uvicorn `--timeout-keep-alive 15`; that may retain more idle sockets and file descriptors, and does not change separate proxy or other-runtime timeouts. The Docker healthcheck uses a fresh urllib request, not that shared pool. Measure actual deployed clients and network settings before drawing a deployment conclusion. [Conditions and events](evaluation/hardening-2026-11-20/transport-boundary/README.md).

### 18. How would you turn this into a production service?

First define the actual workload, risk, access policy, residency and latency targets. Use identity and secrets management; isolate and authenticate both databases, encrypt network traffic, and audit access. Replace the static-restart policy if instant revocation is required. Choose a single-host journal only where its boundaries are acceptable; for multi-host writes, design durable distributed coordination with backend fencing and explicitly versioned index cutovers. Add private uploads and a real parser sandbox when inputs are hostile. Establish independent labelled evaluation sets, attack tests, representative traffic and monitoring before scaling. Test backup/restore, provider outages, rollback, late writes and model drift. The current local controls and single synthetic load run do not establish production readiness.

### 19. How would you explain the database decision?

Chroma stores vectors for dense similarity; Elasticsearch is a search engine used here for BM25 lexical retrieval. BM25 is a ranking algorithm, not another name for Elasticsearch. Hybrid retrieval can find both paraphrases and exact identifiers, and reciprocal rank fusion combines ranks without pretending unlike raw scores share a scale. Two stores cost more to operate and introduce consistency failures. Compare a single engine offering both retrieval methods against this design on the same representative corpus, access-control behavior, retrieval quality, recovery, p95 latency and cost before choosing or migrating.

## Reproduce and keep claims honest

Follow [local reproduction](local-reproduction.md) using fictional documents and isolated resources. Explicitly configure the judge model, inspect `grounding_status`, and keep failed cases in results. Use the source repository's `docs/tenant-authorization.md`, `docs/request-body-limits.md` and `docs/ingestion-recovery.md` for operational configuration. Save source commit and dirty status, model identifiers, dataset hashes, complete logs, fixture labels, denominators and run settings. Provider calls are billable; do not put real confidential text into a provider without appropriate authorization. The previous material and historical experiments remain useful for design foundations, but statements about optional NLI and absent authorization refer to `c375903`, not this hardening work.
