# Production readiness and RAG evaluation matrix

Snapshot: 2026-09-09

## Verdict

The repository is **locally production-validatable, but not production-certified**.
Its deterministic code gates, real Chroma/Elasticsearch integration, CPU image,
and black-box HTTP workflow pass. The remaining blockers need representative
business data, a real deployment target, and owner decisions; passing a synthetic
fixture cannot substitute for those inputs.

## Evidence collected

| Boundary | Result | Evidence |
|---|---:|---|
| Lock, imports, format, lint | Pass | Locked `uv` graph, compile/import smoke, and Ruff gates |
| Unit suite | Pass | 70 tests on Python 3.11, 3.12, and 3.14; 75.55% line coverage against a 70% ratchet |
| Deterministic evaluator | Pass | 94 metrics; explicit `computed`/`not_computed`; dataset digest starts `71e453167c10`; repository-invariant gates pass |
| Real persistence/retrieval integration | Pass | Chroma + Elasticsearch ingest/replay/stale cleanup, vector/BM25/hybrid/parent retrieval, and parity exercised |
| Black-box HTTP E2E | Pass | Auth, ingest + repeat ingest, query, citations, SSE order, readiness failure/recovery, and four-query bounded-concurrency smoke |
| Dual-index parity after E2E | Pass | 134 Chroma IDs and 134 Elasticsearch IDs; no missing or mismatched records |
| CPU image | Pass with risk acceptance | `linux/amd64`, non-root UID 10001, Torch `2.13.0+cpu` with CUDA unavailable, exact default embedding/reranker/NLI models loaded offline; image ID `sha256:c3ee2bcbbd31fd207a48f0783f8f718506717aefcb2ab3bf5667bf4e058b859a`, size 2,396,502,297 bytes |
| Python security gates | Pass | `pip-audit`: 111 dependencies, 0 known vulnerabilities; Bandit: 0 findings; full-history Gitleaks: 0 findings |
| Container vulnerability gate | Conditional | Trivy: 51 HIGH and 3 CRITICAL Debian findings, all without an available fixed version in the scan. The three critical findings have exact, expiring risk acceptances in `security/trivyignore.yaml`; the unfiltered report remains required evidence |
| Package/legal metadata | Pass | Wheel and sdist build; clean wheel import/CLI smoke; MIT license file and pinned local-model inventory present |
| GitHub CI execution | Not run remotely | Workflow and update configuration exist locally; a hosted run requires committing/pushing or opening a pull request |

The final exact-image run used fresh service volumes. Its four concurrent warm
queries measured p50 826.17 ms and max 1,097.98 ms. These are smoke measurements,
not an approved latency SLO. Default-model cold start and ingestion include model
loading and must be measured on the eventual target.

That run also found a real readiness defect before it passed: Elasticsearch was
reachable while the required index had no active primary shard after this host
crossed the high-disk watermark. Startup and `/ready` now verify the configured
index reaches at least yellow health. The E2E-only Compose override disables disk
allocation thresholds for its tiny disposable index; production deployments must
retain disk protection and alert/capacity controls.

## Component-by-component status

| Pipeline component | Implemented and validated now | What is still missing or should improve |
|---|---|---|
| Parsing and ingestion | PDF/DOCX/text/Markdown parsing; bounded file count and bytes; confined paths; deterministic document/chunk IDs; idempotent upsert; stale-record cleanup; per-item bulk errors; dual-store parity report | A representative malformed/encrypted/OCR-heavy corpus; MIME/content sniffing; malware and PII policy; quarantine/dead-letter flow; delete propagation; freshness/index-lag SLO; backup/restore and reindex drills |
| Chunking | Fixed, semantic, and parent strategies; validated size/overlap; exact source spans; no empty chunks; cached semantic model; parent context survives storage | Compare strategies on real retrieval judgments; track chunks/document, boundary quality, redundancy, context utilization, language/document-type slices, and quality-versus-cost curves |
| Embeddings | Pinned BGE-small revision; normalized 384-dimensional vectors; dimension/model fingerprint stored; exact CPU inference smoke | Corpus/query drift, non-finite/norm checks, batch throughput, cold/warm latency, memory saturation, domain/language recall, and an explicit reindex/migration procedure when the model changes |
| Vector retrieval | Native similarity and vector rank retained; precision/recall/hit-rate/MRR/MAP/NDCG and duplicate rate supported at configurable K | Human relevance judgments at production K values; zero-result and filter-recall rates; approximate-index recall; tenant/ACL filtering; latency by corpus size; hard-negative coverage |
| BM25 retrieval | Native BM25 score/rank retained; the same judged retrieval metrics supported | Tune analyzer/language/stemming/synonyms on real queries; test exact identifiers and rare terms; filter/ACL correctness; index-refresh/freshness behavior |
| RRF fusion | Deterministic tie handling; constituent ranks/scores retained; NDCG delta versus the better individual retriever | Tune rank window and constant; measure unique contribution/overlap of each retriever, diversity, regressions by query type, and cases where fusion is worse |
| Cross-encoder reranking | Pinned BGE reranker; separate score/rank provenance; bounded candidate count; exact CPU smoke; NDCG delta metric | Tune candidate K/top K using quality-latency curves; measure recall lost before reranking, throughput/memory, score calibration, multilingual/domain behavior, and fallback behavior under model failure |
| Context assembly | Parent context and generated citations are confined to retrieved IDs; character budget is enforced | Dedicated token-aware budgeting; redundancy/context precision; evidence coverage; truncation/lost-in-the-middle tests; prompt-injection separation and document trust labels |
| Generation | Structured output; bounded untrusted context; citation normalization; transient-only retries; token counts; exact match/token-F1 plus arbitrary human/judge score channels | Representative answer-correctness, completeness, relevance, style, refusal, safety, and groundedness labels; provider schema/timeout/quota/failover tests; prompt/version tracking; cost budgets and cache policy |
| NLI faithfulness | Dynamic label mapping; declared-source scoping; independent entailment/contradiction maxima; claim support/neutral/not-supported aggregation | Human-calibrated precision/recall/F1 and confusion matrix, especially contradictions; threshold selection by risk tier; multilingual/domain slices; judge disagreement; retain raw scores per evaluation case |
| Offline evaluation | Strict versioned JSONL; corpus/config/model revisions; raw per-case observations; 94 deterministic metrics; baseline comparison; blocking/advisory gates with owners; missing metrics never silently pass; citation gates cannot pass without citation evidence; RAGAS failures are explicit | Replace synthetic smoke data with a reviewed golden set sampled from real traffic; automate execution/capture of those cases against a candidate stack; add confidence intervals and significance/regression policy; slice dashboards; adversarial/no-answer/freshness cases; capture distinct contradiction labels/scores; reviewer agreement and label lifecycle |
| API and orchestration | One application service shared by REST/SSE/CLI; `/live` and dependency/index-aware `/ready`; startup rejects an unavailable required index; off-event-loop blocking work; total timeout; concurrency and input bounds; explicit partial ingest | Distributed quotas/rate limiting; tenant identity and document ACL enforcement; cancellation propagation; overload/backpressure tests; stable API versioning; real provider and gateway failure drills |
| Observability | Parent/stage traces and duration/outcome metrics across the pipeline; raw queries/content omitted in favor of hashes/counts/revisions | Deploy a secured collector; define alertable SLIs/SLOs; sampling and retention policy; model/download/cache metrics; queue and resource saturation; correlate quality regressions with releases without storing sensitive content |
| Deployment and supply chain | Pinned base/service/action/tool digests; non-root CPU image; SBOMs; dependency/source/secret/image scans; weekly update configuration; disposable integration and E2E CI jobs | Run the workflow remotely; choose a hardened target/base or formally accept residual CVEs; sign artifacts and publish provenance; pre-stage model weights; enable store auth/TLS; network policy, secrets manager, autoscaling, canary/rollback, backup/restore, and disaster recovery |

## What mature live RAG systems measure

Mature evaluation separates retrieval from response generation so that a bad
answer can be attributed to the stage that caused it. Elastic's rank-evaluation
API uses representative search requests plus explicit relevance ratings, while
the OpenAI evaluation example reports retrieval hit rate/MRR separately from
response quality. Phoenix likewise evaluates retrieval/re-ranker relevance and
hallucination as different concerns.

The practical scorecard for this project should be:

| Layer | Offline release metrics | Online operational metrics |
|---|---|---|
| Data/ingestion | parse success, provenance/span coverage, store parity, freshness/delete tests, chunks/document | indexing lag, failed/partial writes, drift, stale-document rate |
| Retrieval/ranking | precision/recall/hit-rate/MRR/MAP/NDCG at production K; fusion and reranker delta; slice regressions | zero-result rate, clicked/accepted evidence where appropriate, latency, filter/ACL failures |
| Context | evidence coverage, redundancy, token-budget utilization, truncation and injection suites | context size, truncation rate, cache hit rate, retrieval-to-answer failure correlation |
| Answer/grounding | correctness, relevance, completeness, citation validity/coverage, claim support/not-supported, refusal and safety; contradiction only when captured separately | user feedback, escalation/correction rate, sampled human/judge audits, groundedness drift |
| Reliability/economics | E2E success, timeout/error taxonomy, cold/warm p50/p95/p99, concurrency and soak tests | SLO/error-budget burn, saturation/queue depth, provider failures, tokens and cost per successful answer |

Every score must include sample size and relevant slices. Automated NLI/LLM
judges should be calibrated against independently reviewed human labels rather
than treated as ground truth. Production failures and privacy-approved feedback
should feed new cases back into the versioned offline set.

## Prioritized remaining work

### P0 — required before a real production claim

1. Product/search owners supply a representative corpus snapshot, real query
   sample, graded relevant chunk IDs, expected answer/refusal labels, and an
   adjudication process. Move provisional gates to blocking only after baseline
   review.
2. SRE/product owners approve p95/p99 latency, availability/error, freshness,
   answer-quality, contradiction/refusal, and cost budgets on a named target.
3. Deploy authenticated/TLS-enabled stores behind network policy; add a secrets
   manager, tenant/ACL enforcement, retention/deletion controls, and backup +
   restore evidence.
4. Run capacity, soak, dependency-failure, restart, migration, and rollback tests
   with production-like corpus size and concurrency. The current four-query test
   is intentionally only a smoke test. Keep Elasticsearch disk allocation
   protection enabled and prove watermark alerts plus capacity runbooks on the
   actual target.
5. Decide how to eliminate or formally accept the current base-image CVEs before
   their 2026-10-09 exception expiry, and make model weights available without an
   uncontrolled first-request download.
6. Commit/push the workflow and require all CI jobs on protected branches; add
   artifact signing/provenance if releases are distributed.

### P1 — highest-value quality improvements

1. Run chunking, retriever, RRF, candidate-K, reranker, and context-budget
   ablations against the same frozen judgments; keep only statistically credible
   quality/latency gains.
2. Add human calibration reports for NLI and any LLM judge, including confusion
   matrices and disagreement by language, intent, answerability, and risk tier.
3. Add online quality sampling, feedback triage, drift/freshness alerts, and a
   privacy-reviewed path that promotes production failures into regression cases.
4. Add token-aware context packing, duplicate/redundancy controls, prompt-injection
   tests, and explicit trust/ACL metadata throughout retrieval and generation.
5. Define reindex compatibility, deletion, canary, rollback, and disaster-recovery
   runbooks before model/index/schema upgrades.

## Sources for the evaluation design

- [Elastic search rank evaluation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/search-rank-eval)
- [Elastic reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
- [OpenAI cookbook: evaluating RAG with LlamaIndex](https://developers.openai.com/cookbook/examples/evaluation/evaluate_rag_with_llamaindex/)
- [Phoenix retrieval relevance evaluation](https://arize.com/docs/phoenix/evaluation/running-pre-tested-evals/retrieval-rag-relevance/)
- [Phoenix hallucination evaluation](https://arize.com/docs/phoenix/evaluation/running-pre-tested-evals/hallucinations/)
- [Evidently RAG evaluation guide](https://www.evidentlyai.com/llm-guide/rag-evaluation)
- [OpenTelemetry semantic conventions](https://opentelemetry.io/docs/specs/otel/semantic-conventions/)
- [GitHub Actions secure use reference](https://docs.github.com/en/actions/reference/security/secure-use)
