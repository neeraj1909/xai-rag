# XAI-RAG: quick interview recall

> **Version note:** This document describes the earlier `c375903` or explicitly cited historical source. Statements here that call NLI optional, say document authorization is absent, or describe recovery as only a proposal do not describe the later hardening work. Read the [current hardening questions and answers](hardening-interview-qa.md) first, and verify the source revision before an interview. Historical measurements remain tied to their stated snapshots.

## Hardening recall: current development version

- **Pipeline:** authorized scope → Chroma dense + Elasticsearch BM25 → RRF → cross-encoder → bounded context → claims → NLI **and** separate LLM check for support, relevance, safety and full question coverage → cited answer or refusal.
- **Tenant isolation:** optional server-owned source grants filter both retrievers before top-k. Legacy shared-key mode is unscoped; direct store access bypasses application checks.
- **Grounding:** a valid citation identifies visible evidence, not proven support. Both checks are fallible. The application buffers answer text until checks finish; incomplete answers are withheld together, trading partial-answer utility for caution; an unavailable judge withholds it.
- **Costs:** the judge adds a billable provider call and sees the bounded evidence. API generator token counters omit that call; use a provider-wide ledger for total cost.
- **Ingestion:** a POSIX no-symlink open and resource-limited parser reduce risks, but require a trusted ingest tree and do not form an OS sandbox.
- **Recovery:** an opt-in persistent host-local journal saves source bytes and intent; after writers and remote requests stop, an operator can replay and verify actual stored records. It is not a distributed transaction or fencing mechanism.
- **Evidence:** one synthetic real-store crash/replay passed; on a separate older-baseline 120-call load test with stubbed generation, concurrency 8 p50 was 55.066 s and p95 70.173 s. These do not prove production reliability or current-system latency.
- **Repeated evaluation:** three seeded repeats on the same 40 fictional cases gave faithfulness 0.7417 ± 0.0520, answer relevancy 0.5720 ± 0.0247, and reference context precision 0.9492 ± 0.0105 (mean ± sample SD across run means). Initial scoring was 119/120; a scoring-only retry recovered the missing case. These are model-judged scores, not human-reviewed production outcomes. See [reproducibility Q&A](reproducibility-interview-qa.md).
- **Earlier evaluation:** distinguish retrieval recall, answer relevance, correctness, faithfulness and citation support. One 40-case synthetic HTTP/RAGAS run produced means 0.875 faithfulness, 0.6355 answer relevance and 0.9597 reference context precision; five of 27 author-labelled answerable cases were refused despite visible evidence. Preserve the original cap error; synthetic and model-labelled data are not human evaluation.

Read [hardening Q&A](hardening-interview-qa.md) for explanations, examples and limits. The sections below remain a historical revision aid.

Use this for a short reading session or to practise answers aloud. The committed baseline is [`f4c374d`](https://github.com/neeraj1909/xai-rag/tree/f4c374d51d090beba61d34aba81fcc23b8ef700a); newer safeguards described below are included in [published source `c375903`](https://github.com/neeraj1909/xai-rag/tree/c375903fb46327744d076438cb90f7b02711a94f). The examples are illustrative unless identified as an observed check. Project behaviour and your personal contribution are separate: use first person only for work you can substantiate.

## Say it in 30 seconds

> XAI-RAG answers questions from indexed documents and exposes evidence about retrieval and claim support. It retrieves semantically similar chunks from Chroma and keyword matches from Elasticsearch, combines the rankings using reciprocal rank fusion, then reranks a smaller candidate pool with a cross-encoder. An LLM generates an answer with source IDs. Optional explanations and a claim-level NLI diagnostic help inspect the result. I would assess evidence retrieval, answer correctness, citation support, latency and failure handling separately.

## Follow the request

```text
Ingestion: document → parse → chunk → embed → Chroma
                              └─ text + same IDs → Elasticsearch

Question → embed → Chroma dense search ─────┐
         └──────── Elasticsearch BM25 ─────┴→ RRF → cross-encoder rerank
             → bounded context → answer + citations → optional NLI report
```

### 1. What business problem does this solve?

**What:** Find and explain answers in a collection of documents. **Why:** Reading many documents manually is slow, and an unsupported answer is difficult to check. **How:** Retrieve passages, give them to a generator, return source IDs and expose retrieval and claim diagnostics. The code demonstrates this approach; it does not establish a particular customer's savings or production accuracy.

### 2. Is it an agent, a multi-agent system or a memory assistant?

**What:** The inspected request path is a RAG pipeline coordinated by a Python service. **Why:** The steps are sufficient for the documented workflow; no separate agent loop is needed to perform retrieval and generation. **How:** `RAGService` calls the components and uses `asyncio` for concurrent retrieval. This path does not implement LangGraph orchestration, sub-agents, an MCP tool server, model-led planning or durable user memory. An agent chooses actions during execution; MCP is a protocol for connecting models to tools or data; neither is implied by an ordinary function call. The document indexes are retrieval storage, not a personal memory service.

### 3. Why parse and chunk documents before retrieval?

**What:** Parsing extracts text and chunking creates smaller searchable spans with provenance. **Why:** Searching an entire long document can bury a relevant passage, while sending it all to the model wastes context. **How:** The code supports fixed character spans, semantic grouping and parent/child chunks. Fixed spans are simple; semantic grouping adds an embedding model; parent/child retrieval can match a smaller child and provide larger context. Compare their recall, context quality and cost on the same labelled questions before declaring a winner. The parent/child implementation uses character spans despite a token-oriented docstring.

### 4. What does the embedding model do?

**What:** `BAAI/bge-small-en-v1.5` maps chunk and query text into 384-dimensional vectors. **Why:** Related meanings can be close even if wording differs. **How:** Corpus vectors are calculated during ingestion; a question is embedded at query time and compared against stored vectors. The index records compatibility details such as model revision and dimension. Switching models requires an intentional reindex; semantic closeness does not prove the chunk answers the question.

### 5. Why use two retrieval methods?

**What:** Chroma performs dense vector search and Elasticsearch performs BM25 lexical search. **Why:** Dense search can help with paraphrases; BM25 can preserve exact identifiers, numbers and terms. **How:** The two searches run concurrently and return ranked chunks with shared IDs. Other choices include keyword-only, dense-only or a single system offering both; choose through paired retrieval and answer-quality tests, not by assuming hybrid is always better. Two stores add operational and consistency costs.

### 6. How are the rankings combined?

**What:** Reciprocal rank fusion, or RRF, combines positions rather than raw scores. **Why:** BM25 and vector scores have different scales. **How:** For each chunk, add `1 / (60 + rank)` from each ranked list in which it appears; absent entries add zero. Here 60 smooths rank differences; it is not the number of results. This heuristic loses score magnitude and must be evaluated against alternatives.

### 7. Show me one turn with numbers.

**What:** A user asks, “What must I submit for reimbursement?” **How:** Suppose dense search ranks `[A, B, C]`, keyword search ranks `[B, C, A]`, and B says “Submit a receipt.” Then B scores `1/62 + 1/61 ≈ 0.03252`, versus A's `1/61 + 1/63 ≈ 0.03227`; B ranks above A in fusion. A cross-encoder then scores each question–chunk pair and can change that order. If B survives the context limit, an answer might say: “Submit a receipt [B].” Its structured claim could be “A receipt is required,” citing B; NLI could mark it supported. **Why:** This illustrates retrieval, answer and claim checking; the ranks and verdict are not a recorded run.

### 8. What is a reranker, and why use one?

**What:** The configured `BAAI/bge-reranker-v2-m3` is a cross-encoder: it scores a question and candidate chunk together. **Why:** Joint comparison can distinguish closely related candidates better than a reusable embedding similarity alone. **How:** It evaluates a bounded candidate pool after RRF; the top results become generation context. More candidates mean more inference cost and latency; reranking cannot recover a relevant passage that initial retrieval missed. The code uses this model; whether it outperforms alternatives for a given corpus requires measurement.

### 9. What does generation check?

**What:** The configured OpenAI generation path returns structured answer text, source IDs and claims. **Why:** Structure makes outputs easier to inspect and citations easier to validate. **How:** The code rejects unknown IDs, structured claims without source IDs, and an answer containing structured claims with no bracketed citation anywhere. It does not ensure each statement is in the structured claim list or individually supported. The baseline validated IDs against the entire ranked set, including unseen chunks. Revision `c375903` validates against the prompt-visible set instead, though an included ID still does not prove that the visible text supports the claim. A citation-contract failure receives at most one repair attempt; transport and repair attempts share a three-call budget, and a second contract failure is returned as an error.

### 10. How does the project check faithfulness?

**What:** Optional DeBERTa natural-language inference compares a claim with its cited text and reports a support-related verdict. **Why:** A syntactically valid source ID can still point to text that does not support the claim. **How:** Treat the source as a premise and the claim as a hypothesis; inspect entailment, contradiction and neutral signals. The committed baseline could check a child chunk even when generation saw its parent or omitted it under the context limit. Revision `c375903` sends NLI a one-pass HTML-unescaped interpretation of the per-request prompt-visible evidence snapshot; RAGAS receives the literal escaped prompt text. Decoding cannot restore content truncated from the prompt. NLI maps model labels to normalized entailment, contradiction and neutral meanings and uses one model inference per claim–evidence pair. Normalization and model error can still affect the verdict. This is an imperfect diagnostic, not proof of truth or a blocking release gate. In streaming, the generated answer can arrive before the optional NLI report.

## Recall the evaluation

### 11. What do the evaluation terms mean?

- **Retrieval recall@k:** Of all labelled relevant chunks for a question, what fraction appeared in the first *k*? **Hit@k** instead asks whether at least one appeared. If two of three questions have a hit at five, Hit@5 is `2/3`; it is not `2/5`.
- **Faithfulness:** Are answer claims supported by the supplied evidence? A faithful answer can still repeat an inaccurate or outdated source.
- **Correctness:** Is the answer factually right and complete against the reference or authoritative source? A correct guess may be unsupported by the retrieved context.
- **Answer relevancy:** Does the response address the user's question? A relevant answer can still be wrong.
- **Citation validity vs support:** Does an ID resolve, and does that source actually support its associated claim? These are different checks.

### 12. How should an LLM-as-a-judge and human review work together?

**What:** An LLM judge applies a defined rubric to each question, answer and evidence or reference; human reviewers independently inspect examples. **Why:** Automated judging scales but can reward fluent mistakes, misunderstand evidence or disagree with people. **How:** State each rubric's input and score scale, preserve per-case reasons and failures, compare judge decisions with human labels, and report sample sizes and disagreements. Do not call model-judge results human evaluation or transfer one rubric's score to another metric.

### 13. What has actually been tested?

**What:** On one older frozen then-uncommitted snapshot, three passes over the same 13 synthetic cases produced **12/13 answers each**; H05 failed the citation contract on all three. The final pass recorded an unknown citation ID. A separate judge scored the 36 completed answers **2.0/2 each for faithfulness, correctness and relevance**; the three errors were excluded. In a later generator-only probe with reconstructed H05 context, the before responses used attribute-style citations that the validator rejected; adding a citation-format instruction changed accepted responses from **0/5 to 5/5**. It was one case, not a general reliability estimate.

**Why:** The scores are from small fictional datasets, not independently reviewed human assessments. The snapshots, inputs and execution boundaries differ; the later probe does not prove the full pipeline was fixed. **How to investigate:** A harder retrieval test missed an English question's Hindi source in the top five; a separate dense-model comparison surfaced it, without proving general multilingual quality. A candidate conflict prompt also introduced an unsupported qualifier. See [evaluation and failures](holdout-evaluation.md), [citation-format experiment](evaluation/h05-paired-diagnostic.md), [multilingual diagnostic](multilingual-retrieval-diagnostic.md) and [conflict experiment](conflict-prompt-experiment.md). There is no human-evaluation score.

## Defend a production design

### 14. Can the same databases be used in production?

**What:** Potentially, Chroma and Elasticsearch can remain candidates. **Why:** They already serve the project's dense and lexical retrieval needs; replacing them is not a substitute for evaluating requirements. **How:** Test workload, access controls, durable backups, restore, upgrades, failure recovery and operational ownership for the deployment. The local Compose configuration is not evidence of those guarantees. Chroma and Elasticsearch are written separately, so a partial write can leave inconsistent indexes even if IDs are stable; add durable job state, retries and reconciliation as proposed changes.

### 15. How would you prevent one customer's documents reaching another?

**What:** Enforce document-level authorization before context reaches the model. **Why:** The existing service API key is not a per-document user permission. **How:** Derive identity and allowed scope on the server; enforce equivalent authorization in both retrieval paths before fusion, reranking and generation. Test cross-tenant searches and all degraded paths. This is a proposed multi-tenant design, not a claim about the current code.

### 16. What if a search dependency fails or a document contains a malicious instruction?

**What:** The existing concurrent search can fail the query if one branch fails; retrieved text can also contain prompt injection. **Why:** A hidden fallback changes answer quality, and retrieved content is untrusted. **How:** Decide explicitly between fail-closed and a labelled, evaluated degraded mode, with timeouts and alerts. Keep retrieved text as data, limit privileged actions, validate access before retrieval, and test adversarial documents. A prompt instruction alone is not a complete security boundary.

### 17. What would you measure before release?

**What:** Retrieval, generation, safety and operations each need evidence. **Why:** A correct-looking answer can hide poor recall, unsupported citations or unacceptable latency. **How:** Freeze a representative labelled dataset; compare BM25-only, dense-only, RRF and reranked runs on the same queries. Report answer correctness, faithfulness, relevancy, citation support, refusal errors, p50/p95 latency, cost and failure rates by case type. Review human–judge disagreements, test index drift and tenant isolation, and define workload-specific release and rollback thresholds. Those thresholds and results must come from the actual run; they are not established by the smoke fixture.

For source details and longer answers, use the [architecture Q&A](architecture-interview-qa.md), [database Q&A](vector-database-interview-qa.md), [evaluation Q&A](evaluation-interview-qa.md), [guardrail Q&A](guardrails-interview-qa.md) and [production Q&A](production-interview-qa.md).
