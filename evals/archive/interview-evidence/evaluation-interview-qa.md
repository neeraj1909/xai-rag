# XAI-RAG: evaluation interview questions and answers

> **Version note:** This document describes the earlier `c375903` or explicitly cited historical source. Statements here that call NLI optional, say document authorization is absent, or describe recovery as only a proposal do not describe the later hardening work. Read the [current hardening questions and answers](hardening-interview-qa.md) first, and verify the source revision before an interview. Historical measurements remain tied to their stated snapshots.

The linked [`f4c374d`](https://github.com/neeraj1909/xai-rag/tree/f4c374d51d090beba61d34aba81fcc23b8ef700a) is the historical baseline; later changes are included in [published source `c375903`](https://github.com/neeraj1909/xai-rag/tree/c375903fb46327744d076438cb90f7b02711a94f). Links pinned to the baseline do not establish later changes. Production evaluation designs are proposals, and the numbers below identify their datasets and denominators.

## What exists today?

### 1. Question: What evaluation data and tooling does the project actually have?

**Answer:** It has a deterministic offline evaluator that reads captured observations from JSONL. Its manifest records dataset, corpus, configuration and model revisions; query records can contain graded relevant IDs, results at each retrieval stage, citations, verdicts and latencies; ingestion records contain outcomes, spans and index parity. The checked-in smoke fixture has **two synthetic queries and one synthetic ingestion record**. It tests the evaluator and gate wiring, not RAG quality on actual users or documents. [Dataset contract and fixture](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/evals/README.md), [fixture rows](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/evals/smoke/evaluator-contract.jsonl).

### 2. Question: Did you run the evaluator, and what does that result establish?

**Answer:** The following local command completed with exit code 0. Its six configured blocking gates passed on the two synthetic query records and one ingestion record. It **reads prewritten observations**: it does not retrieve documents, call a model, measure live latency or establish production quality.

```bash
uv run --offline xai-rag evaluate evals/smoke/evaluator-contract.jsonl \
  --gates evals/gates.toml --k 2 --output /tmp/xai-eval-interview-k2.json
```

On this fixture, `--k 2` computes the two @2 advisory gates. The synthetic numbers and their `N` values in the report show the evaluator can process that input; they are not a benchmark. This command was run against the checked-out source without provider calls. [CLI test](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/tests/unit/test_evaluation_cli.py), [evaluation runner](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/evaluation/runner.py).

### 3. Question: What did the evaluation run reveal about the default cutoffs?

**Answer:** The evaluator defaults to `k=(1, 3, 5, 10)`, while two advisory gates refer to reranked `recall@2` and `ndcg@2`. With the default command those two gates appeared as **`not_computed`**. Passing `--k 2` computed them in the fixture run. This is an evaluator-configuration mismatch, not evidence that retrieval is weak. A production CI configuration should align requested cutoffs and gates and verify no required metric is missing. [Default cutoffs](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/evaluation/runner.py), [gate configuration](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/evals/gates.toml).

### 4. Question: What is the difference between blocking and advisory gates?

**Answer:** The configured blocking gates check smoke invariants such as error rate, citation validity and coverage, index parity, provenance and duplicate rate. An unavailable or failing **blocking** metric makes the report fail. Advisory targets such as retrieval quality, claim support, refusal accuracy and latency are provisional; they can pass, fail or remain uncomputed without approving production release. The repository says product/domain owners should agree thresholds against representative evidence. For example, the configured p95 of 1,000 ms is a placeholder, not an established service objective. [Gates](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/evals/gates.toml).

## Retrieval and answer quality

### 5. Question: How do you separate retrieval quality from answer quality?

**Answer:** For each question, record IDs at vector, BM25, fusion and reranking stages and compare each list with independently labelled relevant IDs. Then inspect the generated answer against its references and supporting source text. If the evidence is absent from the early lists, investigate parsing and retrieval. If it appears early but disappears after reranking, inspect ranking. If it reaches the prompt and the answer is wrong, inspect context formatting and generation. This isolates which component needs work instead of changing the prompt blindly. [Recorded fields](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/evaluation/models.py).

### 6. Question: Show how Recall@3, Precision@3, Hit@3 and MRR differ.

**Answer:** This is an **illustrative** case. Suppose human labels say A and B are relevant, and the returned order is `[X, A, C]`. At rank three there is one relevant hit out of two labelled relevant chunks, so Recall@3 = `1/2`. Precision@3 = `1/3` because one of three requested positions is relevant. Hit@3 = `1` because at least one relevant item appears. The first relevant item is at rank two, so its reciprocal rank is `1/2`; MRR is the mean of that value across evaluated queries. In this implementation, Precision@k divides by requested `k` even for a short list, duplicate IDs cannot earn multiple hits, and retrieval averages exclude cases with no positive relevance labels. [Metric implementation](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/evaluation/metrics.py), [aggregation](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/evaluation/runner.py).

### 7. Question: When is NDCG useful compared with recall?

**Answer:** Recall asks whether labelled relevant evidence was retrieved, regardless of its position within the cutoff. NDCG uses graded relevance and a rank discount, so retrieving a strongly relevant chunk near the top scores better than placing a weakly relevant chunk first. Here the evaluator computes NDCG per stage and calculates per-query deltas for fusion and reranking. It uses fusion NDCG minus the **better individual retriever for each query**, not merely minus the overall mean of one retriever. Labels and the chosen cutoff still shape the result. [Metric and deltas](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/evaluation/runner.py).

### 8. Question: How do you know whether hybrid retrieval and reranking justify their cost?

**Answer:** Freeze the corpus, questions, models and configuration. Compare BM25-only, vector-only, fusion without reranking and fusion plus reranking on the **same labelled queries**. Report retrieval quality per query and by slice, the reranker's wins and regressions, stage/end-to-end latency, model resource use and cost. Make an intentional deployment decision: a quality gain that improves exact-ID cases but misses the latency budget may call for different candidate counts or query routing. The checked-in smoke fixture is too small to decide this.

### 9. Question: Are valid citations equivalent to supported claims?

**Answer:** No. The offline evaluator's citation validity measures the fraction of **unique cited IDs** found in the recorded reranked context. In contrast, the `c375903` runtime validator checks against IDs actually included in the bounded prompt, which may be a smaller set. For each non-refusal answer, citation coverage measures the fraction of **captured structured claims** whose recorded `claim_source_ids` include at least one ID in that context; the report then averages those per-answer values. If an answer contains a factual claim that the model omitted from its structured claims, this metric cannot detect the omission. Neither metric checks that a source entails the text. For example, citing a real refund-policy chunk for the wrong refund amount can pass both checks. Separately assess claim extraction completeness, claim-to-source support, and whether the source is authoritative and current. [Citation metrics](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/evaluation/metrics.py), [generation aggregation](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/evaluation/runner.py), [generator validation](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/generation/generator.py).

### 10. Question: Why not use exact match as the only answer metric?

**Answer:** The evaluator offers normalized exact match and bag-of-token F1, but a correct paraphrase can differ from a reference and high word overlap can still hide incorrect numbers or negation. Use them as cheap diagnostics, then combine human review of factual correctness, completeness and source support with carefully validated automated judges. An automated result should show its sample size and the reference used; do not turn token overlap into a truth score. [Implementation](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/evaluation/metrics.py).

### 11. Question: How would you validate the NLI faithfulness diagnostic?

**Answer:** Label claim–evidence pairs as supported, contradicted or insufficient evidence with trained reviewers. Include numbers, negation, long passages, domain terms, unrelated evidence and conflicting dates. Compare the diagnostic with labels using a confusion matrix and per-class precision/recall; review disagreement and test drift when models or corpus change. A `neutral` result differs from contradiction. In `c375903`, NLI receives a one-pass HTML-unescaped version of the visible evidence prefix; RAGAS uses the literal escaped prompt evidence. This cannot restore truncated text. A real NLI run matched 9/10 authored fictional pair labels, not independent human labels; see [NLI evaluation](nli-evaluation.md). In this project NLI is an **optional diagnostic**, not a blocking release gate; for streaming, the answer can be emitted before the diagnostic. A high NLI score is not proof of truth. [Faithfulness implementation](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/explainability/faithfulness.py), [service](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/service.py).

### 12. Question: What does optional RAGAS add, and what are its limits?

**Answer:** When requested and configured, the adapter can provide faithfulness, answer relevancy and context precision. Reference-based and reference-free settings answer different questions, and the metrics can use model calls. They cost time and may cost money; their judge can be biased or mistaken. Validate judge agreement with a human-reviewed sample and inspect per-case results. Do not claim RAGAS runs by default or replace human source-support assessment with a single score. [RAGAS adapter](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/evaluation/ragas_eval.py).

### 12a. Question: What did the live LLM-as-a-judge experiment actually show?

**Answer:** On eight fictional questions, a baseline run used the real hybrid retrieval and generator; all eight generated answers. A separate model judged each answer 2/2 on each rubric criterion. That is a small, easy synthetic sample, not a human-validated production score. A later then-uncommitted snapshot produced seven of eight answers and one generation contract error; scores computed on those seven exclude the failure and must not be presented as an eight-case success rate. Code changed after that snapshot, so it is not a final release score. See [live evaluation](live-evaluation.md).

A separate guardrail set illustrates judge instability: the same 12 answers scored 9/12 with the first rubric and 12/12 after its clarification; one of 13 generation attempts had already failed. The two judges agreed on 10/12; both wrongly rejected G08 even though its answer correctly cited visible source `[g]` and ignored the document’s untrusted instruction to cite nonexistent `[h]`. Those observations motivate independent human review and failure-inclusive reporting, rather than treating judge agreement as ground truth. See [judge calibration](judge-calibration.md).

### 12b. Question: What is the difference between correctness, faithfulness and relevance?

**Answer:** **Correctness** asks whether the answer matches the facts or an independently verified reference. **Faithfulness** asks whether the answer's claims follow from the supplied evidence; a faithful answer can repeat an outdated source. **Answer relevance** asks whether the answer addresses the question, even if its facts are wrong. For the live experiment, the judge applied the documented 0–2 rubric to the sample; the recorded 2/2 results are judge assessments, not human labels. A useful evaluation retains per-case rationales, the source and rubric versions, error cases and human disagreement, rather than collapsing them into one number. See [evaluation primer](evaluation-primer.md) and [live evaluation](live-evaluation.md).

## Building a production evaluation

### 13. Question: How would you build a representative dataset?

**Answer:** Collect permitted, privacy-reviewed samples of real query types and representative source documents; add synthetic and adversarial cases for missing evidence, ambiguous requests, conflicting or stale sources, prompt injection and rare high-risk workflows. Freeze corpus and configuration versions. Have domain reviewers label answerability, relevant chunk IDs and acceptable answer/evidence; use double review and adjudication on a subset. Keep a held-out test set separated by document or topic where that prevents leakage. Report label source, sample count, annotation disagreement and coverage by query type, document type, language, freshness and risk. Do not describe a proposed dataset as already collected.

### 14. Question: How would you measure refusals and avoid reward for refusing everything?

**Answer:** Keep answerable and unanswerable cases and score them separately: appropriate refusal among unanswerable cases, false refusal among answerable cases, and answer correctness on the questions actually answered. Include retrieval-empty and irrelevant-but-nonempty context as distinct cases. For an **illustrative** set of 20 answerable cases, refusing 8 gives false-refusal rate `8/20 = 40%`; a high refusal rate would not itself indicate safety. The current evaluator calculates refusal accuracy when expected and actual flags are available; the additional sliced rates are a proposed reporting improvement. [Runner](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/evaluation/runner.py).

### 15. Question: How would you evaluate guardrails and prompt-injection resistance?

**Answer:** Add labelled cases with instructions embedded in documents, malformed or fabricated citation IDs, conflicting evidence, attempts to expose private data, and answerable versus unanswerable requests. Check what the user receives, whether unauthorized content reaches context, citation support and appropriate refusal. Run a separate streaming test because answer text may arrive before an NLI report. API authentication, schema/citation checks and a prompt that treats retrieved text as data are distinct controls; none proves resistance to all attacks. Existing tests and a proposed adversarial benchmark must be reported separately.

### 16. Question: How would you evaluate reliability, latency and the two indexes?

**Answer:** Exercise full ingestion and queries in a controlled environment at defined corpus size and concurrency. Record p50/p95/p99 by stage and end to end, timeout/error rates, token use, actual cost, plus source/ID parity. Inject an Elasticsearch outage, a Chroma outage and a failure between the two index writes; check the user's response, alerts and recovery. Confirm permissions on both retrieval paths and test stale or deleted records. A measured p95 must include workload, sample size, hardware and warm/cold conditions; the smoke fixture's latency fields are synthetic.

### 17. Question: How would offline evaluation and production monitoring work together?

**Answer:** Run the frozen labelled set before a change to detect regressions, and monitor operational errors, latency, retrieval failures and privacy-safe feedback after release. Sample traces under retention and access rules for human review, add failure cases to a future development set, then re-evaluate against a separately protected test set. Track model/index versions so a change in results is attributable. Online indicators do not replace source-level labels and must not expose private queries or documents.

### 18. Question: What would you require before saying the system is ready for a specific production use?

**Answer:** Agree on use case, risk, access policy and owners first. Define representative, reviewed datasets and chosen blocking thresholds for correctness, source support, refusal, isolation and reliability; baseline the current system, test realistic dependency failures and recovery, and document monitoring and rollback. Thresholds are product decisions informed by risk, not universal percentages. A passing synthetic gate and a running demo are evidence of useful engineering checks, not evidence of acceptable production quality.

## Five-minute rehearsal

Explain one metric with the worked `[X, A, C]` example, identify why citation validity is not source support, describe the @2 gate mismatch, compare hybrid against its component baselines, and finish with one failure-injection test and one honest limit on the evidence.
