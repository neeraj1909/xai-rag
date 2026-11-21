# XAI-RAG evaluation: what, why, how

This chapter is for explaining evaluation without opening the code. It describes the [inspected XAI-RAG revision `f4c374d`](https://github.com/neeraj1909/xai-rag/tree/f4c374d51d090beba61d34aba81fcc23b8ef700a). Examples marked **illustrative** teach the calculation; they are not experimental results. For a particular execution, read [runtime evidence](runtime-evidence.md) and, if available, the separate live-evaluation report; keep those results distinct from the sample fixture below.

## The short interview answer

> I evaluate the pipeline in layers. At retrieval, I compare vector, BM25, fusion and reranking against labelled relevant chunks using recall and ranking metrics. At generation, I assess correctness against a reference, faithfulness against the actual context, and whether the answer addresses the question. I separately test citation validity, refusal and system failures. For open-ended answers I can use an LLM judge with a defined rubric, but I calibrate it against independent human labels and inspect disagreements. I report the dataset, configuration, sample count and failure cases with each result.

This describes a **sound evaluation plan**; the evidence recorded for the checked-in project is narrower, as explained below.

## 1. What do the major dimensions mean?

Consider a fictional policy: “Submit a receipt within 30 days.” The question is “What must I submit, and by when?” The assistant answers: “Submit a receipt within 60 days [A].”

| Dimension | What and why | How to assess it |
| --- | --- | --- |
| **Retrieval relevance** | Did the system find chunks that answer the question? If the evidence never arrives, the generator has little basis to answer. | A reviewer marks relevant chunk IDs; calculate recall and ranking metrics at each retrieval stage. |
| **Faithfulness / groundedness** | Are the answer's factual claims supported by the supplied evidence? This is about agreement with context, not whether the source itself is correct. | Break the answer into claims and compare each against the exact context the generator saw. A cited claim with `60 days` is unsupported by a context stating `30 days`. |
| **Answer correctness** | Is the answer factually correct and complete with respect to an approved reference or authoritative source? A faithful answer can still repeat a stale or erroneous source. | Compare with a reference answer and its underlying source. Check amounts, negation, exceptions, time and completeness; use expert review where the stakes demand it. |
| **Answer relevance** | Does the answer address what the user asked, without irrelevant material or avoiding a needed part? | Judge the question and response together. An accurate but unrelated description of an expense tool is not responsive; relevance alone cannot guarantee factual accuracy. |
| **Citation validity and support** | Does an ID resolve, and does the cited passage support the specific claim? A resolvable source is not automatically supportive. | First validate IDs and permissions; separately assess claim-to-passage support and coverage. |
| **Refusal quality** | Does the system decline an unanswerable request without refusing valid ones? | Report appropriate refusal for unanswerable cases and false refusal for answerable cases separately. |

**What is the numerical score?** There is no universal “faithfulness 0.9” scale. A claim-based faithfulness measure may be `supported claims / assessed claims`; a correctness rubric might be categorical; RAGAS uses its own implementation. Always name the scorer, inputs, version, scale and denominator. Do not treat a score as a calibrated probability of truth unless separately calibrated.

## 2. What is actually implemented in XAI-RAG?

| Capability | What the inspected code does | What the result cannot establish |
| --- | --- | --- |
| Offline JSONL evaluator | Reads a manifest and **previously captured** query/ingestion records, then computes stage metrics, summary values and configured gates. It does not itself run the retrieval/generation pipeline. | A gate pass alone is not a live-model or product-quality evaluation. [Dataset contract](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/evals/README.md), [runner](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/evaluation/runner.py). |
| Retrieval | Computes precision, recall, hit rate, MAP and NDCG at selected cutoffs, and MRR for vector, BM25, fused and reranked IDs. | These scores require relevant IDs. Retrieval means exclude cases without positive relevance labels; refusal cases need separate reporting. |
| Generation records | Computes citation validity, captured structured-claim coverage, exact match, token F1 and recorded claim verdict rates. It also averages `judge_scores` and `human_scores` already supplied in records. | Storing a score under `human_scores` or `judge_scores` does **not** run a human or model judge. Citation checks do not prove entailment or detect factual claims missing from the structured claim list. |
| Optional RAGAS | The adapter requests `Faithfulness`, `AnswerRelevancy` and `ContextPrecision` using configured model clients; without a reference, it uses `ContextUtilization` under the returned context-precision key. It reports `computed`, `unavailable`, or `error`. | It is optional, makes model calls when used, and a label/score does not establish independent human agreement. [Adapter](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/evaluation/ragas_eval.py). |
| Claim-level NLI | An optional model compares claims with evidence and produces support/contradiction/neutral diagnostic verdicts. | It is not an infallible fact-check or a hard release gate; a streaming answer can be emitted before its diagnostic. [Implementation](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/explainability/faithfulness.py). |

The two uses of “faithfulness” are related but different implementations: the optional RAGAS score and the separate claim-level NLI diagnostic do not become interchangeable merely because both compare answer claims with evidence. Check inputs and status for the particular run.

## 3. Which dataset was evaluated, and what were the scores?

The repository's checked-in [`evaluator-contract-smoke`](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/evals/smoke/evaluator-contract.jsonl) contains **two synthetic query records** and **one synthetic ingestion record**. Its manifest explicitly says the examples are synthetic and not representative of production. The file already contains the answer, relevant IDs, simulated judge and human-score fields, verdicts, and latency values. The evaluator reads those values; it does not produce them from fresh model calls.

An offline run requesting `@1` and `@2` produced these **fixture-contract results**, not live accuracy:

| Output | Fixture score and denominator | Meaning and limitation |
| --- | --- | --- |
| Citation validity | `1.0`, one non-refusal answer | Its recorded cited IDs occur in recorded reranked IDs. No semantic support or prompt-context completeness is proven. |
| Citation coverage | `1.0`, one non-refusal answer | Recorded structured claims had IDs in the recorded reranked list. It cannot discover an assertion omitted from the claims list. |
| Recorded human correctness | `1.0`, one record | The value was **prewritten synthetic input** under `human_scores`. It was not human assessment during this run. |
| Recorded judge relevance | `1.0`, one record | The value was **prewritten synthetic input** under `judge_scores`. It was not produced by an LLM judge during this run. |
| Refusal accuracy | `1.0`, two records | Fixture's actual and expected refusal flags match for both records. This is neither a real-user refusal rate nor an estimate on a representative dataset. |
| Index parity | `1.0`, one ingestion record | The fixture reports parity. It does not prove all records in live indexes agree. |

Six configured blocking gates passed; with `@2` requested, the six advisory gates passed too. The default cutoffs are `1,3,5,10`, yet two advisory gates refer to `@2`; under those defaults they are `not_computed`. This is a configuration mismatch, not a retrieval-quality conclusion. Exact invocation and validation boundaries are documented in [runtime evidence](runtime-evidence.md).

**Human evaluation status:** the smoke fixture is synthetic even though a field is named `human_scores`. No independent human review is evidenced by that fixture or this chapter. A human-reviewed benchmark and agreement study remain to be conducted; a machine judge's output must not be relabelled as a human score.

## 4. Worked retrieval metrics

**Illustrative example:** Relevant chunks are `{A, B}`; the top three returned are `[X, A, C]`.

- `Recall@3 = 1 / 2 = 0.50`: one of two relevant chunks was found. It measures **coverage**, not ranking quality or correctness of the final answer.
- `Precision@3 = 1 / 3 ≈ 0.33`: one of the three retrieved slots was relevant. In this code, the requested `k` is the denominator even if fewer results are returned.
- `Hit@3 = 1`: at least one relevant chunk appeared; it does not show whether the second one was missed.
- `Reciprocal rank = 1 / 2 = 0.50`: the first relevant result appeared at rank two. **MRR** is the mean reciprocal rank across evaluated queries.

**NDCG** uses graded relevance and discounts lower ranks. For example, if relevance grades for `[A, B]` are `[3, 1]`, placing A first yields higher DCG than placing B first; NDCG divides by the best ordering's DCG. This project computes it for each retrieval stage. Its fusion delta compares fused NDCG against the **better of BM25 or vector for each query**, while its reranker delta compares reranked with fused NDCG. These are per-query comparisons, not proof that every hybrid system improves every query. [Metric definitions in code](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/evaluation/metrics.py).

## 5. What is an LLM-as-a-judge?

An LLM judge is a separate model call that receives a task-specific rubric and the relevant inputs—for example question, answer, reference and exact retrieved context—and returns a structured assessment. It can grade a single answer, classify claim support, or compare two anonymized answers. It scales review but may prefer verbose responses, depend on response ordering, miss numbers/negation, or reward incorrect but confident answers. A judge's score is an **opinion produced by that model under that rubric**, not ground truth.

**Proposed reproducible rubric, not an existing XAI-RAG benchmark:**

| Dimension | 0 | 0.5 | 1 | Required input |
| --- | --- | --- | --- | --- |
| Correctness | A key fact is wrong or answer is unrelated. | Core fact correct but a required detail or exception is missing. | All required facts correct; no material error. | Question, response, reviewed reference and source. |
| Faithfulness | At least one material claim contradicts or is unsupported by supplied evidence. | Some supported facts, but another material claim is ambiguous or lacks adequate context. | Every material factual claim is supported by evidence. | Response and **actual prompt context**, including chunk IDs. |
| Answer relevance | Does not address the requested information. | Addresses some, but not all, requested parts. | Directly addresses all requested parts without material irrelevant content. | Question and response. |

For each score require a short cited reason and an `unscorable` outcome for missing inputs. Unscorable, error, and refusal must not silently become zero or be excluded without reporting counts. A numeric average across dimensions can hide a severe unsupported statement; retain individual dimensions and a separate material-error flag. The table is an example for discussion, not the RAGAS implementation's exact scoring formula.

**Why a rubric rather than “rate this answer”?** It gives the judge an explicit target, improves auditability, allows controlled tests with obvious wrong-number and omitted-exception cases, and makes disagreement review actionable. Freeze the rubric, judge model and settings; randomize A/B order in comparisons; record model/version, prompt, response, latency and failures. The [OpenAI evaluation guidance](https://developers.openai.com/api/docs/guides/evaluation-best-practices/) discusses task-specific tests, judge biases and calibrating automated scoring against human labels. The indexed page was last fetched 14 November 2026; local XAI-RAG pins RAGAS `0.4.3`. Documentation and library versions may differ, so code and the actual run configuration govern this project.

## 6. What is human evaluation and how do I use it responsibly?

A human evaluator independently reviews a question, source, reference and answer against a documented rubric. A domain reviewer may catch stale policies, domain-specific exceptions or unsafe recommendations that a model judge misses. Human labels are also imperfect: reviewers can disagree or overlook context, so publish annotation instructions and adjudicate disagreement.

A defensible proposed protocol is:

1. Freeze a permitted corpus and representative questions; include answerable, unanswerable, multi-hop, exact-ID, adversarial and stale/conflicting-source cases. Split development and held-out sets by document or topic when leakage is a concern.
2. Two reviewers independently label answerability, relevant evidence and reference answers for a subset; show each the exact source version and rubric. Do not allow generated output to define the reference answer.
3. Record first-pass disagreement, adjudicate hard cases with a named reviewer, and retain both original and final labels. Report agreement by dimension, sample sizes and case selection; high raw agreement on heavily imbalanced labels can hide poor performance on rare failures.
4. Run the same cases through the system and separately through the judge, storing generated outputs, supplied context, reviewer/judge decisions and failures. Compare judge versus adjudicated labels using a confusion matrix and per-class false-positive and false-negative rates; inspect examples where the judge calls an unsupported answer supported.
5. Change one pipeline factor at a time, compare paired results and confidence intervals where sample size supports them, and keep final held-out cases untouched during tuning. Include latency, cost, refusal and authorization/error outcomes. Thresholds belong to the use case and risk owner.

LLM-generated synthetic questions can broaden edge-case coverage, but their references must be checked before presenting them as human gold labels. Neither agreement with the same model family nor a tiny convenience set proves a system is safe for unseen users.

## 7. How would I explain evaluation in an interview?

**Question: How do you know if the reranker is useful?**

**Answer:** Freeze the same labelled corpus and questions. Record the fused candidate list before reranking and final order after it; compute per-query NDCG/recall and paired differences, compare wins and regressions, then measure reranker latency and cost. Check lexical-only and dense-only baselines too. A rise in aggregate NDCG can coexist with regressions on exact identifiers or tail cases, so inspect slices. This experiment is proposed unless accompanied by a saved run and dataset.

**Question: Can a response score high on relevance and low on faithfulness?**

**Answer:** Yes. “Submit a receipt within 60 days” directly answers the question but conflicts with evidence stating 30 days. It is relevant yet not supported by the supplied context. Conversely, a faithful quotation can be irrelevant if it does not answer what was asked.

**Question: Do citation validity and faithfulness measure the same thing?**

**Answer:** No. Validity checks an ID exists in the selected available set; faithfulness asks whether evidence supports the claim. XAI-RAG's validator and offline metrics also use retrieved/reranked IDs, which may include chunks omitted from the actual bounded model context; test against the exact presented context if that is the intended guarantee.

**Question: What does a 1.0 smoke score prove?**

**Answer:** Only that the deterministic evaluator processed the synthetic fields consistently for those one or two records and satisfied its configured thresholds. The synthetic `human_scores` and `judge_scores` were prefilled, not newly assessed by a person or LLM. It does not prove production accuracy, fairness, security or latency.

**Question: What should I disclose if asked for a real human-evaluation result?**

**Answer:** State whether independent human labels were actually collected for the run, how many reviewers and items were involved, their rubric and agreement. If none were conducted, say so and propose the review protocol; never call a model's score a human score.
