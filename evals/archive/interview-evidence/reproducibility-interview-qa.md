# XAI-RAG: evaluation reproducibility interview questions

Use this companion with the [evaluation evidence](evaluation-hardening-evidence.md) and the experiment's own manifest. The questions include illustrative examples and, in the measured-results section, the recorded three-run experiment. Check its manifest before claiming a seed, score or model version was used.

## A one-minute answer

> “A seed makes a controlled source of randomness repeatable; it does not freeze the system. For XAI-RAG, I preserve the question set and document corpus, source and dependency revisions, index contents, retrieval and prompt settings, model identifiers and per-case responses. I repeat a fixed evaluation and report each run, failures, metric denominators, mean and variation. For hosted models, I treat deterministic sampling as provider-dependent and best-effort. I keep historical outputs so someone can recompute the original aggregate separately from running a new experiment.”

## Questions to rehearse

### 1. What does a seed control, and why isn't it enough?

A seed initializes a particular random-number generator. For example, the earlier HTTP evaluation used `20261120` to shuffle case order in its harness. That makes the order repeatable given the same input and algorithm; it does not by itself control the generator, grounding judge, RAGAS judge, index construction, or hosted model backend. I record **which component consumed which seed**, not just “seed 20261120.” Saving the ordered case IDs is stronger evidence than assuming a future implementation shuffles them identically. The earlier recorded aggregate can be recomputed from saved scores; that operation does not call a model or reproduce a live run.

### 2. How would you identify the exact system being tested?

I capture the Git commit and any uncommitted patch or a source-tree digest, dependency lock and relevant package versions, runtime and hardware, configuration, and the corpus and case-set hashes. I record ingestion strategy, chunking settings, retrieval cutoffs, fusion, reranker and NLI revisions, and generation, judge and evaluator prompts and parameters. The earlier 40-case study was run on a modified tree at `c375903`; HEAD alone was insufficient to identify it. A [reconstructed Python snapshot](evaluation/hardening-2026-11-20/evaluated-python-source/README.md) matches its recorded digest, but it is not a complete environment or deployment snapshot.

### 3. What does an “index hash” actually prove?

The answer depends on what was hashed. A canonical digest of sorted chunk IDs, stored text and metadata can detect a change in those records, but it does not automatically prove that vector values, embedding model, approximate-index graph, analyzer configuration or search-server version are identical. I would include those separately where available, state precisely what's covered, and compare both Chroma and Elasticsearch. Quiesce writers or use a consistent snapshot; otherwise a multi-page read can combine different index states. A matching corpus hash alone does not prove a matching indexed representation.

### 4. Why hash prompts and parameters as well as the dataset?

The same question and passages can lead to different answers when instructions, evidence ordering, schema, token limits, sampling settings or guardrail thresholds change. XAI-RAG also has a second LLM judge, so I would record **both** the generation and judge request contracts, plus RAGAS metric prompts/configuration where accessible. Hash canonicalized settings for comparison and preserve a readable, non-secret copy or a versioned source location for interpretation. Never place API keys, access tokens or private document text in a public manifest.

### 5. Can we use one seed for OpenAI generation and get identical outputs?

Only if the selected endpoint and model accept such a parameter, and even then it may provide best-effort rather than guaranteed determinism. The Chat Completions seed parameter is documented as deprecated; model and endpoint support must be checked before sending it. A provider alias and its backend can change, and calls can fail or be retried. Distinguish **case-order seed** from a model request seed, record whether the latter was accepted, and do not claim a model was seeded merely because the runner accepted `--seed`. Preserve request parameters, returned model identifier and available provider metadata. [OpenAI Chat Completions reference](https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions/methods/create).

### 6. Why run the same evaluation more than once?

Repeated live runs help measure observed variation from model generation, judging and other nondeterministic runtime behavior on a **fixed** question set. If the system is unchanged and the same seed is used for a component that supports it, a repeatability test asks how much that setup still varies. If different seeds are used, that is a different experiment: it samples some seed-dependent variation. Neither automatically measures how the system performs on different users or documents. Report what changes and what stays fixed for each series.

### 7. What statistics would you show for three repeated runs?

Show each run's score and denominator first, then the arithmetic mean and sample standard deviation across the three run-level scores; also show the minimum and maximum. For illustration, `0.70`, `0.75`, `0.80` have mean `0.75` and sample SD `0.05`. Those numbers are an example, not XAI-RAG measurements. With three runs, SD is itself uncertain; it is not a confidence interval and it does not measure dataset-to-dataset variation. Record whether a run or metric failed and avoid calculating a clean-looking summary that silently excludes it.

### 8. Is run-to-run standard deviation a confidence interval?

No. The SD describes spread among the observed run scores under the chosen setup. A confidence interval concerns uncertainty in a specified estimand under a stated sampling procedure and assumptions. With only a few repeated runs on the same 40 questions, treating all question-by-run rows as independent inflates the apparent sample size. For a model comparison, retain case-level results and consider a **paired bootstrap over questions**, keeping repeated outcomes together by case. That describes case-sampling uncertainty conditional on the observed fixture and runs; it does not independently measure model-run variation. A synthetic, author-labelled fixture that was not sampled representatively cannot establish a confidence interval for a real customer population.

### 9. How do you compare two versions fairly?

Use the same held-out question IDs, source corpus, access scope and reference labels; record any unavoidable environmental differences. Capture results per question for both versions and evaluate **paired deltas**, not only separate aggregate means. For instance, an increase in overall relevance can hide a regression on unanswerable or Hindi questions. Compare refusal, errors, grounding and latency as well as averages, and predefine how missing cases and ties are handled. If the prompts or model alias changed between runs, report that as a changed condition rather than attributing all movement to the code patch.

### 10. How do you prevent failures from making the average look better?

Keep one planned record per case and run, with statuses such as completed, refused, provider error, timeout, metric error and unattempted. Report counts for every status and each metric's eligible denominator. Do not convert a missing score to zero unless a documented metric explicitly defines that behavior; do not drop failures and call the remaining mean an overall score. Report conditional quality on successfully scored cases alongside completion rate and failure-inclusive outcomes. For refusals, use answerability labels to separate an appropriate abstention from a false refusal.

### 11. If every result is HTTP 200, did the evaluation pass?

No. HTTP 200 says the request completed at the protocol level. An answer can still be unsupported, off-topic, unsafe or unnecessarily refused. In the earlier 40-question fictional study, all 40 requests returned 200, while five of 27 cases labelled directly answerable were refused even though the labelled source was visible; those are candidates for independent adjudication. The same study's RAGAS means describe that older modified source and do not measure the later integrated safety-and-coverage judge. [Results and provenance](evaluation-hardening-evidence.md#current-application-http-and-ragas-evaluations).

### 12. Why can an LLM judge give different scores to the same answer?

The judge may sample different wording or classifications, misunderstand a rubric, be influenced by the supplied evidence, or change behavior after a model/backend update. Repeating the judge on a fixed answer and evidence measures grader variability separately from repeating generation. Save rubric and prompt versions, model identifier, per-case scores and errors. Calibrate the judge against independently reviewed examples, including cases where factual correctness, answer relevance and source faithfulness disagree. A generator and judge from the same model family can share blind spots; agreement is not independent human verification.

### 13. What are the metrics measuring?

**Faithfulness** assesses whether an answer's claims follow from the provided context; a faithful answer can repeat a wrong or stale document. **Answer relevancy** assesses whether the response addresses the question; it does not independently establish truth. **Reference-based context precision** assesses whether useful retrieved contexts are ranked above less useful ones relative to a reference; it depends on the reference and metric implementation. For the original 40-case RAGAS run, means were 0.8750, 0.6355 and 0.9597 respectively, each on 40 fictional cases. These are prior single-study aggregate scores, not a multi-run mean or human-reviewed accuracy. RAGAS may score an appropriate refusal poorly, so report answerable, abstention and other slices alongside denominators; the author's historical labels may not match the current whole-answer-refusal policy.

### 14. How do you make retrieval results explainable across runs?

For each query, retain the principal/scope, permitted source IDs, dense and BM25 candidate IDs and scores, fusion order, reranked list and the **actual prompt-visible** chunk IDs. A chunk in a reranked candidate list may be removed during evidence packing, so a citation check should use what the generator saw. Preserve index and query settings to investigate ranking drift, and state whether IDs or full contents are captured. Protect private material: save synthetic test cases publicly, and apply access controls and retention rules to real traces.

### 15. How do you keep repeated tests comparable without leaking access-controlled data?

Use a fixed, permitted synthetic corpus for public experiments. Derive tenant and source scope from server-side test credentials, not from the question text, and test both dense and lexical filters before content reaches the reranker or provider. Hashing a sensitive value does not automatically anonymize it: predictable names and paths can be guessed. Keep keys and policy files out of artifacts, limit what goes to an external judge, and record the scope of the access-control test rather than claiming all tenant-isolation paths are proven.

### 16. What should a repeatable experiment command and artifact contain?

The command should identify the fixture, explicit seed or seed list, run count, model and endpoint settings, output directory and provider-call budget. It should write a manifest, immutable input references, per-run/per-case records, attempted and failed calls, usage, and a summary that can be recomputed offline. Document prerequisites and cleanup for disposable stores and explain any paid calls. Verify that a second operator can follow the command; saving a script with hardcoded `/tmp` paths and temporary credentials is not enough. The older archived harness is evidence, but its README explicitly says it is not a portable rerun.

### 17. What would you tell an architect about cost and performance?

One user query can trigger generation, a separate grounding judge, and multiple RAGAS metric calls and embeddings. XAI-RAG's response token fields count generation only, so they cannot represent total end-to-end spend. Record attempts, successful usage and any unpriced failed requests by purpose, and put an explicit cap in the experiment. Measure latency with a defined corpus, warmup, concurrency, hardware and error denominator; a seed does not make a shared CPU or network deterministic. Keep model-quality evaluation separate from capacity testing.

### 18. What would convince you that the result is production-ready?

Not a single seeded score. I would agree on use-case-specific correctness, source support, refusal, isolation, latency and cost thresholds; use an independently reviewed, representative held-out set; calibrate the judge; test failures and access boundaries; and monitor drift after deployment. I would also investigate known false refusals, partial-answer behavior and resource limits. Seeds and hashes enable someone to audit the experiment, while representative data and appropriate tests determine whether its results support a deployment decision.

## Repeated live evaluation: recorded results

Three runs used the same 40 fictional, author-labelled questions and one scoped index. The case-order seed and evaluation-only model-request seed were each `20261120`. The evaluation wrapper recorded that it sent that seed on all 1,426 chat attempts across generation, grounding and RAGAS; 242 embedding attempts did not carry it. This was a property of the **evaluation wrapper**, not a general production seed setting. The provider returned the `gpt-6-luna` alias and no system fingerprint. A separate same-seed three-call probe returned two distinct outputs: accepted seed parameters did not make the responses identical.

The HTTP API returned 200 for all 120 requests. In the first scoring pass RAGAS completed 40, 40 and **39** cases: run 3 case H08 failed due to a provider connection error. A separately recorded scoring retry used the exact saved answer and context for H08 and succeeded; it did not rerun generation. The following table uses the recovered score, while preserving the initial failure and 39-case first-pass result. Each run-level mean has 40 scored cases **after that retry**.

| RAGAS metric | Run 1 | Run 2 | Run 3, recovered | Mean of runs | Sample SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Faithfulness | 0.7250 | 0.8000 | 0.7000 | 0.7417 | 0.0520 |
| Answer relevancy | 0.5570 | 0.6005 | 0.5586 | 0.5720 | 0.0247 |
| Reference-based context precision | 0.9418 | 0.9447 | 0.9612 | 0.9492 | 0.0105 |

These are three run-level means on the same 40 synthetic cases, not 120 independent user questions. The sample SD measures observed spread across these three runs; it is not a confidence interval or a production-quality estimate. The generator and judge use the same model family, and labels have not been independently reviewed. Appropriate refusals can receive low RAGAS scores; the table is not a stand-alone safety or correctness score.

Among 27 cases labelled directly answerable, the gate returned `insufficient_evidence` for 6, 4 and 5 cases respectively. These are possible false refusals under the author's labels, not independently adjudicated errors. A passing HTTP status does not establish a correct answer. The separate index snapshots recorded 25 text/metadata records per store, matching record digests and a digest of exported Chroma float32 vectors; they did not create an atomic cross-store snapshot or prove the complete HNSW index state was identical. [Repetition manifests, run results and recovery provenance](https://github.com/neeraj1909/xai-rag/tree/main/evals/results/2026-11-20-seeded-repeated).

### 19. A seed was accepted on every chat request. Why did results still change?

An accepted request field shows what was sent, not a promise of determinism. The model alias, provider backend, inference execution and evaluation pipeline can still vary. The same-seed probe returned two outputs in three calls; the full runs also produced different run-level scores and refusals. We did not isolate each source of variation, so we should not claim a specific root cause. The comparison uses the same case set and fixed scoped index evidence; it does not establish behavior across rebuilt indexes, different datasets or future backend versions.

### 20. How do you report the H08 retry in an interview?

I distinguish operational completion from answer quality: all HTTP queries returned 200, while one separate RAGAS scoring operation encountered a connection error in the initial pass. I preserve the failed record, then label the successful scoring-only retry as recovery. I report the initial 40/40/39 scoring denominators as well as the recovered 40/40/40 table, and disclose that the original failure still occurred. Calling it a model-quality failure or saying the first attempt scored all 120 cases would both misstate the evidence.
