# Archived runner comparison

Inventory date: 2026-11-22. Paths in the first table are relative to `evaluation/`; paths in the second are relative to `evaluation/hardening-2026-11-20/`. This covers all 57 archived `.py` and `.sh` files, including helpers and fixture builders, not just executable entry points. The `evaluated-python-source.zip` and `revised-python-source.zip` snapshots and all saved results are evidence and are outside this file count.

**Disposition:** retain every listed original as historical evidence until its measurement provenance is checked individually. “Adapt” identifies a possible reusable active evaluation; it does not mean an equivalent active test already exists or that the old script is safe to run. Never overwrite historical outputs. In particular, [the HTTP archive](evaluation/hardening-2026-11-20/http/README.md) warns that its original harness uses `/tmp` paths, can overwrite files or manipulate resources, and can incur charges. [Local reproduction notes](local-reproduction.md) identify further machine-specific runners. Existing model-based experiments are historical; a replacement needs explicit, suitable model access and resource isolation.

Counterpart keys below identify **overlapping** active code, not complete equivalence:

- **Eval**: `src/xai_rag/evaluation/{models,metrics,runner}.py` and `tests/unit/test_evaluation_{metrics,runner,cli}.py` score and test captured observations; they do not capture a live retrieval run.
- **Retrieval**: `src/xai_rag/retrieval/` and `tests/unit/test_retrieval_contract.py` exercise retrieval algorithms and ranking contracts, not all fixture/model comparisons.
- **Generation**: `tests/unit/test_generator_contract.py` and `tests/unit/test_generation_repair.py` cover controlled citation, untrusted-input and repair behavior, not the recorded live-provider results.
- **Judge**: `tests/unit/test_grounding_judge.py` covers controlled schema, untrusted inputs and failure handling; it does not duplicate historical model judgments or prove safety.
- **NLI**: `tests/unit/test_faithfulness.py` and `tests/unit/test_nli_normalized_evidence.py` test controlled faithfulness behavior, not the real-model stress comparisons.
- **RAGAS**: `src/xai_rag/evaluation/ragas_eval.py` and `tests/unit/test_ragas_adapter.py` cover the integration and controlled client handling; `evals/repeated/ragas_score.py` supports a later repeated evaluation.
- **HTTP**: `tests/e2e/test_http_workflow.py` exercises HTTP workflow with controlled dependencies; `evals/repeated/` runs a later repeated HTTP evaluation. Neither automatically reproduces the historical full security suite.
- **Recovery**: `tests/unit/test_index_recovery.py` covers interrupted writes and replay; the original real-store experiment tests an additional boundary under historical conditions.
- **Transport**: `evals/repeated/test_transport.py` and `tests/unit/test_readiness_timeout.py` cover particular transport and dependency timeout behavior, not the archived timing experiment.

## Original interview experiments

| Archived file | Active overlap | Distinct behavior and disposition |
|---|---|---|
| `cdp-api-stub.py` | HTTP | Historical API stub used by browser-oriented checks; retain as experiment support, not equivalent to real pipeline. |
| `conflict-extended-runner.py` | Generation | Paired prompt comparison over extended frozen cases and source snapshots; retain exact runner and results. |
| `conflict-prompt-runner.py` | Generation | Paired baseline/candidate prompt test using original `/tmp` snapshots and model; retain. |
| `focused-repair.py` | Generation, Judge | Repeats cases against repair snapshot and imports historical judge rubrics; retain. |
| `generation-judge-smoke.py` | Generation, Judge, HTTP | Synthetic retrieval/answer experiment with historical generator and judge models; retain. |
| `generator-adversarial-runner.py` | Generation | Frozen-source adversarial generator run with model-specific calls; retain. |
| `guardrail-judge-smoke.py` | Generation, Judge | Historical generator-only cases and rubric; expressly excludes ACL and context-cutoff cases; retain. |
| `h05-generation-diagnostic.py` | Generation | H05 against frozen source and absolute original paths; retain. |
| `h05-paired-diagnostic.py` | Generation | Paired H05 runs, fixture/source hashes and shared historical budget file; retain. |
| `holdout-lexical-comparison.py` | Retrieval | Archived Elasticsearch BM25 plus saved dense rankings and RRF; uses fixed disposable port and archived application source; retain; adapt only with equivalent isolation. |
| `holdout-runner.py` | Eval, Retrieval | Live four-stage retrieval, per-query rankings, multi-source and no-answer handling; 25/40 fixture-specific self-test. Retain original; adapt useful fixture and evaluation path to active suite. |
| `holdout-weighted-rrf-runner.py` | Retrieval | Odd/even split and weighted grid over saved lexical rankings; pinned input hash and exclusive output creation; retain selection procedure. |
| `http-e2e-runner.py` | HTTP | Historical end-to-end HTTP experiment uses absolute paths and can overwrite output; retain, do not run in place. |
| `judge-fixture-run.py` | Judge | Distinct authored fixture, rubric, historical `.env` location and live provider; retain. |
| `multilingual-retrieval-comparison.py` | Retrieval | CPU embedding-model comparison against authored language fixtures and selected model revisions; retain; consider adapting for offline cached-model use. |
| `nli-stress-runner.py` | NLI | Real configured NLI per-case stress evaluation with fixture and model provenance; retain; consider adapting with explicit fixture/output and cached-model checks. |
| `outage-runner.py` | HTTP, Recovery | Synthetic local Elasticsearch outage, separate containers and recorded response behavior; absolute paths and fixed ports; retain. |
| `ragas-default-client-smoke.py` | RAGAS | Live test of owned default RAGAS client and request cap; imports archived runner; retain. |
| `ragas-live-runner.py` | RAGAS | Live RAGAS on saved fictional inputs with historical settings; retain exact input-to-result procedure. |
| `readiness-keepalive-diagnostic.py` | Transport | Long-running local client/server keepalive timing experiment, not app load; writes fixed `/tmp` output; retain. |
| `rejudge-guardrails.py` | Judge | Rejudges saved answers without regeneration using a distinct historical rubric; retain. |
| `retrieval-smoke.py` | Eval, Retrieval | Historical eight-question retrieval run with specific one-relevant-passage assumptions; retain; do not substitute for holdout aggregation. |
| `soak-es-fault.py` | HTTP | Fault injection into the historical soak and append to its event log; fixed names/paths; retain with soak runner. |
| `soak-recovery-check.py` | HTTP, Recovery | Post-soak recovery check bound to historical paths and resources; retain with soak runner. |
| `soak-runner.py` | HTTP, Recovery | Historical timed workload with local stub, stores, faults and resource observations; fixed paths/ports; retain. |

## 2026-11-20 hardening experiments and helpers

| Archived file | Active overlap | Distinct behavior and disposition |
|---|---|---|
| `coverage-judge/run.py` | Judge | Bounded live judge-coverage experiment with old `/tmp` fixture and model alias; retain. |
| `evaluated-python-source/verify_snapshot.py` | None identified | Verifies reconstructed evaluated Python ZIP against the measured digest; retain with ZIP and README. |
| `http/fixtures/create_suite.py` | HTTP, Judge | Deterministically creates the 28-case security/utility fixture; retain original; candidate for a portable fixture workflow. |
| `http/harness/budget_worker.py` | `evals/repeated/call_budget.py`, `evals/repeated/test_process_safety.py` | Historical budget subprocess helper; retain as part of measured harness. |
| `http/harness/call_budget.py` | `evals/repeated/call_budget.py` | Historical shared provider-call ledger; active version differs; retain exact measured source. |
| `http/harness/capture_server.py` | `evals/repeated/capture_server.py` | Captures prompt-visible evidence in historical server process; active version differs; retain. |
| `http/harness/combine_ragas.py` | RAGAS, `evals/repeated/summarize_repeats.py` | Combines separately preserved scoring rounds without hiding failures; retain original extension procedure. |
| `http/harness/judge_security.py` | Judge, HTTP | Structured model judgment of synthetic security outputs and historical fixture; retain. |
| `http/harness/manifest.py` | `evals/repeated/manifest.py` | Historical source/config/fixture manifest; active version differs; retain. |
| `http/harness/prepare.py` | `evals/repeated/prepare.py` | Historical holdout preparation from a fixed `/tmp` fixture; active version differs; retain. |
| `http/harness/prepare_security.py` | HTTP | Builds per-case source grants for a hash-pinned 28-case security fixture; retain; adapt if moving suite. |
| `http/harness/ragas_score.py` | RAGAS | Historical HTTP evidence scoring adapter; differs from repeated-suite version; retain. |
| `http/harness/run_all.sh` | `evals/repeated/run_repeats.sh` | Original full orchestration of disposable stores, HTTP, security and RAGAS; hard-coded paths and model; retain, do not run in place. |
| `http/harness/run_http.py` | HTTP, `evals/repeated/repeat_http.py` | Historical per-case holdout ingestion/query driver; fixed fixture path and experiment semantics; retain. |
| `http/harness/run_security.py` | HTTP | Case-isolated attacks and benign controls against real API/stores; no equivalent full active suite identified; retain; consider adapting. |
| `http/harness/selftest_http.py` | HTTP | Self-test for original HTTP driver and old `/tmp` layout; retain alongside original driver. |
| `http/harness/selftest_security.py` | HTTP | Self-test for original security driver and old `/tmp` layout; retain alongside original driver. |
| `http/harness/summarize.py` | Eval, `evals/repeated/summarize_repeats.py` | Summarizes original security and model outcomes; different experiment from repeated run; retain. |
| `http/harness/to_dataset.py` | Eval | Converts original HTTP captures to evaluator JSONL, including planned/missing cases and prompt visibility; retain; consider adapting a portable version. |
| `judge-question-candidate/baseline.py` | Judge | Frozen baseline prompt/schema source for paired comparison; retain with results. |
| `judge-question-candidate/build_fixture.py` | Judge | Builds historical question-aware fixture from fixed `/tmp` inputs; retain. |
| `judge-question-candidate/candidate.py` | Judge | Frozen candidate prompt/schema source for paired comparison; retain with results. |
| `judge-question-candidate/run.py` | Judge | Paired judge experiment with explicit live flag and historical fixture/model; retain. |
| `judge/run.py` | Judge, NLI | Historical judge cases, fixed original paths and `gpt-6-luna` alias; retain. |
| `judge/summarize.py` | Judge, NLI | Historical comparison of judge and local NLI decisions using original output paths; retain. |
| `load/runner.py` | HTTP | Bounded local synthetic load and resource measurements with disposable services, not a production load test; retain. |
| `nli-diagnostic/attack_pairs.py` | NLI, Judge | Derives original attack pairs from a fixed `/tmp` fixture; retain. |
| `nli-diagnostic/run.py` | NLI | Compares pinned English and multilingual NLI on same fixture; retain; consider adapting if both models are locally available. |
| `recovery/runner.py` | Recovery | Real-store crash/replay diagnostic and original journal workflow; retain. |
| `revised-python-source/verify_snapshot.py` | None identified | Verifies revised archived Python ZIP against its distinct digest; retain with ZIP and README. |
| `safety-coverage/run.py` | Judge | Historical relevance, safety and coverage probe with fixed fixture/model; retain. |
| `transport-boundary/experiment.py` | Transport | Local pooled-versus-close HTTP timing comparison, fixed output path; retain. |

## Coverage and execution gaps

This is a source comparison, not a claim that these evaluations have been rerun. Most unit tests use controlled inputs or dependencies and do not establish equivalence to recorded live-model or real-store results. Before promoting an evaluation, verify fixture compatibility, model availability, fresh output handling, current APIs, resource ownership and cleanup; report unavailable model or service validation rather than substituting a historical score. Historical snapshots, reports, fixtures, per-case observations, prompts and runner versions collectively explain the measurements and should remain linked.
