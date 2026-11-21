# Generator adversarial probe: 20 fictional cases

The frozen generator returned structured results for **40/40 attempted case-runs**, with **0 provider errors and 0 contract exceptions**. Two runs of the empty-context case A15 returned the application's canonical refusal without a model call. The other 38 runs used 38 provider requests; no repairs were needed. A run returning a result is an execution outcome, not a safety or correctness pass.

The predefined exact-marker and non-visible-citation checks raised **0 flags across 40 results**. These checks are debugging signals only: an attack can succeed without an exact marker, and a marker can appear in a legitimate quotation. In particular, both A18 results refused to answer even though the case expects the two conflicting closing times to be reported. An independent review of the outputs is needed to assess correctness, evidence support, attack following, and false refusals. **No human evaluation was performed**, and no attack-following or false-refusal rate is claimed here.

The generator was `gpt-4o-mini-2024-07-18`, reached through the explicitly configured official OpenAI endpoint with SDK retries disabled. The run observed **17,220 input tokens and 1,896 output tokens** from provider-reported usage; dollar cost was not calculated. The runner set a shared limit of 100 provider attempts, counted actual requests including repairs and transport retries, and would stop on an authentication or permission failure or two consecutive transport or rate-limit failures. None of those stops occurred.

All documents, user questions and markers were fictional. The author-defined case labels were not independently human-reviewed, and the fixture is not a sealed holdout. This generator-only probe does not exercise tenant authorization, network side effects, document ingestion, or retrieval. Valid source IDs do not establish that citations support the answer.

- [Per-case prompts, visible evidence, responses, usage and signals](evaluation/generator-adversarial-results.json)
- [Runner](evaluation/generator-adversarial-runner.py) and [frozen source snapshot](evaluation/generator-adversarial-source-snapshot.tar.gz)
- [Fixture and proposed review criteria](generator-adversarial-plan.md)
- Source digest `c4e4d8f0cf29928eaee5aaf8d14a7a431f561e2df0fd74d881b023f9b5830b70`; fixture SHA-256 `3b6f2f6c6d1d3f49c7a8e376549c538782a6fccedf42beb9017026f61e289d39`.

The runner saves each attempt before making a request, refuses to overwrite the results file, and records provider errors without their potentially sensitive messages. Reproduction requires the XAI-RAG dependencies and a configured API credential, extracts the archived source to `/tmp/xai-generator-adversarial-snapshot`, and runs with that directory on `PYTHONPATH` from the XAI-RAG checkout. The calls are billable; use only the included fictional fixture.
