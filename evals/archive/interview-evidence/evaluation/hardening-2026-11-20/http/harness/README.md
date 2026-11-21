# Isolated HTTP evaluation candidate

These scripts evaluate the current project source through its real HTTP API,
tenant authorization, document parser, Chroma and Elasticsearch. The input
fixtures are entirely fictional and author-labeled, with no human review.
The tests are exploratory, not proof of production quality or safety.

`capture_server.py` temporarily instruments the generator in its own process
to record the exact chunks visible to generation. This writes synthetic text
to a private test directory and must not be used with private corpus data.
The application source itself is unmodified. Capture occurs before grounding
verification; output may later be refused by that gate.

`prepare.py` creates a temporary scope and 25 source documents for the 40-case
holdout. `prepare_security.py` creates per-case reader grants for 28 attack,
conflict and benign controls; exact source paths map fixture IDs to ingested
chunks. `run_http.py` and `run_security.py` ingest and query over localhost.
The latter preserves each case's visible source paths and actual chunk IDs;
absence of an attack marker is not a safety verdict.

`ragas_score.py` invokes this repository's real RAGAS evaluation on the
captured evidence and synthetic reference. `judge_security.py` provides a
structured **model** judgment over instruction-following, utility and
attribution, never a human evaluation. `to_dataset.py` emits the project
evaluator contract from holdout observations: actual prompt-visible chunk IDs
are explicit, missing captures fail closed, and planned requests without an
observation remain error records. Source-level relevance does not define every
chunk's relevance, so this adapter does not fabricate retrieval labels.

Set `XAI_EVAL_BUDGET_FILE` to the same private, fresh ledger path and
`XAI_EVAL_MAX_PROVIDER_CALLS` to a positive cap for every process. The shared
ledger counts all OpenAI SDK request attempts from generation, the mandatory
grounding judge, RAGAS and the security judge. OpenAI SDK retries must remain
disabled; the application's explicit generation retries are counted as new
attempts. Each completed call records token usage where the API returns it;
failed requests can still be billable even when no usage is returned.

All runs must use separately owned, disposable stores and loopback ports,
fresh output filenames, a source-tree digest and explicit model/configuration
versions. For the application process set `XAI_EVAL_CAPTURE_PATH`,
`XAI_EVAL_API_PORT`, `XAI_RAG_INGEST_ROOT`, `XAI_RAG_AUTH_POLICY_FILE`,
the chosen disposable store hosts/ports/names, and an explicit grounding judge
model. Use an inherited provider key only through environment or a local
configuration file; never print or copy it into artifacts. On hosts with a
broken inherited proxy, remove proxy variables for that process only.

Record individual errors, timeouts, retrieval misses, support/utility results,
planned and scored denominators, and model/token usage. Judge outputs must be
qualified as uncalibrated model judgments. A full investigation also needs
independently labeled representative cases and human adjudication. Stop and
retain incomplete evidence on resource pressure or repeated provider failures.
