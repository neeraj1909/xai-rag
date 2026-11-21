# XAI-RAG index-write investigation and prototype

Worktree: `/tmp/xai-index-worktree`, baseline `c375903`; no commits and no shared-checkout edits.

## Ready to extract independently

`src/xai_rag/ingestion/store.py` checks Elasticsearch delete-by-query results on both stale-chunk removal and whole-source deletion. A timeout, version conflict, or item failure raises `StorageWriteError`, even when the API returns HTTP 200. `tests/unit/test_store.py` covers each failure for both paths. This detects partial writes; it does not reverse them.

## Prototype only — not integrated

`src/xai_rag/ingestion/journal.py` implements a private host-local fsync-backed source snapshot and write-intent marker with a cooperative Unix flock. Tests demonstrate fresh-instance marker recovery, tamper detection and process exclusion. `source_matches_expected()` compares both stores against the intended ID/hash manifest, because parity alone can accept equal stale indexes. It has a matching-stale regression test. These are building blocks, not an operational recovery feature: production calls do not use them.

Do not merge the prototype as a completed fix. Remaining work: capture bytes through the verified parser and parse exactly the saved bytes; bind an intent and lock to canonical tenant/index/Chroma backend/ES namespace and configuration; integrate fail-closed writes and an explicit operator recovery entry point with exact intent replay; verify intended manifests in both stores before clearing; add interruption-between-stores, actual restart/replay and concurrent-writer tests; configure persistent private local storage; document retention. Do not use a process-local lock as a multi-host guarantee. An already-sent external store write may complete after a client crash; without backend fencing, explicitly establish quiescence before repair, and do not promise atomic transactions.

## Evidence

`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /home/neeraj/Documents/xai-rag/.venv/bin/pytest -q tests/unit/test_store.py tests/unit/test_ingestion_journal.py tests/unit/test_expected_source.py --disable-warnings --maxfail=1`: 14 passed, 1 warning.

`/home/neeraj/Documents/xai-rag/.venv/bin/ruff check src/xai_rag/ingestion/journal.py src/xai_rag/ingestion/store.py tests/unit/test_store.py tests/unit/test_ingestion_journal.py tests/unit/test_expected_source.py`: passed.

Independent critique: `/tmp/xai-db-review.md`. Initial design: `/tmp/xai-index-design.md`.

## Integrated candidate and real-store follow-up (2026-11-20)

The journal and actual-record verification were subsequently wired into the shared checkout following an explicit integration-owner handoff. Configured with `XAI_RAG_INGESTION_JOURNAL_ROOT`, API and CLI writers stage bounded verified bytes, record an intent and intended full-content/metadata manifest, reject pending work, and expose operator-acknowledged authorized replay. Scoped recovery uses current source grants; `validate-indexes` requires a granted source in scoped mode. Without this configuration, the journal is disabled. `docker-compose.journal.yml` is an opt-in overlay requiring a pre-provisioned UID 10001, mode 0700 host directory. `/home/neeraj/Documents/xai-rag/docs/ingestion-recovery.md` documents retention, privacy, assumptions and limitations.

On final targeted integrated code: 44 passed, 2 skipped (auth/store integration tests); Ruff and `git diff --check` passed. New fault tests check denied/async authorization, stale/corrupt content and metadata, incomplete Elasticsearch shard results, duplicate IDs, restart of an OS process and host-local lock exclusion.

Real-store result: attempt 1 failed before a write because the isolated candidate had not yet inherited tenant-aware `get_chroma_collection` signature. Attempt 2 intentionally exited 29 after Chroma, but ES index creation timed out on the default client timeout. Attempt 3 passed after a 60-second client timeout and a transient disk-watermark override **only** on the disposable ES test node: its filesystem had 13.7 GB free but exceeded 90% usage, leaving the tiny synthetic index red. Test used 512 MiB ES heap, loopback ports 18121/19231, Chroma 1.5.9, Elasticsearch 8.17.0, synthetic local 384-dimensional vectors and no provider calls. Containers launched at approximately 10:11:07 UTC; both health endpoints passed at 10:11:55 UTC. Child exited 29 after Chroma and before any ES write; the original source file was changed; a fresh process recovered the saved original source, verified actual text and full metadata, and reported 1 Chroma / 1 ES record, parity consistent. Both disposable containers were removed and test ports were verified closed at approximately 10:12:38 UTC. Evidence and first two failure logs: `/tmp/xai-index-real-evidence/`; harness: `/tmp/xai-index-real-test.py`.

This is not a distributed transaction or a delayed-in-flight-request test. Recovery relies on an operator establishing remote quiescence, a private persistent local journal shared by all writers with identical canonical configured endpoints, and no bypass writers. It cannot detect a lost marker, fence a revocation racing an authorized replay, guarantee parity during another process's batch, validate embedding values, or coordinate across hosts. A pending snapshot is retained until recovery or manual handling; deletion is not secure erasure. Full integrated test suite and final review remain with the integration owner.

## Fsync race follow-up

Independent review identified a race if one process created a namespace and a second observed it before its parent directory entry was synced. The journal now requires an operator-preprovisioned private durable root, creates only the namespace, and unconditionally fsyncs namespace, root, and root parent at each lock acquisition. The runbook explains the durable-root requirement. A subprocess creates the namespace without syncing it and the next writer's fsync calls are observed; a separate test rejects a missing root. Targeted journal/replay tests: 11 passed; Ruff and git diff --check passed. As always this relies on filesystem/storage honoring fsync. The earlier real-store run preceded this small hardening and its test harness requires pre-creating the journal root if repeated.
