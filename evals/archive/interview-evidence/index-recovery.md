# XAI-RAG: detecting and recovering index drift

> **Version note:** This document describes the earlier `c375903` or explicitly cited historical source. Statements here that call NLI optional, say document authorization is absent, or describe recovery as only a proposal do not describe the later hardening work. Read the [current hardening questions and answers](hardening-interview-qa.md) first, and verify the source revision before an interview. Historical measurements remain tied to their stated snapshots.

## What can go wrong?

The source ingestion path writes Chroma first and Elasticsearch second. The two stores do not share a transaction. If a request is cancelled while a Chroma thread is writing, revision `c375903` waits for that thread to finish before releasing its per-source lock; cancellation then propagates before the Elasticsearch write and before the final parity check. The caller does not receive a successful ingestion result, but the stores can contain different versions.

A controlled local probe against the current `RAGService`, with temporary source data and in-memory fake store adapters, observed `Chroma=new`, `Elasticsearch=old`, and zero parity calls after cancellation. Replaying the same source converged those fake stores and the subsequent parity check passed. This tests the application control path, not crash durability or a real distributed recovery guarantee.

## What can an operator do today?

1. **Quiesce all ingestion writers first**, including other processes and scheduled jobs. The parity checker reads the two stores separately, and Elasticsearch pagination is not a cross-store snapshot. Checking during writes can report temporary drift or miss a change between reads. The current in-process lock does not coordinate other processes.
2. Retain the ingestion error, source identity, intended document version, chunking strategy, embedding revision, index configuration, and application revision. Establish which source version is authoritative before replaying. Treat a timed-out Elasticsearch write as an **unknown outcome**; it may have succeeded remotely.
3. From the XAI-RAG checkout, with the intended store configuration loaded, run:

   ```bash
   uv run xai-rag validate-indexes --source 'relative/source/path.txt' --json-output
   ```

   The `--source` value is the application's relative `source_file` identity, not an arbitrary absolute path. Omit it to inspect all sources. A nonzero exit or a failed dependency check requires investigation; an unavailable checker is not evidence of consistency.
4. If the approved authoritative document is available, replay ingestion for that document using its original strategy and chunking settings. For example, only when these are the intended settings:

   ```bash
   uv run xai-rag ingest ./sample_docs/policy.txt --strategy semantic
   uv run xai-rag validate-indexes --source 'policy.txt' --json-output
   ```

   **CLI identity caveat:** the CLI uses the selected file's parent as its document root. A single-file replay of `sample_docs/nested/policy.txt` therefore uses `policy.txt`, not `nested/policy.txt`, and can create an additional logical source. To preserve a nested identity, use an ingestion path that retains the original root, or replay the original directory after reviewing the scope and its effect on other files. The example above applies only when `policy.txt` was originally indexed at the root. Replaying the current file when an earlier version was intended can silently index a different version. A source that has been deleted cannot be recreated by replaying a missing file; use an approved deletion or rebuild procedure instead of inventing a replacement.
5. Before resuming writes, inspect the parity report and run representative source and retrieval checks. The checker compares chunk IDs and stored content hashes, not every vector value, actual source text, authorization policy, or answer. A consistent report is not proof that either store represents the intended document or that indexing is complete.

## What would make recovery durable?

The smallest reliable production design depends on deployment and consistency requirements, but it needs a durable source of truth beyond the two independent search indexes. One proposed design is a durable ingestion ledger keyed by a stable source identity **and document/version identifier**. Record intent before either write; record each stage, failure, and unknown remote outcome; retry idempotent operations; and verify the intended version in both stores before marking the job complete. A background reconciler revisits incomplete jobs after process death. Store the source version or immutable artifact reference needed for replay, and handle deletion as a versioned operation.

For multiple workers or hosts, enforce per-source ownership using a shared lease with fencing or another coordination mechanism whose failure semantics are explicitly tested. Fencing only works if stale writes are rejected or isolated at the storage/activation boundary; checking a token solely before sending a request does not stop an old request arriving late. A local `asyncio.Lock` neither survives restart nor prevents a second process from writing. Stronger serving guarantees may also require versioned indexes and an atomic active-version pointer so queries never mix partial generations; that is a design proposal, not an existing XAI-RAG guarantee.

Test crashes before and after each write, cancellation while a worker is active, remote success followed by client timeout, duplicate delivery, changed and deleted source versions, concurrent writers in separate processes, stale lease owners, and restore from a saved artifact. Confirm that reconciliation eventually converges to the authorized intended version and that incomplete states are visible. These tests are not established by the local fake-adapter probe above.

**Sources:** [Ingestion orchestration at baseline](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/service.py), [store and parity implementation](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/ingestion/store.py), and [CLI](https://github.com/neeraj1909/xai-rag/blob/f4c374d51d090beba61d34aba81fcc23b8ef700a/src/xai_rag/cli.py). The process-local lock discussed above is included in [published source `c375903`](https://github.com/neeraj1909/xai-rag/tree/c375903fb46327744d076438cb90f7b02711a94f); the baseline links above show earlier code.
