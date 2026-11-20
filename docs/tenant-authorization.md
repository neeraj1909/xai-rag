# Tenant and document authorization

By default, XAI-RAG retains its existing single-user API key and shared indexes. That mode does not provide tenant or document isolation. For multiple users or private document sets, configure `XAI_RAG_AUTH_POLICY_FILE` to an absolute path controlled by the server operator. A missing or invalid configured file prevents startup; it never falls back to the shared API key.

The JSON file contains credential digests and exact, case-sensitive paths relative to `XAI_RAG_INGEST_ROOT`:

```json
{
  "principals": [
    {
      "key_sha256": "replace-with-lowercase-sha256-hex-of-a-strong-api-key",
      "tenant": "team-a",
      "sources": ["manuals/product.txt", "policies/refunds.pdf"],
      "role": "reader"
    }
  ]
}
```

Generate a separate random API key of at least 32 UTF-8 bytes for each principal, calculate its SHA-256 digest locally, and replace the example placeholder with the 64 lowercase hexadecimal characters. Store and distribute the original key securely; send it as the `X-API-Key` header. Do not commit the policy file or credentials. Each credential is unique, and its server-defined scope—not a tenant supplied in an HTTP request—determines access. Allowed sources are an explicit list; an empty list grants no document access. `reader` can query; `ingester` can query and ingest only its allowed sources. A directory request selects granted paths under that prefix without enumerating ungranted files. Missing or invalid granted files fail individually.

Each tenant gets its own Chroma collection and Elasticsearch index; both retrieval queries also apply source filters before top-k selection. The service rejects unexpected sources before they reach fusion or generation. Existing shared-index documents do not automatically migrate into a tenant: re-ingest approved documents with an authorized ingester. An index/collection is not deleted when a principal is removed. Policy changes take effect only after restarting every application replica; protect access to the policy and document root.

This is a static credential-and-grant mechanism, not an identity provider, distributed access-control service, or online revocation system. Keep Chroma and Elasticsearch inaccessible to clients and restrict their administrative credentials: direct store access bypasses application checks. Secure transport, secret rotation, separate environments, audit policy, backups, deletion and retention procedures, and a production identity provider remain deployment responsibilities. The ingestion lock is process-local; multiple application workers need external write coordination for the same tenant and source. Treat documents within a tenant as visible to a principal only through its explicit grants, but do not put data in indexes if operators or storage administrators are not trusted.

On POSIX systems the service opens each component without following symlinks and parses a private snapshot. The ingestion root, its ancestors, and source inodes must still be controlled by trusted operators: same-UID processes, hardlinks, and concurrent writes to an already-open inode are not prevented by this staging mechanism. The parser subprocess has resource bounds but is not an OS security sandbox.

For Compose, place the policy file in an operator-owned host directory, set `XAI_RAG_AUTH_POLICY_MOUNT` to that directory and `XAI_RAG_AUTH_POLICY_FILE` to its container path (for example `/run/xai-rag-policy/policy.json`). The directory is mounted read-only; an invalid configured path prevents startup. In policy mode the CLI query command is disabled; use the authenticated HTTP API. CLI ingest and recovery require an ingester credential in `XAI_RAG_INGEST_API_KEY` and the configured ingestion root; index validation also requires an explicitly granted source. See [ingestion recovery](ingestion-recovery.md).
