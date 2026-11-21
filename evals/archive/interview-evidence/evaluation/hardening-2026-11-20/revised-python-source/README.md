# Python source used in the revised synthetic evaluation

This archive preserves the `src/xai_rag` Python files from the isolated
`/tmp/xai-safety-completeness` candidate used in the revised evaluation.
Its tree digest matches that run's manifest:

```text
0d845a57bbbc9f695c3590407469a8dc352ca49316a1675033040c664a78ddaa
```

Run `python verify_snapshot.py` in this directory to verify the archive without
extracting it. The algorithm hashes each sorted relative Python filename, a
null byte and the SHA-256 digest of that file, matching the evaluation's
manifest script. The archive excludes `.env`, credentials, datasets, other
repository files and dependencies. It records the evaluated Python source,
not a complete deployable environment or a guarantee against model-alias drift.
