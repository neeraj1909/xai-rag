# Python source used in the synthetic HTTP evaluation

The HTTP run was performed on a modified working tree whose Git HEAD was
`c375903fb46327744d076438cb90f7b02711a94f`. That commit alone does **not**
reproduce the run. This archive preserves the evaluated `src/xai_rag` Python
files, reconstructed from the working tree by reversing the later generator
compatibility change and token-description formatting. Its tree digest matches
the run manifest exactly:

```text
6e3c87bd0f650459d490c4f0d2c975f11ee18ce47686284f70014c374dbe9abd
```

To verify without extracting it, run `python verify_snapshot.py` in this
directory. The algorithm hashes each sorted relative Python filename, a null
byte and the SHA-256 digest of that file, matching the evaluation's manifest
script. The source archive excludes `.env`, credentials, test data, other
repository files and runtime dependencies. It is evidence of the Python code
state, not a complete deployable snapshot or a guarantee that provider model
aliases have not changed.
