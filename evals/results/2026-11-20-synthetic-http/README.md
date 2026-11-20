# Synthetic HTTP and RAGAS evaluation — 20 November 2026

This record contains fictional, author-labelled data; it has not been
independently human-reviewed and is not a production benchmark.

| File | Contents |
| --- | --- |
| `holdout-cases.json` | 40-question fixture; SHA-256 `b666492a41e218d6fefa0fcca0fa6ab26277bfb7dfba34413ffadb672afdc8cf` |
| `holdout-http.json` | HTTP observations and prompt-visible evidence from the 40 cases |
| `combined-ragas.json` | Per-case RAGAS scores, including the initial run and extension, and aggregate summary |

The original HTTP run used real Chroma and Elasticsearch, local embedding,
reranking and NLI models, and the `gpt-6-luna` alias for generation and
judging. RAGAS used the same model alias and `text-embedding-3-small` for
answer relevancy. All 40 HTTP requests returned 200; five of 27 cases labelled
directly answerable were refused with relevant evidence visible. Scores are
model-generated, not human judgments.

The initial attempt stopped after three HTTP 400 responses because the model
rejected a legacy token parameter. After a compatibility fix, a 350-attempt cap
left the first RAGAS pass with 15 computed cases, one cap error and 24 unscored.
A separately budgeted extension scored those 25 outstanding cases. The combined
file retains the original and extension labels; its `original` and `extension`
paths are historical `/tmp` locations, not executable instructions.

The run's Git HEAD was `c375903fb46327744d076438cb90f7b02711a94f` with
uncommitted source changes. Its Python source-tree SHA-256 was
`6e3c87bd0f650459d490c4f0d2c975f11ee18ce47686284f70014c374dbe9abd`.
The later integrated judge includes additional safety and coverage checks;
these historical scores do not evaluate that newer source. The original live
harness used temporary paths and is not a portable, self-contained rerun. Use
the command in the [project README](../../../README.md#reproduce-the-evaluation)
to recompute the saved aggregate, and treat a fresh live run as a new experiment.
