# Dense multilingual comparison, 22 November 2026

[Full per-case output](comparison.json) contains exact source-level rankings and
fixture, runner and model metadata. The [active runner](../../multilingual/README.md)
was executed offline against the archived 31-passage, 28-query challenge, using
CPU and two OpenMP/MKL threads. It used the cached
`BAAI/bge-small-en-v1.5` revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
and the local snapshot for
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. The snapshot directory name records
that revision; the runner does not independently verify local snapshot content.
The recorded runner hash matches the active runner at execution.

Of 28 planned queries, 26 have relevance labels. BGE found at least one labeled
source in its top 1 for 22/26 and top 5 for 25/26; multilingual MiniLM achieved
21/26 and 26/26 respectively. All relevant sources were within the top 3 for
24/26 and 26/26 respectively. The two unlabeled queries are not treated as
misses. The output has all rankings and their scores.

The fictional passages, queries and labels were authored together and have no
independent human review. This compares exact embedding similarity, not full
hybrid retrieval, answer quality, abstention, or production performance. Do not
infer model superiority or make an embedding switch from these 26 cases. The
separate [historical report](../../archive/interview-evidence/multilingual-retrieval-diagnostic.md)
retains its original code and measurements. Timings reflect concurrent load and
are not benchmarks.
