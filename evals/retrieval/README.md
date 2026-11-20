# Replay saved retrieval rankings

This offline diagnostic applies the **current** application RRF implementation to
saved top-five multilingual dense and BM25 rankings from the archived lexical
experiment. It uses the active evaluation metrics to report Hit@1, Hit@3 and
Hit@5 overall and by category, with sample counts and per-case rankings. It
rejects missing labels and failed branch observations instead of counting them
as retrieval misses. It reads its input and writes JSON to stdout; it makes no
model, database or provider calls.

From the repository root:

```bash
uv run python evals/retrieval/replay.py \
  evals/archive/interview-evidence/evaluation/holdout-lexical-results.json \
  > /tmp/xai-rag-retrieval-replay.json
uv run pytest -q evals/retrieval/test_replay.py
```

The output includes the input SHA-256. Source IDs stand in for chunk IDs, and
current RRF breaks equal-score ties by ID. Its rankings may differ from the
historical experiment's original implementation. This is a replay of saved branch
rankings, **not** a fresh retrieval, embedding, reranker or live-model evaluation.
The source's 40 synthetic, author-labelled cases are exploratory and not an
independently reviewed or representative benchmark. Historical results and
original runner remain unchanged in the archive.
