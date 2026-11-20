# Fictional citation-aware judge sensitivity

`cases.json` is a byte-identical copy of the 16 authored cases in
`../archive/interview-evidence/evaluation/judge-fixture-cases.json`. The cases
and expectations are synthetic, not independent human labels or a held-out
benchmark. The judge is asked to score a supplied answer; this does **not**
run or assess the application's generation or grounding judge. Existing
`tests/unit/test_grounding_judge.py` tests a different interface with controlled
responses. Retain the historical results and runner in the archive.

From the repository root, using the project OpenAI credential in `.env` or
`XAI_RAG_OPENAI_API_KEY`, run:

```bash
mkdir -p artifacts
uv run --no-sync python evals/judge/run.py --model gpt-4.1-mini \
  --output artifacts/judge-sensitivity.json
```

Each run creates its output exclusively and writes each case as it completes;
it refuses to overwrite existing evidence. It records the fixture hash, rubric,
response schema, requested and returned models, token usage if available,
author expectations, per-case judgments, errors and the planned denominator.
Expected labels are never sent to the judge. Report failure counts separately;
a failed request is not a pass or a zero score. The exit code is nonzero if any
case could not be judged. The model and prompt can give inconsistent results;
a 0-temperature model-judged score is not human review.

An alternative using the existing authenticated Codex CLI is:

```bash
uv run --no-sync python evals/judge/run.py --backend codex --model gpt-5.6-luna \
  --output artifacts/judge-codex.json
```

The Codex backend runs one ephemeral read-only agent turn per case with the
same fictional case and schema, prompts it not to call tools, and clears proxy
variables only in the child process. It has an agent instruction context and
CLI transport rather than the API's chat-completions interface, and may use
different model controls (temperature and token limits are **not** passed to
the CLI). The CLI records usage where available but does not return a model
revision. Results are a separate experiment and cannot reproduce or be directly
compared as a controlled paired study with the original API experiment. Usage
may incur costs according to the account's terms.

[Results for 22 November 2026](../results/2026-11-22-judge/) include failed
OpenAI API attempts and a separately conducted Codex CLI run.
Test the runner without model calls with:

```bash
uv run --no-sync pytest -q evals/judge/test_run.py
```
