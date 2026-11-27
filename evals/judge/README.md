# Fictional citation-aware judge sensitivity

`cases.json` is a byte-identical copy of the 16 authored cases in
`../archive/interview-evidence/evaluation/judge-fixture-cases.json`. The cases
and expectations are synthetic, not independent human labels or a held-out
benchmark. The judge is asked to score a supplied answer; this does **not**
run or assess the application's generation or grounding judge. Existing
`tests/unit/test_grounding_judge.py` tests a different interface with controlled
responses. Retain the historical results and runner in the archive.

## Results

| Judge and interface | Faithfulness, all | Faithfulness, factual answers | Correctness, all |
| --- | ---: | ---: | ---: |
| GPT-4.1 Mini, OpenAI API | 12/16 | 11/14 | 16/16 |
| GPT-5.6 Luna, Codex CLI | 16/16 | 14/14 | 16/16 |

Both runs judged all 16 cases against the same authored labels. The two
remaining cases contain no factual claim and expect a null faithfulness score;
GPT-4.1 Mini matched one of those two. Its other faithfulness disagreements
were J02, J07 and J12: each received 0 where the expected mixed-support score
was 1. The [API judgments](../results/judge-sensitivity/openai-judgments.json)
and [CLI judgments](../results/2026-11-22-judge/codex-judgments.json) include
case-level reasons and usage. The API returned `gpt-4.1-mini-2025-04-14`; the
CLI did not report a model revision. The API run used the runner at commit
`86f094a` and the fixture hash recorded in its report.

The cases and labels are author-created rather than independent human review.
The interfaces and model controls differ, so this is an exploratory comparison,
not a controlled benchmark or a measure of production quality.

## Run

From the repository root, using the project OpenAI credential in `.env` or
`XAI_RAG_OPENAI_API_KEY`, run:

```bash
mkdir -p artifacts
env -u OPENAI_BASE_URL uv run --no-sync python evals/judge/run.py --model gpt-4.1-mini \
  --output artifacts/judge-sensitivity.json
```

This command selects OpenAI's default API endpoint for this process. Omit
`env -u OPENAI_BASE_URL` when intentionally using a compatible custom endpoint.

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

Test the runner without model calls with:

```bash
uv run --no-sync pytest -q evals/judge/test_run.py
```
