# XAI-RAG: NLI diagnostic probe

This is a small, exploratory model run on 19 November 2026, not a benchmark or human evaluation. It uses ten authored fictional premise–claim pairs, including two raw/escaped pairs. We assigned expected labels ourselves; there is no independent human annotation, representative corpus, or confidence interval. The project source was based on commit [`f4c374d`](https://github.com/neeraj1909/xai-rag/tree/f4c374d51d090beba61d34aba81fcc23b8ef700a), with the then-uncommitted `FaithfulnessChecker` copied to an isolated directory. That file's SHA-256 was `d26495e4e23ccfaebcebc8aa5f6922c47fa52b430b6cccca65175bdd20302c63`.

## Method

The project's actual `FaithfulnessChecker._check_single_claim` loaded **`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`**, pinned revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`, with the default 0.5 entailment and contradiction thresholds and 512-token truncation. Each claim pointed to exactly one synthetic evidence chunk. The checkpoint was downloaded from Hugging Face using a process-local network configuration; no OpenAI API was called. Full premises, claims, NLI outputs, per-case times and configuration are in [the raw JSON](evaluation/nli-evaluation.json).

## Observed outcomes

| Synthetic case | Expected label | Observed verdict | Entailment | Contradiction |
| --- | --- | --- | ---: | ---: |
| Receipt rule, exact | Supported | Supported | 0.9883 | 0.0015 |
| Password sharing, negated | Not supported | Not supported | 0.0003 | 0.9990 |
| Refund days, 14 vs 30 | Not supported | Not supported | 0.0019 | 0.9946 |
| Refund missing from handbook | Neutral | Neutral | 0.0003 | 0.0047 |
| `x < 5`, claim “x is below 5” | Supported | Supported | 0.9878 | 0.0011 |
| `x &lt; 5`, same claim | Supported | **Neutral** | 0.2024 | 0.4727 |
| `R&D`, exact claim | Supported | Supported | 0.9951 | 0.0009 |
| `R&amp;D`, same claim | Supported | Supported | 0.9531 | 0.0299 |
| Not confirmed vs not ready | Neutral | Neutral | 0.4468 | 0.0500 |
| 14 units, exact | Supported | Supported | 0.9966 | 0.0005 |

**Why the raw/escaped comparison matters:** In the first staged, then-uncommitted snapshot, the guardrail escaped retrieved text when building its prompt and passed that literal prompt-visible text to NLI. This preserved the outer tag structure and aligned the checked text with what the answer model saw. In the `x < 5` pair, literal HTML-escaped input changed the NLI verdict from supported to neutral. The `R&D` pair did not change verdict, although its scores changed. These two pairs do not establish how often this happens or which representation is generally superior.

## Follow-up: decode only the visible evidence

A second staged snapshot changed the diagnostic to decode the already-selected, already-truncated prompt evidence **once** before NLI. It does not recover omitted chunks or omitted text. The generator still sees the escaped prompt; this is a semantic normalization for NLI, not the exact literal string sent to the generator. RAGAS continues to receive the escaped representation. A literal entity in a source, such as the characters `&lt;`, is escaped when formatting and returns to those original characters after one decode; it must not be decoded a second time.

An independent agent reran four synthetic pairs on the pinned checkpoint. These are **raw entailment probabilities**, rounded to four decimals—not calibrated accuracy, NLI verdict accuracy, or an end-to-end answer evaluation:

| Source and claim | Literal escaped evidence | One-pass decoded visible evidence |
| --- | ---: | ---: |
| `x < 5`; “x is below 5” | 0.5278 | 0.9917 |
| Pressure `> 10` bar; “pressure exceeds 10 bar” | 0.0575 | 0.9937 |
| Alice `&` Bob signed; “Alice and Bob signed” | 0.9980 | 0.9980 |
| Source literally displays `&lt;`; claim repeats that literal text | 0.9888 | 0.9917 |

In the pressure example, the literal escaped evidence also received a contradiction score of 0.7202; the decoded form received 0.0007. With the configured 0.5 thresholds, that single example would move from `not_supported` to `supported`. The first comparison's `x < 5` text and claim were worded differently, so its numerical values are a separate observation rather than a rerun of the same pair. The [raw JSON](evaluation/nli-evaluation.json) retains every premise, claim and score, as well as the runner and tested source patch. These four cases were selected for this diagnosis and were not independently human-reviewed. They show a plausible failure mode and an improvement on these examples, not a general quality gain or proof of prompt-injection resistance.

The NLI probabilities and thresholded labels are model outputs, not calibrated probabilities that an answer is true. The expected labels were created for this probe and are not independent human judgments. A faithful claim may also rely on an outdated or wrong source. The observed inference times were measured in a shared host with one CPU thread and other work running; do not present them as production latency.

## Reproduce

From the XAI-RAG source checkout, with the pinned model revision cached, set `INTERVIEW_REPO` to the absolute path of this interview-materials checkout and run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
from xai_rag.explainability.faithfulness import FaithfulnessChecker
from xai_rag.models import Claim, RankedResult

fixture = Path(os.environ['INTERVIEW_REPO']) / 'projects/xai-rag/evaluation/nli-evaluation.json'
checker = FaithfulnessChecker()
for case in json.loads(fixture.read_text())['cases']:
    verdict = checker._check_single_claim(
        Claim(text=case['hypothesis'], source_chunk_ids=['c1']),
        [RankedResult(id='c1', content=case['premise'])],
    )
    print(case['id'], verdict.verdict, verdict.nli_entailment, verdict.nli_contradiction)
PY
```

Record the model revision, thresholds, code hash, environment, and any errors. Do not silently substitute another NLI checkpoint. Review a larger, independently labelled set—especially numbers, negation, HTML encoding, parent/child chunks and truncated context—before deciding how to normalize evidence for diagnostics or setting release thresholds.

To reconstruct the second staged comparison without changing the source checkout, set `XAI_SOURCE` and `INTERVIEW_REPO` to the absolute paths of the source and interview-materials repositories:

```bash
NLI_PROBE_DIR=$(mktemp -d)
git -C "$XAI_SOURCE" archive f4c374d | tar -x -C "$NLI_PROBE_DIR"
NLI_PROBE_DIR="$NLI_PROBE_DIR" "$XAI_SOURCE/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

fixture = Path(os.environ['INTERVIEW_REPO']) / 'projects/xai-rag/evaluation/nli-evaluation.json'
comparison = json.loads(fixture.read_text())['normalization_comparison']
directory = Path(os.environ['NLI_PROBE_DIR'])
(directory / 'source.diff').write_text(comparison['tracked_three_file_patch'])
(directory / 'src/xai_rag/async_utils.py').write_text(comparison['required_untracked_async_utils_py'])
(directory / 'run.py').write_text(comparison['runner'])
PY
git -C "$NLI_PROBE_DIR" apply "$NLI_PROBE_DIR/source.diff"
cd "$NLI_PROBE_DIR"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  PYTHONPATH="$NLI_PROBE_DIR/src" "$XAI_SOURCE/.venv/bin/python" run.py
```

The JSON records the three-file patch hash and the captured helper file separately. The helper is imported by the checked module but is not called by this synchronous score probe. Exact floating-point results can vary with the environment; retain errors and output when rerunning. Remove the temporary directory after review.
