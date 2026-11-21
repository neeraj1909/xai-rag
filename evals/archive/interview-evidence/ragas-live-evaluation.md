# RAGAS evaluation on saved synthetic answers

The project already has an optional RAGAS integration. This experiment ran its
actual `evaluate_response` function with **RAGAS 0.4.3**, real collection metrics,
and a counted OpenAI client. It scored four saved synthetic answers from the
historical `f4c374d` pipeline and one deliberately incorrect answer. It did not
run retrieval or generate new XAI-RAG answers in this experiment.

| Case | Faithfulness | Answer relevancy | Context precision |
| --- | ---: | ---: | ---: |
| q01 | 1.000 | 0.739 | 1.000 |
| q02 | 1.000 | 0.756 | 1.000 |
| q03 | 1.000 | 0.901 | 1.000 |
| q04 | 1.000 | 0.764 | 1.000 |
| “Within 365 days” control | 0.500 | 0.739 | 1.000 |
| Explicitly permitted on day 365 (separate control) | 0.000 | 0.512 | 1.000 |

The four saved answers averaged **1.000 faithfulness, 0.790 answer relevancy,
and approximately 1.000 context precision**; denominator is four, and the
controls are excluded. The first five case evaluations returned 15 scores. The
first control said “within 365 days”; its faithfulness was **0.5**, but “within”
also includes 30 days, making the wording logically ambiguous. A separately run
control explicitly asserted a return on day 365 was permitted and cited the
30-day policy; it scored **0.0** faithfulness. Relevancy and
reference-based context precision do not independently establish that an answer
is correct. The same contexts and reference were used for q01 and the control.

- [Faithfulness](https://docs.ragas.io/en/v0.4.3/concepts/metrics/available_metrics/faithfulness/)
  uses model-extracted claims and an LLM support judgment against context.
- [Answer relevancy](https://docs.ragas.io/en/v0.4.3/concepts/metrics/available_metrics/answer_relevance/)
  compares generated questions with the user's question using embeddings; cosine
  similarities are not a correctness score.
- [Context precision](https://docs.ragas.io/en/v0.4.3/concepts/metrics/available_metrics/context_precision/)
  here judges the order of retrieved passages against the provided reference
  answer. It is not the fraction of answer claims that are true.

## Boundaries and reproduction

The inputs and references are fictional and AI-authored, not independent human
labels. This is a tiny, non-representative sample; no human review or production
quality estimate was performed. Scoring uses `gpt-4o-mini-2024-07-18` and
`text-embedding-3-small`. RAGAS and the earlier custom judge use different
prompts and scales; do not merge their scores. No answer-correctness metric was
run in this experiment. A high judge score does not establish citation support,
document authority, or prompt-injection resistance.

RAGAS is pinned to 0.4.3 locally. TechDocs did not contain a RAGAS tenant, so the
links above point to the official versioned documentation. The scorer objects
are injected into the project function to count and cap actual HTTP attempts;
they are real RAGAS scorers, not mocks. The OpenAI client uses the official
endpoint and SDK retries are disabled. The run recorded **45 successful HTTP
requests** (35 chat, 10 embeddings), with a hard cap of 60 attempts. Recorded
usage: 29,506 chat input tokens, 1,636 chat output tokens and 241 embedding input
tokens; account billing was not retrieved.

The separate explicit-day control made nine additional successful requests
(seven chat and two embeddings), with 5,902 chat input, 258 chat output, and
36 embedding input tokens. Its 12-request hard cap was not reached. The control
used the same fictional evidence but a later local source snapshot, where client
cleanup had been added; it is one example, not a judge accuracy estimate.

A final [one-case default-client smoke test](evaluation/ragas-default-client-smoke.json)
then called the normal project factory with real RAGAS scorers and a 12-request
cap. The evaluation was computed in nine successful requests, and its one
project-owned OpenAI client reported closed on return. This single case checked
the lifecycle and provider path, not general accuracy. It used 5,922 chat input,
266 output and 39 embedding input tokens. Its
[runner](evaluation/ragas-default-client-smoke.py) and
[source patch](evaluation/ragas-default-client-project.patch) are retained; at
that run `ragas_eval.py` SHA-256 was
`99f7907a3bac95fc47937ef2d221308f1cf4786994a81a4d408f3621b22c36e1`.

The [runner](evaluation/ragas-live-runner.py), [per-case inputs, scores and
request usage](evaluation/ragas-live-results.json), [original answer
fixture](evaluation/generation-judge-baseline.json), and [working-tree source
patch](evaluation/ragas-live-project.patch) are retained. The
[explicit-day control result](evaluation/ragas-strong-negative-results.json) and
its [separate source patch](evaluation/ragas-strong-negative-project.patch) are
also retained. At the time of the first run,
`src/xai_rag/evaluation/ragas_eval.py` SHA-256 was
`9750ce942e4aaa4926fa6e746d21b50910af19e586ea54fed5d8bd264dc9dd0a`, and
`src/xai_rag/config.py` SHA-256 was
`8b2ddcd7ec840f9f6701673b066f134ecd4933bd5067802cf695957bcb52191b`.
For the explicit-day run `ragas_eval.py` SHA-256 was
`8bf4aed364582ea19b8dc65e10573e00b7565c42eda236e6a0f3f304776eb563`;
the configuration hash was unchanged.
The patch applies over XAI-RAG revision `f4c374d`; the runner explicitly sets
the endpoint and scorer models. Run it from the XAI-RAG checkout with its
optional evaluation dependencies and a configured key:

```bash
RAGAS_DO_NOT_TRACK=true PYTHONPATH=src .venv/bin/python \
  /path/to/ragas-live-runner.py /path/to/new-results.json
```

Re-running incurs provider usage and may produce different scores. Do not send
private document content to the evaluation provider without the appropriate
data-handling decisions.
