# GPT-6 Luna verifier: same frozen cases

This repeats the 50 fictional cases in `../fixture.json` (SHA-256 `c7efa755984497cf9f47a5cc9d7f660187e0e391951cdb47330b434ba0b227ce`), with the same untrusted-data system prompt and strict schema as the earlier experiment. Calls use `gpt-6-luna`, `reasoning_effort=none`, `max_completion_tokens=500`, and no retries; requests contain no real user data. A probe plus the full run made 51 total API attempts. Raw responses are in `raw/` and `probe-N12.json`. No shared project file or caller credential was changed.

The [official model page](https://developers.openai.com/api/docs/models/gpt-6-luna) lists GPT-6 Luna as current, supports structured outputs, and lists `gpt-6-luna` only; it does **not** document a dated snapshot. The live API returned `gpt-6-luna` rather than a versioned ID. This is an alias experiment, not a reproducibly pinned snapshot; results may change as the alias changes. The [deprecations page](https://developers.openai.com/api/docs/deprecations) lists it as GPT-4.1 mini's replacement. API access was verified by the 51 successful requests.

| Release gate | Cases | False accepts | False rejects | API errors |
|---|---:|---:|---:|---:|
| Luna judge | original 42 | 0 | 0 | 0 |
| Luna judge | prompt-injection 8 | 0 | 0 | 0 |
| Local NLI AND Luna judge | original 42 | 0 | 2 (N31, N34) | 0 |

On N12, where configured NLI incorrectly accepted a universal statement at entailment 0.9043, Luna returned insufficient, blocking release. Luna returned insufficient for N14 (14 days versus 30 days) and A06 (Monday versus Friday) while their original labels were contradiction; both remain blocked by the binary gate. These subtype differences also expose ambiguity in the synthetic labels: a 14-day allowance does not explicitly prohibit a longer period, and an event may recur. This requires independent adjudication, not assuming the authored labels or the judge are ground truth. The experiment did not simulate network errors, refusals or invalid schemas, so failure behavior for those cases is a recommendation, not tested behavior: fail closed and mark claims unverified rather than fall back to NLI.

The 51 successful calls used 10,875 input and 1,898 output tokens, with no cached input or cache writes; standard-token estimate $0.0020365 at $0.10 input and $0.50 output per million per the model page, excluding tax, regional uplift, and any negotiated rates. Median per-request time for the 50 cases was 1.203 s (range 0.963–3.567 s) with four concurrent requests. The prior NLI latency used a different environment/workload, so no controlled speed comparison is claimed.

These are small synthetic, single-run tests using author-created labels, not proof of robustness or production accuracy. Assess independently labeled held-out data, multi-claim full answers, evidence truncation, attacks, model/version drift, outage/timeout/refusal fail-closed handling, and the application's actual release path before adoption. No app integration is claimed.
