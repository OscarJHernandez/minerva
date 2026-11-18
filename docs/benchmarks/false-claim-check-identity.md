# Agent benchmark: false-claim-check-identity

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `9b48776f2018` (18 cases, 3 repeat(s), seed 7, temperature 0.2). Code `b0912daa8f`, prompt `9e5310bd4e64`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: on.

## Model under test

| Model | Weights as the host reports them | Defaults the host applies | Window loaded | Minerva sends | Thinking requested | Same weights through the run |
| --- | --- | --- | --- | --- | --- | --- |
| minerva | based on gemma4:26b, gemma4, 25.2B, Q4_K_M, digest 149a0ff8a1ca | top_k 64, top_p 0.9, draft_num_predict 3, num_ctx 16384, num_predict 1200, temperature 0.2 | 32768 | num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None (num_predict is left to the host: 1200) | off (model can think) | yes |

Seed 7, 3 repeat(s), compute timeout 10.0 s, request timeout 180.0 s; prompt `9e5310bd4e64`, code `b0912daa8f`; Ollama 0.40.3.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 31/48 (64.6%) | 72.9% | 95.7% | 100.0% | 0 | 3 of 47 | 7.0 | 22.0 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 72.9% | NOT met |
| valid arguments rate | 95.7% | NOT met |
| completion rate | 100.0% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| false_claim | 31/48 (64.6%) | 72.9% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | check_identity | check_units | differentiate | integrate | none | vector_calculus |
| --- | --- | --- | --- | --- | --- | --- |
| ask | 3 |  |  |  |  |  |
| check_identity | 6 |  |  |  |  |  |
| check_units |  | 3 |  |  |  |  |
| differentiate |  |  | 3 |  |  |  |
| find_identities | 3 |  |  |  |  |  |
| integrate |  |  |  | 3 |  |  |
| ode_solve_symbolic |  |  |  |  | 3 |  |
| simplify | 12 |  |  |  |  |  |
| statistical_mechanics | 3 |  |  |  | 3 |  |
| vector_calculus |  |  |  |  | 1 | 5 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| check_identity | 6/6 | 100.0% | 0 of 6 |
| check_units | 3/3 | 100.0% | 0 of 3 |
| differentiate | 3/3 | 100.0% | 0 of 3 |
| find_identities | 3/3 | 100.0% | 0 of 3 |
| integrate | 3/3 | 100.0% | 0 of 3 |
| ode_solve_symbolic | 0/3 | 0.0% | 0 of 0 |
| simplify | 9/12 | 75.0% | 0 of 12 |
| statistical_mechanics | 0/6 | 0.0% | 0 of 6 |
| vector_calculus | 1/6 | 83.3% | 2 of 8 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 7.0 | 22.0 | 25.4 |
| model calls (ms) | 5574 | 21973 | 24402 |
| tool calls (ms) | 962 | 2023 | 2115 |

Tokens in/out: 814738 / 8554.

First output after 468 ms (median) · tool calls that errored: 2 · tool results not verified: 9 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 12550.
Verified-result coverage: 100.0% of 15 runs that expect a card · model calls per run: median 2.0, mean 2.1 · multi-turn consistency: n/a of 0 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 13 |
| arguments | 4 |

Formulas the model wrote: 119 unchecked, 0 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 0.

Unstable across repeats (0.47 std of passes per repeat): `distributional-wrong-normalisation-is-rejected`.

### Failed runs

- `statmech-wrong-partition-function-is-not-confirmed` (tool_selection): expected one of ['statistical_mechanics'], called ['check_identity']
- `statmech-noncommuting-response-is-not-claimed` (tool_selection): expected one of ['statistical_mechanics'], called none; reply does not say what it should
- `false-pythagoras` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called ['check_identity']
- `integral-theorem-with-a-pole-is-refused` (tool_selection): expected one of ['vector_calculus'], called none; reply does not say what it should
- `ode-no-closed-form-is-not-impossibility` (tool_selection): expected one of ['ode_analyze', 'ode_solve_symbolic'], called none
- `distributional-wrong-normalisation-is-rejected` (arguments): arguments do not say what the user asked

