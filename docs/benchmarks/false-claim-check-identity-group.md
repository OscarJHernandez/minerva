# Agent benchmark: false-claim-check-identity-group

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `895fd34a5cb0` (18 cases, 3 repeat(s), seed 7, temperature 0.2). Code `02a95b3da7`, prompt `1ee9e9a10f17`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: on.

## Model under test

| Model | Weights as the host reports them | Defaults the host applies | Window loaded | Minerva sends | Thinking requested | Same weights through the run |
| --- | --- | --- | --- | --- | --- | --- |
| minerva | based on gemma4:26b, gemma4, 25.2B, Q4_K_M, digest 149a0ff8a1ca | top_k 64, top_p 0.9, draft_num_predict 3, num_ctx 16384, num_predict 1200, temperature 0.2 | 32768 | num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None (num_predict is left to the host: 1200) | off (model can think) | yes |

Seed 7, 3 repeat(s), compute timeout 10.0 s, request timeout 180.0 s; prompt `1ee9e9a10f17`, code `02a95b3da7`; Ollama 0.40.3.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 33/48 (68.8%) | 62.5% | 81.8% | 100.0% | 0 | 17 of 55 | 8.9 | 25.3 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 62.5% | NOT met |
| valid arguments rate | 81.8% | NOT met |
| completion rate | 100.0% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| false_claim | 33/48 (68.8%) | 62.5% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | check_identity | integrate | matrix_calculus | none | units | vector_calculus |
| --- | --- | --- | --- | --- | --- | --- |
| ask | 3 |  |  |  |  |  |
| check_identity | 6 |  |  |  |  |  |
| check_units |  |  |  |  | 3 |  |
| differentiate |  |  |  | 3 |  |  |
| find_identities | 3 |  |  |  |  |  |
| integrate |  | 3 |  |  |  |  |
| ode_solve_symbolic |  |  |  | 3 |  |  |
| simplify | 12 |  |  |  |  |  |
| statistical_mechanics | 3 |  | 3 |  |  |  |
| vector_calculus |  |  |  | 3 |  | 3 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| check_identity | 6/6 | 100.0% | 0 of 6 |
| check_units | 3/3 | 0.0% | 3 of 6 |
| differentiate | 0/3 | 0.0% | 0 of 0 |
| find_identities | 3/3 | 100.0% | 0 of 3 |
| integrate | 3/3 | 100.0% | 0 of 3 |
| ode_solve_symbolic | 0/3 | 0.0% | 0 of 0 |
| simplify | 12/12 | 100.0% | 0 of 12 |
| statistical_mechanics | 0/6 | 0.0% | 7 of 16 |
| vector_calculus | 3/6 | 50.0% | 0 of 6 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 8.9 | 25.3 | 35.2 |
| model calls (ms) | 6991 | 24430 | 34306 |
| tool calls (ms) | 927 | 1985 | 2029 |

Tokens in/out: 891134 / 10073.

First output after 468 ms (median) · tool calls that errored: 12 · tool results not verified: 10 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 12523.
Verified-result coverage: 100.0% of 12 runs that expect a card · model calls per run: median 2.0, mean 2.2 · multi-turn consistency: n/a of 0 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 15 |

Formulas the model wrote: 144 unchecked, 0 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 1.

### Failed runs

- `statmech-wrong-partition-function-is-not-confirmed` (tool_selection): expected one of ['statistical_mechanics'], called ['check_identity']
- `statmech-noncommuting-response-is-not-claimed` (tool_selection): expected one of ['statistical_mechanics'], called ['matrix_calculus']; reply does not say what it should; reply too long (221 words)
- `false-derivative` (tool_selection): expected one of ['differentiate'], called none
- `integral-theorem-with-a-pole-is-refused` (tool_selection): expected one of ['vector_calculus'], called none; reply does not say what it should
- `ode-no-closed-form-is-not-impossibility` (tool_selection): expected one of ['ode_analyze', 'ode_solve_symbolic'], called none

