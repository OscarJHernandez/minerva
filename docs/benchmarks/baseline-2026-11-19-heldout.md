# Agent benchmark: baseline-HEAD-heldout

Generated 2026-11-19 by `scripts/eval_models.py`. Case set `aa9c5446ddfd` (6 cases, 3 repeat(s), seed 7, temperature 0.2). Code `c7f00a3e95`, prompt `3125a4c9ab20`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: on.

## Model under test

| Model | Weights as the host reports them | Defaults the host applies | Window loaded | Minerva sends | Thinking requested | Same weights through the run |
| --- | --- | --- | --- | --- | --- | --- |
| minerva | based on gemma4:26b, gemma4, 25.2B, Q4_K_M, digest 149a0ff8a1ca | top_k 64, top_p 0.9, draft_num_predict 3, num_ctx 16384, num_predict 1200, temperature 0.2 | 32768 | num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None (num_predict is left to the host: 1200) | off (model can think) | yes |

Seed 7, 3 repeat(s), compute timeout 10.0 s, request timeout 180.0 s; prompt `3125a4c9ab20`, code `c7f00a3e95`; Ollama 0.40.3.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 17/18 (94.4%) | 100.0% | 100.0% | 100.0% | 0 | 0 of 27 | 6.2 | 11.6 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 100.0% | met |
| valid arguments rate | 100.0% | met |
| completion rate | 100.0% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| multistep | 3/3 (100.0%) | 100.0% |
| calculus | 3/3 (100.0%) | 100.0% |
| tensor_matrix | 3/3 (100.0%) | 100.0% |
| ambiguous | 2/3 (66.7%) | 100.0% |
| followup | 6/6 (100.0%) | 100.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | matrix_calculus | ode_analyze | series | simplify |
| --- | --- | --- | --- | --- | --- |
| analyze_lagrangian | 3 |  |  |  |  |
| matrix_calculus |  | 3 |  |  |  |
| ode_analyze |  |  | 3 |  |  |
| series |  |  |  | 3 |  |
| simplify |  |  |  |  | 6 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| analyze_lagrangian | 3/3 | 100.0% | 0 of 6 |
| matrix_calculus | 3/3 | 100.0% | 0 of 3 |
| ode_analyze | 2/3 | 100.0% | 0 of 3 |
| series | 3/3 | 100.0% | 0 of 3 |
| simplify | 6/6 | 100.0% | 0 of 12 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 6.2 | 11.6 | 11.6 |
| model calls (ms) | 3262 | 9839 | 9839 |
| tool calls (ms) | 1360 | 3268 | 3268 |

Tokens in/out: 286109 / 2670.

First output after 641 ms (median) · tool calls that errored: 0 · tool results not verified: 6 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 10648.
Verified-result coverage: 100.0% of 18 runs that expect a card · model calls per run: median 2.0, mean 2.2 · multi-turn consistency: 100.0% of 9 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| model_honesty | 1 |

Formulas the model wrote: 44 unchecked, 1 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 0.

Unstable across repeats (0.47 std of passes per repeat): `heldout-ode-classify`.

### Failed runs

- `heldout-ode-classify` (model_honesty): 1 equation(s) that do not hold as written

