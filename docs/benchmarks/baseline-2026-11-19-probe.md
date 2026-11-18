# Agent benchmark: baseline-HEAD-probe

Generated 2026-11-19 by `scripts/eval_models.py`. Case set `6c095266ac4c` (38 cases, 3 repeat(s), seed 7, temperature 0.2). Code `5a3def49e0`, prompt `3125a4c9ab20`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: on.

## Model under test

| Model | Weights as the host reports them | Defaults the host applies | Window loaded | Minerva sends | Thinking requested | Same weights through the run |
| --- | --- | --- | --- | --- | --- | --- |
| minerva | based on gemma4:26b, gemma4, 25.2B, Q4_K_M, digest 149a0ff8a1ca | top_k 64, top_p 0.9, draft_num_predict 3, num_ctx 16384, num_predict 1200, temperature 0.2 | 32768 | num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None (num_predict is left to the host: 1200) | off (model can think) | yes |

Seed 7, 3 repeat(s), compute timeout 10.0 s, request timeout 180.0 s; prompt `3125a4c9ab20`, code `5a3def49e0`; Ollama 0.40.3.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 98/114 (86.0%) | 94.7% | 100.0% | 100.0% | 0 | 21 of 135 | 4.2 | 12.6 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 94.7% | NOT met |
| valid arguments rate | 100.0% | met |
| completion rate | 100.0% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| symbolic | 18/21 (85.7%) | 85.7% |
| multistep | 9/9 (100.0%) | 100.0% |
| calculus | 21/24 (87.5%) | 100.0% |
| retrieval | 6/6 (100.0%) | 100.0% |
| tensor_matrix | 3/3 (100.0%) | 100.0% |
| physics | 3/3 (100.0%) | 100.0% |
| ambiguous | 9/9 (100.0%) | 100.0% |
| unsupported | 3/9 (33.3%) | 100.0% |
| false_claim | 20/24 (83.3%) | 87.5% |
| followup | 6/6 (100.0%) | 100.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | approximation | check_identity | differentiate | find_identities | geometry | integrate | isolate_variable | none | search_handbook | series | simplify | solve_polynomial | substitute |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analyze_lagrangian | 6 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ask |  |  |  |  |  |  |  |  | 6 |  |  |  |  |  |
| check_identity |  |  | 20 |  |  |  |  |  |  |  |  | 1 |  |  |
| differentiate |  |  |  | 9 |  |  |  |  |  |  |  |  |  |  |
| find_identities |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |
| geometry |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |
| integrate |  |  |  |  |  |  | 15 |  |  |  |  |  |  |  |
| isolate_variable |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |
| matrix_calculus |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  |
| none |  |  |  |  |  |  |  |  | 9 |  |  |  |  |  |
| search_handbook |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |
| series |  | 3 |  |  |  |  |  |  |  |  | 3 |  |  |  |
| simplify |  |  |  |  |  |  |  |  | 3 |  |  | 15 |  |  |
| solve_polynomial |  |  |  |  |  |  |  | 3 |  |  |  |  | 3 |  |
| substitute |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| analyze_lagrangian | 6/6 | 100.0% | 0 of 12 |
| check_identity | 20/21 | 100.0% | 0 of 24 |
| differentiate | 9/9 | 100.0% | 0 of 12 |
| find_identities | 3/3 | 100.0% | 0 of 3 |
| geometry | 3/3 | 100.0% | 0 of 7 |
| integrate | 12/15 | 100.0% | 0 of 15 |
| isolate_variable | 3/3 | 100.0% | 0 of 3 |
| matrix_calculus | 0/3 | 0.0% | 0 of 0 |
| search_handbook | 3/3 | 100.0% | 0 of 3 |
| series | 3/6 | 100.0% | 0 of 6 |
| simplify | 15/18 | 83.3% | 0 of 38 |
| solve_polynomial | 6/6 | 100.0% | 0 of 6 |
| substitute | 3/3 | 100.0% | 0 of 3 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 4.2 | 12.6 | 14.6 |
| model calls (ms) | 2460 | 10372 | 13942 |
| tool calls (ms) | 928 | 3526 | 6619 |

Tokens in/out: 1723009 / 13910.

First output after 446 ms (median) · tool calls that errored: 3 · tool results not verified: 21 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 11088.
Verified-result coverage: 96.5% of 86 runs that expect a card · model calls per run: median 2.0, mean 2.2 · multi-turn consistency: 100.0% of 18 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 6 |
| model_honesty | 6 |
| invented_assumption | 4 |

Formulas the model wrote: 170 unchecked, 0 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 0.

Unstable across repeats (0.47 std of passes per repeat): `probe-log-power-needs-sign`.

### Failed runs

- `probe-false-matrix-square` (tool_selection): expected one of ['matrix_calculus'], called none
- `probe-log-power-needs-sign` (invented_assumption): assumption(s) the user did not state were passed for ['x']
- `probe-exponential-integral-needs-sign` (invented_assumption): arguments do not say what the user asked; the tool's result is not the expected one (different input or status); assumption(s) the user did not state were passed for ['k']
- `probe-laurent-expansion` (model_honesty): reply does not say what it should
- `probe-turbulence-pde` (model_honesty): reply does not say what it should
- `probe-unchanged-is-an-answer` (tool_selection): expected one of ['simplify'], called none

