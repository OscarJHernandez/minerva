# Agent benchmark: before-421998e

Generated 2026-11-16 by `scripts/eval_models.py`. Case set `fe7f15587719` (18 cases, 3 repeat(s), seed 7, temperature 0.2). Code `421998e`, prompt `4cb7f67d126b`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 39/54 (72.2%) | 77.8% | 91.1% | 100.0% | 0 | 13 of 79 | 20.4 | 40.7 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 77.8% | NOT met |
| valid arguments rate | 91.1% | NOT met |
| completion rate | 100.0% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| symbolic | 6/6 (100.0%) | 100.0% |
| multistep | 6/12 (50.0%) | 100.0% |
| calculus | 6/6 (100.0%) | 100.0% |
| tensor_matrix | 6/9 (66.7%) | 66.7% |
| physics | 6/6 (100.0%) | 50.0% |
| ambiguous | 3/6 (50.0%) | 50.0% |
| false_claim | 3/6 (50.0%) | 100.0% |
| followup | 3/3 (100.0%) | 0.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | differentiate | integrate | load_tools | matrix_calculus | none | ode_analyze | simplify | substitute |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analyze_lagrangian |  |  |  | 6 |  |  |  |  |  |
| ask |  | 3 |  |  |  |  |  |  |  |
| differentiate |  | 3 |  |  |  |  |  |  |  |
| expand_motion | 3 |  |  |  |  |  |  |  |  |
| integrate |  |  | 6 |  |  |  |  |  |  |
| matrix_calculus |  |  |  |  | 6 |  |  |  |  |
| none |  |  |  |  |  | 6 |  |  |  |
| ode_analyze |  |  |  |  |  | 3 |  |  |  |
| ode_solve_numerical |  |  |  |  |  |  | 3 |  |  |
| simplify |  |  |  |  |  |  |  | 12 |  |
| substitute |  |  |  |  |  |  |  |  | 3 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 20.4 | 40.7 | 45.6 |
| model calls (ms) | 18428 | 36549 | 41461 |
| tool calls (ms) | 1581 | 2843 | 2923 |

Tokens in/out: 941231 / 8509.

First output after 5508 ms (median) · tool calls that errored: 7 · tool results not verified: 9 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 13712.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 6 |
| result_reuse | 3 |
| model_honesty | 3 |
| invented_assumption | 3 |

Formulas the model wrote: 272 unchecked, 0 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 0.

### Failed runs

- `reuse-and-restriction-after-simplify` (result_reuse): turn 2: an earlier verified result was retyped instead of reused as @rN; turn 2: the reply does not state the restriction the result carries
- `oscillating-integral-stops-when-not-verified` (model_honesty): 7 unchecked formula(s) in the reply
- `declared-matrices-no-invented-commutation` (tool_selection): turn 2: expected one of ['matrix_calculus'], called none
- `followup-resolves-it-from-earlier-turns` (invented_assumption): turn 2: assumption(s) the user did not state were passed for ['x']; turn 2: an earlier verified result was retyped instead of reused as @rN
- `ode-bare-equation-asks-which-method` (tool_selection): turn 1: expected one of ['ode_analyze'], called none

