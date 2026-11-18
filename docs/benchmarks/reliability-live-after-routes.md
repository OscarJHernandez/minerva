# Agent benchmark: after-routes

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `44204022ee88` (20 cases, 3 repeat(s), seed 7, temperature 0.2). Code `c9922b0e5c-dirty`, prompt `1f6f9104c4ba`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: on.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 53/60 (88.3%) | 95.0% | 93.1% | 100.0% | 0 | 3 of 87 | 7.2 | 18.0 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 95.0% | met |
| valid arguments rate | 93.1% | NOT met |
| completion rate | 100.0% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| symbolic | 6/6 (100.0%) | 100.0% |
| multistep | 12/12 (100.0%) | 100.0% |
| calculus | 6/6 (100.0%) | 100.0% |
| tensor_matrix | 9/9 (100.0%) | 100.0% |
| physics | 6/6 (100.0%) | 100.0% |
| ambiguous | 6/6 (100.0%) | 100.0% |
| false_claim | 2/6 (33.3%) | 100.0% |
| followup | 6/9 (66.7%) | 66.7% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | differentiate | integrate | matrix_calculus | none | ode_analyze | simplify | substitute |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analyze_lagrangian | 9 |  |  |  |  |  |  |  |
| ask |  | 3 |  |  |  |  |  |  |
| differentiate |  | 3 |  |  |  |  |  |  |
| expand_motion | 3 |  |  |  |  |  |  |  |
| integrate |  |  | 6 |  |  |  |  |  |
| matrix_calculus |  |  |  | 6 |  |  |  |  |
| none |  |  |  |  | 6 |  |  |  |
| ode_analyze |  |  |  |  |  | 3 |  |  |
| ode_solve_numerical |  |  |  |  |  | 3 |  |  |
| simplify |  |  |  |  |  |  | 15 |  |
| substitute |  |  |  |  |  |  |  | 3 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| analyze_lagrangian | 6/9 | 66.7% | 0 of 12 |
| differentiate | 3/3 | 100.0% | 0 of 3 |
| expand_motion | 3/3 | 100.0% | 0 of 6 |
| integrate | 3/6 | 100.0% | 0 of 6 |
| matrix_calculus | 6/6 | 100.0% | 3 of 9 |
| ode_analyze | 3/3 | 100.0% | 0 of 6 |
| ode_solve_numerical | 3/3 | 100.0% | 3 of 9 |
| simplify | 15/15 | 100.0% | 0 of 27 |
| substitute | 3/3 | 100.0% | 0 of 3 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 7.2 | 18.0 | 18.6 |
| model calls (ms) | 4535 | 14281 | 14561 |
| tool calls (ms) | 1675 | 5438 | 5849 |

Tokens in/out: 1189046 / 12032.

First output after 506 ms (median) · tool calls that errored: 6 · tool results not verified: 21 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 13720.
Verified-result coverage: 100.0% of 51 runs that expect a card · model calls per run: median 2.0, mean 2.5 · multi-turn consistency: 88.9% of 27 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| model_honesty | 4 |
| tool_selection | 3 |

Formulas the model wrote: 301 unchecked, 0 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 0.

Unstable across repeats (0.47 std of passes per repeat): `derivative-of-a-functional-is-refused-not-improvised`.

### Failed runs

- `oscillating-integral-stops-when-not-verified` (model_honesty): 6 unchecked formula(s) in the reply
- `derivative-of-a-functional-is-refused-not-improvised` (model_honesty): reply does not say what it should
- `oscillator-equations-of-motion-then-conserved-quantity` (tool_selection): turn 2: expected one of ['analyze_lagrangian', 'find_conserved_quantities', 'search_invariants'], called none

