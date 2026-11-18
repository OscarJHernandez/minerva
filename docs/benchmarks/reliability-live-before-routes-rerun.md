# Agent benchmark: before-pr81-rerun

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `44204022ee88` (20 cases, 3 repeat(s), seed 7, temperature 0.2). Code `c63059a658-dirty`, prompt `1f6f9104c4ba`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 39/60 (65.0%) | 85.0% | 92.6% | 100.0% | 0 | 16 of 94 | 6.6 | 17.1 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 85.0% | NOT met |
| valid arguments rate | 92.6% | NOT met |
| completion rate | 100.0% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| symbolic | 6/6 (100.0%) | 100.0% |
| multistep | 6/12 (50.0%) | 100.0% |
| calculus | 6/6 (100.0%) | 100.0% |
| tensor_matrix | 6/9 (66.7%) | 66.7% |
| physics | 6/6 (100.0%) | 100.0% |
| ambiguous | 3/6 (50.0%) | 50.0% |
| false_claim | 1/6 (16.7%) | 100.0% |
| followup | 5/9 (55.6%) | 66.7% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | differentiate | expand_motion | integrate | matrix_calculus | none | ode_analyze | simplify | substitute |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analyze_lagrangian | 9 |  |  |  |  |  |  |  |  |
| ask |  | 3 |  |  |  |  |  |  |  |
| differentiate |  | 3 |  |  |  |  |  |  |  |
| expand_motion |  |  | 3 |  |  |  |  |  |  |
| integrate |  |  |  | 6 |  |  |  |  |  |
| matrix_calculus |  |  |  |  | 6 |  |  |  |  |
| none |  |  |  |  |  | 6 |  |  |  |
| ode_analyze |  |  |  |  |  | 3 |  |  |  |
| ode_solve_numerical |  |  |  |  |  |  | 3 |  |  |
| simplify |  |  |  |  |  |  |  | 15 |  |
| substitute |  |  |  |  |  |  |  |  | 3 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| analyze_lagrangian | 6/9 | 66.7% | 0 of 21 |
| differentiate | 3/3 | 100.0% | 0 of 3 |
| expand_motion | 3/3 | 100.0% | 0 of 6 |
| integrate | 3/6 | 100.0% | 0 of 6 |
| matrix_calculus | 6/6 | 100.0% | 3 of 9 |
| ode_analyze | 0/3 | 0.0% | 0 of 6 |
| ode_solve_numerical | 3/3 | 100.0% | 3 of 9 |
| simplify | 8/15 | 100.0% | 1 of 28 |
| substitute | 3/3 | 100.0% | 0 of 3 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 6.6 | 17.1 | 17.6 |
| model calls (ms) | 3865 | 13417 | 15195 |
| tool calls (ms) | 1406 | 5353 | 5852 |

Tokens in/out: 1493584 / 12444.

First output after 523 ms (median) · tool calls that errored: 7 · tool results not verified: 18 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 13305.
Verified-result coverage: 97.9% of 48 runs that expect a card · model calls per run: median 3.0, mean 3.1 · multi-turn consistency: 40.7% of 27 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 9 |
| model_honesty | 5 |
| invented_assumption | 4 |
| result_reuse | 3 |

Formulas the model wrote: 296 unchecked, 0 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 0.

Unstable across repeats (0.82 std of passes per repeat): `assumption-added-then-dropped-in-followups`, `derivative-of-a-functional-is-refused-not-improvised`.

### Failed runs

- `reuse-and-restriction-after-simplify` (result_reuse): turn 2: an earlier verified result was retyped instead of reused as @rN; turn 2: the reply does not state the restriction the result carries
- `oscillating-integral-stops-when-not-verified` (model_honesty): 6 unchecked formula(s) in the reply
- `declared-matrices-no-invented-commutation` (tool_selection): turn 2: expected one of ['matrix_calculus'], called none
- `followup-resolves-it-from-earlier-turns` (invented_assumption): turn 2: assumption(s) the user did not state were passed for ['x']; turn 2: an earlier verified result was retyped instead of reused as @rN
- `ode-bare-equation-asks-which-method` (tool_selection): turn 1: expected one of ['ode_analyze'], called none
- `derivative-of-a-functional-is-refused-not-improvised` (model_honesty): reply does not say what it should
- `assumption-added-then-dropped-in-followups` (invented_assumption): turn 3: arguments do not say what the user asked; turn 3: the tool's result is not the expected one (different input or status); turn 3: assumption(s) the user did not state were passed for ['x']
- `oscillator-equations-of-motion-then-conserved-quantity` (tool_selection): turn 2: expected one of ['analyze_lagrangian', 'find_conserved_quantities', 'search_invariants'], called none

