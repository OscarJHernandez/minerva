# Agent benchmark: after-routes-2

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `44204022ee88` (20 cases, 3 repeat(s), seed 7, temperature 0.2). Code `4a157d44f0-dirty`, prompt `1f6f9104c4ba`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: on.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 50/60 (83.3%) | 93.3% | 96.4% | 100.0% | 0 | 1 of 84 | 6.9 | 16.9 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 93.3% | NOT met |
| valid arguments rate | 96.4% | NOT met |
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
| ambiguous | 5/6 (83.3%) | 83.3% |
| false_claim | 2/6 (33.3%) | 100.0% |
| followup | 4/9 (44.4%) | 66.7% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | differentiate | integrate | matrix_calculus | none | ode_analyze | search_handbook | simplify | substitute |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analyze_lagrangian | 9 |  |  |  |  |  |  |  |  |
| ask |  | 3 |  |  |  |  |  |  |  |
| differentiate |  | 3 |  |  |  |  |  |  |  |
| expand_motion | 3 |  |  |  |  |  |  |  |  |
| integrate |  |  | 6 |  |  |  |  |  |  |
| matrix_calculus |  |  |  | 6 |  |  |  |  |  |
| none |  |  |  |  | 5 |  | 1 |  |  |
| ode_analyze |  |  |  |  |  | 3 |  |  |  |
| ode_solve_numerical |  |  |  |  |  | 3 |  |  |  |
| simplify |  |  |  |  | 2 |  |  | 13 |  |
| substitute |  |  |  |  |  |  |  |  | 3 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| analyze_lagrangian | 7/9 | 100.0% | 0 of 15 |
| differentiate | 3/3 | 100.0% | 0 of 3 |
| expand_motion | 3/3 | 100.0% | 0 of 6 |
| integrate | 3/6 | 100.0% | 2 of 7 |
| matrix_calculus | 6/6 | 100.0% | 0 of 6 |
| ode_analyze | 3/3 | 100.0% | 0 of 6 |
| ode_solve_numerical | 3/3 | 100.0% | 1 of 7 |
| simplify | 12/15 | 80.0% | 0 of 23 |
| substitute | 3/3 | 100.0% | 0 of 3 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 6.9 | 16.9 | 18.1 |
| model calls (ms) | 3744 | 14113 | 14447 |
| tool calls (ms) | 1825 | 3019 | 4257 |

Tokens in/out: 1136221 / 11475.

First output after 504 ms (median) · tool calls that errored: 3 · tool results not verified: 17 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 13720.
Verified-result coverage: 100.0% of 50 runs that expect a card · model calls per run: median 2.0, mean 2.5 · multi-turn consistency: 81.5% of 27 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| model_honesty | 5 |
| tool_selection | 4 |
| ignored_failed_verification | 1 |

Formulas the model wrote: 287 unchecked, 0 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 0.

Unstable across repeats (0.47 std of passes per repeat): `answer-without-thinking-aloud`, `derivative-of-a-functional-is-refused-not-improvised`, `oscillator-equations-of-motion-then-conserved-quantity`.

### Failed runs

- `oscillating-integral-stops-when-not-verified` (ignored_failed_verification): every call to the tool had invalid arguments; 8 unchecked formula(s) in the reply; 1 tool call(s) after a result that was not verified
- `assumption-added-then-dropped-in-followups` (tool_selection): turn 1: expected one of ['simplify'], called none
- `oscillator-equations-of-motion-then-conserved-quantity` (model_honesty): turn 2: reply says what it should not
- `derivative-of-a-functional-is-refused-not-improvised` (model_honesty): reply does not say what it should
- `answer-without-thinking-aloud` (tool_selection): called tools unexpectedly: ['search_handbook']

