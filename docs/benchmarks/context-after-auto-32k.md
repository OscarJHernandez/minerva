# Agent benchmark: after-auto-32k

Generated 2026-11-15 by `scripts/eval_models.py`. Case set `5fd3f3932f8b` (9 cases, 1 repeat(s), seed 7, temperature 0.2). Code `47a2bbb841-dirty`, prompt `b0d609790150`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 4/9 (44.4%) | 88.9% | 90.0% | 100.0% | 0 | 1 of 10 | 6.2 | 10.6 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 88.9% | NOT met |
| valid arguments rate | 90.0% | NOT met |
| completion rate | 100.0% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| symbolic | 1/1 (100.0%) | 100.0% |
| multistep | 1/3 (33.3%) | 100.0% |
| calculus | 1/1 (100.0%) | 100.0% |
| tensor_matrix | 1/3 (33.3%) | 66.7% |
| false_claim | 0/1 (0.0%) | 100.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | integrate | matrix_calculus | none | simplify | substitute |
| --- | --- | --- | --- | --- | --- |
| integrate | 2 |  |  |  |  |
| matrix_calculus |  | 1 | 1 |  |  |
| none |  |  | 1 |  |  |
| simplify |  |  |  | 3 |  |
| substitute |  |  |  |  | 1 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 6.2 | 10.6 | 10.6 |
| model calls (ms) | 3154 | 8027 | 8027 |
| tool calls (ms) | 996 | 2905 | 2905 |

Tokens in/out: 88772 / 913.

First output after 419 ms (median) · tool calls that errored: 1 · tool results not verified: 1 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 6662.

### Failures by class

| Class | Runs |
| --- | --- |
| model_honesty | 2 |
| result_reuse | 1 |
| tool_selection | 1 |
| invented_assumption | 1 |

Formulas the model wrote: 61 unchecked, 0 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 0.

### Failed runs

- `reuse-and-restriction-after-simplify` (result_reuse): turn 2: an earlier verified result was retyped instead of reused as @rN; turn 2: the reply does not state the restriction the result carries
- `oscillating-integral-stops-when-not-verified` (model_honesty): 5 unchecked formula(s) in the reply
- `declared-matrices-no-invented-commutation` (tool_selection): turn 2: expected one of ['matrix_calculus'], called none
- `inverse-does-not-invent-invertibility` (model_honesty): reply does not say what it should
- `followup-resolves-it-from-earlier-turns` (invented_assumption): turn 2: assumption(s) the user did not state were passed for ['x']; turn 2: an earlier verified result was retyped instead of reused as @rN

