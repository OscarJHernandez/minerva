# Agent benchmark: before-all-16k

Generated 2026-11-15 by `scripts/eval_models.py`. Case set `5fd3f3932f8b` (9 cases, 1 repeat(s), seed 7, temperature 0.2). Code `47a2bbb841-dirty`, prompt `b0d609790150`. The targets are goals, not results.

Model parameters: num_ctx 16384, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset all.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 0/9 (0.0%) | 11.1% | n/a | 100.0% | 0 | 0 of 0 | 17.0 | 20.3 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 11.1% | NOT met |
| valid arguments rate | n/a | not measured |
| completion rate | 100.0% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| symbolic | 0/1 (0.0%) | 0.0% |
| multistep | 0/3 (0.0%) | 0.0% |
| calculus | 0/1 (0.0%) | 0.0% |
| tensor_matrix | 0/3 (0.0%) | 33.3% |
| false_claim | 0/1 (0.0%) | 0.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | none |
| --- | --- |
| integrate | 2 |
| matrix_calculus | 2 |
| none | 1 |
| simplify | 3 |
| substitute | 1 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 17.0 | 20.3 | 20.3 |
| model calls (ms) | 15893 | 17729 | 17729 |
| tool calls (ms) | n/a | n/a | n/a |

Tokens in/out: 73755 / 2116.

First output after 6215 ms (median) · tool calls that errored: 0 · tool results not verified: 0 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 16384 tokens, largest prompt 8195; earlier messages were left out in 5 run(s).

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 8 |
| model_honesty | 1 |

Formulas the model wrote: 124 unchecked, 0 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 0.

### Failed runs

- `pole-cancelled-by-simplification-is-reported` (tool_selection): expected one of ['simplify'], called none
- `reuse-and-restriction-after-simplify` (tool_selection): turn 1: expected one of ['simplify'], called none; turn 2: expected one of ['differentiate'], called none; turn 2: an earlier verified result was retyped instead of reused as @rN; turn 2: the reply does not state the restriction the result carries
- `reciprocal-integral-invents-no-assumption` (tool_selection): expected one of ['integrate'], called none
- `oscillating-integral-stops-when-not-verified` (tool_selection): expected one of ['integrate'], called none; reply says what it should not; 11 unchecked formula(s) in the reply
- `declared-matrices-then-false-expansion` (tool_selection): turn 1: expected one of ['matrix_calculus'], called none
- `declared-matrices-no-invented-commutation` (tool_selection): turn 2: expected one of ['matrix_calculus'], called none
- `inverse-does-not-invent-invertibility` (model_honesty): reply too long (388 words)
- `premise-stays-conditional` (tool_selection): turn 1: expected one of ['substitute'], called none; turn 1: reply does not say what it should; turn 2: reply does not say what it should
- `followup-resolves-it-from-earlier-turns` (tool_selection): turn 1: expected one of ['simplify'], called none; turn 2: expected one of ['integrate'], called none; turn 2: an earlier verified result was retyped instead of reused as @rN; turn 3: reply does not say what it should; turn 3: reply says what it should not

