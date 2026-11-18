# Agent benchmark: heldout-before

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `aa9c5446ddfd` (6 cases, 3 repeat(s), seed 7, temperature 0.2). Code `9720423f4d-dirty`, prompt `1f6f9104c4ba`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: off (the model alone).

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 9/18 (50.0%) | 66.7% | 88.5% | 100.0% | 0 | 5 of 26 | 7.1 | 11.3 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 66.7% | NOT met |
| valid arguments rate | 88.5% | NOT met |
| completion rate | 100.0% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| multistep | 0/3 (0.0%) | 100.0% |
| calculus | 3/3 (100.0%) | 100.0% |
| tensor_matrix | 3/3 (100.0%) | 100.0% |
| ambiguous | 0/3 (0.0%) | 0.0% |
| followup | 3/6 (50.0%) | 50.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | matrix_calculus | none | series | simplify |
| --- | --- | --- | --- | --- | --- |
| analyze_lagrangian | 3 |  |  |  |  |
| matrix_calculus |  | 3 |  |  |  |
| ode_analyze |  |  | 3 |  |  |
| series |  |  |  | 3 |  |
| simplify |  |  |  |  | 6 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| analyze_lagrangian | 0/3 | 0.0% | 0 of 4 |
| matrix_calculus | 3/3 | 100.0% | 3 of 7 |
| ode_analyze | 0/3 | 0.0% | 0 of 0 |
| series | 3/3 | 100.0% | 0 of 3 |
| simplify | 3/6 | 100.0% | 0 of 12 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 7.1 | 11.3 | 11.3 |
| model calls (ms) | 3612 | 10346 | 10346 |
| tool calls (ms) | 968 | 3604 | 3604 |

Tokens in/out: 385104 / 3077.

First output after 508 ms (median) · tool calls that errored: 3 · tool results not verified: 6 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 10356.
Verified-result coverage: 100.0% of 15 runs that expect a card · model calls per run: median 3.0, mean 2.9 · multi-turn consistency: 33.3% of 9 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 6 |
| invented_assumption | 3 |

Formulas the model wrote: 76 unchecked, 0 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 0.

### Failed runs

- `heldout-reuse-and-restriction` (invented_assumption): turn 2: assumption(s) the user did not state were passed for ['t']; turn 2: an earlier verified result was retyped instead of reused as @rN
- `heldout-ode-classify` (tool_selection): expected one of ['ode_analyze'], called none
- `heldout-pendulum-conserved` (tool_selection): turn 2: expected one of ['analyze_lagrangian', 'find_conserved_quantities', 'search_invariants'], called none

