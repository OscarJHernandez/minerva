# Agent benchmark: unsupported-prompt-before

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `e04bf392c3e3` (12 cases, 3 repeat(s), seed 7, temperature 0.2). Code `5253e5ab25-dirty`, prompt `1f6f9104c4ba`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: on.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 5/30 (16.7%) | 50.0% | 65.5% | 100.0% | 0 | 14 of 29 | 4.6 | 14.4 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 50.0% | NOT met |
| valid arguments rate | 65.5% | NOT met |
| completion rate | 100.0% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| unsupported | 5/30 (16.7%) | 50.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | creative_telescoping | integrate | isolate_variable | none | search_handbook | statistical_mechanics | vector_calculus |
| --- | --- | --- | --- | --- | --- | --- | --- |
| creative_telescoping | 3 |  |  |  |  |  |  |
| differentiate |  |  |  | 3 |  |  |  |
| integrate |  | 3 |  |  |  |  |  |
| isolate_variable |  |  |  | 3 |  |  |  |
| matrix_calculus |  |  |  |  | 3 |  |  |
| search_handbook |  |  |  | 3 |  |  |  |
| series |  |  |  | 3 |  |  |  |
| solve_polynomial |  |  | 3 |  |  |  |  |
| statistical_mechanics |  |  |  |  |  | 3 |  |
| vector_calculus |  |  |  |  |  |  | 3 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| creative_telescoping | 3/3 | 100.0% | 0 of 3 |
| differentiate | 0/3 | 0.0% | 0 of 0 |
| integrate | 1/3 | 100.0% | 0 of 3 |
| isolate_variable | 0/3 | 0.0% | 0 of 0 |
| matrix_calculus | 0/3 | 0.0% | 0 of 3 |
| search_handbook | 0/3 | 100.0% | 0 of 0 |
| series | 0/3 | 0.0% | 0 of 0 |
| solve_polynomial | 0/3 | 0.0% | 0 of 3 |
| statistical_mechanics | 0/3 | 100.0% | 9 of 6 |
| vector_calculus | 1/3 | 100.0% | 1 of 11 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 4.6 | 14.4 | 24.3 |
| model calls (ms) | 3868 | 14015 | 22226 |
| tool calls (ms) | 358 | 1648 | 1648 |

Tokens in/out: 502735 / 7542.

First output after 427 ms (median) · tool calls that errored: 7 · tool results not verified: 16 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 12393.
Verified-result coverage: 100.0% of 9 runs that expect a card · model calls per run: median 2.0, mean 2.1 · multi-turn consistency: n/a of 0 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 15 |
| model_honesty | 7 |
| arguments | 3 |

Formulas the model wrote: 238 unchecked, 0 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 3.

Unstable across repeats (0.47 std of passes per repeat): `distributional-unsupported-field-is-refused`, `unsupported-no-closed-form`.

### Failed runs

- `statmech-divergent-series-is-refused` (arguments): every call to the tool had invalid arguments
- `isolate-unsupported-is-not-answered` (tool_selection): expected one of ['isolate_variable'], called none; reply does not say what it should
- `unsupported-derivative-abs` (tool_selection): expected one of ['differentiate'], called none
- `unsupported-no-closed-form` (model_honesty): reply says what it should not
- `unsupported-series-pole` (tool_selection): expected one of ['series'], called none
- `unsupported-transcendental` (tool_selection): expected one of ['solve_polynomial'], called ['isolate_variable']
- `unsupported-magnus` (tool_selection): expected one of ['matrix_calculus'], called ['search_handbook']; reply does not say what it should; 10 unchecked formula(s) in the reply
- `unsupported-quantum` (model_honesty): reply does not say what it should; 5 unchecked formula(s) in the reply
- `distributional-unsupported-field-is-refused` (model_honesty): reply does not say what it should

