# Agent benchmark: new

Generated 2026-11-13 by `scripts/eval_models.py`. Case set `e89efb8efe53` (78 cases, 2 repeat(s), seed 7, temperature 0.2). Code `abaeb9fd4b`, prompt `ed0574ddeeb0`. The targets are goals, not results.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4:26b | 74/154 (48.1%) | 77.3% | 91.9% | 94.8% | 4 | 27 of 124 | 26.7 | 55.6 |
| qwen3.5:35b | 102/154 (66.2%) | 85.1% | 99.3% | 99.4% | 3 | 21 of 142 | 26.7 | 72.1 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## gemma4:26b

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 77.3% | NOT met |
| valid arguments rate | 91.9% | NOT met |
| completion rate | 94.8% | met |
| false verified | 4 | NOT met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| symbolic | 14/18 (77.8%) | 88.9% |
| multistep | 12/16 (75.0%) | 87.5% |
| calculus | 4/22 (18.2%) | 90.9% |
| retrieval | 14/18 (77.8%) | 77.8% |
| tensor_matrix | 2/10 (20.0%) | 20.0% |
| physics | 11/18 (61.1%) | 100.0% |
| ambiguous | 8/12 (66.7%) | 100.0% |
| unsupported | 0/12 (0.0%) | 16.7% |
| false_claim | 5/18 (27.8%) | 61.1% |
| followup | 4/10 (40.0%) | 100.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | check_units | differentiate | dimensional_analysis | expand_motion | find_identities | get_entry | integrate | integrate_motion | none | perturbation_solve | search_handbook | series | simplify | solve_polynomial | tensor_canonicalize |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analyze_lagrangian | 2 |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |
| ask |  |  |  |  |  |  |  |  |  | 10 |  |  |  |  |  |  |
| check_units |  | 4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| differentiate |  |  | 15 |  |  |  |  |  |  | 3 |  |  |  |  |  |  |
| dimensional_analysis |  |  |  | 4 |  |  |  |  |  |  |  |  |  |  |  |  |
| expand_motion |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |  |
| find_identities |  |  |  |  |  | 6 |  |  |  | 4 |  | 2 |  |  |  |  |
| get_entry |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |
| integrate |  |  |  |  |  |  |  | 16 |  | 4 |  |  |  |  |  |  |
| integrate_motion |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |
| matrix_calculus |  |  |  |  |  |  |  |  |  | 6 |  | 2 |  |  |  |  |
| none |  |  |  |  |  |  |  |  |  | 8 |  |  |  |  |  |  |
| perturbation_solve |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |
| search_handbook |  |  |  |  |  |  |  |  |  | 4 |  | 4 |  |  |  |  |
| series |  |  |  |  |  |  |  |  |  | 2 |  |  | 2 |  |  |  |
| simplify |  |  |  |  |  |  |  |  |  | 6 |  |  |  | 30 |  |  |
| solve_polynomial |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  | 4 |  |
| tensor_canonicalize |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  | 2 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 26.7 | 55.6 | 71.3 |
| model calls (ms) | 25969 | 55305 | 70615 |
| tool calls (ms) | 474 | 1927 | 2841 |

Tokens in/out: 1707666 / 21116.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 33 |
| model_honesty | 23 |
| arguments | 12 |
| unproductive_loop | 8 |
| verification | 4 |

Formulas the model wrote: 557 unchecked, 3 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 10.

Unstable across repeats (2.00 std of passes per repeat): `dimensional-missing`, `false-derivative`, `motion-values-given`, `units-check`.

### Failed runs

- `simplify-rational` (model_honesty): 1 unchecked formula(s) in the reply
- `sqrt-no-assumptions` (tool_selection): expected one of ['simplify'], called none
- `definite-by-tool` (model_honesty): 1 unchecked formula(s) in the reply
- `lagrangian-then-numbers` (tool_selection): expected one of ['analyze_lagrangian'], called none
- `differentiate-first` (model_honesty): 3 unchecked formula(s) in the reply
- `differentiate-second` (arguments): the tool's result is not the expected one (different input or status)
- `differentiate-product` (model_honesty): 2 unchecked formula(s) in the reply
- `integrate-indefinite` (model_honesty): 2 unchecked formula(s) in the reply
- `integrate-definite` (model_honesty): 1 unchecked formula(s) in the reply
- `integrate-gaussian` (arguments): the tool's result is not the expected one (different input or status)
- `integrate-improper` (arguments): the tool's result is not the expected one (different input or status)
- `series-tan` (arguments): every call to the tool had invalid arguments
- `integrate-needs-assumptions` (tool_selection): expected one of ['integrate'], called none
- `handbook-constant` (tool_selection): expected one of ['get_entry', 'search_handbook'], called none
- `identities-blocked` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `tensor-contraction` (tool_selection): expected one of ['tensor_canonicalize'], called none
- `matrix-merge-needs-commutation` (verification): expected one of ['matrix_calculus'], called none; 1 false statement(s) shown as verified
- `matrix-commuting-declared` (tool_selection): expected one of ['matrix_calculus'], called none
- `matrix-integral-not-evaluated` (tool_selection): expected one of ['matrix_calculus'], called none; reply does not say what it should
- `motion-values-given` (model_honesty): 1 equation(s) that do not hold as written
- `dimensional-pendulum` (arguments): arguments do not say what the user asked
- `dimensional-missing` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status)
- `units-check` (unproductive_loop): the turn ended early: repeated_calls; every call to the tool had invalid arguments
- `assumptions-carried` (unproductive_loop): the turn ended early: repeated_calls
- `ask-when-ambiguous` (model_honesty): reply does not say what it should
- `ambiguous-simplify` (model_honesty): reply does not say what it should
- `unsupported-derivative-abs` (tool_selection): expected one of ['differentiate'], called none
- `unsupported-no-closed-form` (tool_selection): expected one of ['integrate'], called none
- `unsupported-series-pole` (tool_selection): expected one of ['series'], called none
- `unsupported-transcendental` (tool_selection): expected one of ['solve_polynomial'], called none
- `unsupported-magnus` (tool_selection): expected one of ['matrix_calculus'], called ['search_handbook']; reply does not say what it should; 11 unchecked formula(s) in the reply
- `unsupported-quantum` (model_honesty): 5 unchecked formula(s) in the reply
- `false-pythagoras` (model_honesty): 1 equation(s) that do not hold as written
- `false-log-sum` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none; reply does not say what it should
- `false-sqrt-square` (tool_selection): expected one of ['find_identities', 'simplify'], called none
- `false-log-exp` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `false-matrix-exp` (verification): 1 false statement(s) shown as verified
- `false-derivative` (tool_selection): expected one of ['differentiate'], called none
- `false-energy-units` (unproductive_loop): the turn ended early: repeated_calls; every call to the tool had invalid arguments
- `followup-keeps-assumptions` (unproductive_loop): the turn ended early: repeated_calls
- `followup-second-derivative` (arguments): arguments do not say what the user asked
- `followup-explain-no-tool` (model_honesty): 7 unchecked formula(s) in the reply

## qwen3.5:35b

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 85.1% | NOT met |
| valid arguments rate | 99.3% | met |
| completion rate | 99.4% | met |
| false verified | 3 | NOT met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| symbolic | 16/18 (88.9%) | 100.0% |
| multistep | 14/16 (87.5%) | 100.0% |
| calculus | 10/22 (45.5%) | 100.0% |
| retrieval | 16/18 (88.9%) | 88.9% |
| tensor_matrix | 2/10 (20.0%) | 30.0% |
| physics | 17/18 (94.4%) | 100.0% |
| ambiguous | 12/12 (100.0%) | 100.0% |
| unsupported | 3/12 (25.0%) | 58.3% |
| false_claim | 6/18 (33.3%) | 61.1% |
| followup | 6/10 (60.0%) | 80.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | check_units | differentiate | dimensional_analysis | expand_motion | find_identities | get_entry | integrate | integrate_motion | none | perturbation_solve | search_handbook | series | simplify | solve_polynomial | tensor_canonicalize |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analyze_lagrangian | 4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ask |  |  |  |  |  |  |  |  |  | 10 |  |  |  |  |  |  |
| check_units |  | 2 |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |
| differentiate |  |  | 17 |  |  |  |  |  |  | 1 |  |  |  |  |  |  |
| dimensional_analysis |  |  |  | 4 |  |  |  |  |  |  |  |  |  |  |  |  |
| expand_motion |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |  |
| find_identities |  |  |  |  |  | 5 |  |  |  | 2 |  | 5 |  |  |  |  |
| get_entry |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |
| integrate |  |  |  |  |  |  |  | 20 |  |  |  |  |  |  |  |  |
| integrate_motion |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |
| matrix_calculus |  |  |  |  |  |  |  |  |  | 7 |  | 1 |  |  |  |  |
| none |  |  |  |  |  |  |  |  |  | 8 |  |  |  |  |  |  |
| perturbation_solve |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |
| search_handbook |  |  |  |  |  |  |  |  |  | 2 |  | 6 |  |  |  |  |
| series |  |  |  |  |  |  |  |  |  | 1 |  |  | 3 |  |  |  |
| simplify |  |  |  |  |  |  |  |  |  | 5 |  | 3 |  | 28 |  |  |
| solve_polynomial |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  | 4 |  |
| tensor_canonicalize |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  | 3 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 26.7 | 72.1 | 91.9 |
| model calls (ms) | 25458 | 71077 | 91517 |
| tool calls (ms) | 502 | 1706 | 2824 |

Tokens in/out: 2334283 / 32267.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 22 |
| model_honesty | 21 |
| arguments | 3 |
| verification | 3 |
| missing_assumptions | 2 |
| resource_limit | 1 |

Formulas the model wrote: 846 unchecked, 5 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 0.

Unstable across repeats (1.00 std of passes per repeat): `integrate-gaussian`, `integrate-improper`, `motion-values-given`, `unsupported-series-pole`.

### Failed runs

- `simplify-rational` (model_honesty): 2 unchecked formula(s) in the reply
- `definite-by-tool` (model_honesty): 1 unchecked formula(s) in the reply
- `differentiate-first` (model_honesty): 3 unchecked formula(s) in the reply
- `differentiate-product` (model_honesty): 3 unchecked formula(s) in the reply
- `integrate-indefinite` (model_honesty): 3 unchecked formula(s) in the reply
- `integrate-definite` (model_honesty): 2 unchecked formula(s) in the reply
- `integrate-gaussian` (arguments): the tool's result is not the expected one (different input or status)
- `integrate-needs-assumptions` (missing_assumptions): the tool's result is not the expected one (different input or status)
- `handbook-constant` (tool_selection): expected one of ['get_entry', 'search_handbook'], called none
- `tensor-contraction` (tool_selection): expected one of ['tensor_canonicalize'], called none
- `matrix-merge-needs-commutation` (tool_selection): expected one of ['matrix_calculus'], called none
- `matrix-commuting-declared` (tool_selection): expected one of ['matrix_calculus'], called none
- `matrix-integral-not-evaluated` (tool_selection): expected one of ['matrix_calculus'], called none
- `motion-values-given` (model_honesty): 1 equation(s) that do not hold as written
- `unsupported-derivative-abs` (tool_selection): expected one of ['differentiate'], called none
- `unsupported-series-pole` (tool_selection): expected one of ['series'], called none
- `unsupported-transcendental` (tool_selection): expected one of ['solve_polynomial'], called ['differentiate', 'search_handbook', 'simplify']; reply too long (349 words)
- `unsupported-magnus` (model_honesty): reply does not say what it should; 10 unchecked formula(s) in the reply
- `unsupported-quantum` (resource_limit): the turn ended early: step_limit
- `false-square` (tool_selection): expected one of ['find_identities', 'simplify'], called none
- `false-pythagoras` (model_honesty): 2 equation(s) that do not hold as written
- `false-sqrt-square` (model_honesty): reply too long (227 words)
- `false-log-exp` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `false-matrix-exp` (verification): 1 false statement(s) shown as verified
- `false-energy-units` (tool_selection): expected one of ['check_units'], called none
- `followup-uses-context` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `followup-explain-no-tool` (model_honesty): 12 unchecked formula(s) in the reply
- `integrate-improper` (arguments): the tool's result is not the expected one (different input or status)

