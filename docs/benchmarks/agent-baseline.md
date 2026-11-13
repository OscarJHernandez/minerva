# Agent benchmark: baseline

Generated 2026-11-13 by `scripts/eval_models.py`. Case set `e89efb8efe53` (78 cases, 2 repeat(s), seed 7, temperature 0.2). Code `baseline-7b19980`, prompt `ba9767c5db19`. The targets are goals, not results.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4:26b | 79/154 (51.3%) | 75.3% | 84.9% | 98.7% | 4 | 34 of 139 | 23.8 | 53.7 |
| qwen3.5:35b | 102/154 (66.2%) | 89.0% | 99.3% | 99.4% | 3 | 16 of 140 | 27.1 | 62.3 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## gemma4:26b

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 75.3% | NOT met |
| valid arguments rate | 84.9% | NOT met |
| completion rate | 98.7% | met |
| false verified | 4 | NOT met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| symbolic | 14/18 (77.8%) | 88.9% |
| multistep | 12/16 (75.0%) | 87.5% |
| calculus | 4/22 (18.2%) | 90.9% |
| retrieval | 14/18 (77.8%) | 77.8% |
| tensor_matrix | 1/10 (10.0%) | 10.0% |
| physics | 14/18 (77.8%) | 94.4% |
| ambiguous | 8/12 (66.7%) | 100.0% |
| unsupported | 0/12 (0.0%) | 16.7% |
| false_claim | 6/18 (33.3%) | 55.6% |
| followup | 6/10 (60.0%) | 100.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | check_units | differentiate | dimensional_analysis | expand_motion | find_identities | get_entry | integrate | integrate_motion | none | perturbation_solve | search_handbook | series | simplify | solve_polynomial | tensor_canonicalize |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analyze_lagrangian | 2 |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |
| ask |  |  |  |  |  |  |  |  |  | 10 |  |  |  |  |  |  |
| check_units |  | 4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| differentiate |  |  | 14 |  |  |  |  |  |  | 4 |  |  |  |  |  |  |
| dimensional_analysis |  |  |  | 4 |  |  |  |  |  |  |  |  |  |  |  |  |
| expand_motion |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |  |
| find_identities |  |  |  |  |  | 6 |  |  |  | 4 |  | 2 |  |  |  |  |
| get_entry |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |
| integrate |  |  |  |  |  |  |  | 16 |  | 4 |  |  |  |  |  |  |
| integrate_motion |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |
| matrix_calculus |  |  |  |  |  |  |  |  |  | 6 |  | 2 |  |  |  |  |
| none |  |  |  |  |  |  |  |  |  | 8 |  |  |  |  |  |  |
| perturbation_solve |  |  |  |  |  |  |  |  |  | 1 | 1 |  |  |  |  |  |
| search_handbook |  |  |  |  |  |  |  |  |  | 4 |  | 4 |  |  |  |  |
| series |  |  |  |  |  |  |  |  |  | 2 |  |  | 2 |  |  |  |
| simplify |  |  |  |  |  |  |  |  |  | 6 |  |  |  | 30 |  |  |
| solve_polynomial |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  | 4 |  |
| tensor_canonicalize |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  | 1 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 23.8 | 53.7 | 70.3 |
| model calls (ms) | 23118 | 51570 | 68889 |
| tool calls (ms) | 476 | 1930 | 2483 |

Tokens in/out: not reported by the host / not reported by the host.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 36 |
| model_honesty | 22 |
| arguments | 11 |
| verification | 4 |
| resource_limit | 2 |

Formulas the model wrote: 575 unchecked, 2 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 15.

Unstable across repeats (0.50 std of passes per repeat): `dimensional-missing`, `perturbation-duffing`, `tensor-riemann-pair-symmetry`.

### Failed runs

- `simplify-rational` (model_honesty): 1 unchecked formula(s) in the reply
- `sqrt-no-assumptions` (tool_selection): expected one of ['simplify'], called none
- `definite-by-tool` (model_honesty): 1 unchecked formula(s) in the reply
- `lagrangian-then-numbers` (tool_selection): expected one of ['analyze_lagrangian'], called none
- `differentiate-first` (model_honesty): 3 unchecked formula(s) in the reply
- `differentiate-second` (arguments): the tool's result is not the expected one (different input or status)
- `differentiate-product` (model_honesty): 3 unchecked formula(s) in the reply
- `integrate-indefinite` (model_honesty): 2 unchecked formula(s) in the reply
- `integrate-definite` (model_honesty): 1 unchecked formula(s) in the reply
- `integrate-gaussian` (arguments): the tool's result is not the expected one (different input or status)
- `integrate-improper` (arguments): the tool's result is not the expected one (different input or status)
- `series-tan` (arguments): every call to the tool had invalid arguments
- `integrate-needs-assumptions` (tool_selection): expected one of ['integrate'], called none
- `handbook-constant` (tool_selection): expected one of ['get_entry', 'search_handbook'], called none
- `identities-blocked` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `tensor-contraction` (tool_selection): expected one of ['tensor_canonicalize'], called none
- `tensor-riemann-pair-symmetry` (tool_selection): expected one of ['tensor_canonicalize'], called none
- `matrix-merge-needs-commutation` (verification): expected one of ['matrix_calculus'], called none; 1 false statement(s) shown as verified
- `matrix-commuting-declared` (tool_selection): expected one of ['matrix_calculus'], called none
- `matrix-integral-not-evaluated` (tool_selection): expected one of ['matrix_calculus'], called none
- `dimensional-pendulum` (arguments): arguments do not say what the user asked
- `ask-when-ambiguous` (model_honesty): reply does not say what it should
- `ambiguous-simplify` (model_honesty): reply does not say what it should
- `unsupported-derivative-abs` (tool_selection): expected one of ['differentiate'], called none
- `unsupported-no-closed-form` (tool_selection): expected one of ['integrate'], called none
- `unsupported-series-pole` (tool_selection): expected one of ['series'], called none
- `unsupported-transcendental` (tool_selection): expected one of ['solve_polynomial'], called none
- `unsupported-magnus` (tool_selection): expected one of ['matrix_calculus'], called ['search_handbook']; reply does not say what it should; 10 unchecked formula(s) in the reply
- `unsupported-quantum` (model_honesty): 4 unchecked formula(s) in the reply
- `false-pythagoras` (model_honesty): 1 equation(s) that do not hold as written
- `false-log-sum` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none; reply does not say what it should
- `false-sqrt-square` (tool_selection): expected one of ['find_identities', 'simplify'], called none
- `false-log-exp` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `false-matrix-exp` (verification): 1 false statement(s) shown as verified; reply does not say what it should
- `false-derivative` (tool_selection): expected one of ['differentiate'], called none
- `followup-second-derivative` (arguments): arguments do not say what the user asked
- `followup-explain-no-tool` (model_honesty): 9 unchecked formula(s) in the reply
- `perturbation-duffing` (tool_selection): expected one of ['perturbation_solve'], called none
- `dimensional-missing` (resource_limit): hit the step or time limit; the tool's result is not the expected one (different input or status); reply does not say what it should

## qwen3.5:35b

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 89.0% | NOT met |
| valid arguments rate | 99.3% | met |
| completion rate | 99.4% | met |
| false verified | 3 | NOT met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| symbolic | 14/18 (77.8%) | 88.9% |
| multistep | 14/16 (87.5%) | 100.0% |
| calculus | 8/22 (36.4%) | 100.0% |
| retrieval | 16/18 (88.9%) | 88.9% |
| tensor_matrix | 5/10 (50.0%) | 50.0% |
| physics | 17/18 (94.4%) | 100.0% |
| ambiguous | 11/12 (91.7%) | 100.0% |
| unsupported | 5/12 (41.7%) | 100.0% |
| false_claim | 5/18 (27.8%) | 61.1% |
| followup | 7/10 (70.0%) | 90.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | check_units | differentiate | dimensional_analysis | expand_motion | find_identities | get_entry | integrate | integrate_motion | matrix_calculus | none | perturbation_solve | search_handbook | series | simplify | solve_polynomial | tensor_canonicalize |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analyze_lagrangian | 4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ask |  |  |  |  |  |  |  |  |  |  | 10 |  |  |  |  |  |  |
| check_units |  | 3 |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |
| differentiate |  |  | 18 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| dimensional_analysis |  |  |  | 4 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| expand_motion |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| find_identities |  |  |  |  |  | 5 |  |  |  |  | 3 |  | 4 |  |  |  |  |
| get_entry |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |
| integrate |  |  |  |  |  |  |  | 20 |  |  |  |  |  |  |  |  |  |
| integrate_motion |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  |
| matrix_calculus |  |  |  |  |  |  |  |  |  | 2 | 6 |  |  |  |  |  |  |
| none |  |  |  |  |  |  |  |  |  |  | 8 |  |  |  |  |  |  |
| perturbation_solve |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |
| search_handbook |  |  |  |  |  |  |  |  |  |  | 2 |  | 6 |  |  |  |  |
| series |  |  |  |  |  |  |  |  |  |  |  |  |  | 4 |  |  |  |
| simplify |  |  |  |  |  | 1 |  |  |  |  | 6 |  | 4 |  | 25 |  |  |
| solve_polynomial |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 6 |  |
| tensor_canonicalize |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 4 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 27.1 | 62.3 | 107.5 |
| model calls (ms) | 26137 | 61434 | 107006 |
| tool calls (ms) | 402 | 1657 | 2168 |

Tokens in/out: not reported by the host / not reported by the host.

### Failures by class

| Class | Runs |
| --- | --- |
| model_honesty | 24 |
| tool_selection | 16 |
| arguments | 6 |
| verification | 3 |
| missing_assumptions | 2 |
| resource_limit | 1 |

Formulas the model wrote: 889 unchecked, 3 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 0.

Unstable across repeats (1.00 std of passes per repeat): `ask-when-ambiguous`, `false-energy-units`, `followup-uses-context`, `matrix-commuting-declared`, `motion-values-given`, `unsupported-derivative-abs`, `unsupported-no-closed-form`, `unsupported-transcendental`.

### Failed runs

- `simplify-rational` (model_honesty): 1 unchecked formula(s) in the reply
- `sqrt-no-assumptions` (tool_selection): expected one of ['simplify'], called none
- `definite-by-tool` (model_honesty): 1 unchecked formula(s) in the reply
- `differentiate-first` (model_honesty): 5 unchecked formula(s) in the reply
- `differentiate-product` (model_honesty): 3 unchecked formula(s) in the reply
- `integrate-indefinite` (model_honesty): 4 unchecked formula(s) in the reply
- `integrate-definite` (model_honesty): 2 unchecked formula(s) in the reply
- `integrate-gaussian` (arguments): the tool's result is not the expected one (different input or status)
- `integrate-improper` (arguments): the tool's result is not the expected one (different input or status)
- `integrate-needs-assumptions` (missing_assumptions): the tool's result is not the expected one (different input or status)
- `handbook-constant` (tool_selection): expected one of ['get_entry', 'search_handbook'], called none
- `matrix-merge-needs-commutation` (tool_selection): expected one of ['matrix_calculus'], called none
- `matrix-commuting-declared` (tool_selection): expected one of ['matrix_calculus'], called none
- `matrix-integral-not-evaluated` (tool_selection): expected one of ['matrix_calculus'], called none
- `unsupported-derivative-abs` (model_honesty): reply too long (253 words)
- `unsupported-no-closed-form` (model_honesty): reply says what it should not
- `unsupported-transcendental` (model_honesty): reply too long (329 words)
- `unsupported-magnus` (arguments): every call to the tool had invalid arguments; reply does not say what it should; 18 unchecked formula(s) in the reply
- `unsupported-quantum` (resource_limit): hit the step or time limit
- `false-square` (tool_selection): expected one of ['find_identities', 'simplify'], called none
- `false-pythagoras` (model_honesty): 1 equation(s) that do not hold as written
- `false-log-sum` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `false-sqrt-square` (tool_selection): expected one of ['find_identities', 'simplify'], called none
- `false-log-exp` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `false-matrix-exp` (verification): 1 false statement(s) shown as verified
- `followup-uses-context` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `followup-explain-no-tool` (model_honesty): 15 unchecked formula(s) in the reply
- `motion-values-given` (model_honesty): 1 equation(s) that do not hold as written
- `ask-when-ambiguous` (model_honesty): reply does not say what it should
- `false-energy-units` (tool_selection): expected one of ['check_units'], called none

