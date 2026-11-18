# Agent benchmark: core-routing-off

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `d2d3164e5b44` (161 cases, 1 repeat(s), seed 7, temperature 0.2). Code `e5f9a5011e`, prompt `1f6f9104c4ba`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: off (the model alone).

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 85/148 (57.4%) | 81.1% | 79.4% | 91.2% | 0 | 86 of 218 | 6.6 | 20.6 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 81.1% | NOT met |
| valid arguments rate | 79.4% | NOT met |
| completion rate | 91.2% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| symbolic | 8/10 (80.0%) | 90.0% |
| multistep | 9/10 (90.0%) | 100.0% |
| calculus | 18/31 (58.1%) | 90.3% |
| retrieval | 8/9 (88.9%) | 88.9% |
| tensor_matrix | 4/8 (50.0%) | 62.5% |
| physics | 19/35 (54.3%) | 82.9% |
| ambiguous | 9/15 (60.0%) | 93.3% |
| unsupported | 0/10 (0.0%) | 50.0% |
| false_claim | 5/14 (35.7%) | 42.9% |
| followup | 5/6 (83.3%) | 100.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | approximation | creative_telescoping | differentiate | dimensional_analysis | distribution | electromagnetism | expand_motion | find_conserved_quantities | find_identities | get_entry | integrate | integrate_motion | isolate_variable | matrix_calculus | none | ode_analyze | perturbation_solve | search_handbook | series | simplify | solve_polynomial | statistical_mechanics | substitute | tensor_canonicalize | time_derivative | units | variation | vector_calculus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analyze_lagrangian | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| approximation |  | 4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |
| ask |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 8 |  |  |  |  |  |  |  |  |  |  |  | 1 |  |
| check_units |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |
| creative_telescoping |  |  | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| differentiate |  |  |  | 7 |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| dimensional_analysis |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| distribution |  |  |  |  |  | 4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| electromagnetism |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  | 1 |  |  | 1 |  |  |  |  |  |  |  |  |  |  |
| expand_motion |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| find_conserved_quantities |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| find_identities |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  | 2 |  |  | 1 |  |  |  |  |  |  |  |  |  |  |
| get_entry |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| integrate |  |  |  |  |  |  |  |  |  |  |  | 11 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| integrate_motion |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| isolate_variable |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| matrix_calculus |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 | 3 |  |  | 1 |  |  |  |  |  |  |  |  |  |  |
| none |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 5 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ode_analyze |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| ode_solve_numerical |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| ode_solve_symbolic |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| perturbation_solve |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |
| search_handbook |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  | 3 |  |  |  |  |  |  |  |  |  |  |
| series |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 1 |  |  |  |  |  |  |  |  |  |
| simplify |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  | 15 |  |  |  |  |  |  |  |  |
| solve_polynomial |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |
| statistical_mechanics |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 2 |  |  |  |  |  |  | 6 |  |  |  |  |  |  |
| substitute |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |
| tensor_canonicalize |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  |
| time_derivative |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |
| variation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |
| vector_calculus |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |  |  | 13 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| analyze_lagrangian | 3/3 | 100.0% | 0 of 6 |
| approximation | 3/6 | 66.7% | 1 of 15 |
| check_units | 2/2 | 0.0% | 2 of 4 |
| creative_telescoping | 0/3 | 100.0% | 9 of 8 |
| differentiate | 5/9 | 77.8% | 0 of 13 |
| dimensional_analysis | 1/2 | 100.0% | 0 of 2 |
| distribution | 2/4 | 100.0% | 5 of 8 |
| electromagnetism | 2/5 | 60.0% | 1 of 9 |
| expand_motion | 1/1 | 100.0% | 0 of 1 |
| find_conserved_quantities | 2/2 | 100.0% | 0 of 4 |
| find_identities | 4/6 | 66.7% | 0 of 4 |
| get_entry | 1/1 | 100.0% | 0 of 1 |
| integrate | 7/11 | 100.0% | 0 of 15 |
| integrate_motion | 0/1 | 100.0% | 0 of 1 |
| isolate_variable | 2/3 | 66.7% | 0 of 2 |
| matrix_calculus | 2/6 | 33.3% | 0 of 6 |
| ode_analyze | 1/1 | 100.0% | 0 of 1 |
| ode_solve_numerical | 2/2 | 100.0% | 1 of 7 |
| ode_solve_symbolic | 2/3 | 66.7% | 0 of 4 |
| perturbation_solve | 1/1 | 100.0% | 0 of 2 |
| search_handbook | 3/4 | 100.0% | 0 of 3 |
| series | 1/2 | 50.0% | 0 of 1 |
| simplify | 14/18 | 83.3% | 0 of 22 |
| solve_polynomial | 2/3 | 66.7% | 0 of 3 |
| statistical_mechanics | 0/9 | 66.7% | 21 of 23 |
| substitute | 2/2 | 100.0% | 0 of 3 |
| tensor_canonicalize | 2/2 | 100.0% | 1 of 4 |
| time_derivative | 1/1 | 100.0% | 0 of 1 |
| variation | 2/3 | 100.0% | 0 of 5 |
| vector_calculus | 4/17 | 76.5% | 2 of 31 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 6.6 | 20.6 | 28.6 |
| model calls (ms) | 4667 | 16563 | 28571 |
| tool calls (ms) | 1070 | 5308 | 12039 |

Tokens in/out: 3216892 / 31610.

First output after 542 ms (median) · tool calls that errored: 40 · tool results not verified: 52 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 15639.
Verified-result coverage: 88.2% of 93 runs that expect a card · model calls per run: median 2.0, mean 2.6 · multi-turn consistency: n/a of 0 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 22 |
| arguments | 14 |
| model_honesty | 12 |
| unproductive_loop | 11 |
| resource_limit | 2 |
| missing_assumptions | 1 |
| parsing | 1 |

Formulas the model wrote: 546 unchecked, 1 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 17.

### Failed runs

- `simplify-rational` (model_honesty): 1 unchecked formula(s) in the reply
- `sqrt-no-assumptions` (tool_selection): expected one of ['simplify'], called none
- `definite-by-tool` (model_honesty): 1 unchecked formula(s) in the reply
- `differentiate-first` (model_honesty): 1 unchecked formula(s) in the reply
- `differentiate-second` (arguments): the tool's result is not the expected one (different input or status)
- `integrate-definite` (model_honesty): 1 unchecked formula(s) in the reply
- `integrate-needs-assumptions` (missing_assumptions): arguments do not say what the user asked; the tool's result is not the expected one (different input or status)
- `identities-blocked` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `matrix-merge-needs-commutation` (tool_selection): expected one of ['matrix_calculus'], called none
- `matrix-integral-not-evaluated` (tool_selection): expected one of ['matrix_calculus'], called none
- `matrix-evolution-asks-for-the-span` (resource_limit): the turn ended early: step_limit; reply does not say what it should
- `motion-values-given` (model_honesty): 1 equation(s) that do not hold as written
- `statmech-two-level-free-energy` (unproductive_loop): the turn ended early: repeated_calls; every call to the tool had invalid arguments
- `statmech-oscillator-heat-capacity` (arguments): every call to the tool had invalid arguments
- `statmech-ideal-gas-conventions-are-the-users` (tool_selection): expected one of ['statistical_mechanics'], called none
- `statmech-divergent-series-is-refused` (unproductive_loop): the turn ended early: repeated_calls; every call to the tool had invalid arguments; reply does not say what it should
- `statmech-wrong-partition-function-is-not-confirmed` (tool_selection): expected one of ['statistical_mechanics'], called none
- `statmech-noncommuting-response-is-not-claimed` (unproductive_loop): the turn ended early: repeated_calls; expected one of ['statistical_mechanics'], called ['matrix_calculus']; reply does not say what it should
- `statmech-fermi-number-and-variance` (arguments): every call to the tool had invalid arguments
- `statmech-ising-thermodynamic-limit` (unproductive_loop): the turn ended early: repeated_calls; every call to the tool had invalid arguments
- `statmech-stirling-is-an-approximation` (arguments): every call to the tool had invalid arguments
- `dimensional-pendulum` (arguments): arguments do not say what the user asked
- `ask-when-ambiguous` (model_honesty): reply does not say what it should
- `ambiguous-simplify` (model_honesty): reply does not say what it should
- `isolate-unsupported-is-not-answered` (tool_selection): expected one of ['isolate_variable'], called none; reply does not say what it should
- `unsupported-derivative-abs` (tool_selection): expected one of ['differentiate'], called none
- `unsupported-no-closed-form` (model_honesty): reply says what it should not
- `unsupported-series-pole` (tool_selection): expected one of ['series'], called none
- `unsupported-transcendental` (tool_selection): expected one of ['solve_polynomial'], called ['isolate_variable']
- `unsupported-magnus` (tool_selection): expected one of ['matrix_calculus'], called ['search_handbook']; reply does not say what it should; 12 unchecked formula(s) in the reply
- `unsupported-quantum` (model_honesty): reply does not say what it should; 7 unchecked formula(s) in the reply
- `false-log-sum` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none; reply does not say what it should
- `false-sqrt-square` (tool_selection): expected one of ['find_identities', 'simplify'], called none
- `false-log-exp` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `false-derivative` (tool_selection): expected one of ['differentiate'], called none
- `followup-explain-no-tool` (model_honesty): 7 unchecked formula(s) in the reply
- `delta-integral-uses-the-tool` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status)
- `delta-on-a-limit-asks-for-a-convention` (arguments): the tool's result is not the expected one (different input or status)
- `matrix-noncommuting-explicit-product` (tool_selection): expected one of ['matrix_calculus'], called none
- `vector-curl-grad-needs-smoothness` (arguments): the tool's result is not the expected one (different input or status)
- `vector-curl-grad-with-smoothness` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status)
- `vector-curl-curl-rewrite` (arguments): arguments do not say what the user asked
- `vector-curl-curl-rewrite-needs-smoothness` (parsing): the tool's result is not the expected one (different input or status)
- `em-adopting-premises-is-the-users-choice` (tool_selection): expected one of ['electromagnetism'], called none
- `em-gauge-invariance` (arguments): arguments do not say what the user asked
- `es-laplace-rectangle` (tool_selection): expected one of ['electromagnetism'], called ['search_handbook']; reply does not say what it should
- `integral-flux-through-a-sphere` (tool_selection): expected one of ['vector_calculus'], called none; reply does not say what it should
- `integral-theorem-with-a-pole-is-refused` (arguments): every call to the tool had invalid arguments; reply does not say what it should
- `ode-no-closed-form-is-not-impossibility` (tool_selection): expected one of ['ode_analyze', 'ode_solve_symbolic'], called none
- `telescoping-gaussian-parameter` (arguments): every call to the tool had invalid arguments; 6 unchecked formula(s) in the reply
- `telescoping-refuses-trigonometric-integrand` (unproductive_loop): the turn ended early: repeated_calls; every call to the tool had invalid arguments
- `telescoping-undeclared-sign-is-not-assumed` (unproductive_loop): the turn ended early: repeated_calls; every call to the tool had invalid arguments
- `brachistochrone-euler-lagrange-leibniz-form` (model_honesty): 6 unchecked formula(s) in the reply
- `approximation-relativistic-energy` (unproductive_loop): the turn ended early: repeated_calls; expected one of ['approximation'], called ['series']
- `approximation-laurent-residue` (arguments): arguments do not say what the user asked; the tool's result is not the expected one (different input or status)
- `approximation-large-x` (tool_selection): expected one of ['approximation'], called ['series', 'simplify']
- `distributional-translated-source` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status); reply does not say what it should
- `distributional-point-charge-gauss` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status); reply does not say what it should
- `distributional-classical-versus-distributional` (resource_limit): the turn ended early: output_limit; expected one of ['vector_calculus'], called none; reply does not say what it should
- `distributional-wrong-normalisation-is-rejected` (arguments): arguments do not say what the user asked
- `distributional-unsupported-field-is-refused` (model_honesty): reply does not say what it should
- `homepage-coulomb-divergence` (arguments): the tool's result is not the expected one (different input or status); reply does not say what it should
- `homepage-stokes-curved-surface` (tool_selection): expected one of ['vector_calculus'], called none; reply does not say what it should

