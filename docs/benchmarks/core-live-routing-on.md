# Agent benchmark: core-routing-on

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `d2d3164e5b44` (161 cases, 1 repeat(s), seed 7, temperature 0.2). Code `e5f9a5011e`, prompt `1f6f9104c4ba`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: on.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 87/148 (58.8%) | 82.4% | 79.5% | 90.5% | 0 | 93 of 224 | 6.6 | 19.6 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 82.4% | NOT met |
| valid arguments rate | 79.5% | NOT met |
| completion rate | 90.5% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| symbolic | 9/10 (90.0%) | 100.0% |
| multistep | 9/10 (90.0%) | 100.0% |
| calculus | 19/31 (61.3%) | 90.3% |
| retrieval | 7/9 (77.8%) | 77.8% |
| tensor_matrix | 5/8 (62.5%) | 75.0% |
| physics | 19/35 (54.3%) | 82.9% |
| ambiguous | 10/15 (66.7%) | 93.3% |
| unsupported | 0/10 (0.0%) | 50.0% |
| false_claim | 5/14 (35.7%) | 50.0% |
| followup | 4/6 (66.7%) | 100.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | approximation | creative_telescoping | differentiate | dimensional_analysis | distribution | electromagnetism | find_conserved_quantities | find_identities | get_entry | integrate | integrate_motion | isolate_variable | matrix_calculus | none | ode_analyze | ode_solve_numerical | perturbation_solve | search_handbook | series | simplify | solve_polynomial | statistical_mechanics | substitute | tensor_canonicalize | time_derivative | units | variation | vector_calculus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analyze_lagrangian | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| approximation |  | 4 |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  |
| ask | 1 |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 7 |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |
| check_units |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |
| creative_telescoping |  |  | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| differentiate |  |  |  | 7 |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| dimensional_analysis |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| distribution |  |  |  |  |  | 4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| electromagnetism |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| expand_motion | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| find_conserved_quantities |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| find_identities |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  | 2 |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |
| get_entry |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| integrate |  |  |  |  |  |  |  |  |  |  | 11 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| integrate_motion |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| isolate_variable |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| matrix_calculus |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 | 2 |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |
| none |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ode_analyze |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ode_solve_numerical |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| ode_solve_symbolic |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| perturbation_solve |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |
| search_handbook |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |
| series |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  |
| simplify |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  | 16 |  |  |  |  |  |  |  |  |
| solve_polynomial |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |
| statistical_mechanics |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 2 |  |  |  |  |  |  |  | 6 |  |  |  |  |  |  |
| substitute |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |
| tensor_canonicalize |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  |
| time_derivative |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |
| variation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |
| vector_calculus |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |  |  |  | 14 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| analyze_lagrangian | 3/3 | 100.0% | 0 of 3 |
| approximation | 3/6 | 66.7% | 1 of 14 |
| check_units | 2/2 | 0.0% | 2 of 4 |
| creative_telescoping | 0/3 | 100.0% | 10 of 9 |
| differentiate | 5/9 | 77.8% | 0 of 13 |
| dimensional_analysis | 1/2 | 100.0% | 0 of 2 |
| distribution | 2/4 | 100.0% | 4 of 8 |
| electromagnetism | 2/5 | 60.0% | 3 of 10 |
| expand_motion | 1/1 | 100.0% | 0 of 2 |
| find_conserved_quantities | 2/2 | 100.0% | 0 of 4 |
| find_identities | 4/6 | 66.7% | 0 of 4 |
| get_entry | 1/1 | 100.0% | 0 of 1 |
| integrate | 7/11 | 100.0% | 0 of 15 |
| integrate_motion | 0/1 | 100.0% | 0 of 2 |
| isolate_variable | 2/3 | 66.7% | 0 of 2 |
| matrix_calculus | 3/6 | 50.0% | 0 of 7 |
| ode_analyze | 1/1 | 100.0% | 0 of 1 |
| ode_solve_numerical | 2/2 | 100.0% | 0 of 5 |
| ode_solve_symbolic | 2/3 | 66.7% | 0 of 4 |
| perturbation_solve | 1/1 | 100.0% | 0 of 2 |
| search_handbook | 2/4 | 75.0% | 0 of 2 |
| series | 1/2 | 50.0% | 0 of 1 |
| simplify | 15/18 | 88.9% | 0 of 20 |
| solve_polynomial | 2/3 | 66.7% | 0 of 3 |
| statistical_mechanics | 0/9 | 66.7% | 19 of 27 |
| substitute | 2/2 | 100.0% | 0 of 3 |
| tensor_canonicalize | 2/2 | 100.0% | 1 of 4 |
| time_derivative | 1/1 | 100.0% | 0 of 1 |
| variation | 2/3 | 100.0% | 0 of 5 |
| vector_calculus | 5/17 | 82.4% | 3 of 39 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 6.6 | 19.6 | 25.9 |
| model calls (ms) | 4960 | 16029 | 24121 |
| tool calls (ms) | 1187 | 4419 | 12992 |

Tokens in/out: 3257489 / 31565.

First output after 546 ms (median) · tool calls that errored: 43 · tool results not verified: 54 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 15639.
Verified-result coverage: 91.4% of 93 runs that expect a card · model calls per run: median 2.0, mean 2.6 · multi-turn consistency: n/a of 0 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 22 |
| model_honesty | 13 |
| unproductive_loop | 12 |
| arguments | 10 |
| resource_limit | 2 |
| missing_assumptions | 1 |
| parsing | 1 |

Formulas the model wrote: 598 unchecked, 1 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 16.

### Failed runs

- `simplify-rational` (model_honesty): 1 unchecked formula(s) in the reply
- `definite-by-tool` (model_honesty): 1 unchecked formula(s) in the reply
- `differentiate-first` (model_honesty): 1 unchecked formula(s) in the reply
- `integrate-definite` (model_honesty): 1 unchecked formula(s) in the reply
- `integrate-needs-assumptions` (missing_assumptions): arguments do not say what the user asked; the tool's result is not the expected one (different input or status)
- `handbook-constant` (tool_selection): expected one of ['get_entry', 'search_handbook'], called none
- `identities-blocked` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `matrix-merge-needs-commutation` (tool_selection): expected one of ['matrix_calculus'], called none
- `matrix-evolution-asks-for-the-span` (model_honesty): reply does not say what it should
- `motion-values-given` (model_honesty): 1 equation(s) that do not hold as written
- `statmech-two-level-free-energy` (unproductive_loop): the turn ended early: repeated_calls; every call to the tool had invalid arguments
- `statmech-oscillator-heat-capacity` (unproductive_loop): the turn ended early: repeated_calls; every call to the tool had invalid arguments
- `statmech-ideal-gas-conventions-are-the-users` (tool_selection): expected one of ['statistical_mechanics'], called none
- `statmech-divergent-series-is-refused` (unproductive_loop): the turn ended early: repeated_calls; every call to the tool had invalid arguments; reply does not say what it should
- `statmech-wrong-partition-function-is-not-confirmed` (unproductive_loop): the turn ended early: repeated_calls; every call to the tool had invalid arguments; reply too long (233 words)
- `statmech-noncommuting-response-is-not-claimed` (unproductive_loop): the turn ended early: repeated_calls; expected one of ['statistical_mechanics'], called ['matrix_calculus']; reply does not say what it should
- `statmech-fermi-number-and-variance` (tool_selection): expected one of ['statistical_mechanics'], called none
- `statmech-ising-thermodynamic-limit` (arguments): every call to the tool had invalid arguments
- `statmech-stirling-is-an-approximation` (arguments): every call to the tool had invalid arguments
- `dimensional-pendulum` (arguments): arguments do not say what the user asked
- `ask-when-ambiguous` (model_honesty): reply does not say what it should
- `ambiguous-simplify` (model_honesty): reply does not say what it should
- `isolate-unsupported-is-not-answered` (tool_selection): expected one of ['isolate_variable'], called none; reply does not say what it should
- `unsupported-derivative-abs` (tool_selection): expected one of ['differentiate'], called none
- `unsupported-no-closed-form` (model_honesty): reply says what it should not
- `unsupported-series-pole` (tool_selection): expected one of ['series'], called none
- `unsupported-transcendental` (tool_selection): expected one of ['solve_polynomial'], called ['isolate_variable']
- `unsupported-magnus` (tool_selection): expected one of ['matrix_calculus'], called ['search_handbook']; reply does not say what it should; 10 unchecked formula(s) in the reply
- `unsupported-quantum` (model_honesty): reply does not say what it should; 6 unchecked formula(s) in the reply
- `false-log-sum` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none; reply does not say what it should
- `false-sqrt-square` (tool_selection): expected one of ['find_identities', 'simplify'], called none
- `false-log-exp` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `false-derivative` (tool_selection): expected one of ['differentiate'], called none
- `followup-second-derivative` (arguments): arguments do not say what the user asked
- `followup-explain-no-tool` (model_honesty): 10 unchecked formula(s) in the reply
- `delta-integral-uses-the-tool` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status)
- `delta-on-a-limit-asks-for-a-convention` (arguments): the tool's result is not the expected one (different input or status)
- `matrix-noncommuting-explicit-product` (tool_selection): expected one of ['matrix_calculus'], called none
- `vector-curl-grad-with-smoothness` (arguments): the tool's result is not the expected one (different input or status)
- `vector-curl-curl-rewrite` (arguments): arguments do not say what the user asked
- `vector-curl-curl-rewrite-needs-smoothness` (parsing): the tool's result is not the expected one (different input or status)
- `em-adopting-premises-is-the-users-choice` (tool_selection): expected one of ['electromagnetism'], called none
- `em-gauge-invariance` (unproductive_loop): the turn ended early: repeated_calls; arguments do not say what the user asked
- `es-laplace-rectangle` (tool_selection): expected one of ['electromagnetism'], called none; reply does not say what it should
- `integral-flux-through-a-sphere` (tool_selection): expected one of ['vector_calculus'], called none; reply does not say what it should
- `integral-theorem-with-a-pole-is-refused` (arguments): every call to the tool had invalid arguments; reply does not say what it should
- `ode-no-closed-form-is-not-impossibility` (tool_selection): expected one of ['ode_analyze', 'ode_solve_symbolic'], called none
- `telescoping-gaussian-parameter` (arguments): every call to the tool had invalid arguments; 12 unchecked formula(s) in the reply
- `telescoping-refuses-trigonometric-integrand` (unproductive_loop): the turn ended early: repeated_calls; every call to the tool had invalid arguments
- `telescoping-undeclared-sign-is-not-assumed` (unproductive_loop): the turn ended early: repeated_calls; every call to the tool had invalid arguments
- `brachistochrone-euler-lagrange-leibniz-form` (model_honesty): 6 unchecked formula(s) in the reply
- `approximation-relativistic-energy` (unproductive_loop): the turn ended early: repeated_calls; expected one of ['approximation'], called ['series']
- `approximation-laurent-residue` (unproductive_loop): the turn ended early: repeated_calls; arguments do not say what the user asked; the tool's result is not the expected one (different input or status)
- `approximation-large-x` (tool_selection): expected one of ['approximation'], called none
- `distributional-translated-source` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status); reply does not say what it should
- `distributional-point-charge-gauss` (tool_selection): expected one of ['vector_calculus'], called ['electromagnetism']
- `distributional-classical-versus-distributional` (resource_limit): the turn ended early: step_limit; the tool's result is not the expected one (different input or status); reply does not say what it should
- `distributional-wrong-normalisation-is-rejected` (arguments): arguments do not say what the user asked
- `distributional-unsupported-field-is-refused` (model_honesty): reply does not say what it should; reply says what it should not
- `homepage-coulomb-divergence` (resource_limit): the turn ended early: step_limit; reply does not say what it should
- `homepage-stokes-curved-surface` (tool_selection): expected one of ['vector_calculus'], called none; reply does not say what it should

