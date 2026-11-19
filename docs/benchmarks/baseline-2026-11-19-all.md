# Agent benchmark: baseline-HEAD

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `091eb4bf23e4` (183 cases, 3 repeat(s), seed 7, temperature 0.2). Code `c7f00a3e95`, prompt `3125a4c9ab20`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: on.

## Model under test

| Model | Weights as the host reports them | Defaults the host applies | Window loaded | Minerva sends | Thinking requested | Same weights through the run |
| --- | --- | --- | --- | --- | --- | --- |
| minerva | based on gemma4:26b, gemma4, 25.2B, Q4_K_M, digest 149a0ff8a1ca | top_k 64, top_p 0.9, draft_num_predict 3, num_ctx 16384, num_predict 1200, temperature 0.2 | 32768 | num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None (num_predict is left to the host: 1200) | off (model can think) | yes |

Seed 7, 3 repeat(s), compute timeout 10.0 s, request timeout 180.0 s; prompt `3125a4c9ab20`, code `c7f00a3e95`; Ollama 0.40.3.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 363/510 (71.2%) | 86.1% | 88.0% | 93.7% | 0 | 258 of 798 | 7.0 | 20.2 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 86.1% | NOT met |
| valid arguments rate | 88.0% | NOT met |
| completion rate | 93.7% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| symbolic | 33/36 (91.7%) | 91.7% |
| multistep | 42/42 (100.0%) | 100.0% |
| calculus | 65/99 (65.7%) | 92.9% |
| retrieval | 21/27 (77.8%) | 77.8% |
| tensor_matrix | 28/33 (84.8%) | 100.0% |
| physics | 67/111 (60.4%) | 85.6% |
| ambiguous | 41/51 (80.4%) | 86.3% |
| unsupported | 8/30 (26.7%) | 50.0% |
| false_claim | 38/54 (70.4%) | 68.5% |
| followup | 20/27 (74.1%) | 100.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | approximation | check_identity | check_units | creative_telescoping | differentiate | dimensional_analysis | distribution | electromagnetism | find_conserved_quantities | find_identities | get_entry | integrate | integrate_motion | isolate_variable | matrix_calculus | none | ode_analyze | ode_solve_numerical | ode_solve_symbolic | perturbation_solve | search_handbook | series | simplify | solve_polynomial | statistical_mechanics | substitute | tensor_canonicalize | time_derivative | units | variation | vector_calculus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analyze_lagrangian | 18 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| approximation |  | 15 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |
| ask | 3 |  | 3 |  |  | 3 |  |  |  |  |  |  |  |  |  | 4 | 18 | 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| check_identity |  |  | 6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| check_units |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 5 |  |  |
| creative_telescoping |  |  |  |  | 9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| differentiate |  |  | 3 |  |  | 24 |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| dimensional_analysis |  |  |  |  |  |  | 6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| distribution |  |  |  |  |  |  |  | 12 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| electromagnetism |  |  |  |  |  |  |  |  | 10 |  |  |  |  |  |  |  | 3 |  |  |  |  | 2 |  |  |  |  |  |  |  |  |  |  |
| expand_motion | 6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| find_conserved_quantities |  |  |  |  |  |  |  |  |  | 6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| find_identities |  |  | 3 |  |  |  |  |  |  |  | 9 |  |  |  |  |  | 3 |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |
| get_entry |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| integrate |  |  |  |  |  |  |  |  |  |  |  |  | 39 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| integrate_motion |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| isolate_variable |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 7 |  | 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| matrix_calculus |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 21 |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |
| none |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 18 |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |
| ode_analyze |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ode_solve_numerical |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 8 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ode_solve_symbolic |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 | 5 |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| perturbation_solve |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |  |
| search_handbook |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 6 |  |  |  |  | 6 |  |  |  |  |  |  |  |  |  |  |
| series |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |
| simplify |  |  | 15 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 54 |  |  |  |  |  |  |  |  |
| solve_polynomial |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  | 6 |  |  |  |  |  |  |  |
| statistical_mechanics |  |  | 3 |  |  |  |  |  |  |  |  |  |  |  |  | 3 | 10 |  |  |  |  |  |  |  |  | 11 |  |  |  |  |  |  |
| substitute |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 9 |  |  |  |  |  |
| tensor_canonicalize |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 6 |  |  |  |  |
| time_derivative |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |
| variation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 9 |  |
| vector_calculus |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  | 6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 42 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| analyze_lagrangian | 17/18 | 100.0% | 0 of 24 |
| approximation | 9/18 | 83.3% | 0 of 41 |
| check_identity | 6/6 | 100.0% | 0 of 6 |
| check_units | 6/6 | 16.7% | 5 of 11 |
| creative_telescoping | 6/9 | 100.0% | 3 of 15 |
| differentiate | 19/30 | 80.0% | 0 of 39 |
| dimensional_analysis | 3/6 | 100.0% | 0 of 6 |
| distribution | 8/12 | 100.0% | 11 of 24 |
| electromagnetism | 5/15 | 66.7% | 10 of 37 |
| expand_motion | 6/6 | 100.0% | 0 of 12 |
| find_conserved_quantities | 6/6 | 100.0% | 0 of 11 |
| find_identities | 15/18 | 83.3% | 0 of 15 |
| get_entry | 3/3 | 100.0% | 0 of 3 |
| integrate | 27/39 | 100.0% | 2 of 51 |
| integrate_motion | 0/3 | 100.0% | 0 of 6 |
| isolate_variable | 7/9 | 77.8% | 0 of 7 |
| matrix_calculus | 19/24 | 87.5% | 10 of 42 |
| ode_analyze | 6/6 | 100.0% | 0 of 10 |
| ode_solve_numerical | 9/9 | 100.0% | 3 of 26 |
| ode_solve_symbolic | 6/9 | 66.7% | 0 of 15 |
| perturbation_solve | 2/3 | 100.0% | 1 of 9 |
| search_handbook | 6/12 | 75.0% | 0 of 6 |
| series | 3/6 | 50.0% | 0 of 3 |
| simplify | 65/69 | 95.7% | 0 of 103 |
| solve_polynomial | 6/9 | 66.7% | 0 of 9 |
| statistical_mechanics | 6/27 | 40.7% | 26 of 64 |
| substitute | 9/9 | 100.0% | 0 of 12 |
| tensor_canonicalize | 6/6 | 100.0% | 0 of 10 |
| time_derivative | 3/3 | 100.0% | 0 of 3 |
| variation | 8/9 | 100.0% | 0 of 13 |
| vector_calculus | 23/51 | 82.4% | 13 of 123 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 7.0 | 20.2 | 28.0 |
| model calls (ms) | 4832 | 15801 | 25811 |
| tool calls (ms) | 1080 | 3789 | 12956 |

Tokens in/out: 11213883 / 107240.

First output after 525 ms (median) · tool calls that errored: 98 · tool results not verified: 217 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 15655.
Verified-result coverage: 90.1% of 364 runs that expect a card · model calls per run: median 2.0, mean 2.6 · multi-turn consistency: 96.3% of 27 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 57 |
| model_honesty | 27 |
| arguments | 24 |
| unproductive_loop | 23 |
| resource_limit | 9 |
| parsing | 4 |
| missing_assumptions | 3 |

Formulas the model wrote: 1305 unchecked, 3 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 33.

Unstable across repeats (2.16 std of passes per repeat): `ambiguous-matrix-sizes`, `assumptions-carried`, `brachistochrone-euler-lagrange-leibniz-form`, `delta-integral-uses-the-tool`, `derivative-of-a-functional-is-refused-not-improvised`, `differentiate-first`, `distributional-coulomb-divergence`, `distributional-point-charge-gauss`, `distributional-unsupported-field-is-refused`, `em-wave-equation-from-maxwell`, `isolate-unsupported-is-not-answered`, `matrix-noncommuting-explicit-product`, `oscillator-equations-of-motion-then-conserved-quantity`, `perturbation-duffing`, `statmech-divergent-series-is-refused`, `statmech-ising-thermodynamic-limit`, `statmech-oscillator-heat-capacity`, `statmech-two-level-free-energy`, `vector-curl-curl-rewrite`.

### Failed runs

- `hyperbolic` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called ['check_identity']
- `differentiate-second` (arguments): the tool's result is not the expected one (different input or status)
- `integrate-definite` (model_honesty): 1 unchecked formula(s) in the reply
- `integrate-needs-assumptions` (missing_assumptions): arguments do not say what the user asked; the tool's result is not the expected one (different input or status)
- `handbook-constant` (tool_selection): expected one of ['get_entry', 'search_handbook'], called none
- `identities-blocked` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `matrix-evolution-asks-for-the-span` (unproductive_loop): the turn ended early: repeated_calls; reply does not say what it should
- `motion-values-given` (model_honesty): 1 equation(s) that do not hold as written
- `statmech-oscillator-heat-capacity` (arguments): the tool's result is not the expected one (different input or status)
- `statmech-ideal-gas-conventions-are-the-users` (tool_selection): expected one of ['statistical_mechanics'], called none
- `statmech-wrong-partition-function-is-not-confirmed` (tool_selection): expected one of ['statistical_mechanics'], called ['check_identity']
- `statmech-noncommuting-response-is-not-claimed` (unproductive_loop): the turn ended early: repeated_calls; expected one of ['statistical_mechanics'], called ['matrix_calculus']; reply does not say what it should
- `statmech-fermi-number-and-variance` (tool_selection): expected one of ['statistical_mechanics'], called none
- `statmech-ising-thermodynamic-limit` (resource_limit): the turn ended early: step_limit
- `statmech-stirling-is-an-approximation` (tool_selection): expected one of ['statistical_mechanics'], called none
- `dimensional-pendulum` (arguments): arguments do not say what the user asked
- `unsupported-derivative-abs` (tool_selection): expected one of ['differentiate'], called none
- `unsupported-no-closed-form` (model_honesty): reply says what it should not
- `unsupported-series-pole` (tool_selection): expected one of ['series'], called none
- `unsupported-transcendental` (tool_selection): expected one of ['solve_polynomial'], called ['isolate_variable']
- `unsupported-magnus` (tool_selection): expected one of ['matrix_calculus'], called ['search_handbook']; reply does not say what it should; 11 unchecked formula(s) in the reply
- `unsupported-quantum` (model_honesty): reply does not say what it should
- `followup-second-derivative` (arguments): arguments do not say what the user asked
- `followup-explain-no-tool` (model_honesty): 6 unchecked formula(s) in the reply
- `delta-on-a-limit-asks-for-a-convention` (arguments): the tool's result is not the expected one (different input or status)
- `matrix-noncommuting-explicit-product` (arguments): every call to the tool had invalid arguments
- `vector-curl-grad-with-smoothness` (resource_limit): the turn ended early: output_limit; the tool's result is not the expected one (different input or status)
- `vector-curl-curl-rewrite-needs-smoothness` (parsing): the tool's result is not the expected one (different input or status)
- `em-adopting-premises-is-the-users-choice` (tool_selection): expected one of ['electromagnetism'], called none
- `em-gauge-invariance` (unproductive_loop): the turn ended early: repeated_calls; arguments do not say what the user asked
- `es-laplace-rectangle` (tool_selection): expected one of ['electromagnetism'], called ['search_handbook']; reply does not say what it should
- `integral-flux-through-a-sphere` (tool_selection): expected one of ['vector_calculus'], called none; reply does not say what it should
- `integral-theorem-with-a-pole-is-refused` (arguments): every call to the tool had invalid arguments; reply does not say what it should
- `ode-no-closed-form-is-not-impossibility` (tool_selection): expected one of ['ode_analyze', 'ode_solve_symbolic'], called none
- `telescoping-gaussian-parameter` (model_honesty): 7 unchecked formula(s) in the reply
- `brachistochrone-euler-lagrange-leibniz-form` (model_honesty): 5 unchecked formula(s) in the reply
- `approximation-relativistic-energy` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status)
- `approximation-laurent-residue` (arguments): arguments do not say what the user asked; the tool's result is not the expected one (different input or status)
- `approximation-large-x` (tool_selection): expected one of ['approximation'], called ['series', 'simplify']
- `distributional-translated-source` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status); reply does not say what it should
- `distributional-classical-versus-distributional` (arguments): arguments do not say what the user asked; reply too long (222 words)
- `homepage-coulomb-divergence` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status); reply does not say what it should
- `homepage-stokes-curved-surface` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status); reply does not say what it should
- `oscillating-integral-stops-when-not-verified` (model_honesty): 6 unchecked formula(s) in the reply
- `derivative-of-a-functional-is-refused-not-improvised` (model_honesty): reply does not say what it should
- `answer-without-thinking-aloud` (tool_selection): called tools unexpectedly: ['search_handbook']
- `oscillator-equations-of-motion-then-conserved-quantity` (model_honesty): turn 2: reply says what it should not
- `differentiate-first` (model_honesty): 1 unchecked formula(s) in the reply
- `ambiguous-matrix-sizes` (tool_selection): called forbidden tool(s) ['matrix_calculus']
- `isolate-unsupported-is-not-answered` (tool_selection): expected one of ['isolate_variable'], called none; reply does not say what it should
- `vector-curl-curl-rewrite` (arguments): arguments do not say what the user asked
- `distributional-unsupported-field-is-refused` (unproductive_loop): the turn ended early: repeated_calls; reply does not say what it should
- `perturbation-duffing` (resource_limit): the turn ended early: step_limit; the tool's result is not the expected one (different input or status)
- `statmech-two-level-free-energy` (arguments): the tool's result is not the expected one (different input or status)
- `statmech-divergent-series-is-refused` (tool_selection): expected one of ['statistical_mechanics'], called none; reply does not say what it should
- `assumptions-carried` (resource_limit): the turn ended early: step_limit
- `delta-integral-uses-the-tool` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status)
- `em-wave-equation-from-maxwell` (arguments): arguments do not say what the user asked
- `distributional-coulomb-divergence` (model_honesty): reply says what it should not
- `distributional-point-charge-gauss` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status)

