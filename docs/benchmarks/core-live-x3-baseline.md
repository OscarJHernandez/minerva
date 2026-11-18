# Agent benchmark: core-live-x3-baseline

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `d2d3164e5b44` (161 cases, 3 repeat(s), seed 7, temperature 0.2). Code `064ec6e5c4`, prompt `1f6f9104c4ba`. The targets are goals, not results.

Model parameters: history_messages 20, history_tokens None, num_ctx 32768, num_predict None, temperature 0.2, think False, toolset auto, top_k None, top_p None.
Deterministic routing: on.

## Model under test

| Model | Weights as the host reports them | Defaults the host applies | Window loaded | Minerva sends | Thinking requested | Same weights through the run |
| --- | --- | --- | --- | --- | --- | --- |
| minerva | based on gemma4:26b, gemma4, 25.2B, Q4_K_M, digest 149a0ff8a1ca | top_k 64, top_p 0.9, draft_num_predict 3, num_ctx 16384, num_predict 1200, temperature 0.2 | 32768 | num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None (num_predict is left to the host: 1200) | off (model can think) | yes |

Seed 7, 3 repeat(s), compute timeout 10.0 s, request timeout 180.0 s; prompt `1f6f9104c4ba`, code `064ec6e5c4`; Ollama 0.40.3.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 267/444 (60.1%) | 82.7% | 83.7% | 91.7% | 0 | 257 of 673 | 6.8 | 20.6 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 82.7% | NOT met |
| valid arguments rate | 83.7% | NOT met |
| completion rate | 91.7% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| ambiguous | 29/45 (64.4%) | 93.3% |
| calculus | 59/93 (63.4%) | 93.5% |
| false_claim | 15/42 (35.7%) | 42.9% |
| followup | 13/18 (72.2%) | 100.0% |
| multistep | 27/30 (90.0%) | 100.0% |
| physics | 56/105 (53.3%) | 86.7% |
| retrieval | 22/27 (81.5%) | 81.5% |
| symbolic | 24/30 (80.0%) | 90.0% |
| tensor_matrix | 15/24 (62.5%) | 70.8% |
| unsupported | 7/30 (23.3%) | 50.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | analyze_lagrangian | approximation | check_units | creative_telescoping | differentiate | dimensional_analysis | distribution | electromagnetism | find_conserved_quantities | find_identities | get_entry | integrate | integrate_motion | isolate_variable | matrix_calculus | none | ode_analyze | ode_solve_numerical | perturbation_solve | search_handbook | series | simplify | solve_polynomial | statistical_mechanics | substitute | tensor_canonicalize | time_derivative | units | variation | vector_calculus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analyze_lagrangian | 9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| approximation |  | 13 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 5 |  |  |  |  |  |  |  |  |  |
| ask | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  | 4 | 19 | 2 |  |  |  |  |  |  |  |  |  |  |  | 2 |  |
| check_units |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 5 |  |  |
| creative_telescoping |  |  |  | 9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| differentiate |  |  |  |  | 21 |  |  |  |  |  |  |  |  |  |  | 6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| dimensional_analysis |  |  |  |  |  | 6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| distribution |  |  |  |  |  |  | 12 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| electromagnetism |  |  |  |  |  |  |  | 9 |  |  |  |  |  |  |  | 3 |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |
| expand_motion | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| find_conserved_quantities |  |  |  |  |  |  |  |  | 6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| find_identities |  |  |  |  |  |  |  |  |  | 9 |  |  |  |  |  | 5 |  |  |  | 3 |  | 1 |  |  |  |  |  |  |  |  |
| get_entry |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| integrate |  |  |  |  |  |  |  |  |  |  |  | 33 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| integrate_motion |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| isolate_variable |  |  |  |  |  |  |  |  |  |  |  |  |  | 6 |  | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| matrix_calculus |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 8 | 7 |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |
| none |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 15 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ode_analyze |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ode_solve_numerical |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 4 | 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| ode_solve_symbolic |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 | 6 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| perturbation_solve |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |  |
| search_handbook |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 5 |  |  |  | 7 |  |  |  |  |  |  |  |  |  |  |
| series |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |
| simplify |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 10 |  |  |  |  |  | 44 |  |  |  |  |  |  |  |  |
| solve_polynomial |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  | 6 |  |  |  |  |  |  |  |
| statistical_mechanics |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 | 7 |  |  |  |  |  |  |  | 17 |  |  |  |  |  |  |
| substitute |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 6 |  |  |  |  |  |
| tensor_canonicalize |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 6 |  |  |  |  |
| time_derivative |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |
| variation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 9 |  |
| vector_calculus |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  | 45 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| analyze_lagrangian | 9/9 | 100.0% | 0 of 9 |
| approximation | 9/18 | 72.2% | 3 of 51 |
| check_units | 6/6 | 16.7% | 5 of 11 |
| creative_telescoping | 4/9 | 100.0% | 3 of 17 |
| differentiate | 13/27 | 77.8% | 0 of 37 |
| dimensional_analysis | 3/6 | 100.0% | 0 of 6 |
| distribution | 7/12 | 100.0% | 11 of 24 |
| electromagnetism | 5/15 | 60.0% | 3 of 27 |
| expand_motion | 3/3 | 100.0% | 0 of 6 |
| find_conserved_quantities | 6/6 | 100.0% | 0 of 10 |
| find_identities | 13/18 | 72.2% | 0 of 13 |
| get_entry | 3/3 | 100.0% | 0 of 3 |
| integrate | 22/33 | 100.0% | 2 of 45 |
| integrate_motion | 0/3 | 100.0% | 0 of 5 |
| isolate_variable | 6/9 | 66.7% | 0 of 6 |
| matrix_calculus | 8/18 | 44.4% | 0 of 22 |
| ode_analyze | 3/3 | 100.0% | 0 of 3 |
| ode_solve_numerical | 6/6 | 100.0% | 2 of 18 |
| ode_solve_symbolic | 6/9 | 66.7% | 0 of 12 |
| perturbation_solve | 3/3 | 100.0% | 0 of 6 |
| search_handbook | 7/12 | 83.3% | 0 of 7 |
| series | 3/6 | 50.0% | 0 of 3 |
| simplify | 41/54 | 81.5% | 0 of 65 |
| solve_polynomial | 6/9 | 66.7% | 0 of 9 |
| statistical_mechanics | 0/27 | 63.0% | 59 of 72 |
| substitute | 6/6 | 100.0% | 0 of 9 |
| tensor_canonicalize | 6/6 | 100.0% | 3 of 12 |
| time_derivative | 3/3 | 100.0% | 0 of 3 |
| variation | 7/9 | 100.0% | 0 of 14 |
| vector_calculus | 19/51 | 88.2% | 13 of 120 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 6.8 | 20.6 | 71.1 |
| model calls (ms) | 4848 | 17993 | 67050 |
| tool calls (ms) | 1111 | 3762 | 17518 |

Tokens in/out: 9611789 / 95660.

First output after 575 ms (median) · tool calls that errored: 112 · tool results not verified: 179 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 15639.
Verified-result coverage: 88.4% of 292 runs that expect a card · model calls per run: median 2.0, mean 2.6 · multi-turn consistency: n/a of 0 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 61 |
| model_honesty | 36 |
| arguments | 34 |
| unproductive_loop | 33 |
| parsing | 6 |
| resource_limit | 4 |
| missing_assumptions | 3 |

Formulas the model wrote: 1765 unchecked, 4 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 36.

Unstable across repeats (1.41 std of passes per repeat): `brachistochrone-euler-lagrange-leibniz-form`, `delta-integral-uses-the-tool`, `differentiate-mixed`, `differentiate-second`, `distributional-coulomb-divergence`, `em-wave-equation-from-maxwell`, `false-log-sum`, `false-pythagoras`, `followup-second-derivative`, `handbook-constant`, `homepage-coulomb-divergence`, `matrix-evolution-asks-for-the-span`, `matrix-integral-not-evaluated`, `matrix-noncommuting-explicit-product`, `telescoping-undeclared-sign-is-not-assumed`, `unsupported-no-closed-form`, `vector-curl-curl-rewrite`, `vector-curl-grad-needs-smoothness`.

### Failed runs

- `simplify-rational` (model_honesty): 1 unchecked formula(s) in the reply
- `sqrt-no-assumptions` (tool_selection): expected one of ['simplify'], called none
- `definite-by-tool` (model_honesty): 1 unchecked formula(s) in the reply
- `differentiate-first` (model_honesty): 1 unchecked formula(s) in the reply
- `differentiate-second` (arguments): the tool's result is not the expected one (different input or status)
- `integrate-definite` (model_honesty): 1 unchecked formula(s) in the reply
- `integrate-needs-assumptions` (missing_assumptions): arguments do not say what the user asked; the tool's result is not the expected one (different input or status)
- `handbook-constant` (tool_selection): expected one of ['get_entry', 'search_handbook'], called none
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
- `telescoping-gaussian-parameter` (model_honesty): 7 unchecked formula(s) in the reply
- `telescoping-undeclared-sign-is-not-assumed` (parsing): the tool's result is not the expected one (different input or status)
- `brachistochrone-euler-lagrange-leibniz-form` (model_honesty): 8 unchecked formula(s) in the reply
- `approximation-relativistic-energy` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status)
- `approximation-laurent-residue` (unproductive_loop): the turn ended early: repeated_calls; arguments do not say what the user asked; the tool's result is not the expected one (different input or status)
- `approximation-large-x` (unproductive_loop): the turn ended early: repeated_calls; expected one of ['approximation'], called ['series']
- `distributional-translated-source` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status); reply does not say what it should
- `distributional-point-charge-gauss` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status)
- `distributional-classical-versus-distributional` (unproductive_loop): the turn ended early: repeated_calls; every call to the tool had invalid arguments; arguments do not say what the user asked; reply does not say what it should
- `distributional-wrong-normalisation-is-rejected` (arguments): arguments do not say what the user asked
- `homepage-coulomb-divergence` (model_honesty): reply says what it should not
- `differentiate-mixed` (resource_limit): the turn ended early: output_limit; reply too long (679 words)
- `followup-second-derivative` (resource_limit): the turn ended early: step_limit; arguments do not say what the user asked
- `distributional-coulomb-divergence` (model_honesty): reply says what it should not
- `false-pythagoras` (tool_selection): expected one of ['find_identities', 'search_handbook', 'simplify'], called none
- `em-wave-equation-from-maxwell` (arguments): arguments do not say what the user asked

