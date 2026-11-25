# Evaluation matrix

One row per registered agent tool: what independently checks its mathematics, which evaluation exercises the
model's use of it, what that evidence cannot show, and the rule for a regression. A test
(`services/api/tests/test_reliability_docs.py`) fails if a registered tool has no row.

Layers, from cheapest to dearest:

| Layer | Runs | What it shows | What it cannot show |
| --- | --- | --- | --- |
| Engine oracles (`packages/engine/tests`, `minerva_engine/reference/`, domain `benchmark.py`) | every PR | the result agrees with a route that shares no code with the one that built it | that the model chose the tool |
| Adversarial and property tests | PR (derandomized), nightly (extended) | false candidates stay rejected, true twins stay accepted | cases nobody thought of |
| Reference calls of the agent benchmark (`cases.yaml`, `cases_reliability.yaml`) | every PR (`test_evaluation.py`) | the benchmark's ground truth still matches the engine | model behaviour |
| Offline orchestration (`scripts/eval_offline.py`, replayed model) | every PR (`--check`) | the real loop and tools, tool names, notation, routing, evidence, tokens, for a *scripted* model | that a real model behaves so |
| Corruption (`scripts/eval_corruptions.py`) | every PR (`test_argument_corruption.py`) | a known model mistake in a reference call gives the clean result or a typed refusal | mistakes not in the operator list |
| Fault injection (`test_fault_injection.py`) | every PR | a stalled or failing host, a dead or hung sandbox child, a client that leaves, concurrent turns: bounded turn, one stop reason, nothing falsely verified | provider behaviour on the wire |
| Failure fixtures (`tests/fixtures/failures`, `test_failure_fixtures.py`) | every PR | a recorded failure class stays classified and handled | new failures |
| Live model (`scripts/eval_models.py`) | by hand (`model-eval.yml`) | pass rate, tool choice and latency of a real model on this host | anything about the code on its own: runs are noisy |

A scripted or replayed run is not evidence about a real model, and a live run is not evidence about the engine.

## What a live run is scored on

`passed` is the conjunction of every check of a case and stays the headline number. Next to it each run carries five
dimensions that say *what* went wrong (`CaseResult` in `agent/evaluation/score.py`, summarised per run set in the report
as "Dimensions"; completion has always been its own metric):

| Dimension | True when | Not the same as |
| --- | --- | --- |
| operation_ok | an accepted tool was used with arguments that say what the user asked (no tool-selection, argument or parse failure) | `tool_selected` (only the *first* call) |
| math_ok | no false or contradicted statement, no correct statement marked contradicted, and the card is the expected one where the case states one | the reply being well worded |
| evidence_ok | nothing false shown as verified; the restriction, the stop after a failed verification and the reuse of a result were respected | `math_ok`: a right answer can still lack its evidence |
| efficient | no call beyond what the case needs and no repeated call | failing: waste never fails a case by itself |
| consistent | every turn of a multi-turn case passed (None for one turn) | |

`accept:` lists other tools that answer correctly, each with its own `args`, `expect_card` and `reference`
(checked against the engine by `--check-cases`). Unlike `also_ok` (acceptable, never looked at), an alternative is
checked like the expected tool, and a run passes if any accepted tool met its own checks; a correct answer through
an accepted alternative is no longer a tool-selection failure. A case without `accept` scores exactly as before, and
the field is left out of the case digest while empty.

`claim_marks:` (a case or a follow-up; issue #172) pins the mark a displayed formula of the reply must carry and,
optionally, the `evidence_rule` the claim checker recorded for it (`debug.claims`): `[{formula, status,
evidence_rule}]`, the formula compared as `wrong_formulas` are. A verified mark where another is pinned is a
`verification` failure, any other difference a `claim_checker_false_positive`. It is for deterministic
conversations (`test_conserved_quantity_multiturn.py`); no live case needs a model to produce a particular
mistake. Like `accept`, it is left out of the case digest while empty, in a follow-up too.

`--compare` states, for every change in passes, whether it is within the run-to-run noise: the spread (population
standard deviation) of the passes per repeat on each side, a limit of twice the larger spread and at least one case. With
one repeat on a side the noise cannot be measured and nothing is called larger than it. Cases that are unstable across
repeats are marked in the lists of improved and worse cases.

## Per tool

Cases: benchmark cases that expect the tool; refs: those with a reference call that the PR tests run against the
engine. "Oracle" is the independent check named in `AGENTS.md`.

| Tool | Group | Cases / refs | Oracle | Limits | Regression rule |
| --- | --- | --- | --- | --- | --- |
| `simplify` | core | 22 / 19 | numeric sampling plus symbolic zero, targeted probes | sampling is evidence, not proof | adversarial corpus row (false identity and true twin) |
| `check_identity` | identity (keyword group) | 2 / 2 | the claim checker's gates, then sampled points plus targeted probes and a symbolic zero; a witness is confirmed at 60 and 120 digits | a disagreement is shown only with its exact point; a relation, matrix, vector, distribution or sum is never called true or false | adversarial pairs in `test_identity_check.py` |
| `find_identities` | core | 6 / 4 | rule application re-verified by the engine | retrieval only proposes | benchmark case |
| `isolate_variable` | core | 3 / 3 | back-substitution and root count | no systems, no periodic inverses | reference case |
| `solve_polynomial` | core | 3 / 3 | substitution and numerical roots | degree 3+ symbolic is unsupported | reference case |
| `differentiate` | core | 10 / 10 | independent numerical derivative, real and imaginary direction | partial derivative only | wrong-candidate test |
| `series` | core | 2 / 2 | Cauchy-integral coefficients and remainder decay | finite real point, no poles | wrong-by-1e-9 test |
| `integrate` | core | 12 / 11 | numerical derivative (antiderivative), quadrature (definite) | "no closed form" is not "none exists" | reference problem |
| `creative_telescoping` | core | 3 / 3 | exact certificate check plus quadrature | hyperexponential integrands only | benchmark case |
| `substitute` | core | 3 / 3 | whole-expression re-verification, restrictions carried | a premise is never checked | reference case |
| `rewrite_part` | core | 1 / 1 | whole-expression re-verification through the rule matcher | one step chosen by the user | reference case |
| `search_handbook` | core | 5 / 3 | entries are machine-checked at their level | "no entry matched" is an answer | handbook `verify` |
| `get_entry` | core | 1 / 1 | the same | `stated` entries are never applied | handbook `verify` |
| `load_tools` | core | none (offline suite) | not mathematical | loads groups only | offline trajectories |
| `analyze_lagrangian` | mechanics | 6 / 4 | numerical differentiation of L, confirmed equations | at most the stated coordinate count | reference case, corruption sweep |
| `routh_reduce` | mechanics | 1 / 1 | momenta verified conserved; R read back and checked on motions of L (Euler-Lagrange residual, -dR/dp, R = L - Σ p ċ); symbolic leg reported apart | only coordinates the caller names; linear momenta, nonsingular block; p_c stays a symbol | wrong twins (sign, dropped 1/f, 1e-6, +C, +f(t), +dF/dt); reference problem; reliability case `schwarzschild-routh-reduction-by-time` |
| `effective_potential` | reduction | none yet (see #206) | plane shown consistent by computation; Routh verification; two numerical legs on the full geodesic equations (equation on g(v, v) = kappa, r_ddot = -V_eff'/2); symbolic leg reported apart | the normalisation is the user's; explicit metric, no declared functions; one radial coordinate | wrong twins (dropped f, sign, 1e-6); reference problems; offline suite pending a case |
| `time_derivative` | mechanics | 2 / 2 | central difference along random motions | no accelerations in the input | wrong-candidate test; multi-turn case `kepler-discovery-then-derivative-of-a-relation-among-the-constants` and `test_conserved_quantity_multiturn.py` |
| `expand_motion` | mechanics | 2 / 2 | `check_series` on every expansion | no stability claims | reference case |
| `perturbation_solve` | mechanics | 1 / 1 | residual decay at the claimed order | one coordinate, no damping | wrong-frequency test |
| `integrate_motion` | mechanics | 1 / 1 | second integrator, time reversal, conserved quantities | numbers are the user's | reference case |
| `find_conserved_quantities` | mechanics | 2 / 2 | `Flow.verify` on the equations of motion | within the explored families only | discovery benchmark; first turn of `kepler-discovery-then-derivative-of-a-relation-among-the-constants` |
| `search_invariants` | mechanics | none | the same verification, fixed ansatz | no reference call (`corruption.EXEMPT`) | invariant benchmark |
| `ode_analyze` | ode | 2 / 2 | independent method checks | local, from the linearisation | `reference_benchmark` problem |
| `ode_solve_symbolic` | ode | 3 / 3 | residual with numerical derivatives | "no closed form found" is not "none" | `reference_benchmark` problem |
| `ode_solve_numerical` | ode | 3 / 3 | second integrator, refinement, integral defect | an approximation on the interval | `reference_benchmark` problem |
| `variation` | variation | 4 / 3 | finite differences of the original action | numerical evidence, never a proof | `variational/benchmark.py` |
| `matrix_calculus` | matrix | 8 / 5 | `mpmath.expm`, quadrature, central differences on random matrices | never assumes commuting or invertible | `matrix-benchmark`, wrong-merge tests |
| `vector_calculus` | vector | 8 / 8 | independent Cartesian numerical evaluation | nothing assumed smooth | `vector/benchmark.py` |
| `electromagnetism` | vector | 5 / 5 | two methods that must agree, exact-solution consistency | premises are the user's | `vector/benchmark.py` |
| `geometry` | geometry | 2 / 0 | numerical recomputation and identity legs | Levi-Civita only; no reference call | `geometry/benchmark.py` |
| `tensor_canonicalize` | geometry | 2 / 2 | monoterm symmetries only | multi-term identities out of scope | reference case |
| `statistical_mechanics` | statmech | 9 / 9 | enumeration and quadrature oracle | conventions are the user's | `statmech/benchmark.py` |
| `expectation` | statistics | 1 / 1 | explicit finite joint distributions | one independence group | reference case |
| `statistical_model` | statistics | 2 / 2 | quadrature, direct log-likelihood scores | independence never assumed | `stat-benchmark` |
| `sums` | statistics | none | explicit instances at several sizes | no reference call (`corruption.EXEMPT`); not proved for every n | adversarial corpus |
| `distribution` | distributions | 3 / 2 | test-function pairing | notation confirmed by the user | pairing tests (1e-6 error) |
| `dimensional_analysis` | units | 2 / 2 | rebuilt groups found dimensionless | coherent SI only | reference case |
| `check_units` | units | 2 / 2 | exact dimension arithmetic | no unit conversion | reference case |
| `approximation` | approximation | 6 / 6 | second implementation, residual decay | errors are measurements, not bounds | `approx/benchmark.py` |

## Rules for changing this matrix

- A new tool needs a row, a reference call in the benchmark or a reason in `corruption.EXEMPT`, and a regression
  rule that names a test that fails when the tool is wrong.
- Numbers in the "Cases / refs" column come from `cases.yaml` and `cases_reliability.yaml`; update them when
  cases change (the case-set digest changes too: say so when comparing runs).
- Never regenerate an expected answer from the code under test.
