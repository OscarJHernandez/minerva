# Ordinary differential equations

Minerva reads an ordinary differential equation (or a first-order-reducible system), classifies it with exact
tests, ranks the methods that can be tried, runs the one the user chooses, and shows a result only when checks
that do not use the producing route pass. The language model orchestrates: it never computes and never decides
whether a result is correct. Package: `packages/engine/src/minerva_engine/ode/`.

## Entry points

| Operation | Engine | HTTP | Agent tool |
| --- | --- | --- | --- |
| Reading, classification, ranked methods, equilibria and linearisation | `Engine.ode_analyze` | `POST /v1/compute/ode/analyze` | `ode_analyze` |
| Closed form, Taylor series, perturbation expansion | `Engine.ode_solve_symbolic` | `POST /v1/compute/ode/solve` | `ode_solve_symbolic` |
| Initial-value and boundary-value problems, numerically | `Engine.ode_solve_numerical` | `POST /v1/compute/ode/numeric` | `ode_solve_numerical` |

All three run in the sandbox. The workbench has a "Differential equations" mode; chat cards for the numerical
and symbolic tools have forms that post the user's numbers straight to
`/v1/chat/conversations/{id}/tools/<tool>`. Examples with real output: [capabilities.md](capabilities.md).

## Reading the equation

Derivative notation is interpreted in one place (`reading.py`): `y'`, `y''`, `\dot y`, `\ddot y`,
`\frac{d^n y}{dt^n}`, `y^{(n)}`, `diff(y, t, n)`, `y_dot`. Nothing else in Minerva reads these, and the claim
checker leaves a statement with a derivative `unverified`. The reading is returned with every analysis
(unknowns, independent variable, order, parameters, normalised equations) so it can be inspected.

Refused as `unsupported` or `parse_error`: delayed or composed arguments (`y(t-1)`, `y(2t)`), partial
derivatives, derivatives of expressions, an independent variable that cannot be named, more than 8 unknowns,
order above 10. Conditions are accepted as `y(0) = 1`, `y'(0) = 0` or structured. A symbol that appears only in
a condition becomes a parameter.

## Analysis and method ranking

Classification is decided by exact tests, not by names: order, linearity, constant coefficients, homogeneity,
autonomy, whether the highest derivative can be solved for (confirmed at sample points), and position-only
second order (`y'' = a(y)`, which has an energy integral). First-order classes (separable, linear, exact,
Bernoulli, homogeneous) come from SymPy's `classify_ode`, mapped to a fixed list. The problem is labelled
general, initial-value, boundary-value, incomplete or overdetermined.

Each method in the ranked list carries what it needs and what limits it. "Applicable" means "can be tried", never
"will succeed"; a method that does not apply is listed with its reason. Stiffness is not predicted: the
numerical integrator is chosen from measurements during the run.

When several methods are applicable and the message names none (closed form, numerically, series, perturbation, phase
plane, equilibria), the assistant lists them in this order and asks which to run; the server refuses a solver call
until the user chooses (error category `needs_user_choice`). The card's sections run each method with the user's own
numbers.

## Methods, checks, limits

| Method | What it needs | Shown only if | Limits |
| --- | --- | --- | --- |
| Closed form (`dsolve`) | nothing numeric | the candidate put into the original equation leaves a residual at the round-off scale (derivatives by mpmath finite differences at random points, two tranches, at least 4 points); conditions checked separately | Deadline and size cap. "No closed form found" never says none exists. An unevaluated integral is a representation (`evaluated=false`). Fewer constants than the order is "a family", not "the general solution". Singular solutions are not searched |
| Implicit closed form | as above | the curve is solved numerically near each sample and the slope matches the equation | Order at most 2 |
| Numerical IVP (SciPy RK45, DOP853, Radau, BDF, LSODA; `auto`) | every parameter, every initial value, the end time | initial values reproduced; a second, independent integrator agrees (Minerva's own Dormand–Prince, or BDF for stiff runs); a tighter tolerance changes the answer by less; the integral-form defect is small; a verified first integral does not drift | At most 12 states, 20 000 numbers, 400 samples, evaluation and time budgets. Always an approximation on the given interval; blow-up and step collapse are reported, not smoothed over |
| Boundary-value (`solve_bvp`) | every parameter, both boundary conditions | boundary conditions hold; the residual on a fine grid is small; a refined mesh agrees; shooting from the left end reaches the right-end value | A solution, not claimed unique. Guesses: zero, linear, one |
| Taylor series | all initial values, a regular point | the residual falls like h^(N+1-n) when the step is halved twice; conditions hold | A singular point is `unsupported` (no Frobenius). The error table is a measurement, not a bound |
| Perturbation | the small parameter named by the user | regular: order-by-order solution whose residual falls like ε^(N+1). With an equilibrium: Poincaré–Lindstedt (existing `perturbation_solve`) | Scalar equations, order at most 4; secular terms are flagged and validity is not claimed for all times. Multiple scales is **not implemented** |
| Equilibria and linearisation | autonomous system, at most 3 states | each equilibrium substituted back; the Jacobian matches a numerical Jacobian; eigenvalues match the characteristic polynomial | "Equilibria found", complete only for one polynomial equation. Classification is local, from the linearisation; a zero real part is "the linearisation does not decide". Nothing says "globally stable" |
| Phase plane | two states, a window the user chooses | direction field on a 15x15 grid; trajectories are verified numerical runs | Autonomous only |
| Cross-method comparison | a verified closed form and a numerical run | the closed form is evaluated on the numerical grid | The difference is a measurement, not a bound |

## What is never done

- No initial condition, parameter value, interval or small parameter is invented. The model must ask, and the
  tool refuses to run without the numbers.
- A candidate that fails a check is dropped, not shown with a warning. The statuses are the usual ones: `ok`,
  `conditional`, `not_verified`, `unsupported`, `parse_error`, `resource_limit`.
- Numerical agreement is evidence, not proof; text says "proved" only for a symbolic zero.
- Diagnostics and logs hold counters and fixed vocabulary only, never an equation or a value.

## Not implemented

Delay, partial and stochastic equations; differential-algebraic systems; Frobenius series at singular points;
multiple scales; Laplace and Green's-function solvers; stability of the full nonlinear system; more than three
states in the qualitative analysis; bifurcation analysis. They are refused with a reason; they are not
approximated.

## Benchmark

`python -m minerva_engine.reference_benchmark [--check]` includes 23 ODE problems whose answers come from outside
the solvers (closed forms evaluated with mpmath, `mp.odefun`, `mp.expm`, Airy functions, published Robertson and
Lorenz values, a quarter-period integral, a conserved quantity), plus refusal cases. A wrong answer or a lost one
fails CI. Report: [reference-benchmark.md](reference-benchmark.md). Tests that doctor the engine's result and
require the judge to call it wrong: `packages/engine/tests/test_ode_benchmark.py`.
