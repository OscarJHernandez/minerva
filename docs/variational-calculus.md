# Variational calculus

Minerva treats an action `S[x] = ∫_a^b L(t, x_i, x_i', …, x_i^(n_i)) dt (+ end terms)` as a functional of
**several functions with derivatives of any order up to four** and fixed bounds. It is not a collection of
mechanical formulas: the integrand is read, varied, integrated by parts and checked, whatever it is.
The operation is `variation` (engine `packages/engine/src/minerva_engine/variational/`, endpoint
`POST /v1/compute/variation`, chat tool `variation`, workbench mode "Variational calculus").

## Conventions

* `EL_i = Σ_k (-1)^k D^k (∂L/∂x_i^(k))`, `D` the **total** derivative (it acts on `t` and on every derivative
  of every function; `∂L/∂x^(k)` is a partial derivative). For `n = 1` this is `∂L/∂x − d/dt ∂L/∂ẋ`, the
  negative of the form used by `analyze_lagrangian`.
* `δS = ∫ Σ EL_i η_i dt + [Σ Π_{i,j} η_i^(j)]_a^b`, with
  `Π_{i,j} = Σ_{k>j} (−1)^{k−j−1} D^{k−j−1} ∂L/∂x_i^(k)`. End terms `G` add `∂G/∂x^(j)` at that end (with a
  minus sign at the lower end of `[·]_a^b`).
* `S[x + εη] = S + ε δS + ½ ε² δ²S`, so `δ²S = d²/dε² S[x + εη]` at `ε = 0`.
* Notation is strict. Unknown functions are declared; `\dot{x}`, `\ddot{x}`, `\dddot{x}`, `x^{(4)}` (and
  `x_dot`, `x_ddot`, `x_d3` in text) are derivatives only of declared unknowns, and the reading is echoed back.
  A given function such as `V(x)` must be declared; an undeclared `V(x)` is an error, never the product `V·x`.
* With another independent variable (`y` in the brachistochrone `x(y)`), the derivative of the unknown `x` is
  `\frac{dx}{dy}`, `\frac{d^2x}{dy^2}` or `x'`, `x''`: the same order above and below, the unknown over the
  independent variable, nothing else. Any other use of `d` in a fraction is refused (read as letters the d's would
  cancel, so `\frac{dx}{dy}` would be `x/y`).
* Without bounds the numerical check draws them. They are drawn again on an interval entirely above or below zero
  (then reseeded) when the first draw makes the action's quadrature inaccurate, as when it straddles a singularity
  such as `1/√y` at `y = 0`. If no interval works the answer is `not_verified` and says it could not be checked, which
  is different from "did not pass" (a real disagreement). Bounds you give are never replaced. Every status carries the
  integrand as Minerva read it.

## What a result means

A result is shown only when an independent numerical check passed. The deciding check uses the **original
integrand** only: for polynomial paths and perturbations (exact derivatives, no `sp.diff` of a path) it compares
`(S[x + hη] − S[x − hη]) / 2h` with the claimed `∫ EL·η + boundary terms`. Perturbations that vanish with their
lower derivatives at both ends isolate the Euler–Lagrange expressions; perturbations that do not vanish test
every boundary coefficient. Quadratures are two independent rules. This is **numerical evidence, not a proof**,
and every result says so; SymPy's own `euler_equations`, a pointwise evaluation and each integration by parts
are reported separately and never decide.

## A worked example: a higher-derivative Lagrangian

`L = (m/2) ẋ² + (α/2) ẍ² − V(x)`, with `V` declared as a given function. The derivation lists each integration
by parts (one derivative moved per step, with the smoothness it needs), then returns

* `α x⁗ − m ẍ − V′(x) = 0`,
* the boundary form `[(m ẋ − α x‴) η + α ẍ η̇]_a^b`, kept whole.

Nothing about the ends is assumed. Each `η^(j)` at an end is independent of the others, so **fixing `x` does not
fix `ẋ`**: with only `x(a), x(b)` declared fixed, the two terms with `η̇` are kept and reported as "no condition
was declared". Declare a derivative `free` and its coefficient becomes a natural boundary condition; declare it
`fixed` and the term is removed. For a cantilever `½ẍ² − qx` with `x, ẋ` fixed at 0 and both free at 1 the result
is `x⁗ = q` with natural conditions `ẍ(1) = 0`, `x‴(1) = 0`.

A total derivative (`xẋ`) has no interior equation and its variation is pure boundary. An integrand linear in its
highest derivative drops the order of the equation, and the result says so.

## Numerical steps and who chooses the numbers

Evaluating the action, verifying along your path, solving, and the second-variation diagnostics need numbers
only the **user** can choose: the interval, endpoint conditions with values, parameter values, definitions of
given functions, trial paths and a starting guess. The model's tool has no fields for them and cannot run those
actions; the card's form posts them through `POST /v1/chat/conversations/{id}/tools/variation`, and the workbench
has them in its own form. Missing values are asked for (`conditions_not_satisfied`), never defaulted.

* **evaluate**: the action of analytic, sampled (not-a-knot cubic spline, or cubic Hermite with slopes) and
  parameterised trial paths. Two quadrature rules must agree. Several paths are compared under the same
  conditions, and the statement says the smallest of a few trials is not claimed to be a minimiser.
* **verify**: the finite-difference check along your own path with automatically generated admissible
  perturbations (declared fixed values kept at zero, free ends exercised), or perturbations you give. The
  interior and boundary parts are reported separately.
* **solve**: stationary paths by two independent methods. (B) the action is discretised with Hermite elements
  (degree `2n+1`, `C^n`) and Newton's method is applied to `∇S_h = 0` on three meshes, so saddles and maxima are
  found as readily as minima. (A) the derived Euler–Lagrange equations and natural conditions are solved by
  Chebyshev collocation in extended precision. A path is shown only if its refinement sequence converged and the
  first variation of the original integrand vanishes along admissible perturbations. If the methods end at
  different paths from the same starting guess, both are reported. A resonant problem (the harmonic oscillator
  on `[0, π]` with `x(π) = 1`) has no isolated solution and is `not_verified`. Wording: *a stationary path found
  from this starting guess*; never unique, never "the minimum".
* **second**: the quadratic form (interior and end terms, Legendre matrix) verified by second differences. Along a
  path: stationarity, Legendre samples, the inertia of the form on admissible polynomial perturbations, and
  directions certified three ways (two quadratures, a second difference, the measured change of the action).
  The result is a list of **facts**: a certified negative direction proves the path is not a weak minimum, and
  the absence of one proves nothing. Sufficient conditions (strengthened Legendre, no conjugate points) are listed
  as not established. Nothing in the result type says a path is a minimum.

## Limits

Fixed bounds only (no moving endpoints); smooth integrands (no `|·|`, sign, piecewise); derivative order ≤ 4,
≤ 4 functions; `solve` needs every unknown to appear with a derivative, orders ≤ 2 summing to ≤ 4, and real
parameter values; endpoint terms of the same order as the integrand's highest derivative would add conditions
beyond the Euler–Lagrange problem and are refused; sampled paths support derivatives up to order two. Fields
`φ(x^μ)` and PDEs are not implemented: the jet space already carries its independent variables as a tuple and
`Stage.flux` is the place a divergence will go, but nothing more is claimed.

## Verification of the implementation

`docs/variational-benchmark.md` lists 21 reference cases whose answers are written down before Minerva is
asked (free particle, oscillators, coupled oscillators, explicit time dependence, higher derivatives, endpoint
terms, free ends, beams, a stiff problem, null Lagrangians, mixed orders, a cubic stationary non-minimum, arc
length, the catenary with two stationary paths, a degenerate Legendre determinant, a Ritz family). The tests
include deliberately wrong candidates: a wrong integration-by-parts sign, a dropped or mis-signed boundary term,
`∂` used for `D`, a wrong lower-end sign, a coefficient off by one part in ten million, a missing cross term or
a factor of two in `δ²S`, a wrong discretisation stencil, and classifications that overreach. Three problems with
independent oracles (`mpmath` quadrature and central differences) are also in `docs/reference-benchmark.md`.
