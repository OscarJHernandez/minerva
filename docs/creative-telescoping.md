# Creative telescoping

For an integral that depends on a parameter,

    I(a) = ∫_u^v f(x, a) dx,

Minerva searches for a differential operator `L = Σ_k p_k(a) ∂_a^k` and a certificate `G(x, a)` with

    L f = ∂_x G.

Integrating gives `L I(a) = G(v, a) − G(u, a)`. When the boundary term vanishes this is a homogeneous
ODE for `I`; otherwise it is inhomogeneous. The ODE is solved and the constants are fixed by
independently verified initial values. The code is `packages/engine/src/minerva_engine/creative_telescoping/`;
the operation is `creative_telescoping`, the endpoint `POST /v1/compute/telescoping`, the agent tool
`creative_telescoping` and the workbench page `/workbench/telescoping`.

The relation is **discovered**, not looked up: nothing in the package knows a Gaussian, a Bessel
function or any other integral. A telescoper is searched for in explicit families, and every
candidate is proved before it is shown.

## What is supported

The integrand must be *hyperexponential* in the variable and in the parameter: `f_x / f` and
`f_a / f` are rational functions of `x`, `a` and any further symbols (treated as generic constants).
That covers products of rational functions, powers whose exponent does not contain `x`, and
exponentials of rational functions: `e^{-a x²}`, `x^n e^{-a x}`, `e^{-a x}/(1+x)`, `1/(x²+a²)`,
`e^{-x²−a²/x²}`, `e^{i a x}/√(1−x²)`, `e^{−x³/3−a x}`.

Refused with a reason (`not_hyperexponential`): sums of such terms, trigonometric functions,
logarithms, a parameter inside an exponent of `x` (`x^a`), anything with an unevaluated operator.
Bounds are numbers or `±∞`. The parameter must be declared real; its domain is read from its
declaration (positive, negative, or all of ℝ).

## The method

### 1. Search (`search.py`)

Write `∂_a^k f = R_k f` with `R_0 = 1`, `R_{k+1} = ∂_a R_k + R_k · (f_a / f)`; every `R_k` is a rational
function. Seek `G = y f` with `y = Y / D`, where `D` is the lcm of the denominators of `R_0 … R_N` and
`Y` is a polynomial of degree at most `d`. Because `∂_x (y f) = (y' + (f_x/f) y) f`, the telescoping
equation becomes

    Σ_k c_k(a) R_k = y' + (f_x / f) y,

which is *linear* in the coefficients of `Y` and in the `c_k`, whose coefficients are rational
functions of the parameter and the other symbols. Clearing denominators gives a polynomial identity
in `x`; its coefficients form a matrix over the field of rational functions (Gaussian rationals if
`i` occurs), and the solutions are its nullspace, computed exactly. Families are tried in a fixed
order (operator order `N = 0, 1, …`; certificate degree `d = 0, 1, …`), cheapest first, so the first
solution has minimal order within the families tried. The operator is made primitive (polynomial
coefficients with no common factor) and the certificate is scaled with it.

Bounded: operator order at most 5, certificate degree at most 12, at most 48 unknowns, a degree cap
on the denominators and a deadline. Hitting a bound is `resource_limit`; finding nothing is
`no_telescoper_found`, worded as "within the searched families" and never as "none exists". The
list of families tried is part of the result.

### 2. Verification (`certificate.py`)

Operates on plain data, so a forged certificate can be handed to it in a test. The proof is exact:
`(L f − ∂_x G) / f` must simplify to zero (computed with `L f` by plain differentiation, never through
the search's own recurrences). Two numerical legs that share nothing with the simplification are
reported next to it: sampled agreement of `L f` and `∂_x G`, and `check_derivative` of `G` against
`L f`. They can only **veto**: a numerical disagreement rejects a certificate whose exact test passed.
A zero operator is rejected. A candidate that fails is dropped and counted.

### 3. Convergence and boundary (`boundary.py`)

Everything is decided from the **declared** assumptions; nothing is assumed, and each conclusion is
reported as a condition with how it was established.

* *No singularity inside the interval*: every irreducible factor of every denominator involved must
  have a constant sign on the open interval, shown by a **coefficient certificate** (map the interval to
  `(0, ∞)` and require all coefficients of the numerator to be provably of one sign). A non-integer
  power needs a provably positive base (one real branch).
* *Each end*, through a local variable `t → 0⁺`: with `f = K e^{g} ∏ b_i^{e_i}`, `|f|` behaves like
  `e^{Re g} t^κ`. If `Re g` has negative order its leading coefficient decides decay or blow-up when
  its sign is provable; otherwise `κ` must clear the threshold (`> −1` at a finite end, `> 1` at
  infinity). The same is required of every `∂_a^k f` with `k ≤ N`, which is what justifies differentiating under the
  integral sign (the deciding quantities are continuous in the parameter and the inequalities strict,
  so the bounds are locally uniform).
* *Divergence* is claimed only where the phase does not oscillate, so that `f ~ C t^κ` with `C ≠ 0` and
  a constant phase. An oscillating end is accepted only through absolute convergence; conditional
  convergence is not analysed.
* *Boundary values* of `G = y f`: zero if the end decays, or if `κ_G > 0`; the exact finite limit if
  `κ_G = 0` and the exponential part is tame; otherwise `boundary_not_established`.

### 4. The equation and its solution (`ode.py`, `evaluate.py`)

`Σ_k p_k I^{(k)}(a) = G(v) − G(u)`. The leading coefficient (and any parameter-only factor in the
denominator of `y`) must be provably nonzero on the whole parameter domain; otherwise the result stops
at the equation with `singular_point_in_domain`. The equation is solved with `dsolve`; the solution
is substituted back **exactly** (SymPy sometimes returns only the homogeneous part of an inhomogeneous
solution, and this check is what catches it). The constants come from `I^{(k)}(a₀)` at a rational
`a₀`, each an independent evaluation: the verified definite integral (`integrate.definite_value`,
which checks its answer by quadrature) of `∂_a^k f` at `a = a₀`.

A value is shown only if the solution satisfies the equation exactly, reproduces the initial values,
**and** agrees with quadrature of the original integrand at sampled parameter values
(`integrate.check_definite`). It is worded "derived from a proved certificate, checked numerically",
never "proved": the quadrature leg is numerical evidence.

## Outcomes

| `outcome` | Meaning |
| --- | --- |
| `evaluated` | operator, proved certificate, conditions, equation, initial values and a quadrature-checked value |
| `equation` | the equation is derived and verified; the value is not (no solution in closed form, no independent initial values, singular leading coefficient, or the candidate failed its checks and is withheld) |
| `certificate` | the identity `L f = ∂_x G` is proved; convergence, the boundary or the sign of a parameter is not established, or no bounds were given |
| `none` | nothing is claimed (refused, divergent, outside the class) |

A verified certificate alone is never treated as the value of an integral. `unresolved` lists what
stopped the derivation.

## Benchmark

`python -m minerva_engine.creative_telescoping.benchmark [--check]` (`make bench-telescoping`) runs 29
problems in the ten requested groups against references that do not use the discovery route (closed
forms written in the benchmark, mpmath quadrature of the integrand, and finite-difference derivatives
of that quadrature to test every derived equation). See `docs/creative-telescoping-benchmark.md` for
the current table: discovery success, operator order, whether the certificate was proved exactly,
time and the failure class of each refusal. The test suite fails on a wrong answer or a lost refusal.

## What is adopted from the literature, and what is not

* **Almkvist and Zeilberger [1]** give the algorithm for differentiating under the integral sign:
  find `L` and `G` with `L f = ∂_x G` for hyperexponential `f`, then integrate. The search here is that
  idea in its linear-algebra (ansatz) form for the *rational-in-x certificate* case, with the families
  searched in increasing order. Their full algorithm (a complete decision procedure through the
  Gosper-Petkovšek style denominator bounds, and the treatment of the case in which no certificate of
  the assumed shape exists) is **not** implemented: the denominator of the certificate is bounded by
  the lcm of the denominators of the `R_k`, which is what the supported class needs, and a failed search
  is reported as inconclusive.
* **Zeilberger [2]** is the discrete analogue (hypergeometric summation, recurrences in `n`). It is
  **not** implemented; `DifferentialOperator.kind` and the `Telescoper` protocol are where a shift
  operator and a summation backend would plug in.
* **Zeilberger [3]** (the holonomic systems approach) is the setting in which the method extends to
  holonomic integrands whose derivatives span a finite-dimensional space over the rational functions.
  **Not implemented**: only hyperexponential integrands, where that space is one-dimensional.
* **Koutschan [4]** describes ansatz-based creative telescoping with much faster linear algebra and
  careful degree bounds. The ansatz approach is the same as here; the efficiency techniques (guessing,
  modular and evaluation methods, tailored degree bounds) are **not** implemented.

## Extending it

The search is behind the `Telescoper` protocol (`search.families`), and every candidate, whatever
produced it, passes through `certificate.verify` before it is accepted. A future backend (a
holonomic closure, an Ore-algebra system run out of process) therefore never has to be trusted.

SageMath's `ore_algebra` has the algorithms for general holonomic integrands and recurrences, and
would be the natural backend to investigate. **It is GPL-licensed** (like SageMath itself, see
`licensing.md`), so it is not a dependency, is not imported and is not vendored. If it is ever used it
must be an optional, out-of-process backend over a narrow text protocol, never part of the default
Apache-2.0 distribution or its images, and only after the maintainer's licence review.

## References

1. G. Almkvist and D. Zeilberger, *The Method of Differentiating under the Integral Sign*, Journal of
   Symbolic Computation 10(6), 571–591 (1990). doi:10.1016/S0747-7171(08)80159-9
2. D. Zeilberger, *The Method of Creative Telescoping*, Journal of Symbolic Computation 11(3), 195–204
   (1991). doi:10.1016/S0747-7171(08)80044-2
3. D. Zeilberger, *A Holonomic Systems Approach to Special Functions Identities*, Journal of
   Computational and Applied Mathematics 32(3), 321–368 (1990). doi:10.1016/0377-0427(90)90042-X
4. C. Koutschan, *A Fast Approach to Creative Telescoping* (2010). arXiv:1004.3314
