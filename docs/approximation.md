# Approximations

`approximation` (engine package `minerva_engine.approx`, operation `approximation`, endpoint
`POST /v1/compute/approximation`, chat tool `approximation`, workbench page `/workbench/approximation`)
finds a simpler expression that represents a function in the regime the user names. The language model
only chooses the tool and passes what the user said; every number, formula and check comes from
deterministic code (SymPy, mpmath, NumPy, SciPy).

## The rule behind every method

SymPy (or a formula from the literature) only **proposes**. A candidate is shown only if a check that
does **not** use the route that built it agrees. Otherwise the status is `not_verified` and nothing is
displayed. Errors are **measurements** on sampled points, never bounds, and "the tolerance is met" means
"on the sampled points". The interval, the point, the period, the tolerance, the degrees and the number
of modes belong to the user: none has a default, and a missing one is a question (`domain_unspecified`
with `questions`), not a guess.

Evidence is one of `series_checked` (coefficients and remainder order against independent estimates),
`numeric_estimate` (a measurement), `bound_established` (an inequality that was verified; used only for
continued fractions of a rational) or `none`.

## Methods and what decides them

| Method | Built by | Decided by (does not use the construction) |
| --- | --- | --- |
| `taylor` | `series` | Cauchy-integral coefficients + remainder decay (`series.check_series`) |
| `perturbation` | substitute the small parameter (`v/c` becomes `q`), expand, write back | `check_series` in `q`, other symbols sampled |
| `laurent` | `sp.series` | `check_series` on `h^m f`; pole order and residue are coefficients of it |
| `asymptotic` (`x -> ±oo`) | series in `u = 1/x` | residual falls at the claimed rate when `x` doubles from `10^12` at a precision that resolves it; Cauchy estimates in `u` as a second leg where the function continues analytically |
| `multivariable` | mixed derivatives to total degree `N` | `check_series` along random lines through the point |
| `leading`, `limit` | `as_leading_term`, `sp.limit` | ratio / value along a geometric approach at 60 digits; the error must shrink steadily |
| `pade` `[m/n]` | exact linear system on Taylor coefficients | `mpmath.pade`; residual `f - P/Q` falls like `h^(m+n+1)` on `f` itself; poles located, spurious ones flagged |
| `chebyshev` | interpolation at Chebyshev nodes, kept in the Chebyshev basis | `numpy.polynomial.Chebyshev.interpolate`; error on two disjoint grids that must agree |
| `legendre` | Gauss-Legendre projection | two panel counts agree; numpy `leggauss` (independent nodes) |
| `fourier` | composite Gauss-Legendre over the places the function is not smooth | two panel counts agree; Bessel inequality; closed forms only if they reproduce the quadrature |
| `piecewise` | SciPy `CubicSpline`, `PchipInterpolator`, linear | error off the knots on two disjoint grids; never extrapolates |
| `continued_fraction` | high-precision quotients | two precisions + `sympy.continued_fraction_iterator`; `abs(x - p/q) <= 1/(q q')` |
| `euler_maclaurin` | verified integral + Bernoulli corrections | direct summation (error consistent with the first neglected term and better than the plain integral); exactness on polynomials up to degree `2p+1` |
| `accelerate` | Richardson, Shanks, `mpmath.nsum` | at least two accelerators with small own error estimates must agree |
| `integral_expansion` | expand the integrand, integrate each term with the verified integrator | quadrature of the **original** integral: the error falls like `eps^(n+1)` |
| `root_expansion`, `reversion` | order-by-order solution from a simple root | residual of the equation and a separately computed (Newton) root |
| `laplace` | Gaussian moments about an interior maximum | quadrature of the original integral: relative error falls like `M^-(m+1)` |
| `compare` | the above on one interval | each candidate measured on the same two grids; the simplest that meets the tolerance is chosen |
| `auto` | rules over the shape of the request (`strategy.py`) | the chosen method's own checks |

`auto` reads what the user said: a number gives continued fractions; several variables give a
multivariable expansion; infinity gives an asymptotic expansion; a small parameter gives a perturbation
expansion; a period gives Fourier; a point with an order gives Taylor, and Laurent when Taylor refuses a
pole; an interval with a degree or a tolerance gives `compare`. Every rule is returned with whether it
applied and whether it was tried (at most three are). A request that names no regime gets a question.
`compare` includes a piecewise candidate when the function jumps.

## Limits (also in `docs/capabilities.md`)

- Errors are sampled. A rigorous global bound is not produced (no Remez or interval arithmetic).
- Asymptotic expansions are asymptotic. Exponentially small or large behaviour is refused, not guessed.
- A Padé approximant may have a pole the function does not; it is reported, and Padé is not assumed to
  beat Taylor (both are measured when an interval is given).
- Fourier series of a function that jumps overshoot by about 9% of the jump (Gibbs) and do not converge
  uniformly; Lanczos sigma factors are optional (`kind: sigma`).
- Interval methods refuse poles, undefined points and non-real values on the interval; a jump is a
  warning, with the large measured error shown.
- Not covered: Puiseux and essential singularities, endpoint or degenerate Laplace maxima, stationary
  phase, perturbative solutions of differential equations (use the ODE tools), spherical harmonics,
  multiscale methods, general symbolic summation.
- The interval methods compute with mpmath at 40 digits, except SciPy splines, which are double precision.

## Testing and the benchmark

`packages/engine/tests/test_approx_*.py` test each method against closed forms and give each checker
deliberately wrong candidates (a 1e-9 error in a coefficient, a wrong residue, a dropped Bernoulli term, a
tampered Chebyshev coefficient, a perturbed Padé denominator). `python -m minerva_engine.approx.benchmark`
(`make bench-approximation`) runs a corpus whose expected answers come from outside Minerva and writes
`docs/benchmarks/approximation.json` and `docs/approximation-benchmark.md`; `--check` is part of the test
suite. The agent's tool choice is scored by the `approximation-*` cases in the evaluation set.

## Where things are

- Engine: `approx/models.py` (input, report), `operation.py` (dispatch), `strategy.py`, `expand.py` (Taylor,
  Laurent, asymptotic, perturbation), `leading.py`, `multivar.py`, `pade.py`, `chebyshev.py`, `legendre.py`,
  `fourier.py`, `piecewise.py`, `compare.py`, `cfrac.py`, `sums.py`, `composed.py`, `laplace.py`, and the
  shared `interval.py` (reference function, domain scan, error measurement), `checks.py`, `quadnodes.py`.
- API: `services/api/.../agent/approx_tools.py`, the `/v1/compute/approximation` route, the direct route
  `POST /v1/chat/conversations/{id}/tools/approximation` used by the card's form, and `approximation_log.py`
  (method name and counters only; never an expression, interval or tolerance).
- Web: `ApproximationPanel`, `ApproximationCard` (a form for the user's numbers), `ApproximationWorkbench`.

## References

- SymPy series, limits, Fourier series and number theory:
  <https://docs.sympy.org/latest/modules/series/series.html>,
  <https://docs.sympy.org/latest/modules/series/limits.html>,
  <https://docs.sympy.org/latest/modules/series/fourier.html>,
  <https://docs.sympy.org/latest/modules/ntheory.html>
- mpmath approximation (`pade`, `chebyfit`, `richardson`, `shanks`, `nsum`):
  <https://mpmath.org/doc/current/calculus/approximation.html>
- NumPy polynomials (`Chebyshev.interpolate`, `leggauss`):
  <https://numpy.org/doc/stable/reference/routines.polynomials.html>
- SciPy interpolation (`CubicSpline`, `PchipInterpolator`):
  <https://docs.scipy.org/doc/scipy/tutorial/interpolate.html>
- Digital Library of Mathematical Functions, for the Euler-Maclaurin formula (§2.10), continued fractions
  (§1.12), Padé approximants (§3.11) and Laplace's method (§2.3).
