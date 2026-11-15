# Integration strategies

`integrate` first runs SymPy's route, verified as before. Where that route fails (`unsupported` or
`not_verified`), or when a strategy is requested, a deterministic strategy engine
(`packages/engine/src/minerva_engine/integration_strategy/`) looks at the *structure* of the
integrand, tries a few promising techniques, and keeps the first result that passes the independent
checkers against the **original** integrand. No language model takes part: detection, the
techniques and the verification are ordinary code, so a small local model only has to ask for an
integral (or pick one of the suggestions returned with the answer).

## Pipeline

1. **Detect** (`detect.py`): a pure function of the expression tree that returns ordered
   `Suggestion(technique, why)` pairs, `why` a fixed code. Specific, cheap-to-confirm techniques come
   first.
2. **Try** (`explore.py`, `transforms.py`): at most four techniques, in that order (or exactly the one
   requested). Each returns a candidate or nothing.
3. **Verify** (`explore._verified`): an antiderivative must pass `check_antiderivative` (an independent
   numerical derivative, in the imaginary direction too for a variable that may be complex); a
   definite value must pass `check_definite` (independent adaptive quadrature at sampled parameter
   values). A candidate that fails is discarded and counted, never shown. The simplest equivalent
   form that still passes is kept.
4. **Report** (`StrategyReport` on `IntegrateResult`): the technique, why it applies, the steps with
   their formulas, the suggestions, every attempt with its outcome, the verification level and the
   conditions that were established.

Bounds: at most four techniques, at most eight substitution or parts candidates each, a size cap on
rewritten forms, a visited set so the same remaining integral is not tried twice, a depth cap on
nested sub-integrals, and the request's deadline. The whole exploration runs inside the sandbox.

`strategy` argument of `integrate` (tool, endpoint, engine): absent = SymPy's route, strategies only
on failure; `auto` = also explain a result SymPy found; a technique name = only that technique.

## Techniques

| Technique | Cue (`detect`) | How a candidate is found | What must hold |
| --- | --- | --- | --- |
| `substitution` | a function or power of a non-linear inner expression | `u` from SymPy's own rule tree (`manualintegrate.integral_steps`, `URule`) and from subexpressions; accepted if `f / u'` depends on `x` only through `u` | the result passes the derivative check |
| `parts` | polynomial times exp, log, trig, atan | `u` and `dv` from `PartsRule`; `v` checked by derivative; a remaining integral that repeats is discarded | derivative check of the whole result |
| `trigonometric` | trigonometric powers or products | power reduction, product-to-sum, double angle, Pythagorean rewrites (`sympy.simplify.fu`), alone or two in a row; each rewrite must agree with the integrand | derivative check; the simplest integrable form wins |
| `partial_fractions` | rational function with a reducible or repeated denominator | `apart`; each term integrated | decomposition agrees with the integrand |
| `parity` | symmetric interval and `f(-x) = ±f(x)` (exact zero test) | odd: `0`; even: twice the half-range integral | the **half-range integral must converge** (it is evaluated by the verified route); otherwise `blocked: convergence_not_established`, never `0` |
| `reflection` | finite interval where `(f(x) + f(a+b−x))/2` is strictly simpler (King's rule) | integrate the average | quadrature of the original agrees |
| `frullani` | `∫₀^∞`, `x f = g(a x) − g(b x)` | `(g(0⁺) − g(∞)) ln(b/a)` | `a, b > 0` **declared**; `g` provably continuous on `(0, ∞)`; both limits finite (`sp.limit`, confirmed numerically); a limit that does not exist (`cos`) is refused |
| `schwinger` | definite integral with a factor `A^{-n}`, `A` not a bare power | `A^{-n} = Γ(n)^{-1} ∫₀^∞ t^{n−1} e^{−tA} dt`; inner integral over `x`, outer over `t`, each by the verified route | `A > 0` on the interval and `n > 0` provable; the other factor nonnegative, so swapping the integrals is justified by Tonelli (anything else: `interchange_not_justified`) |
| `parameter_differentiation` | a free parameter and a hyperexponential integrand | [creative telescoping](creative-telescoping.md) | its own conditions; only an `evaluated` outcome is used |
| `numerical` | a definite integral with no free symbols | two independent rules (tanh-sinh, Gauss-Legendre) must agree to the reported error; the tail beyond a cutoff must be negligible | otherwise `divergent_or_unreliable` and no number |

## What is shown, and how it is worded

`verification` is one of `proved` (a symbolic zero of `F' − f` as well as the numerical derivative
check), `numerically_checked` (quadrature, or the derivative check without a symbolic zero) and
`numeric_estimate`. A definite value is checked by quadrature, never by the fundamental theorem, so
it is never called proved. A `blocked` attempt names the condition that was not established (and
nothing is applied). The numerical estimate has no formula: found automatically it rides along with
the unchanged `unsupported` status (`result_kind: numeric`, `numeric_error`); it becomes the result
only when `strategy: numerical` or `auto` is requested.

## Not covered

Gröbner bases, Lie symmetries, general holonomic algorithms, contour integration, special-function
identities, principal values (an odd integrand with a divergent half-range is refused, never `0`),
Feynman parameters for products of several denominators, and conditionally convergent oscillatory
integrals in the Schwinger and parity routes.
