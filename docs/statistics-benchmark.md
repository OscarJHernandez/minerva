# Statistical derivation benchmark

Minerva derives statistical results (sums, expectations, moments of probability families, maximum-likelihood
estimators, Fisher information, Bayesian updates, matrix gradients of Gaussian log-likelihoods) from the model,
and every step is verified by a check that does not use the route that produced it. This benchmark was written
**before** those operations, so that the work is measured against answers that were fixed in advance.

```bash
uv run python -m minerva_knowledge stat-benchmark                       # run everything
uv run python -m minerva_knowledge stat-benchmark --group mle           # one group
uv run python -m minerva_knowledge stat-benchmark --compare docs/benchmarks/statistics.json
uv run python -m minerva_knowledge stat-benchmark --json docs/benchmarks/statistics.json --label "Phase N: ..."
```

## How a case is judged

Cases are data (`packages/knowledge/src/minerva_knowledge/data/benchmark/stat_cases.yaml`), grouped as `sums`,
`expectation`, `families`, `mle`, `fisher`, `posterior` and `matrix`.

| Kind | The operation must |
| --- | --- |
| `supported` | answer `ok` with passed verification; every named output agrees with the hand-written reference at concrete data; the evidence, optimum and reported conditions match when stated |
| `unsupported` | refuse with the stated status and reason, and show no result |
| `invalid_candidate` | reject a wrong answer to the same request (`not_verified` or `conditions_not_satisfied`) **and** accept a correct `control`, so a checker that rejects everything fails |

A request that checks an answer is the same request with `candidate` set: the operation then verifies the given
answer instead of deriving one. A request the engine cannot express yet (no input model, field, option or method)
is `not_expressible`. `pending: true` marks a known gap: it is reported and not asserted, and a pending case that
passes fails the suite, so progress has to be recorded in the case file.

## Independence of the references

* **The oracle** (`stat_oracle.py`) reads answer texts with Python's `ast` module (never `eval`) and evaluates them
  at concrete numbers with mpmath. It shares no code with the engine's parsers or checkers. Sums and products are
  loops over indexed data; matrices are mpmath matrices; `E[...]`, `Var(...)` and `Cov(...)` are computed over a
  finite joint distribution that is **dependent** unless the case declares independence, so an answer that quietly
  assumes independence disagrees with the reference. Two answers are equal if they agree to 25 digits at five
  random draws (sizes 2 to 5), which also rejects the false identity `sum x^2 = (sum x)^2 / n` that holds at n = 1.
* **The references themselves** are checked by `tests/test_stat_references.py` without the engine: sum and
  expectation identities by evaluating both sides, matrix gradients, Hessians and Jacobians by finite differences of
  the function (for a symmetric variable: directional derivatives along symmetric directions plus symmetry of the
  gradient), family moments by quadrature or explicit sums, estimators by the score and the curvature of the
  log-likelihood, Fisher information as the variance of the score, and posteriors as prior times likelihood divided
  by its integral. Every wrong candidate is required to differ from the truth.
* Sources are textbook **chapter-level locators** or "hand derivation"; nothing is quoted, and section numbers are to
  be added only after they are confirmed against the book.

## Conventions the cases fix

* Text syntax: `Sum(f, (i, a, b))`, `Product(...)`, `x[i]`, `E[X*Y]`, `Var(X)`, `Cov(X, Y)`; matrices use `*` for the
  product and `transpose`, `inv`, `tr`, `det`, `logdet` (log of the absolute determinant, with the same derivative).
* A candidate or output is a flat map of names to texts: `result`, `estimate:<parameter>`, `moment:<name>`,
  `posterior:<parameter>`, `matrix[i][j]`, `alternative:<convention>`.
* Normal families have two parametrisations (`sigma` or variance `v`); gamma is shape and rate.
* Matrix derivatives are over the reals. For a symmetric variable the result is the gradient within the symmetric
  matrices; the independent-entries form `2G - diag(G)` is a labelled alternative. The unconstrained gradient is never
  shown for a symmetric variable.
* The five derivations that make up the minimum successful outcome are listed in `MINIMUM_OUTCOME`.

## Baseline

Phase 0, before any implementation: 108 cases (51 supported, 28 refused by design, 29 wrong-candidate), all pending
and not expressible. The committed baseline is [benchmarks/statistics.json](benchmarks/statistics.json); the suite
fails on a regression against it (a case that passed no longer passing). Timing is recorded and never asserted. Peak
memory is not measured per case: each operation runs in a sandboxed child process whose address space is capped by the
engine's limits.

| Group | Cases | Passing |
| --- | --- | --- |
| sums | 16 | 0 |
| expectation | 15 | 0 |
| families | 14 | 0 |
| mle | 15 | 0 |
| fisher | 11 | 0 |
| posterior | 9 | 0 |
| matrix | 28 | 0 |

Results by phase are appended below as each phase lands.

## Phase 1: sums, products, indexed data and expectations

Sums and products (`Sum(f, (i, a, n))`, `\sum_{i=a}^{n}`), indexed data (`x[i]`, `x_i` inside a sum) and moments
(`E[...]`, `Var(...)`, `Cov(...)`) are read only by the `sums` and `expectation` operations; every other operation and
every pointwise checker refuses them (`tests/test_sum_guard.py`). The reason is measured: `verify_equal` used to accept
`sum x_i^2 = (sum x_i)^2 / n` because it gave every `x[i]` the same number.

* `sums` verifies a rewrite by expanding both sides explicitly at several sizes (always including n of at least 2), with
  a separate symbol for every `x[k]`, and running the existing sampled check and symbolic zero test on every instance.
  The result says "checked for n = 1; 2; 3; 5", never "proved for every n".
* `expectation` verifies against explicit finite joint distributions with exact rational weights. They are dependent
  unless independence was declared, so an answer that quietly assumes independence is rejected, and a result that used
  declared independence is `conditional` and says so. `sympy.stats.symbolic_probability` was measured and not used as
  the container (it needs probability spaces, leaves `Cov(X, X)` unreduced and has no notion of declared independence).

| Group | Cases | Passing |
| --- | --- | --- |
| sums | 16 | 16 |
| expectation | 15 | 15 |

All 31 pass against the independent oracle; 77 cases of the later phases remain pending.

## Phase 2: gradients, Jacobians and Hessians of matrix functions

The `matrix_calculus` operation gained the actions `gradient`, `jacobian` and `hessian` (plain-text syntax:
`tr`, `det`, `logdet`, `transpose`, `inv`, `*` as the matrix product), and the declarations `symmetric` and
`positive_definite`. They use their own small expression tree (`matrix/mexpr.py`) because the existing noncommutative
SymPy terms cannot hold a transpose, an inverse, a trace or a determinant.

* The derivative is built by differentials, not by SymPy's matrix-expression `diff`, which was measured to return 0
  for `log det X` and for the Gaussian log-density.
* It is shown only if central differences of the original function at random real matrices of the declared sizes
  agree. For a symmetric variable the candidate must also be symmetric: on symmetric directions the unconstrained
  gradient has the same directional derivatives as its symmetric part, so a directional check alone cannot tell them
  apart (`test_the_symmetry_requirement_is_what_rejects_it`). The result is the gradient within the symmetric matrices
  (Frobenius); the independent-entries form `2G - diag(G)` is verified separately and labelled.
* Real field only (a transpose, not a conjugate transpose); `logdet` is `log |det|`. A matrix is never assumed
  invertible: `inv`, `logdet` and the derivative of `det` need a declared invertible matrix (`invertibility_not_declared`),
  declarations are checked against explicit entries, and a 1x1 product is a scalar factor (refused as ambiguous if it
  could be grouped two ways). The exact (symbolic) leg is not run: `symbolic_zero` is `None`.
* Not covered: LaTeX input for these actions, derivatives of matrix-valued functions by a matrix (a fourth-order
  tensor), solving the stationarity equation for a matrix, Hermitian or Wirtinger derivatives.

| Group | Cases | Passing |
| --- | --- | --- |
| matrix | 28 | 28 |

Cumulative: 59 of 108 cases pass (sums 16, expectation 15, matrix 28); 49 of the later phases remain pending. The
existing matrix benchmark is unchanged (`matrix-benchmark` still reports the same coverage).

## Phase 3: probability families

`statistical_model` with action `moments` (and `candidate`, to check an answer) covers Bernoulli, Binomial, Poisson,
Normal (standard deviation or variance), Beta, Gamma (shape and rate), Cauchy and Uniform(0, theta). The registry
(`stats/families.py`) holds only the parameters, their constraints, the support and the density.

* Moments are **derived**: SymPy `integrate` for continuous families, `summation` for discrete ones. A size parameter
  (Binomial's n) is fitted as a polynomial from exact sums at small n and verified at further n, because SymPy returns
  a `Piecewise` with an unevaluated sum for symbolic n. A result with an unevaluated integral or sum is
  `no_closed_form`, never shown.
* Absolute convergence is established **before** a moment is trusted: quadrature of the Cauchy mean returns 0 with
  an error estimate of 0, so the tails of `|x| f(x)` must shrink (they do not: the Cauchy tail stays near 20 across
  five decades). Such a moment is `convergence_not_established`.
* Each moment and the normalisation are checked by mpmath quadrature or summation of the density at sampled parameter
  values; numerical agreement is evidence, not a proof. A wrong density fails the normalisation check
  (`test_a_wrong_density_is_caught_by_the_normalisation_check`).
* Parameter constraints are decided by `stats/domain.py`: a provable violation is `parameter_invalid`, a missing value
  `parameter_missing`, and a constraint SymPy's assumptions cannot express (`0 < p < 1`) is shown as a premise of the
  result, never silently assumed and never used to refuse.

| Group | Cases | Passing |
| --- | --- | --- |
| families | 14 | 14 |

Cumulative: 76 of 108 cases pass; the three refusing `mle` cases that depend only on reading the model now pass too.

## Phase 4: complete derivations

`statistical_model` gains `log_likelihood`, `score`, `mle`, `fisher_information` and `posterior`. Each action returns the
steps of the derivation with the evidence for them, and anything that cannot be verified is a refusal with a reason.

* **Likelihood.** A likelihood of several observations is a product only under independence, which the user must state
  (`iid`); otherwise `independence_not_declared`. The logarithm is expanded with `expand_log(force=True)` and the
  expansion is checked against a direct numerical sum of log densities.
* **Maximum likelihood.** Sums over the data become symbols, the score is differentiated, and the score equations are
  solved one parameter at a time (triangular systems only: `system_not_triangular`). The estimator is then checked on
  explicit data against the *direct* log-likelihood: zero slope and a negative definite Hessian (second differences),
  inside the parameter space. The variance estimate comes out as `sum (x_i - xbar)^2 / n`, derived, with the data
  condition `sum (x_i - xbar)^2 > 0` stated.
* **Optimum.** `global` only when concavity on the whole parameter space is proved (one parameter, Hessian provably
  non-positive; Bernoulli uses `n - sum x >= 0` and `0 < p < 1`, Poisson `sum x >= 0`). The joint Gaussian
  log-likelihood is not concave, so it is `local` (proved from the Hessian at the point given the admissibility
  condition), and a stationary point that is not a maximum is never shown.
* **Boundary and nonexistent estimates.** With concrete data whose estimate lies outside the parameter space (a Bernoulli
  sample of all zeros or all ones, identical normal observations) the supremum is not attained:
  `candidate_inadmissible`, no estimate shown. A family whose support depends on the parameter (Uniform(0, theta)) has no
  score or information: `regularity_not_satisfied`.
* **Fisher information** is `-E[Hessian of the log density]`, with the expectation taken from the family's *derived*
  moments, and is checked against the variance of the score (quadrature or summation, numerically differentiated log
  density) with the expected score equal to 0. For n independent observations it is n times that of one (a stated
  premise). A Hessian that is not polynomial in the observation (Cauchy location) is `no_closed_form`. Note that a
  result can contain `polygamma(1, alpha)` (the Gamma shape): the input parsers do not read it back.
* **Posteriors** are found by matching the kernel of prior times likelihood (factors free of the parameter dropped and
  recorded) against each family with the same support, solving for the parameters that make the derivative of the
  difference vanish identically. There is no conjugacy table: a Beta prior on a Poisson rate is
  `kernel_not_recognised`. The result is checked against prior times likelihood divided by its integral (quadrature) at
  several values of the parameter, and it must integrate to 1. The prior is an assumption of the result and says so.

| Group | Cases | Passing |
| --- | --- | --- |
| mle | 15 | 15 |
| fisher | 11 | 11 |
| posterior | 9 | 9 |

**All 108 cases pass (51 supported, 28 refused as designed, 29 wrong answers rejected), and the five derivations of the
minimum successful outcome are complete**: Gaussian mean and variance, Bernoulli, Beta-Binomial, Normal Fisher
information and the multivariate Gaussian gradient. Latency: median 40 ms, p95 2.3 s (the slowest are Beta moments and
Fisher information, a few seconds each, under the 10 s sandbox timeout; the tests run the five required derivations at
the product's default limits).
