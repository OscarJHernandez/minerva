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
