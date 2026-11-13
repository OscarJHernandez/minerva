# Benchmark: controlled symbolic transformations

Cases: `packages/knowledge/src/minerva_knowledge/data/benchmark/step_cases.yaml` (65 scored cases and 3
pending ones). They are worked by hand from the mathematics, never copied from an engine run, and kept
apart from the implementation tests. The same file scores two things:

- the **existing functionality**, i.e. what Minerva could do before these operations (the baseline), and
- the **new operations** (`substitute`, `collect`, `isolate`, rewrite one part).

```bash
uv run python -m minerva_knowledge step-benchmark --adapter existing            # the baseline
uv run python -m minerva_knowledge step-benchmark --adapter steps               # the new operations
uv run python -m minerva_knowledge step-benchmark --adapter steps --repeat 5    # + latency and effort (local only)
```

CI runs the correctness side through `pytest` (`packages/knowledge/tests/test_step_benchmark.py`): no
scored case may fail, nothing the baseline solved may be lost, a pending case that starts passing must be
promoted, and the recorded baseline must reproduce. **No timing is asserted in CI**; latency is measured
locally with `--repeat`.

## What a case is

| Field | Meaning |
| --- | --- |
| `operation` | `substitute`, `collect`, `isolate`, or `rewrite` (apply `action` to one `part`) |
| `status` | The expected outcome. Refusals are cases: `unsupported`, `unchanged`, `parse_error`, `conditions_not_satisfied` |
| `results` / `solutions` | Accepted result forms; for `isolate` the **complete** solution set (a dropped, extra or repeated solution fails) |
| `match` | `structural`: the requested form (collect and rewrite); `equivalent`: the same function, by a numeric check |
| `conditions` | The conditions a correct answer must report (checked by the new operations) |
| `pending` | A capability known to be missing. Reported, never asserted |

The numeric equivalence used for scoring compares an output with the hand-worked answer. It is never how
the engine decides that a result is correct (see `AGENTS.md`, "Mathematical correctness").

Domains: algebra, physics (energy conservation, Ohm's law, ideal gas, lens equation, Kepler's third law,
kinematics, inverse-square law, relativistic momentum, pendulum, exponential decay), applied mathematics
(Arrhenius rate, compound interest, logarithmic response) and operators (noncommuting symbols).

## Baseline: the existing functionality

Each case is given to the nearest operation that existed: `simplify`, `expand` and `factor` goals (the
best of the three is taken for `collect`), the polynomial solver for `isolate` (exact roots only),
applying a handbook identity to the whole expression for rewriting by rule. There was no way to give a
relation, select a part, collect, or declare noncommuting symbols, so the baseline cannot use them. It is
as generous as the old operations allow.

| Operation | Passed | Scored |
| --- | ---: | ---: |
| substitute | 1 | 17 |
| collect | 2 | 10 |
| isolate | 14 | 27 |
| rewrite one part | 3 | 11 |
| **all** | **20** | **65** |

Of the 50 cases that expect a result, 12 pass (26 return another status, 12 return a different result);
of the 15 cases that expect a refusal, 8 pass. Resource limits hit: 0. Pending cases: 3, none satisfied.

What the old operations did pass:

- Polynomial equations with symbolic or numeric coefficients up to degree 4, solved for one symbol
  (`iso_linear`, `iso_ideal_gas`, `iso_kinematics_time`, `iso_relativistic_momentum`, the numeric
  quadratic, cubic and `x^3 = 2`).
- Refusals that coincide with its own limits (an unknown absent from the equation, transcendental
  equations, noncommutative input, a pattern that does not occur).
- Rewrites where acting on the whole expression happens to be the same (`part_identity_nested_control`),
  and an identity that is blocked by missing assumptions.

What it could not do, and why the cases exist:

- **Any substitution.** `simplify` knows nothing about `y^2 = 1 - x^2`; it returned a different but
  unrelated form or no change.
- **Collect.** No operation groups by a chosen variable or power.
- **Rational equations and any non-polynomial equation** (`iso_lens_equation`, `iso_kepler_third_law`,
  exponentials, roots, logarithms): status `unsupported`.
- **Choosing the place.** Applying `sin(2x) = 2 sin x cos x` to the second term, or to one of two equal
  occurrences, was impossible: the old operation picks the first match and replaces every copy.
- **Saying that a result is only conditional.** Nothing recorded a user-supplied premise.

The per-case pass/fail is recorded in `data/benchmark/step_baseline.json` (no timings, so it is
reproducible) and re-checked by a test: if an older operation changes behaviour the test fails, so a
change in the baseline is never mistaken for an improvement.

## Results with the new operations

`step-benchmark --adapter steps`, the same 65 scored cases:

| Operation | Baseline (existing) | New operations |
| --- | ---: | ---: |
| substitute | 1 / 17 | 17 / 17 |
| collect | 2 / 10 | 10 / 10 |
| isolate | 14 / 27 | 27 / 27 |
| rewrite one part | 3 / 11 | 11 / 11 |
| **all** | **20 / 65** | **65 / 65** |
| of which: expected a result | 12 / 50 | 50 / 50 |
| of which: expected a refusal | 8 / 15 | 15 / 15 |

Scoring is strict where it should be. A result that is the right function in another form fails when the case asks for
a form (collect and rewrite); `isolate` needs the *complete* solution set (a dropped, an extra and a repeated solution all
fail); a refusal case passes only with the refusal status; the conditions a correct answer must report are checked
(a reported `n R \neq 0` covers `n \neq 0` and `R \neq 0`). The scorer is independent of the engine: it compares with the
hand-worked answer and never decides what the engine shows.

What the new operations do that the baseline cannot, in terms of the cases: substitute a relation (and say that the result is
conditional), group terms by a chosen variable or power, solve rational equations and equations with `exp`, `log` and roots (with
every branch and condition, and without the extraneous candidate), change one chosen part (including the second of two equal
places), refuse an order-dependent set of relations, and handle noncommuting symbols in order.

### What is still not solved (pending cases)

Three cases describe wanted behaviour that is not implemented, and a test fails if one starts passing without being promoted:

| Case | Why it is not solved |
| --- | --- |
| `sub_pending_commutator_relation` | `AB - BA = 1` would have to be used as a commutation rule, not as a pattern that occurs in the expression |
| `iso_pending_periodic` | `sin x = a` needs the periodic inverse with its integer family; periodic inverses are `unsupported` |
| `iso_pending_real_branch` | Solutions are over the complex numbers; declared real quantities do not select one branch |

Other limits are covered by refusal cases that pass (`unsupported`): a transcendental equation with the unknown in two places,
Lambert-type equations, symbolic polynomials of degree five, equations with noncommuting symbols, an order-dependent
substitution, a relation that can never hold.

## Latency and effort (local, not asserted in CI)

`make bench` (`step-benchmark --adapter steps --repeat 5`), one run on a laptop, each case five times, sandbox start-up
included, so these are costs of the whole request:

| Operation | Cases | Median | p95 | Max | Candidates | Rejected | Time verifying |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| substitute | 18 | 123 ms | 270 ms | 364 ms | 13 | 5 | 1.1 s |
| collect | 10 | 153 ms | 201 ms | 264 ms | 9 | 1 | 0.8 s |
| isolate | 29 | 82 ms | 257 ms | 378 ms | 33 | 6 | 1.4 s |
| rewrite one part | 11 | 143 ms | 206 ms | 207 ms | 11 | 2 | 0.8 s |
| **all (new)** | 68 | 112 ms | 216 ms | 378 ms | 66 | 14 | 4.1 s |
| all (existing, for scale) | 68 | 106 ms | 1534 ms | 3188 ms | | | |

Resource-limit status was never returned on these cases. A whole-expression search in the existing operations reaches seconds
for cases that a single chosen step answers in about a tenth of a second, because a step does not search. "Candidates" are
the parts, branches or roots considered, "rejected" those dropped for a reason code (extraneous roots, a pattern that does not
occur, ...), and the same counters are in every result's `diagnostics` and in the server log (see `AGENTS.md`).

Timings depend on the machine and are not compared in CI: only correctness is (`packages/knowledge/tests/test_step_benchmark.py`).
