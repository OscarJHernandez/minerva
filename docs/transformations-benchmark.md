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

Of the 50 cases that expect a result, 12 pass (27 return another status, 11 return a different result);
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

Cases the new operations are **not** expected to solve yet (`pending`): substituting with a commutation
relation, periodic inverses (`sin x = a`), and selecting the real branch from declared real quantities.
