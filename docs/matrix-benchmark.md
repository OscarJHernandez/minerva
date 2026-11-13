# Matrix calculus benchmark

`uv run python -m minerva_knowledge matrix-benchmark [--repeat N] [--section LABEL] [--out FILE]` runs
hand-written cases (`packages/knowledge/src/minerva_knowledge/data/benchmark/matrix_cases.yaml`)
through the real engine and sandbox. Groups: `existing` (behaviour that shipped before this work),
`frechet`, `duhamel`, `evolution`, `unsupported` (gaps that must stay explicit) and `limits`.

A case passes when status, and where given method, result kind and the `evaluated` flag, match.
`pending` cases are known gaps: reported, never asserted. A pending case that already passes fails the
test suite until `pending` is removed. A request the engine cannot express (for example a field that
does not exist yet) is reported as `not_expressible`, so the final case file can be run against an
older commit for the "before" run. Each `## Run:` section below is one run of the same case file; the
"before" run was made by exporting commit `7b19980` (the code before this work), copying the final
benchmark module and case file into it and running that engine, so both runs measure the same cases.
There the new-capability cases show as failing, which is the measured gap, not a stale `pending` flag.

Limits of this evidence: timings are medians over the repeats on one machine (the first call in a
process is excluded, forking the sandbox is included), are noisy and are never asserted; the cases are
small hand-written problems, not a sample of real use; agreement with the expected status and method
shows the engine behaves as designed, not that every result is right: correctness is established by the
checks inside the engine and by the regression tests.

## Before and after

`Before (baseline, commit 7b19980)` against `After (this change)`, the same case file.

| Measure | Before | After |
| --- | --- | --- |
| Cases passing (all, incl. pending) | 22 / 40 | 40 / 40 |
| Scored cases passing | 22 / 40 | 40 / 40 |
| Pending (known gaps) | 0 | 0 |
| Results with status ok | 21 | 31 |
| Verification passed (where one ran) | 21 / 21 | 31 / 31 |
| Method selected as expected | 13 / 29 | 29 / 29 |
| Resource-limit results | 1 | 1 |
| Not expressible | 7 | 0 |
| Latency median / p95 / max (ms) | 66 / 465 / 827 | 94 / 703 / 1438 |
| Max verification residual | - | 4.27e-14 |

Cases passing per group (the `existing` group is the regression check against the implementation before this work):

| Group | Cases | Passing before | Passing after |
| --- | --- | --- | --- |
| existing | 19 | 19 | 19 |
| frechet | 8 | 0 | 8 |
| duhamel | 5 | 0 | 5 |
| evolution | 4 | 0 | 4 |
| unsupported | 3 | 2 | 3 |
| limits | 1 | 1 | 1 |

## Run: Before (baseline, commit 7b19980)

40 cases, 5 repeat(s) each.

| Measure | Value |
| --- | --- |
| Cases passing (all, incl. pending) | 22 / 40 |
| Scored cases passing | 22 / 40 |
| Pending (known gaps) | 0 |
| Results with status ok | 21 |
| Verification passed (where one ran) | 21 / 21 |
| Method selected as expected | 13 / 29 |
| Resource-limit results | 1 |
| Not expressible | 7 |
| Latency median / p95 / max (ms) | 66 / 465 / 827 |

| Group | Cases | Passing | Pending |
| --- | --- | --- | --- |
| existing | 19 | 19 | 0 |
| frechet | 8 | 0 | 0 |
| duhamel | 5 | 0 | 0 |
| evolution | 4 | 0 | 0 |
| unsupported | 3 | 2 | 0 |
| limits | 1 | 1 | 0 |

| Case | Group | Status | Method | Kind | Pass | ms |
| --- | --- | --- | --- | --- | --- | --- |
| exp-diagonal | existing | ok | exp_explicit | explicit | yes | 87 |
| exp-nilpotent | existing | ok | exp_explicit | explicit | yes | 24 |
| exp-defective-repeated-eigenvalue | existing | ok | exp_explicit | explicit | yes | 79 |
| exp-ordered-product | existing | ok | exp_explicit | explicit | yes | 74 |
| exp-abstract-cannot-be-evaluated | existing | unsupported | - | - | yes | 8 |
| derivative-abstract | existing | ok | exp_derivative | expression | yes | 118 |
| derivative-product-keeps-order | existing | ok | exp_derivative | expression | yes | 148 |
| derivative-scalar-function-of-t | existing | ok | exp_derivative | expression | yes | 66 |
| derivative-commuting-declared | existing | ok | exp_derivative | expression | yes | 88 |
| merge-blocked-without-commutation | existing | conditions_not_satisfied | - | - | yes | 8 |
| merge-declared | existing | ok | exp_merge_commuting | expression | yes | 44 |
| integral-invertible | existing | ok | exp_integral_invertible | expression | yes | 162 |
| integral-not-invertible-is-a-representation | existing | ok | block_exponential | representation | yes | 213 |
| integral-singular-explicit | existing | ok | block_exponential | explicit | yes | 90 |
| integral-polynomial | existing | ok | integrate_polynomial | expression | yes | 50 |
| integral-sylvester-explicit | existing | ok | sylvester_reduction | explicit | yes | 227 |
| integral-sylvester-abstract | existing | ok | block_exponential | representation | yes | 371 |
| dimension-mismatch | existing | parse_error | - | - | yes | 6 |
| bare-e-is-rejected | existing | parse_error | - | - | yes | 6 |
| frechet-abstract | frechet | unsupported | - | - | NO | 61 |
| frechet-explicit-nilpotent-pair | frechet | unsupported | - | - | NO | 62 |
| frechet-explicit-defective | frechet | unsupported | - | - | NO | 62 |
| frechet-time-dependent-matrix-abstract | frechet | not_expressible | - | - | NO | 0 |
| frechet-time-dependent-self-commuting | frechet | not_expressible | - | - | NO | 0 |
| frechet-time-dependent-explicit | frechet | parse_error | - | - | NO | 59 |
| frechet-time-dependent-commutes-from-entries | frechet | parse_error | - | - | NO | 9 |
| frechet-time-dependent-undeclared-product | frechet | ok | exp_derivative | expression | NO | 22 |
| duhamel-abstract | duhamel | ok | block_exponential | representation | NO | 467 |
| duhamel-commuting-declared | duhamel | ok | block_exponential | representation | NO | 220 |
| duhamel-nilpotent | duhamel | ok | block_exponential | explicit | NO | 195 |
| duhamel-diagonal | duhamel | ok | block_exponential | explicit | NO | 718 |
| duhamel-defective | duhamel | ok | block_exponential | explicit | NO | 355 |
| evolution-constant-abstract | evolution | not_expressible | - | - | NO | 0 |
| evolution-commuting-family | evolution | not_expressible | - | - | NO | 0 |
| evolution-self-commuting-declared | evolution | not_expressible | - | - | NO | 0 |
| evolution-numeric-noncommuting | evolution | not_expressible | - | - | NO | 0 |
| unsupported-sine-of-a-matrix | unsupported | unsupported | - | - | yes | 7 |
| unsupported-time-ordered-integral | unsupported | unsupported | - | - | yes | 8 |
| unsupported-evolution-abstract-noncommuting | unsupported | not_expressible | - | - | NO | 0 |
| limit-power-too-large | limits | resource_limit | - | - | yes | 6 |

## Run: After (this change)

40 cases, 5 repeat(s) each.

| Measure | Value |
| --- | --- |
| Cases passing (all, incl. pending) | 40 / 40 |
| Scored cases passing | 40 / 40 |
| Pending (known gaps) | 0 |
| Results with status ok | 31 |
| Verification passed (where one ran) | 31 / 31 |
| Method selected as expected | 29 / 29 |
| Resource-limit results | 1 |
| Not expressible | 0 |
| Latency median / p95 / max (ms) | 94 / 703 / 1438 |
| Max verification residual | 4.27e-14 |

| Group | Cases | Passing | Pending |
| --- | --- | --- | --- |
| existing | 19 | 19 | 0 |
| frechet | 8 | 8 | 0 |
| duhamel | 5 | 5 | 0 |
| evolution | 4 | 4 | 0 |
| unsupported | 3 | 3 | 0 |
| limits | 1 | 1 | 0 |

| Case | Group | Status | Method | Kind | Pass | ms |
| --- | --- | --- | --- | --- | --- | --- |
| exp-diagonal | existing | ok | exp_explicit | explicit | yes | 89 |
| exp-nilpotent | existing | ok | exp_explicit | explicit | yes | 25 |
| exp-defective-repeated-eigenvalue | existing | ok | exp_explicit | explicit | yes | 73 |
| exp-ordered-product | existing | ok | exp_explicit | explicit | yes | 75 |
| exp-abstract-cannot-be-evaluated | existing | unsupported | - | - | yes | 8 |
| derivative-abstract | existing | ok | exp_derivative | expression | yes | 119 |
| derivative-product-keeps-order | existing | ok | exp_derivative | expression | yes | 121 |
| derivative-scalar-function-of-t | existing | ok | exp_derivative | expression | yes | 68 |
| derivative-commuting-declared | existing | ok | exp_derivative | expression | yes | 88 |
| merge-blocked-without-commutation | existing | conditions_not_satisfied | - | - | yes | 8 |
| merge-declared | existing | ok | exp_merge_commuting | expression | yes | 43 |
| integral-invertible | existing | ok | exp_integral_invertible | expression | yes | 161 |
| integral-not-invertible-is-a-representation | existing | ok | block_exponential | representation | yes | 217 |
| integral-singular-explicit | existing | ok | block_exponential | explicit | yes | 91 |
| integral-polynomial | existing | ok | integrate_polynomial | expression | yes | 54 |
| integral-sylvester-explicit | existing | ok | sylvester_reduction | explicit | yes | 236 |
| integral-sylvester-abstract | existing | ok | block_exponential | representation | yes | 376 |
| dimension-mismatch | existing | parse_error | - | - | yes | 7 |
| bare-e-is-rejected | existing | parse_error | - | - | yes | 7 |
| frechet-abstract | frechet | ok | exp_frechet_derivative | representation | yes | 595 |
| frechet-explicit-nilpotent-pair | frechet | ok | exp_frechet_derivative | explicit | yes | 669 |
| frechet-explicit-defective | frechet | ok | exp_frechet_derivative | explicit | yes | 1417 |
| frechet-time-dependent-matrix-abstract | frechet | ok | exp_frechet_derivative | representation | yes | 597 |
| frechet-time-dependent-self-commuting | frechet | ok | exp_derivative | expression | yes | 92 |
| frechet-time-dependent-explicit | frechet | ok | exp_frechet_derivative | explicit | yes | 449 |
| frechet-time-dependent-commutes-from-entries | frechet | ok | exp_derivative | expression | yes | 66 |
| frechet-time-dependent-undeclared-product | frechet | parse_error | - | - | yes | 6 |
| duhamel-abstract | duhamel | ok | duhamel | representation | yes | 497 |
| duhamel-commuting-declared | duhamel | ok | duhamel | expression | yes | 139 |
| duhamel-nilpotent | duhamel | ok | duhamel | explicit | yes | 207 |
| duhamel-diagonal | duhamel | ok | duhamel | explicit | yes | 737 |
| duhamel-defective | duhamel | ok | duhamel | explicit | yes | 370 |
| evolution-constant-abstract | evolution | ok | evolution_commuting | expression | yes | 100 |
| evolution-commuting-family | evolution | ok | evolution_commuting | expression | yes | 120 |
| evolution-self-commuting-declared | evolution | ok | evolution_commuting | representation | yes | 114 |
| evolution-numeric-noncommuting | evolution | ok | evolution_numeric | numeric | yes | 170 |
| unsupported-sine-of-a-matrix | unsupported | unsupported | - | - | yes | 8 |
| unsupported-time-ordered-integral | unsupported | unsupported | - | - | yes | 10 |
| unsupported-evolution-abstract-noncommuting | unsupported | unsupported | - | - | yes | 12 |
| limit-power-too-large | limits | resource_limit | - | - | yes | 7 |

RK4 cross-check, Airy generator on [0, 1.5]: errors 8.5e-11, 5.3e-12, 3.4e-13 at N, 2N, 4N steps; observed order 4.00 (expected 4)
