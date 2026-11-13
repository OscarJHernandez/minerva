# Matrix calculus benchmark

`uv run python -m minerva_knowledge matrix-benchmark [--repeat N] [--section LABEL] [--out FILE]` runs
hand-written cases (`packages/knowledge/src/minerva_knowledge/data/benchmark/matrix_cases.yaml`)
through the real engine and sandbox. Groups: `existing` (behaviour that shipped before this work),
`frechet`, `duhamel`, `evolution`, `unsupported` (gaps that must stay explicit) and `limits`.

A case passes when status, and where given method, result kind and the `evaluated` flag, match.
`pending` cases are known gaps: reported, never asserted. A pending case that already passes fails the
test suite until `pending` is removed. A request the engine cannot express (for example a field that
does not exist yet) is reported as `not_expressible`, so the final case file can be run against an
older commit for the "before" run. Each `## Run:` section below is one run of the same case file.

Limits of this evidence: timings are medians over the repeats on one machine (the first call in a
process is excluded, forking the sandbox is included), are noisy and are never asserted; the cases are
small hand-written problems, not a sample of real use; agreement with the expected status and method
shows the engine behaves as designed, not that every result is right: correctness is established by the
checks inside the engine and by the regression tests.

## Run: Before (baseline, commit 7b19980)

38 cases, 5 repeat(s) each.

| Measure | Value |
| --- | --- |
| Cases passing (all, incl. pending) | 22 / 38 |
| Scored cases passing | 22 / 22 |
| Pending (known gaps) | 16 |
| Results with status ok | 20 |
| Verification passed (where one ran) | 20 / 20 |
| Method selected as expected | 13 / 27 |
| Resource-limit results | 1 |
| Not expressible | 7 |
| Latency median / p95 / max (ms) | 70 / 460 / 689 |

| Group | Cases | Passing | Pending |
| --- | --- | --- | --- |
| existing | 19 | 19 | 0 |
| frechet | 6 | 0 | 6 |
| duhamel | 5 | 0 | 5 |
| evolution | 4 | 0 | 4 |
| unsupported | 3 | 2 | 1 |
| limits | 1 | 1 | 0 |

| Case | Group | Status | Method | Kind | Pass | ms |
| --- | --- | --- | --- | --- | --- | --- |
| exp-diagonal | existing | ok | exp_explicit | explicit | yes | 85 |
| exp-nilpotent | existing | ok | exp_explicit | explicit | yes | 24 |
| exp-defective-repeated-eigenvalue | existing | ok | exp_explicit | explicit | yes | 69 |
| exp-ordered-product | existing | ok | exp_explicit | explicit | yes | 71 |
| exp-abstract-cannot-be-evaluated | existing | unsupported | - | - | yes | 7 |
| derivative-abstract | existing | ok | exp_derivative | expression | yes | 114 |
| derivative-product-keeps-order | existing | ok | exp_derivative | expression | yes | 118 |
| derivative-scalar-function-of-t | existing | ok | exp_derivative | expression | yes | 64 |
| derivative-commuting-declared | existing | ok | exp_derivative | expression | yes | 84 |
| merge-blocked-without-commutation | existing | conditions_not_satisfied | - | - | yes | 8 |
| merge-declared | existing | ok | exp_merge_commuting | expression | yes | 42 |
| integral-invertible | existing | ok | exp_integral_invertible | expression | yes | 153 |
| integral-not-invertible-is-a-representation | existing | ok | block_exponential | representation | yes | 205 |
| integral-singular-explicit | existing | ok | block_exponential | explicit | yes | 84 |
| integral-polynomial | existing | ok | integrate_polynomial | expression | yes | 50 |
| integral-sylvester-explicit | existing | ok | sylvester_reduction | explicit | yes | 217 |
| integral-sylvester-abstract | existing | ok | block_exponential | representation | yes | 361 |
| dimension-mismatch | existing | parse_error | - | - | yes | 6 |
| bare-e-is-rejected | existing | parse_error | - | - | yes | 6 |
| frechet-abstract | frechet | unsupported | - | - | pending | 59 |
| frechet-explicit-nilpotent-pair | frechet | unsupported | - | - | pending | 58 |
| frechet-explicit-defective | frechet | unsupported | - | - | pending | 58 |
| frechet-time-dependent-matrix-abstract | frechet | not_expressible | - | - | pending | 0 |
| frechet-time-dependent-self-commuting | frechet | not_expressible | - | - | pending | 0 |
| frechet-time-dependent-explicit | frechet | parse_error | - | - | pending | 8 |
| duhamel-abstract | duhamel | ok | block_exponential | representation | pending | 460 |
| duhamel-commuting-declared | duhamel | ok | block_exponential | representation | pending | 215 |
| duhamel-nilpotent | duhamel | ok | block_exponential | explicit | pending | 178 |
| duhamel-diagonal | duhamel | ok | block_exponential | explicit | pending | 688 |
| duhamel-defective | duhamel | ok | block_exponential | explicit | pending | 350 |
| evolution-constant-abstract | evolution | not_expressible | - | - | pending | 0 |
| evolution-commuting-family | evolution | not_expressible | - | - | pending | 0 |
| evolution-self-commuting-declared | evolution | not_expressible | - | - | pending | 0 |
| evolution-numeric-noncommuting | evolution | not_expressible | - | - | pending | 0 |
| unsupported-sine-of-a-matrix | unsupported | unsupported | - | - | yes | 7 |
| unsupported-time-ordered-integral | unsupported | unsupported | - | - | yes | 8 |
| unsupported-evolution-abstract-noncommuting | unsupported | not_expressible | - | - | pending | 0 |
| limit-power-too-large | limits | resource_limit | - | - | yes | 6 |
