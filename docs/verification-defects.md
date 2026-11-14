# Verification defects

Mathematical defects found in Minerva's own checks, kept apart from defects of the test harness. Each has an id, a
minimal input, and a status. A defect that is not fixed is pinned by an `xfail(strict=True)` row in
`packages/engine/tests/adversarial/` (or a `pending` reference problem), so fixing it forces the row to be promoted.
Rows are never deleted or loosened to make a suite pass.

Status: `open` (reproduced, not fixed), `fixed` (commit noted), `wontfix` (reason given; the limit is documented).

## Mathematical defects

Found by the first adversarial pass (2026-11-14) against `verify_equal(..., symbolic=False)`. Every input below
passed at 20 of 20 sample points, and so reaches `claims._compare`, where a model's prose claim would be `holds`.

| Id | Class | Minimal input (assumptions) | Why sampling misses it | Status |
| --- | --- | --- | --- | --- |
| V-001 | boundary beyond the wide tranche | `\|x-60\| = 60-x` (x real) | no sample reaches x > 50 | open |
| V-002 | boundary beyond the wide tranche | `sqrt((x-60)^2) = 60-x` (x real) | same | open |
| V-003 | boundary beyond the wide tranche | `Max(x, 49) = 49` (x real) | same | open |
| V-004 | piecewise region | `Piecewise((1, x<48), (2, True)) = 1` (x real) | same | open |
| V-005 | step functions | `floor(x/100) = 0` (x positive) | same | open |
| V-006 | spread shrinks with symbol count | `\|x-5\|+y+z = 5-x+y+z` (real) | with 3 symbols the wide range is 4 | open |
| V-007 | scaled branch argument | `log(exp(x/8)) = x/8` (complex) | `\|Im x\| <= 10` stays in the principal branch | open |
| V-008 | scaled branch argument | `asin(sin(x/40)) = x/40` (complex) | same | open |
| V-009 | integers outside the pool | `KroneckerDelta(n, 7) = 0` (n integer) | integers are drawn from -6..6 | open |
| V-010 | integers outside the pool | `prod_{k=-6}^{6}(n-k) = 0` (n integer) | same | open |
| V-011 | below tolerance | `x + 1/10^11 = x` (real) | `REL_TOL = 1e-9` | open |
| V-012 | below tolerance | `x^2 + 1/10^10 = x^2` (real) | same | open |
| V-013 | thin region near zero | `Piecewise((1, x>1/1000), (2, True)) = 1` (x positive) | positives start at 0.01 | open |

Not defects (checked, behave correctly): `1e10*x + 5 = 1e10*x` is rejected (19 of 20 points); `sqrt(x^2) = x` and
`log x + log y = log(xy)` (complex) are rejected; `(x^2-3025)/(x-55) = x+55` and `gamma(x+1) = x gamma(x)` are accepted.

Plan: V-001..V-010 and V-013 are addressed by targeted probes (stage 1 of the verification plan), V-011/V-012 by an
exact rational probe that does not touch `REL_TOL`. Numerical agreement stays "evidence, not proof" in every case.

## Harness defects

None recorded yet.
