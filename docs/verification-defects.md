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
| V-001 | boundary beyond the wide tranche | `\|x-60\| = 60-x` (x real) | no sample reaches x > 50 | fixed |
| V-002 | boundary beyond the wide tranche | `sqrt((x-60)^2) = 60-x` (x real) | same | fixed |
| V-003 | boundary beyond the wide tranche | `Max(x, 49) = 49` (x real) | same | fixed |
| V-004 | piecewise region | `Piecewise((1, x<48), (2, True)) = 1` (x real) | same | fixed |
| V-005 | step functions | `floor(x/100) = 0` (x positive) | same | fixed |
| V-006 | spread shrinks with symbol count | `\|x-5\|+y+z = 5-x+y+z` (real) | with 3 symbols the wide range is 4 | fixed |
| V-007 | scaled branch argument | `log(exp(x/8)) = x/8` (complex) | `\|Im x\| <= 10` stays in the principal branch | fixed |
| V-008 | scaled branch argument | `asin(sin(x/40)) = x/40` (complex) | same | fixed |
| V-009 | integers outside the pool | `KroneckerDelta(n, 7) = 0` (n integer) | integers are drawn from -6..6 | fixed |
| V-010 | integers outside the pool | `prod_{k=-6}^{6}(n-k) = 0` (n integer) | same | fixed |
| V-011 | below tolerance | `x + 1/10^11 = x` (real) | `REL_TOL = 1e-9` | fixed |
| V-012 | below tolerance | `x^2 + 1/10^10 = x^2` (real) | same | fixed |
| V-013 | thin region near zero | `Piecewise((1, x>1/1000), (2, True)) = 1` (x positive) | positives start at 0.01 | fixed |

Not defects (checked, behave correctly): `1e10*x + 5 = 1e10*x` is rejected (19 of 20 points); `sqrt(x^2) = x` and
`log x + log y = log(xy)` (complex) are rejected; `(x^2-3025)/(x-55) = x+55` and `gamma(x+1) = x gamma(x)` are accepted.

Fixed by `counterexample.py` (targeted probes at exact rational points, run by `verify_equal` after its random
points agree; `REL_TOL` is unchanged, a residual of 1e-11 is found because the search compares at 60 and 120
digits against a noise floor of 1e-20). Each is a row in `packages/engine/tests/adversarial/corpus.py` and a test in
`test_counterexample.py` run against the one probe family meant to catch it. Numerical agreement stays "evidence,
not proof" in every case.

### Open limitations (no fix possible by sampling)

| Id | Class | Minimal input | Status |
| --- | --- | --- | --- |
| V-014 | narrow feature at a point that is neither a boundary nor a simple value | `exp(-10^6 (x - 1/7)^2) = 0` (x real) | open: strict xfail row `precision.bump_at_arbitrary_point`; a pass here is "no disagreement found", never a proof |

Also caught on the way (now rows in the corpus and tests, all passing): identities that fail only ON a branch cut
(`log(1/z) = -log z`, `sqrt(conj z) = conj(sqrt z)`; the cut is a measure-zero set a random complex draw never hits)
and regions narrower than the scan grid (`|x - 1/3| < 1e-6`).

## Harness defects

| Id | Defect | Found by | Status |
| --- | --- | --- | --- |
| H-001 | Probes evaluated points as floats whenever a fractional power was present. On a branch cut (`x^2 + 2x + 1` at `x = -1 + i/100` is exactly negative real) float noise flipped the sqrt branch, and `sqrt(x^2+2x+1) = sqrt((x+1)^2)` was rejected (`test_steps.py::test_a_part_inside_a_function_argument`). A false reject. | full test suite | fixed: floats only for powers with a *symbolic* exponent |
| H-002 | Evaluating `log(a**b)` at exact rational `a`, `b` made SymPy extract roots symbolically; handbook verification did not finish in 10 minutes. | handbook `verify` | fixed: float inputs for symbolic exponents |
| H-003 | One undefined grid point (`1/x` at 0) aborted the whole boundary scan. | review | fixed: per-point evaluation |
| H-004 | A broad `except Exception` around the search hid a `NameError` and silently turned the probes off. | repro run showing no change | fixed: handler removed; errors surface |
