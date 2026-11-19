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

### Found by the expanded reference benchmark

| Id | Defect | Status |
| --- | --- | --- |
| V-016 | `solve` ran `nroots` on the whole polynomial. A root of multiplicity 3 or more made the iteration fail to converge and the request ended in `internal_error` (`(x-1)^3 = 0`, `x^3-3x^2+3x-1 = 0`); a double root was split into two nearby numbers (`2` and `2.000000000000000000001897` for `(x-2)^2(x+1)`), every root was marked multiplicity 1, and `x^4 = 0` was shown as four roots of size 1e-31 and 0. No wrong result was labelled verified, but legitimate queries failed and multiplicities were wrong. | fixed: roots come from the squarefree factors (simple roots) with the multiplicity from `factor_list`; a genuine non-convergence is `unsupported`, never `internal_error`; tests `test_repeated_roots_*` |
| V-017 | The targeted probes after a passing check have a wall-clock safety stop (`verify.PROBE_SECONDS`, 0.5 s) next to their point cap, so the verdict depended on the machine's load: on an overloaded CI runner the search ended before it reached the negative real axis and `log(1/z) = -log(z)` was accepted (`branch.log_reciprocal_cut`, shard 1, 2026-11-19; the same row passed on the runs before and after). A search the clock ended is returned as `exhausted` but nothing reads it, so the pass carries no sign that probes were skipped. | open: the adversarial tests now run the probes with the shared test budget as their stop (`tests/adversarial/run.py: patient_probes`; the production constant is unchanged), which removes the load dependence from the tests only. Strict xfail `test_a_search_the_clock_cut_short_does_not_pass_silently` states what is still wrong: a pass after a clock-cut probe search should say so. A pass is still "no disagreement found", never a proof |

### Agent layer

| Id | Defect | Status |
| --- | --- | --- |
| V-015 | `ResultStore` (`agent/refs.py`) kept only the printed LaTeX of a verified result. Restrictions that an earlier step reported (`simplify` cancelled `(x-2)`, so `x \neq 2`) were dropped when the result was reused as `@r1`: the next tool's card and `for_model` text said nothing about `x = 2`, which the original expression excluded. | fixed: the store keeps each result's restrictions (its own plus inherited), `@rN` resolution collects them, the downstream card gets `inherited_restrictions_latex` and the tool message an "Inherited from @rN" note, and `_history` rebuilds them for the next turn. Tests `test_agent_restrictions.py` and the live case `reuse-and-restriction-after-simplify` |
| A-001 | A call chosen by a routing rule ran before the first step, outside any timer: its trace step said 0 ms with no engine time, the turn totals left it out, and a declined route was neither timed nor counted in the tool-call budget. | fixed: the cost is measured where the call ran and carried to the step, the events and the totals; a declined route has `routing.duration_ms` and uses one tool call of the budget; an unknown route reason is a declined route, not an `internal_error`. Tests `test_loop_defects.py` |
| A-002 | A call the loop recovered (re-read with the other `format`) registered only the original spelling as seen, so the model sending the corrected call itself ran it a third time; a timeout of the re-run replaced the first reading's diagnostic; a call that timed out was registered as having a result above. | fixed: both spellings are registered, the first reading is kept when the re-run times out, a timed-out call is never registered. Tests `test_loop_defects.py` |
| A-003 | A tool that outlived the turn's time limit left its worker thread (and the sandbox child it waited for) running up to the sandbox's own limit, holding one of its few slots and able to start more children. | fixed: the loop sets a cancel event (`sandbox.cancel_event`, part of the context a worker copies) on timeout or cancellation; the sandbox polls it, kills its child, frees the slot and starts nothing further. A child that has already answered still returns its whole result. Tests `test_sandbox_cancel.py`, `test_loop_defects.py` |
| A-004 | The loop wrote the declarations of the current call into the shared `ToolContext` and reset them afterwards while an abandoned worker could still be reading them. | fixed: each call gets its own copy. Test `test_loop_defects.py` |
| A-005 | `prompt_sha` hashed the full static prompt and every schema: it did not identify what a call was sent (a subset of tools, the rule blocks loaded so far) and left out the notices. | fixed: `request_sha` per model call (event and stored debug, with the tool groups loaded); `prompt_sha` covers the notices. Tests `test_loop_defects.py` |

### Open limitations (no fix possible by sampling)

| Id | Class | Minimal input | Status |
| --- | --- | --- | --- |
| V-014 | narrow feature at a point that is neither a boundary nor a simple value | `exp(-10^6 (x - 1/7)^2) = 0` (x real) | open: strict xfail row `precision.bump_at_arbitrary_point`; a pass here is "no disagreement found", never a proof |
| A-006 | A turn stopped by a client that left stores its prose and cards but no trace, no debug and no result ids (`routers_chat._store_stopped`), so its events are lost and a later `@rN` cannot refer to its results; its `stop_reason` `stopped` is outside `StopReason`. | open: pinned as a risk, not fixed; no wrong result can result (nothing from it is marked verified) |
| A-007 | Smaller recording gaps in the loop: a failed wrap-up call after `repeated_calls` is not recorded, `steps` leaves out the routed step, a failed claim check keeps no error site, a tool's `verify_ms` is its whole engine time, and sandbox slot waits are not reported apart (no queue timing). | open: recording only; none can change a result or its mark |

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
