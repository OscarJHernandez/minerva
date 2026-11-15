# Verifying the verifier

Minerva shows a result only if an independent check passed. This document is about the checks of those checks:
what tests exist, which tier runs them, how to reproduce a number, and what the verifier cannot establish.
Defects found along the way are in [verification-defects.md](verification-defects.md); the measured state is in
[verification-report.md](verification-report.md).

**Numerical agreement is evidence, never proof.** A symbolic zero test is the only proof of an identity. Where a
class of inputs cannot be verified by sampling, the answer is `unverified` or `unsupported`, not a pass.

## What exists

| Layer | Where | Oracle (never the code under test) |
| --- | --- | --- |
| Independent reference problems | `packages/engine/src/minerva_engine/reference/` (one module per domain), report in [reference-benchmark.md](reference-benchmark.md) | `mpmath` quadrature, differentiation, Taylor, `polyroots`, `expm`; regularised delta sequences; exact solutions; exact Poisson brackets; plain arithmetic. Every problem cites a source by locator and declares its oracle, the evidence kind and its tier |
| Adversarial corpus | `packages/engine/tests/adversarial/` | The truth of each identity as written (52 identities, 34 false and 18 true, in algebra, singularities, branch cuts, piecewise functions, integers, precision and conditional identities; a true identity that is rejected is a false reject, as bad as a false accept) |
| Targeted probes and counterexample search | `packages/engine/src/minerva_engine/counterexample.py`, run by `verify_equal` | Exact rational points re-evaluated at 60 and 120 digits |
| Property and metamorphic tests | `packages/engine/tests/test_properties*.py`, generators in `packages/engine/tests/strategies/` | Exact noncommutative expansion, textbook curvature, a hand Legendre transformation, `mpmath` finite differences, the independent `bench/` Hamiltonian generator |
| Agent evaluation | `services/api/src/minerva_api/agent/evaluation/` (`cases.yaml`, `cases_reliability.yaml`) | Reference calls checked against the engine; scripted ideal and adversarial trajectories |
| Report | `scripts/verification_report.py` | Aggregates the above; no new oracle |

### Targeted probes

`verify_equal` samples 12 ordinary and 8 wide random points. When they all agree it now also tries up to 24
probes (`counterexample.TARGETED`) at exact rational points that respect the declared assumptions:

- **boundary**: where an argument of `Abs`, `Max`/`Min`, `Piecewise`, `floor`/`ceiling`, `sign`, `Heaviside`,
  `KroneckerDelta`, `log` or a fractional power changes sign or touches zero (found by a scan with local refinement,
  so regions narrower than the grid are found too), on both sides, and between adjacent roots;
- **branch**: large imaginary parts and the negative real axis for symbols without assumptions;
- **special**: integers outside −6..6, and 0, ±1, ±2, ±3, 1/2, 10, 100 for other real symbols;
- **scale**: magnitudes 10², 10³, 10⁻³, 10⁻⁶.

A disagreement counts only if it persists at 60 and 120 digits with a stable residual far above rounding noise
(`NOISE_FLOOR`, relative to the size of the values); one that vanishes with precision is *cancellation*, counted but
never reported. A point where exactly one side is undefined is **not** a counterexample (it can be a removable
singularity). Probes can only turn a pass into a failure; a probe is never agreement. `REL_TOL` is unchanged: an error of
10⁻¹¹ is found because the search compares against a noise floor of 10⁻²⁰, not because the tolerance was tightened.

`counterexample.find_counterexample(a, b)` is available directly; it returns a `Counterexample` (expressions,
assumptions, the exact substitution, both values, the residual, the probe family, the precision, the seed, whether
it was minimised) or `None`, which means *none was found*, never *none exists*. The record holds the user's
expressions: it is returned to the caller and never logged (a test checks that no field name is in a log allow-list).
Noncommuting symbols and distributions are `unsupported` there: use `ncverify` and the distribution operation.

## Tiers

| Tier | Runs | Command | Time |
| --- | --- | --- | --- |
| Pull request | every pull request, in the existing test shards | `make check-py` (or `uv run pytest`) | the suite is within noise of before (measured: 6:18 against 6:31 wall) |
| Extended | nightly and on demand ([extended.yml](../.github/workflows/extended.yml)) | `make test-extended`, `make reference-extended` | tens of minutes |
| Model evaluation | by hand, when a model host is configured ([model-eval.yml](../.github/workflows/model-eval.yml)) | `make eval`, `scripts/eval_models.py` | hours |

`MINERVA_TEST_TIER=extended` runs tests marked `extended` (skipped otherwise, before sharding);
`MINERVA_HYPOTHESIS_PROFILE=extended` runs property tests at 100 random examples with a printed reproduction blob
instead of 8 derandomized ones. A failure found in the extended tier is **pinned**: add the minimal example with
`@example(...)` (or the seed triple to `FOUND` in `test_properties_extended.py`) so the pull-request tier keeps it, and add a
row to the adversarial corpus or a problem to the reference corpus. Reproduce a nightly failure with
`--hypothesis-seed=<seed printed in the job>`.

Reference problems have two tiers (`tier` on the problem). The pull-request tier is also what
`test_reference_benchmark.py` runs; rewrite the committed report only with
`python -m minerva_engine.reference_benchmark --tier extended`. **Expected answers are never regenerated from the engine**:
they are written by hand in the judges with a source locator, and `--check` only compares recorded outcomes. A problem the
engine cannot do yet is `pending` with an id; if it starts to pass, the suite fails until the problem is promoted.

Each reference judge is itself tested: it must reject an engine whose results are shifted by 1/1000 (or whose refusals
are forced to `ok`).

## Reproducing the numbers

```bash
uv run python scripts/verification_report.py                    # rewrite the report from the committed artefacts
uv run python scripts/verification_report.py --check            # fail if it is stale (CI runs this)
make verification-compare                                       # before / after
# the "before", regenerated:
uv run python scripts/verification_report.py --probes off --reference-rev f0015b0 \
    --json-out docs/benchmarks/verification-summary-baseline.json
```

The baseline is measured: the same adversarial corpus through the verifier with probes switched off (exactly the earlier
behaviour), and the reference report as it was at commit `f0015b0`. Result: false accepts 17 → 0, false rejects 0 → 0,
reference problems 9 → 46. **Read this correctly**: the corpus was written after the gaps were found, so the drop shows that
those classes are closed, not what fraction of all false identities the verifier catches. The property tests and the
extended tier are what search for classes nobody thought of; they have found no new defect in the runs recorded here
(100 examples per test, seeds 424242 and 7), and the reference corpus found one (V-016).

## What the verifier cannot establish

Documented, not hidden:

- **Narrow features at arbitrary points** (V-014): `exp(-10⁶ (x − 1/7)²) = 0` has no boundary and no simple value to
  probe. Sampling cannot find it. The strict-xfail row `precision.bump_at_arbitrary_point` pins it.
- **Distributions** are never verified by sampling (`verify_equal` refuses them); the distribution operation's exact
  conditions and pairing apply, and its result is "supporting evidence, not a proof".
- **Noncommuting symbols** are checked on random matrices of a sufficient size (`ncverify`); this is evidence for a
  polynomial identity, not a proof, and the counterexample search does not search them.
- **Branch cuts** are probed on the real axis and at large imaginary parts; identities that fail only on a cut that
  none of these touches are not found.
- **A probe budget is finite**: `exhausted` says the search stopped before trying everything.
- **Equation solving, series and integrals** keep their own independent checks (see `AGENTS.md`); the probes add to
  `verify_equal` only.
- **Statistics** (sums, expectations, likelihoods, Fisher information) is implemented on the unmerged branch
  `claude/minerva-statistical-derivations-c00973`, not here. The reference package has no statistics module yet; add one
  (`reference/statistics.py`, registered in `reference/__init__.py`) when that branch lands, with oracles such as
  exact summation or quadrature and independently constructed log-likelihoods.

## Agent reliability

`cases_reliability.yaml` (run with `--suite reliability`) checks behaviours of the whole loop that single-turn tool
selection does not show, over live multi-turn conversations: the agent's real earlier answers are rebuilt into the next turn
exactly as the chat route does. Beyond the existing checks it reports four failure classes:

| Class | Meaning |
| --- | --- |
| `invented_assumption` | an assumption the user never stated was passed to a tool (positivity, realness, a matrix declared commuting or invertible) |
| `ignored_failed_verification` | a tool result was not verified and the agent carried on instead of stopping |
| `misreported_restriction` | a result that carries a domain restriction was reported without it |
| `result_reuse` | an earlier verified result was retyped instead of reused as `@rN` |

A restriction travels with a result: when `simplify` cancels `(x − 1)` and reports `x ≠ 1`, a later step that reuses
`@r1` inherits it (card field `inherited_restrictions_latex` and a line in the tool message), also across turns (V-015).

The deterministic coverage needs no model: every reliability case has an ideal trajectory that must pass and adversarial ones
that must fail with a named class (`make eval-reliability`). **Real-model runs are not part of this change**; to compare a
model on the new cases against a baseline with the same models and settings:

```bash
uv run python scripts/eval_models.py --suite all --base-url http://HOST:11434 --models a,b --repeat 3 --seed 7 \
    --label BASELINE --json-out docs/benchmarks/BASELINE.json --out docs/benchmarks/BASELINE.md
# change the code or the prompt, then
uv run python scripts/eval_models.py --suite all --base-url http://HOST:11434 --models a,b --repeat 3 --seed 7 \
    --label NEW --json-out docs/benchmarks/NEW.json --out docs/benchmarks/NEW.md --compare docs/benchmarks/BASELINE.json
```

The core suite's case-set digest is unchanged by this work (`04c6550e6ca6`, the value on `main` after the statistical-mechanics and variational cases; it was `1050a8e33d4e` before), so earlier core runs stay comparable; reports that include the
reliability suite carry a different digest, and the comparison says so. The host name never goes into a report.
