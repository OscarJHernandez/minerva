"""Extended-tier searches: many random members of the independent Hamiltonian family, one engine run each.

Marked `extended`: they never run on a pull request (`MINERVA_TEST_TIER=extended` runs them, nightly). The pr tier
covers the same code with fixed seeds in `test_discovery_generated.py`; this file hunts for the seeds that break it.

The generator (`bench/`) shares no code with the engine and every reference invariant is checked by an exact Poisson
bracket before use. A failure is pinned by adding the (base, seed, steps) triple to FOUND below and to the fixed
cases, so the pr tier keeps it.
"""

import pytest
import sympy as sp
from bench import cases as bench_cases
from bench import evaluate as bench_eval
from hypothesis import given, settings
from hypothesis import strategies as st
from minerva_engine.discovery import discover_invariants
from minerva_engine.models import Limits

pytestmark = pytest.mark.extended

LIMITS = Limits(timeout_seconds=120)
# Cheap bases (one or two degrees of freedom); each example runs the discovery ladder once.
BASES = [
    "free2",
    "osc2_iso",
    "osc2_res",
    "osc2_aniso",
    "quartic_sep",
    "central_quartic",
    "cyclic_sep",
]
# Minimal failing examples found by this search, kept as plain cases. Empty while the search finds nothing.
FOUND: list[tuple[str, int, int]] = []


def _evaluate(base: str, seed: int, steps: int) -> bench_eval.Evaluation:
    case = bench_cases.build_case(base, seed, steps)
    inp = case.engine_input()
    report = discover_invariants(
        "hamiltonian",
        inp.hamiltonian_latex,
        inp.coordinates,
        inp.momenta,
        [],
        "standard",
        None,
        "t",
        {},
        LIMITS,
    )
    return bench_eval.evaluate(case, report)


@settings(max_examples=12)
@given(st.sampled_from(BASES), st.integers(0, 10_000), st.integers(0, 2))
def test_no_false_or_trivial_discovery_on_random_hamiltonians(
    base: str, seed: int, steps: int
) -> None:
    ev = _evaluate(base, seed, steps)
    assert ev.false_positives == []  # every reported constant has {I, H} = 0, checked independently
    assert ev.trivial_reported == 0  # none lies in the algebra of the earlier ones


@settings(max_examples=12)
@given(st.sampled_from(BASES), st.integers(0, 10_000), st.integers(0, 1), st.integers(0, 3))
def test_a_broken_invariant_is_never_reported(base: str, seed: int, steps: int, index: int) -> None:
    case = bench_cases.negative_control(
        bench_cases.build_case(base, seed, steps),
        index % len(bench_cases.PERTURBATIONS),
        sp.Rational(1, 3),
    )
    inp = case.engine_input()
    report = discover_invariants(
        "hamiltonian",
        inp.hamiltonian_latex,
        inp.coordinates,
        inp.momenta,
        [],
        "standard",
        None,
        "t",
        {},
        LIMITS,
    )
    ev = bench_eval.evaluate(case, report)
    assert all(r.failure != "false_claim" for r in ev.refs)
    assert ev.false_positives == []


@pytest.mark.parametrize(("base", "seed", "steps"), FOUND)
def test_pinned_failures_stay_fixed(base: str, seed: int, steps: int) -> None:
    ev = _evaluate(base, seed, steps)
    assert ev.false_positives == [] and ev.trivial_reported == 0
