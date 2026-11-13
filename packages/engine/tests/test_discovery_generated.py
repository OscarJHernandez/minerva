"""Discovery on generated Hamiltonians with independently known invariants (fixed seeds, CI).

The reference side (`bench/`) shares no code with the discovery side: it builds a Hamiltonian from a
curated or constructed integrable system, applies a seeded exact canonical map, and verifies every
reference invariant with its own Poisson bracket. Minerva receives only the Hamiltonian and the variable
names (`EngineInput`). Larger randomized sweeps run offline: `scripts/bench_invariants.py sweep`.
"""

import dataclasses
import functools
import json

import pytest
import sympy as sp
from bench import cases as bench_cases
from bench import evaluate as bench_eval
from bench.symplectic import bracket, canonical_brackets_preserved, map_from_recipe, transform
from minerva_engine.discovery import DiscoveryReport, discover_invariants
from minerva_engine.models import Limits, Status

LIMITS = Limits(timeout_seconds=120)
CASES = bench_cases.load_fixed()
BY_ID = {c.id: c for c in CASES}

# Reference invariants that are not recovered as an exact linear combination, with the reason
# (bench/fixtures/pending.json). A case that starts to pass fails the test until it is removed from there.
PENDING = bench_cases.load_pending()


@functools.cache
def run(case_id: str) -> DiscoveryReport:
    inp = BY_ID[case_id].engine_input()
    return discover_invariants(
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


@functools.cache
def evaluation(case_id: str) -> bench_eval.Evaluation:
    return bench_eval.evaluate(BY_ID[case_id], run(case_id))


# --- the reference side is trustworthy -------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_fixture_regenerates_exactly_from_its_seed(case: bench_cases.Case) -> None:
    again = bench_cases.rebuild(case)
    assert json.dumps(bench_cases.to_json(again), sort_keys=True) == json.dumps(
        bench_cases.to_json(case), sort_keys=True
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_the_map_is_canonical_and_the_references_are_conserved(case: bench_cases.Case) -> None:
    q, p = case.q, case.p
    m = bench_cases.map_matrix(case)
    assert canonical_brackets_preserved(m, q, p)
    for ref in case.refs:
        zero = sp.simplify(bracket(ref.expr, case.H, q, p)) == 0
        if case.kind == "negative_control":
            assert zero == (not case.broken[ref.name])
        else:
            assert zero, ref.name


def test_the_engine_is_given_the_hamiltonian_and_names_only() -> None:
    assert {f.name for f in dataclasses.fields(bench_cases.EngineInput)} == {
        "hamiltonian_latex",
        "coordinates",
        "momenta",
    }
    case = BY_ID["osc2_res.s1.k0"]
    text = json.dumps(dataclasses.asdict(case.engine_input()))
    assert str(case.refs[2].expr) not in text


# --- discovery ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_no_false_claims_and_no_trivial_discoveries(case: bench_cases.Case) -> None:
    ev = evaluation(case.id)
    assert ev.false_positives == []  # every reported constant has {I, H} = 0, checked independently
    assert ev.trivial_reported == 0  # none lies in the algebra of the earlier ones
    assert all(r.failure != "false_claim" for r in ev.refs)  # a broken invariant is never reported
    if ev.rank_upper:
        assert ev.rank_lower <= ev.rank_upper <= 2 * case.n - 1


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_complete_within_every_family_that_was_searched(case: bench_cases.Case) -> None:
    """An invariant that lies in a searched family (found independently of the engine) is recovered."""
    ev = evaluation(case.id)
    for ref in ev.refs:
        if case.broken.get(ref.name):
            continue
        if ref.dominated:
            assert ref.recovered == "linear", (case.id, ref)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_recovery_matches_the_recorded_expectations(case: bench_cases.Case) -> None:
    ev = evaluation(case.id)
    actual = {
        r.name: (r.recovered if r.recovered != "no" else f"no:{r.failure}")
        for r in ev.refs
        if r.recovered != "linear" and not case.broken.get(r.name)
    }
    assert actual == PENDING.get(case.id, {})


@pytest.mark.parametrize(
    "case", [c for c in CASES if c.kind == "negative_control"], ids=lambda c: c.id
)
def test_broken_symmetries_are_rejected_and_survivors_stay_eligible(case: bench_cases.Case) -> None:
    ev = evaluation(case.id)
    for ref in ev.refs:
        if case.broken[ref.name]:
            assert ref.recovered == "no"
        else:
            assert ref.recovered == "linear"  # an unbroken invariant is still found
    assert any(case.broken.values())


def test_the_independent_count_is_recovered_for_in_family_systems() -> None:
    for case in CASES:
        if case.id in PENDING or case.kind != "discovery":
            continue
        ev = evaluation(case.id)
        assert ev.independent_found >= case.ref_rank, case.id
        assert ev.rank_lower >= min(case.ref_rank, ev.rank_upper), case.id


def test_unsupported_family_is_reported_without_inventing_results() -> None:
    report = run("kepler_cartesian.s1.k0")
    assert report.status is Status.UNSUPPORTED and report.constants == []
    ev = evaluation("kepler_cartesian.s1.k0")
    assert {r.failure for r in ev.refs} == {"missing_function_family"}


def test_functional_equivalence_across_different_expressions() -> None:
    """Same invariant space, different expressions: the transformed oscillator's constants differ from
    the reference expressions textually, yet span the same space."""
    case = BY_ID["osc2_iso.s2.k2"]
    constants = [str(e) for _, e in bench_eval.parse_constants(run(case.id))]
    texts = {str(r.expr) for r in case.refs}
    assert not set(constants) <= texts  # not the same strings
    assert all(r.recovered == "linear" for r in evaluation(case.id).refs)


# --- metamorphic: canonical maps that keep every coordinate separate preserve the discovery --------------
@pytest.mark.parametrize(
    ("base", "seed", "steps"), [("osc2_iso", 4, 2), ("sw", 3, 2), ("sep_sum", 2, 1)]
)
def test_discovery_commutes_with_a_separable_canonical_map(
    base: str, seed: int, steps: int
) -> None:
    parent = bench_cases.build_case(base, seed, 0, mixing=False)
    child = bench_cases.build_case(base, seed, steps, mixing=False)
    m = map_from_recipe(child.recipe, child.n)
    q, p = child.q, child.p
    assert canonical_brackets_preserved(m, q, p)
    # round trip: mapping the child's Hamiltonian back gives the parent's
    assert sp.expand(transform(child.H, m.inv(), q, p) - parent.H) == 0

    reports = {}
    for label, case in (("parent", parent), ("child", child)):
        inp = case.engine_input()
        reports[label] = discover_invariants(
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
    ev = {k: bench_eval.evaluate(c, reports[k]) for k, c in (("parent", parent), ("child", child))}
    for e in ev.values():
        assert e.false_positives == [] and all(r.recovered == "linear" for r in e.refs)
    ip, ic = reports["parent"].independence, reports["child"].independence
    assert ip is not None and ic is not None
    assert (ip.rank_lower, ip.rank_proved) == (ic.rank_lower, ic.rank_proved)
    assert len(ip.relations) == len(ic.relations)
    # constants found for the parent, carried through the map, are in the child's discovered space
    child_constants = [e for _, e in bench_eval.parse_constants(reports["child"])]
    for _, expr in bench_eval.parse_constants(reports["parent"]):
        mapped = transform(expr, m, q, p)
        assert sp.simplify(bracket(mapped, child.H, q, p)) == 0
        assert bench_eval.in_span(bench_eval.algebra(child_constants), mapped, [*q, *p]), mapped


# --- exceptional parameter values ------------------------------------------------------------------------------
def _oscillator(freq_sq: str) -> DiscoveryReport:
    return discover_invariants(
        "hamiltonian",
        rf"\frac{{p_1^2+q_1^2}}{{2}}+\frac{{p_2^2+{freq_sq} q_2^2}}{{2}}",
        ["q_1", "q_2"],
        ["p_1", "p_2"],
        [],
        "standard",
        None,
        "t",
        {},
        LIMITS,
    )


def test_resonant_frequencies_have_extra_invariants_that_generic_ones_lack() -> None:
    resonant = _oscillator("4")  # frequencies 1 : 2
    generic = _oscillator(r"\frac{49}{9}")  # 1 : 7/3, not resonant
    assert resonant.status is generic.status is Status.OK
    assert max(c.momentum_degree for c in resonant.constants) == 3  # the cubic resonance invariants
    assert max(c.momentum_degree for c in generic.constants) == 2
    assert len(resonant.new_constants) > len(generic.new_constants)
    assert resonant.independence is not None and resonant.independence.rank_lower == 3
    assert generic.independence is not None and generic.independence.rank_lower == 2


def test_symbolic_parameters_are_generic_and_say_so() -> None:
    report = discover_invariants(
        "hamiltonian",
        r"\frac{p_1^2+q_1^2}{2}+\frac{p_2^2+w^2 q_2^2}{2}",
        ["q_1", "q_2"],
        ["p_1", "p_2"],
        [],
        "standard",
        None,
        "t",
        {},
        LIMITS,
    )
    assert report.status is Status.OK
    assert (
        max(c.momentum_degree for c in report.constants) == 2
    )  # no resonance invariant at generic w
    assert "generic" in " ".join(report.notes)
