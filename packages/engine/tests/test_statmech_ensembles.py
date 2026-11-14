"""Grand canonical and microcanonical ensembles: results, refusals, honest comparisons.

References are plain arithmetic computed here (per-state occupation formulas, math.comb, a brute-force
enumeration), never the code under test.
"""

import itertools
import math
import re

import pytest
import sympy as sp
from minerva_engine import Limits
from minerva_engine.models import Status
from minerva_engine.statmech.models import StatMechInput, StatMechResult
from minerva_engine.statmech.operation import statmech

LIMITS = Limits(timeout_seconds=120)
LEVELS = [{"energy": "0"}, {"energy": "eps"}, {"energy": "2*eps", "degeneracy": "2"}]
EPS, B, MU, KB = 1.1, 0.8, -0.3, 1.7
STATES = [(0.0, 1), (EPS, 1), (2 * EPS, 2)]


def run(**kwargs: object) -> StatMechResult:
    return statmech(StatMechInput(**kwargs), LIMITS)  # type: ignore[arg-type]


def grand(statistics: str, **extra: object) -> StatMechResult:
    assumptions = {"eps": ["positive"]}
    if statistics == "bose":
        assumptions["mu"] = ["negative"]
    return run(
        ensemble="grand_canonical",
        system={"kind": "occupation", "statistics": statistics, "levels": LEVELS},
        assumptions=assumptions,
        **extra,
    )


def value(result: StatMechResult, name: str, **symbols: float) -> float:
    q = next(q for q in result.quantities if q.name == name)
    names = {n: sp.Symbol(n) for n in (*symbols, "k_B", "beta", "T", "mu", "eps")}
    expr = sp.sympify(q.text, locals=names)
    return float(expr.subs({sp.Symbol(k): v for k, v in symbols.items()}))


def passed(result: StatMechResult, name: str) -> dict[str, bool | None]:
    return {c.name: c.passed for q in result.quantities if q.name == name for c in q.checks}


def occupation(statistics: str, e: float) -> float:
    x = B * (e - MU)
    return {
        "fermi": 1 / (math.exp(x) + 1),
        "bose": 1 / (math.exp(x) - 1),
        "boltzmann": math.exp(-x),
    }[statistics]


def reference(statistics: str) -> dict[str, float]:
    n = {e: occupation(statistics, e) for e, _ in STATES}
    total = sum(g * n[e] for e, g in STATES)
    energy = sum(g * e * n[e] for e, g in STATES)
    if statistics == "fermi":
        var = sum(g * n[e] * (1 - n[e]) for e, g in STATES)
        entropy = -sum(
            g * (n[e] * math.log(n[e]) + (1 - n[e]) * math.log(1 - n[e])) for e, g in STATES
        )
    elif statistics == "bose":
        var = sum(g * n[e] * (1 + n[e]) for e, g in STATES)
        entropy = sum(
            g * ((1 + n[e]) * math.log(1 + n[e]) - n[e] * math.log(n[e])) for e, g in STATES
        )
    else:
        var = total
        entropy = sum(g * n[e] * (1 - math.log(n[e])) for e, g in STATES)
    return {"N": total, "U": energy, "Var_N": var, "S": KB * entropy}


@pytest.mark.parametrize("statistics", ["fermi", "bose", "boltzmann"])
def test_each_statistic_matches_its_per_state_formulas(statistics: str) -> None:
    r = grand(statistics, action="derive", quantities=["N", "U", "S", "Var_N"])
    assert r.status is Status.OK, r.message
    kw = {"beta": B, "mu": MU, "eps": EPS, "k_B": KB}
    for name, want in reference(statistics).items():
        assert value(r, name, **kw) == pytest.approx(want), name
    # the enumeration of occupation numbers is a different route from the formulas
    assert passed(r, "N")["enumeration"] is True and passed(r, "N")["finite_difference"] is True
    assert passed(r, "S")["finite_difference"] is True  # S = -dOmega/dT at fixed mu, numerically


@pytest.mark.parametrize("statistics", ["fermi", "bose", "boltzmann"])
def test_the_grand_potential_obeys_euler_and_gives_the_number_by_a_derivative(
    statistics: str,
) -> None:
    r = grand(statistics, action="derive", quantities=["Omega", "N", "U", "S"])
    assert passed(r, "Omega")["free_energy_identity"] is True  # Omega = U - T S - mu N
    kw = {"beta": B, "mu": MU, "eps": EPS, "k_B": KB}
    omega = value(r, "Omega", **kw)
    assert omega == pytest.approx(
        value(r, "U", **kw) - value(r, "S", **kw) / (KB * B) - MU * value(r, "N", **kw)
    )


def test_bose_and_fermi_reduce_to_classical_counting_at_low_occupation() -> None:
    """Maxwell-Boltzmann limit, measured: the relative difference falls like the occupation."""
    gaps = []
    for mu in (-0.3, -3.0, -6.0):
        row = {}
        for statistics in ("bose", "fermi", "boltzmann"):
            r = run(
                ensemble="grand_canonical",
                system={
                    "kind": "occupation",
                    "statistics": statistics,
                    "levels": [{"energy": "eps"}],
                },
                assumptions={"eps": ["positive"], "mu": ["negative"]},
                action="derive",
                quantities=["N"],
            )
            row[statistics] = value(r, "N", beta=B, mu=mu, eps=EPS)
        gaps.append(abs(row["bose"] - row["boltzmann"]) / row["boltzmann"])
        assert row["fermi"] < row["boltzmann"] < row["bose"]  # exclusion lowers, bunching raises
    assert gaps[0] > gaps[1] > gaps[2]
    # the gap is the occupation itself, e^{-x}, to leading order: the classical limit is low occupation
    assert gaps[2] == pytest.approx(math.exp(-B * (EPS + 6.0)), rel=0.02)


def test_occupation_numbers_are_checked_per_level() -> None:
    r = grand("fermi", action="partition")
    assert r.status is Status.OK
    q = next(q for q in r.quantities if q.name == "occupations")
    assert len(q.checks) == 3 and all(c.passed for c in q.checks)


def test_particle_number_fluctuations_are_derivatives_of_ln_xi() -> None:
    r = grand("bose", action="expectation", observable="N", statistic="variance")
    assert r.status is Status.OK, r.message
    assert passed(r, "variance_N")["fluctuation_response"] is True
    kw = {"beta": B, "mu": MU, "eps": EPS}
    assert value(r, "variance_N", **kw) == pytest.approx(reference("bose")["Var_N"])
    third = grand("fermi", action="expectation", observable="N", statistic="cumulant", order=3)
    assert third.status is Status.OK and passed(third, "cumulant_N")["enumeration"] is True


def test_heat_capacity_at_fixed_chemical_potential_is_a_held_fixed_derivative() -> None:
    r = grand(
        "fermi",
        action="derive",
        quantities=["U"],
        partials=[{"of": "U", "variable": "T", "held_fixed": ["mu"], "label": "C_mu"}],
    )
    assert r.status is Status.OK, r.message
    q = next(q for q in r.quantities if q.name == "C_mu")
    assert q.held_fixed == ["mu"]

    def u(temperature: float) -> float:
        b = 1 / (KB * temperature)
        return sum(g * e / (math.exp(b * (e - MU)) + 1) for e, g in STATES)

    t0, h = 1 / (KB * B), 1e-5
    want = (u(t0 + h) - u(t0 - h)) / (2 * h)
    assert value(r, "C_mu", beta=B, mu=MU, eps=EPS, k_B=KB) == pytest.approx(want, rel=1e-6)


def test_a_derivative_that_lets_mu_vary_is_not_the_one_asked_for() -> None:
    r = grand(
        "fermi",
        action="derive",
        quantities=[],
        partials=[{"of": "U", "variable": "T", "held_fixed": []}],  # mu is a state variable here
    )
    assert r.reason == "ensemble_mismatch" and "mu" in r.message


def test_bose_statistics_need_mu_below_the_levels() -> None:
    r = run(
        ensemble="grand_canonical",
        system={"kind": "occupation", "statistics": "bose", "levels": LEVELS},
        assumptions={"eps": ["positive"]},
        action="partition",
    )
    assert r.status is Status.CONDITIONS_NOT_SATISFIED and r.reason == "convergence_not_provable"
    assert "below every single-particle energy" in r.message and not r.quantities


def test_the_ensemble_must_match_the_system() -> None:
    occupation_system = {"kind": "occupation", "statistics": "fermi", "levels": LEVELS}
    assert (
        run(system=occupation_system, assumptions={"eps": ["positive"]}, action="partition").reason
        == "ensemble_mismatch"
    )
    levels = {"kind": "levels", "levels": [{"energy": "0"}, {"energy": "1"}]}
    assert (
        run(ensemble="grand_canonical", system=levels, action="partition").reason
        == "ensemble_mismatch"
    )
    classical = {
        "kind": "classical",
        "hamiltonian": "p**2/2",
        "variables": [{"name": "p"}],
        "phase_space_cell": "1",
    }
    assert (
        run(ensemble="microcanonical", system=classical, action="partition").reason
        == "ensemble_mismatch"
    )
    assert run(system=levels, total_energy="2", action="partition").reason == "ensemble_mismatch"


def test_grand_canonical_refusals() -> None:
    copies = grand("fermi", copies={"count": "2", "identical": False}, action="partition")
    assert copies.reason == "unsupported_system"
    energy = grand("fermi", action="expectation", observable="E")
    assert energy.reason == "unknown_observable"
    bad = run(
        ensemble="grand_canonical",
        system={
            "kind": "occupation",
            "statistics": "fermi",
            "levels": [{"energy": "0", "degeneracy": "0"}],
        },
        action="partition",
    )
    assert bad.reason == "invalid_degeneracy"
    same = run(
        ensemble="grand_canonical",
        system={"kind": "occupation", "statistics": "fermi", "levels": LEVELS, "mu": "eps"},
        assumptions={"eps": ["positive"]},
        action="partition",
    )
    assert same.reason == "unsupported_system"  # the energies would depend on mu


# -- microcanonical -------------------------------------------------------------------------
TWO = {"kind": "levels", "levels": [{"energy": "0"}, {"energy": "1"}]}


def micro(
    system: object = TWO, copies: str = "10", energy: str = "4", identical: bool = False
) -> StatMechResult:
    return run(
        ensemble="microcanonical",
        system=system,
        copies={"count": copies, "identical": identical},
        total_energy=energy,
        action="partition",
    )


def omega(r: StatMechResult) -> str:
    return next(q for q in r.quantities if q.name == "Omega").text


@pytest.mark.parametrize("n,k", [(10, 4), (20, 7), (40, 16)])
def test_two_level_counting_is_a_binomial_coefficient(n: int, k: int) -> None:
    r = micro(copies=str(n), energy=str(k))
    assert r.status is Status.OK, r.message
    assert omega(r) == str(math.comb(n, k))
    entropy = next(q for q in r.quantities if q.name == "S")
    assert entropy.text == f"k_B*log({math.comb(n, k)})"
    assert all(c.passed for c in next(q for q in r.quantities if q.name == "Omega").checks)


def test_three_levels_with_degeneracy_agree_with_a_brute_force_count() -> None:
    levels = [(0, 1), (1, 2), (3, 1)]
    system = {
        "kind": "levels",
        "levels": [{"energy": str(e), "degeneracy": str(g)} for e, g in levels],
    }
    n, target = 6, 7
    want = sum(
        math.prod(levels[k][1] for k in assignment)
        for assignment in itertools.product(range(3), repeat=n)
        if sum(levels[k][0] for k in assignment) == target
    )
    r = micro(system, copies=str(n), energy=str(target))
    assert r.status is Status.OK and omega(r) == str(want)


def test_indistinguishable_copies_divide_by_n_factorial() -> None:
    r = micro(copies="4", energy="2", identical=True)
    assert omega(r) == f"{math.comb(4, 2)}/{math.factorial(4)}".replace("6/24", "1/4")
    assert any("N!" in n for n in r.notes)


def test_the_temperature_estimate_is_a_labelled_measurement_that_improves_with_n() -> None:
    def gap(n: int) -> float:
        r = micro(copies=str(n), energy=str(round(0.4 * n)))
        text = next(q for q in r.quantities if q.name == "beta_estimate").text
        assert "finite difference" in text and "relative difference" in text
        return float(re.search(r"relative difference ([0-9.e-]+)", text).group(1))  # type: ignore[union-attr]

    gaps = [gap(10), gap(20), gap(40)]
    assert gaps[0] > gaps[1] > gaps[2]  # equivalence of ensembles is approached, not assumed
    r = micro(copies="40", energy="16")
    estimate = next(q for q in r.quantities if q.name == "beta_estimate")
    assert estimate.result_kind == "numeric" and "not a derivative" in estimate.route


def test_microcanonical_refusals() -> None:
    assert micro(copies="4", energy="9/2").reason == "energy_not_attainable"
    unconfirmed = run(
        ensemble="microcanonical",
        system=TWO,
        copies={"count": "4"},
        total_energy="2",
        action="partition",
    )
    assert unconfirmed.reason == "convention_unconfirmed"
    missing = run(
        ensemble="microcanonical",
        system=TWO,
        copies={"count": "4", "identical": False},
        action="partition",
    )
    assert missing.reason == "parse_error"
    symbolic = micro({"kind": "levels", "levels": [{"energy": "0"}, {"energy": "eps"}]}, "4", "2")
    assert symbolic.status is not Status.OK
    assert micro(copies="41", energy="20").reason == "resource_limit"
