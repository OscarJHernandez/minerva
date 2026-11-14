"""Statistical mechanics of level families, densities of states and classical Hamiltonians.

Expected values are computed here with plain arithmetic or mpmath, never with the code under test.
"""

import math

import mpmath as mp
import pytest
import sympy as sp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from minerva_engine import Limits
from minerva_engine.models import Status
from minerva_engine.statmech.models import StatMechInput, StatMechResult
from minerva_engine.statmech.operation import statmech

LIMITS = Limits(timeout_seconds=120)


def run(**kwargs: object) -> StatMechResult:
    return statmech(StatMechInput(**kwargs), LIMITS)  # type: ignore[arg-type]


def value(result: StatMechResult, name: str, **symbols: float) -> float:
    q = next(q for q in result.quantities if q.name == name)
    names = {n: sp.Symbol(n) for n in (*symbols, "k_B", "beta", "T")}
    expr = sp.sympify(q.text, locals=names)
    return float(expr.subs({sp.Symbol(k): v for k, v in symbols.items()}))


def checks(result: StatMechResult, name: str) -> dict[str, bool | None]:
    q = next(q for q in result.quantities if q.name == name)
    return {c.name: c.passed for c in q.checks}


# -- level families -------------------------------------------------------------------------
def oscillator(**extra: object) -> StatMechResult:
    return run(
        system={"kind": "level_family", "energy": "hbar*omega*(n + 1/2)"},
        assumptions={"hbar": ["positive"], "omega": ["positive"]},
        **extra,
    )


def test_quantum_oscillator_infinite_sum_has_a_closed_form() -> None:
    r = oscillator(action="derive", quantities=["Z", "U", "Var_E"])
    assert r.status is Status.OK, r.message
    b, hw = 0.7, 1.3
    x = b * hw
    assert value(r, "Z", beta=b, hbar=hw, omega=1.0) == pytest.approx(1 / (2 * math.sinh(x / 2)))
    assert value(r, "U", beta=b, hbar=hw, omega=1.0) == pytest.approx(hw / 2 / math.tanh(x / 2))
    assert value(r, "Var_E", beta=b, hbar=hw, omega=1.0) == pytest.approx(
        (hw / 2) ** 2 / math.sinh(x / 2) ** 2
    )
    assert checks(r, "U")["enumeration"] is True
    assert any("ratio" in c.how for c in r.conditions)  # convergence was proved, not assumed


def test_series_conditions_are_stated() -> None:
    r = oscillator(action="partition")
    step = next(s for s in r.steps if s.name == "partition_function")
    assert any(c.established and "ratio test" in c.how for c in step.conditions)


def test_finite_family_is_a_finite_sum() -> None:
    r = run(
        system={"kind": "level_family", "energy": "n*eps", "lower": "0", "upper": "3"},
        assumptions={"eps": ["positive"]},
        action="derive",
        quantities=["Z"],
    )
    assert r.status is Status.OK
    b, e = 0.4, 0.9
    assert value(r, "Z", beta=b, eps=e) == pytest.approx(
        sum(math.exp(-b * e * n) for n in range(4))
    )


def test_divergent_series_is_refused() -> None:
    r = run(system={"kind": "level_family", "energy": "-n"}, action="partition")
    assert r.status is Status.CONDITIONS_NOT_SATISFIED and r.reason == "divergent"
    assert not r.quantities


@pytest.mark.parametrize("energy", ["0", "log(n + 1)"])
def test_inconclusive_convergence_is_not_assumed(energy: str) -> None:
    r = run(system={"kind": "level_family", "energy": energy}, action="partition")
    assert r.reason == "convergence_not_provable"
    assert r.status is Status.CONDITIONS_NOT_SATISFIED


def test_invalid_degeneracy_in_a_family_is_refused() -> None:
    r = run(
        system={"kind": "level_family", "energy": "n", "degeneracy": "n"}, action="partition"
    )  # g(0) = 0
    assert r.reason == "invalid_degeneracy"


def test_series_without_closed_form_gets_numeric_evaluation_only() -> None:
    base = {
        "system": {"kind": "level_family", "energy": "eps*n**2"},
        "assumptions": {"eps": ["positive"]},
    }
    r = run(**base, action="derive", quantities=["U"])
    assert r.status is Status.UNSUPPORTED and "numerical evaluation" in r.message
    r = run(
        **base,
        action="evaluate",
        quantities=["lnZ", "U", "F"],
        values={"beta": "1/2", "eps": "1", "k_B": "1"},
    )
    assert r.status is Status.OK, r.message
    z = mp.nsum(lambda n: mp.exp(-mp.mpf(1) / 2 * n * n), [0, mp.inf])
    assert float(next(q for q in r.quantities if q.name == "lnZ").text) == pytest.approx(
        float(mp.log(z))
    )
    assert checks(r, "U")["finite_difference"] is True  # U from moments vs -d lnZ/d beta
    assert next(q for q in r.quantities if q.name == "U").result_kind == "numeric"


# -- density of states ---------------------------------------------------------------------
def test_three_dimensional_density_of_states() -> None:
    r = run(
        system={"kind": "density_of_states", "density": "sqrt(En)", "lower": "0"},
        action="derive",
        quantities=["Z", "U", "Var_E"],
    )
    assert r.status is Status.OK, r.message
    b = 0.8
    assert value(r, "Z", beta=b) == pytest.approx(math.gamma(1.5) / b**1.5)
    assert value(r, "U", beta=b) == pytest.approx(1.5 / b)
    assert value(r, "Var_E", beta=b) == pytest.approx(1.5 / b**2)
    assert checks(r, "U")["symbolic_zero"] is True  # int E rho e^{-bE} / Z equals -d ln Z / d beta


def test_density_of_states_exponent_needs_a_declared_condition() -> None:
    base = {
        "system": {"kind": "density_of_states", "density": "En**a", "lower": "0"},
        "action": "derive",
    }
    r = run(**base, quantities=["U"])
    assert r.reason == "assumptions_insufficient"  # a could be complex
    r = run(**base, quantities=["U"], assumptions={"a": ["positive"]})
    assert r.status is Status.OK, r.message
    assert value(r, "U", beta=0.5, a=2.0) == pytest.approx(3 / 0.5)  # (a + 1)/beta


def test_exponentially_growing_density_is_not_normalisable_for_every_temperature() -> None:
    # rho = exp(c E): the integral exists only for beta > c, which the assumptions do not imply.
    r = run(
        system={"kind": "density_of_states", "density": "exp(c*En)", "lower": "0"},
        assumptions={"c": ["positive"]},
        action="derive",
        quantities=["U"],
    )
    assert r.status is Status.CONDITIONS_NOT_SATISFIED and r.reason == "convergence_not_provable"
    assert not r.quantities


def test_density_of_states_needs_a_lower_bound() -> None:
    r = run(
        system={"kind": "density_of_states", "density": "1", "lower": "-oo"}, action="partition"
    )
    assert r.reason == "unsupported_system"


def test_text_e_is_euler_number_is_noted() -> None:
    r = run(
        system={"kind": "density_of_states", "density": "sqrt(E)", "lower": "0"},
        action="partition",
    )
    assert any("Euler" in n for n in r.notes)


# -- classical systems ---------------------------------------------------------------------
def classical_oscillator(**extra: object) -> StatMechResult:
    return run(
        system={
            "kind": "classical",
            "hamiltonian": "p**2/(2*m) + m*omega**2*x**2/2",
            "variables": [{"name": "x"}, {"name": "p"}],
            "phase_space_cell": "h",
        },
        assumptions={"m": ["positive"], "omega": ["positive"], "h": ["positive"]},
        **extra,
    )


def test_classical_oscillator_equipartition() -> None:
    r = classical_oscillator(action="derive", quantities=["Z", "U", "Var_E"])
    assert r.status is Status.OK, r.message
    b, m, w, h = 0.9, 1.2, 0.7, 2.0
    kw = {"beta": b, "m": m, "omega": w, "h": h}
    assert value(r, "Z", **kw) == pytest.approx(2 * math.pi / (h * w * b))
    assert value(r, "U", **kw) == pytest.approx(1 / b)  # two quadratic terms: 2 x kT/2
    assert value(r, "Var_E", **kw) == pytest.approx(1 / b**2)
    assert any("factors exactly" in c.how for c in r.conditions)


def test_ideal_gas_pressure_and_entropy() -> None:
    gas = run(
        system={
            "kind": "classical",
            "hamiltonian": "(px**2 + py**2 + pz**2)/(2*m)",
            "variables": [
                {"name": "x", "domain_lower": "0", "domain_upper": "V**(1/3)"},
                {"name": "y", "domain_lower": "0", "domain_upper": "V**(1/3)"},
                {"name": "z", "domain_lower": "0", "domain_upper": "V**(1/3)"},
                {"name": "px"},
                {"name": "py"},
                {"name": "pz"},
            ],
            "phase_space_cell": "h**3",
        },
        copies={"count": "N", "identical": True},
        parameters=["V"],
        assumptions={
            "m": ["positive"],
            "h": ["positive"],
            "V": ["positive"],
            "N": ["integer", "positive"],
        },
        action="derive",
        quantities=["U", "S"],
        partials=[{"of": "F", "variable": "V", "held_fixed": ["T"], "scale": "-1", "label": "p"}],
    )
    assert gas.status is Status.OK, gas.message
    b, m, h, v, n, kB = 0.8, 1.3, 1.1, 2.5, 5, 1.7
    kw = {"beta": b, "m": m, "h": h, "V": v, "N": n, "k_B": kB}
    assert value(gas, "U", **kw) == pytest.approx(1.5 * n / b)  # (3/2) N k T
    assert value(gas, "p", **kw) * v * b == pytest.approx(n)  # p V = N k T, derived
    # exact entropy with ln N! (no Stirling): S/k = N ln(V (2 pi m / beta)^(3/2) / h^3) - ln N! + 3N/2
    expected = kB * (
        n * math.log(v * (2 * math.pi * m / b) ** 1.5 / h**3) - math.lgamma(n + 1) + 1.5 * n
    )
    assert value(gas, "S", **kw) == pytest.approx(expected)


def test_indistinguishability_changes_free_energy_by_ln_n_factorial() -> None:
    system = {
        "kind": "classical",
        "hamiltonian": "p**2/(2*m)",
        "variables": [{"name": "x", "domain_lower": "0", "domain_upper": "L"}, {"name": "p"}],
        "phase_space_cell": "h",
    }
    common = {
        "system": system,
        "parameters": [],
        "assumptions": {
            "m": ["positive"],
            "h": ["positive"],
            "L": ["positive"],
            "N": ["integer", "positive"],
        },
        "action": "derive",
        "quantities": ["lnZ"],
    }
    same = run(**common, copies={"count": "N", "identical": True})
    different = run(**common, copies={"count": "N", "identical": False})
    kw = {"beta": 1.0, "m": 1.0, "h": 1.0, "L": 3.0, "N": 4}
    diff = value(different, "lnZ", **kw) - value(same, "lnZ", **kw)
    assert diff == pytest.approx(math.lgamma(5))
    assert any("not exact Bose or Fermi" in n for n in same.notes)


def test_conventions_are_never_chosen_for_the_user() -> None:
    no_cell = run(
        system={"kind": "classical", "hamiltonian": "p**2/2", "variables": [{"name": "p"}]},
        assumptions={},
        action="partition",
    )
    assert no_cell.reason == "convention_unconfirmed" and no_cell.questions
    assert not no_cell.quantities
    levels = {"kind": "levels", "levels": [{"energy": "0"}, {"energy": "1"}]}
    unknown = run(system=levels, copies={"count": "3"}, action="partition")
    assert (
        unknown.reason == "convention_unconfirmed" and "indistinguishable" in unknown.questions[0]
    )


def test_non_separable_hamiltonian_is_refused() -> None:
    r = run(
        system={
            "kind": "classical",
            "hamiltonian": "(x - y)**2 + p**2",
            "variables": [{"name": "x"}, {"name": "y"}, {"name": "p"}],
            "phase_space_cell": "1",
        },
        action="partition",
    )
    assert r.reason == "not_separable" and r.status is Status.UNSUPPORTED


@pytest.mark.parametrize(
    "hamiltonian, variables",
    [
        (
            "p**2/2",
            [{"name": "x"}, {"name": "p"}],
        ),  # free particle, no box: integral over x diverges
        ("-p**2/2", [{"name": "p"}]),  # unbounded below: Boltzmann factor grows
    ],
)
def test_non_normalisable_boltzmann_factors_are_refused(
    hamiltonian: str, variables: list[dict[str, str]]
) -> None:
    r = run(
        system={
            "kind": "classical",
            "hamiltonian": hamiltonian,
            "variables": variables,
            "phase_space_cell": "1",
        },
        action="partition",
    )
    assert r.status is Status.CONDITIONS_NOT_SATISFIED and r.reason == "convergence_not_provable"
    assert not r.quantities


def test_assumptions_decide_whether_a_gaussian_integral_exists() -> None:
    system = {
        "kind": "classical",
        "hamiltonian": "k*x**2/2",
        "variables": [{"name": "x"}],
        "phase_space_cell": "1",
    }
    refused = run(system=system, action="partition")
    assert refused.status is not Status.OK
    ok = run(system=system, assumptions={"k": ["positive"]}, action="derive", quantities=["U"])
    assert ok.status is Status.OK
    assert value(ok, "U", beta=0.7, k=2.0) == pytest.approx(0.5 / 0.7)


# -- copies and evaluation -----------------------------------------------------------------
def test_distinguishable_copies_multiply_the_partition_function() -> None:
    r = run(
        system={"kind": "levels", "levels": [{"energy": "0"}, {"energy": "eps"}]},
        assumptions={"eps": ["positive"]},
        copies={"count": "3", "identical": False},
        action="derive",
        quantities=["Z", "U", "Var_E"],
    )
    assert r.status is Status.OK, r.message
    b, e = 0.6, 1.4
    z1 = 1 + math.exp(-b * e)
    assert value(r, "Z", beta=b, eps=e) == pytest.approx(z1**3)
    assert value(r, "U", beta=b, eps=e) == pytest.approx(3 * e * math.exp(-b * e) / z1)


def test_observables_with_copies_are_refused() -> None:
    r = run(
        system={
            "kind": "levels",
            "levels": [
                {"energy": "0", "observables": {"M": "1"}},
                {"energy": "1", "observables": {"M": "-1"}},
            ],
        },
        copies={"count": "2", "identical": False},
        action="expectation",
        observable="M",
    )
    assert r.reason == "unknown_observable"


def test_energy_cumulants_for_a_density_of_states() -> None:
    r = run(
        system={"kind": "density_of_states", "density": "1", "lower": "0"},
        action="expectation",
        observable="E",
        statistic="cumulant",
        order=3,
    )
    assert r.status is Status.OK, r.message
    assert value(r, "cumulant_E", beta=0.5) == pytest.approx(2 / 0.5**3)  # exponential distribution


def test_sweep_gives_a_table_checked_against_the_closed_form() -> None:
    r = oscillator(
        action="evaluate",
        quantities=["U"],
        values={"hbar": "1", "omega": "1", "k_B": "1"},
        sweep={"variable": "beta", "start": "1/10", "stop": "5", "points": 7, "scale": "log"},
    )
    assert r.status is Status.OK, r.message
    assert r.table is not None and len(r.table.rows) == 7 and r.table.columns == ["beta", "U"]
    for row in r.table.rows:
        b = float(row[0])
        assert float(row[1]) == pytest.approx(0.5 / math.tanh(b / 2), rel=1e-12)
    assert checks(r, "U")["enumeration"] is True


def test_evaluation_needs_the_users_numbers() -> None:
    base = {
        "system": {"kind": "level_family", "energy": "eps*n"},
        "assumptions": {"eps": ["positive"]},
        "action": "evaluate",
    }
    missing = run(**base, values={"beta": "1"})
    assert missing.reason == "parse_error" and "eps" in missing.message
    wrong = run(**base, values={"beta": "-1", "eps": "1", "k_B": "1"})
    assert wrong.reason == "assumptions_insufficient"


def test_diagnostics_hold_no_expressions_for_any_kind() -> None:
    r = classical_oscillator(action="derive", quantities=["U"])
    text = str(r.diagnostics.model_dump())
    assert "omega" not in text and "x**2" not in text


# -- properties ----------------------------------------------------------------------------
PROFILE = settings(
    max_examples=10, deadline=None, derandomize=True, suppress_health_check=[HealthCheck.too_slow]
)


@PROFILE
@given(
    st.lists(st.tuples(st.integers(-4, 4), st.integers(1, 3)), min_size=2, max_size=4),
    st.integers(1, 4),
    st.booleans(),
)
def test_independent_copies_scale_every_extensive_quantity(
    spectrum: list[tuple[int, int]], count: int, identical: bool
) -> None:
    levels = [{"energy": f"{e}/2", "degeneracy": str(g)} for e, g in spectrum]
    one = run(
        system={"kind": "levels", "levels": levels},
        action="derive",
        quantities=["lnZ", "U", "Var_E"],
    )
    many = run(
        system={"kind": "levels", "levels": levels},
        copies={"count": str(count), "identical": identical},
        action="derive",
        quantities=["lnZ", "U", "Var_E"],
    )
    assert one.status is Status.OK and many.status is Status.OK, many.message
    b = 0.7
    correction = math.lgamma(count + 1) if identical else 0.0
    assert value(many, "lnZ", beta=b) == pytest.approx(
        count * value(one, "lnZ", beta=b) - correction, abs=1e-10
    )
    assert value(many, "U", beta=b) == pytest.approx(count * value(one, "U", beta=b), abs=1e-10)
    assert value(many, "Var_E", beta=b) == pytest.approx(
        count * value(one, "Var_E", beta=b), abs=1e-10
    )
