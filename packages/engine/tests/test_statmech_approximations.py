"""Approximations that state their assumptions and are measured: expansions, Stirling, transfer matrices.

References are plain arithmetic computed here, including a brute-force enumeration of Ising rings, never
the code under test.
"""

import itertools
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
TWO = {"kind": "levels", "levels": [{"energy": "0"}, {"energy": "eps"}]}
POS = {"eps": ["positive"]}
ISING = {"kind": "chain", "states": ["1", "-1"], "bond": "-J*s*t", "site": "-h*s", "length": "N"}
ISING_ASSUME = {"J": ["real"], "h": ["real"], "N": ["integer", "positive"]}


def run(**kwargs: object) -> StatMechResult:
    return statmech(StatMechInput(**kwargs), LIMITS)  # type: ignore[arg-type]


def value(result: StatMechResult, name: str, **symbols: float) -> float:
    q = next(q for q in result.quantities if q.name == name)
    names = {n: sp.Symbol(n) for n in (*symbols, "k_B", "beta", "N", "J", "h")}
    expr = sp.sympify(q.text, locals=names)
    return float(expr.subs({sp.Symbol(k): v for k, v in symbols.items()}))


def checks(result: StatMechResult, name: str) -> dict[str, bool | None]:
    return {c.name: c.passed for q in result.quantities if q.name == name for c in q.checks}


def ising_z(n: int, beta: float, j: float, h: float) -> float:
    """Brute-force partition function of a ring of n spins, H = -J sum s s' - h sum s."""
    total = 0.0
    for spins in itertools.product((1, -1), repeat=n):
        energy = -h * sum(spins) - j * sum(spins[i] * spins[(i + 1) % n] for i in range(n))
        total += math.exp(-beta * energy)
    return total


# -- the transfer matrix ---------------------------------------------------------------------
def test_zero_field_ising_matches_the_textbook_partition_function() -> None:
    r = run(
        system={**ISING, "site": "0"},
        assumptions={"J": ["positive"], "N": ["integer", "positive"]},
        action="derive",
        quantities=["Z"],
    )
    assert r.status is Status.OK, r.message
    for n in (3, 6):
        b, j = 0.7, 1.3
        want = (2 * math.cosh(b * j)) ** n + (2 * math.sinh(b * j)) ** n
        assert value(r, "Z", beta=b, J=j, N=n) == pytest.approx(want)


def test_ising_in_a_field_agrees_with_a_brute_force_enumeration() -> None:
    r = run(system=ISING, assumptions=ISING_ASSUME, action="derive", quantities=["Z", "U", "Var_E"])
    assert r.status is Status.OK, r.message
    b, j, h, n = 0.6, 0.9, 0.4, 7
    assert value(r, "Z", beta=b, J=j, h=h, N=n) == pytest.approx(ising_z(n, b, j, h))
    assert checks(r, "U")["enumeration"] is True and checks(r, "U")["finite_difference"] is True
    # the energy from the enumeration, with the test's own arithmetic
    z = ising_z(n, b, j, h)
    u = 0.0
    for spins in itertools.product((1, -1), repeat=n):
        energy = -h * sum(spins) - j * sum(spins[i] * spins[(i + 1) % n] for i in range(n))
        u += energy * math.exp(-b * energy) / z
    assert value(r, "U", beta=b, J=j, h=h, N=n) == pytest.approx(u)


def test_a_literal_length_is_enumerated_too() -> None:
    r = run(
        system={**ISING, "length": "5"},
        assumptions={"J": ["real"], "h": ["real"]},
        action="derive",
        quantities=["Z"],
    )
    assert r.status is Status.OK
    assert value(r, "Z", beta=0.5, J=1.1, h=-0.3) == pytest.approx(ising_z(5, 0.5, 1.1, -0.3))


def test_transfer_matrix_refusals() -> None:
    long_chain = run(
        system={**ISING, "length": "40"},
        assumptions={"J": ["real"], "h": ["real"]},
        action="partition",
    )
    assert long_chain.reason == "resource_limit"
    asymmetric = run(
        system={"kind": "chain", "states": ["1", "-1"], "bond": "-J*s", "length": "4"},
        assumptions={"J": ["real"]},
        action="partition",
    )
    assert asymmetric.reason == "unsupported_system" and "symmetric" in asymmetric.message
    repeated = run(
        system={"kind": "chain", "states": ["1", "1"], "bond": "-J*s*t", "length": "4"},
        assumptions={"J": ["real"]},
        action="partition",
    )
    assert repeated.reason == "assumptions_insufficient"
    copies = run(
        system=ISING,
        copies={"count": "2", "identical": False},
        assumptions=ISING_ASSUME,
        action="partition",
    )
    assert copies.reason == "unsupported_system"


def test_thermodynamic_limit_comes_from_the_dominant_eigenvalue_with_a_measured_error() -> None:
    r = run(
        system=ISING, assumptions=ISING_ASSUME, action="approximate", method="thermodynamic_limit"
    )
    assert r.status is Status.OK, r.message
    approx = r.approximation
    assert approx is not None and approx.method == "thermodynamic_limit"
    assert any("Perron" in c.how for s in r.steps for c in s.conditions)
    errors = [abs(float(row.error)) for row in approx.error_table]
    assert (
        errors == sorted(errors, reverse=True) and errors[-1] < errors[0] / 100
    )  # exponentially small in N
    assert all(row.note.startswith("leading finite-size term") for row in approx.error_table)
    assert next(q for q in r.quantities if q.name == "f_inf").result_kind == "exact"
    assert checks(r, "f_inf")["residual_order"] is True
    # the free energy per site is a plain number from the dominant eigenvalue (textbook closed form)
    f = value(r, "f_inf", beta=0.6, J=0.9, h=0.4)
    lam = math.exp(0.6 * 0.9) * math.cosh(0.6 * 0.4) + math.sqrt(
        math.exp(2 * 0.6 * 0.9) * math.sinh(0.6 * 0.4) ** 2 + math.exp(-2 * 0.6 * 0.9)
    )
    assert f == pytest.approx(-math.log(lam) / 0.6)


def test_thermodynamic_limit_needs_a_symbolic_length_and_a_chain() -> None:
    literal = run(
        system={**ISING, "length": "8"},
        assumptions={"J": ["real"], "h": ["real"]},
        action="approximate",
        method="thermodynamic_limit",
    )
    assert literal.reason == "assumptions_insufficient"
    levels = run(system=TWO, assumptions=POS, action="approximate", method="thermodynamic_limit")
    assert levels.reason == "nothing_to_approximate"


# -- high temperature ------------------------------------------------------------------------
def test_high_temperature_series_of_a_two_level_system_is_verified_and_measured() -> None:
    r = run(system=TWO, assumptions=POS, action="approximate", method="high_temperature", order=4)
    assert r.status is Status.OK, r.message
    q = next(q for q in r.quantities if q.name == "lnZ_series")
    assert q.result_kind == "asymptotic"  # never an identity
    x = 0.4 * 0.9  # beta * eps
    series = math.log(2) - x / 2 + x**2 / 8 - x**4 / 192  # ln(1 + e^-x), known expansion
    assert value(r, "lnZ_series", beta=0.4, eps=0.9) == pytest.approx(series)
    assert checks(r, "lnZ_series") == {"series_check": True, "residual_order": True}
    approx = r.approximation
    assert approx is not None and "O(beta^5)" in approx.order
    assert approx.error_table[0].label.startswith("beta = ")
    errors = [abs(float(row.error)) for row in approx.error_table]
    assert (
        errors[0] / errors[1] > 2**5 / 3
    )  # the error falls at the claimed order when beta is halved
    assert value(r, "U_series", beta=0.4, eps=0.9) == pytest.approx(
        0.9 / 2 - 0.9 * x / 4 + 0.9 * x**3 / 48
    )


def test_high_temperature_expansion_is_refused_where_ln_z_is_not_analytic() -> None:
    oscillator = run(
        system={"kind": "level_family", "energy": "hbar*omega*(n + 1/2)"},
        assumptions={"hbar": ["positive"], "omega": ["positive"]},
        action="approximate",
        method="high_temperature",
        order=2,
    )
    assert (
        oscillator.status is Status.UNSUPPORTED and oscillator.reason == "singular_expansion_point"
    )
    assert "Laurent" in oscillator.message and not oscillator.quantities
    dos = run(
        system={"kind": "density_of_states", "density": "sqrt(En)", "lower": "0"},
        action="approximate",
        method="high_temperature",
    )
    assert dos.reason == "singular_expansion_point"


# -- low temperature -------------------------------------------------------------------------
def test_low_temperature_limits_and_corrections_are_measured() -> None:
    levels = {
        "kind": "levels",
        "levels": [
            {"energy": "0", "degeneracy": "2"},
            {"energy": "eps"},
            {"energy": "3*eps", "degeneracy": "2"},
        ],
    }
    r = run(system=levels, assumptions=POS, action="approximate", method="low_temperature")
    assert r.status is Status.OK, r.message
    b, e = 4.0, 1.0
    x1, x3 = math.exp(-b * e), math.exp(-3 * b * e)
    want = e * 0.5 * x1 + 3 * e * 1.0 * x3  # sum (g_j/g_0) eps_j e^(-beta eps_j)
    assert value(r, "U_low", beta=b, eps=e) == pytest.approx(want)
    assert checks(r, "U_low")["residual_order"] is True
    condition = next(c for s in r.steps for c in s.conditions if "ln g" in c.latex)
    assert "measured" in condition.how and "asymptotic" in condition.how
    assert r.approximation is not None and "exp(-beta * lowest gap)" in r.approximation.parameter


def test_low_temperature_needs_a_decided_order_of_levels() -> None:
    undecided = run(
        system={"kind": "levels", "levels": [{"energy": "a"}, {"energy": "b"}]},
        assumptions={"a": ["real"], "b": ["real"]},
        action="approximate",
        method="low_temperature",
    )
    assert undecided.reason == "ordering_not_decided"
    single = run(
        system={"kind": "levels", "levels": [{"energy": "0"}]},
        action="approximate",
        method="low_temperature",
    )
    assert single.reason == "nothing_to_approximate"
    classical = run(
        system={
            "kind": "classical",
            "hamiltonian": "p**2/2",
            "variables": [{"name": "p"}],
            "phase_space_cell": "1",
        },
        action="approximate",
        method="low_temperature",
    )
    assert classical.reason == "nothing_to_approximate"


# -- Stirling --------------------------------------------------------------------------------
GAS = {
    "kind": "classical",
    "hamiltonian": "(px**2 + py**2 + pz**2)/(2*m)",
    "variables": [
        *({"name": c, "domain_lower": "0", "domain_upper": "V**(1/3)"} for c in "xyz"),
        *({"name": p} for p in ("px", "py", "pz")),
    ],
    "phase_space_cell": "h**3",
}
GAS_ASSUME = {"m": ["positive"], "h": ["positive"], "V": ["positive"], "N": ["integer", "positive"]}


def test_stirling_gives_the_sackur_tetrode_entropy_and_measures_its_error() -> None:
    r = run(
        system=GAS,
        copies={"count": "N", "identical": True},
        parameters=["V"],
        assumptions=GAS_ASSUME,
        action="approximate",
        method="stirling",
    )
    assert r.status is Status.OK, r.message
    b, m, h, v, n, kb = 0.8, 1.3, 1.1, 2.5, 50, 1.7
    kw = {"beta": b, "m": m, "h": h, "V": v, "N": n, "k_B": kb}
    sackur_tetrode = kb * n * (math.log(v / n * (2 * math.pi * m / (b * h**2)) ** 1.5) + 2.5)
    assert value(r, "S_leading", **kw) == pytest.approx(sackur_tetrode)
    assert value(r, "S_stirling", **kw) == pytest.approx(
        sackur_tetrode - kb * 0.5 * math.log(2 * math.pi * n)
    )
    approx = r.approximation
    assert approx is not None and approx.parameter == "1/N"
    scaled = [float(row.note.split("N * error = ")[1].split(" ")[0]) for row in approx.error_table]
    assert scaled[-1] == pytest.approx(1 / 12, rel=1e-3)  # the error is 1/(12 N)
    assert next(q for q in r.quantities if q.name == "S_leading").result_kind == "asymptotic"
    assert checks(r, "lnZ_stirling")["residual_order"] is True


def test_stirling_differs_from_the_exact_entropy_by_the_stirling_error() -> None:
    exact = run(
        system=GAS,
        copies={"count": "N", "identical": True},
        parameters=["V"],
        assumptions=GAS_ASSUME,
        action="derive",
        quantities=["S"],
    )
    approx = run(
        system=GAS,
        copies={"count": "N", "identical": True},
        parameters=["V"],
        assumptions=GAS_ASSUME,
        action="approximate",
        method="stirling",
    )
    b, m, h, v, n, kb = 0.8, 1.3, 1.1, 2.5, 5, 1.7
    kw = {"beta": b, "m": m, "h": h, "V": v, "N": n, "k_B": kb}
    gap = value(approx, "S_stirling", **kw) - value(exact, "S", **kw)
    stirling_error = math.lgamma(n + 1) - (n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n))
    assert gap == pytest.approx(kb * stirling_error)  # S_stirling = S + k_B (ln N! - Stirling(N))


def test_stirling_applies_only_to_a_ln_n_factorial() -> None:
    without = run(system=TWO, assumptions=POS, action="approximate", method="stirling")
    assert without.reason == "nothing_to_approximate"
    distinguishable = run(
        system=TWO,
        copies={"count": "N", "identical": False},
        assumptions={**POS, "N": ["integer", "positive"]},
        action="approximate",
        method="stirling",
    )
    assert distinguishable.reason == "nothing_to_approximate"


def test_an_approximation_needs_its_method_named() -> None:
    r = run(system=TWO, assumptions=POS, action="approximate")
    assert r.reason == "parse_error" and "high_temperature" in r.message


# -- properties ------------------------------------------------------------------------------
PROFILE = settings(
    max_examples=8, deadline=None, derandomize=True, suppress_health_check=[HealthCheck.too_slow]
)
RATIONAL = st.integers(-8, 8).map(lambda k: k / 4)


@PROFILE
@given(RATIONAL, RATIONAL, st.integers(2, 6))
def test_the_transfer_matrix_matches_enumeration_for_random_couplings(
    j: float, h: float, n: int
) -> None:
    r = run(
        system={**ISING, "length": str(n)},
        assumptions={"J": ["real"], "h": ["real"]},
        action="derive",
        quantities=["Z"],
    )
    assert r.status is Status.OK, r.message
    mp.mp.dps = 30
    b = 0.55
    assert value(r, "Z", beta=b, J=j, h=h) == pytest.approx(ising_z(n, b, j, h), rel=1e-9)


def test_a_chain_expression_that_does_not_parse_says_the_syntax() -> None:
    """Live runs sent `-J s_i s_{i+1}` and read only 'could not parse the expression'."""
    result = run(
        system={
            "kind": "chain",
            "states": ["1", "-1"],
            "bond": "-J s_i s_{i+1}",
            "site": "-h*s",
            "length": "N",
        },
        assumptions={"J": ["real"], "h": ["real"], "N": ["integer", "positive"]},
        action="analyze",
    )
    assert result.reason == "parse_error"
    assert "bond = -J*s*t" in result.message and "no subscripts" in result.message


def test_an_energy_without_declared_assumptions_names_the_symbols_to_declare() -> None:
    """The oscillator run sent hbar and omega with no assumptions and the model never found out which to add."""
    result = run(
        system={"kind": "level_family", "energy": "hbar*omega*(n + 1/2)", "index": "n"},
        action="analyze",
    )
    assert result.reason == "assumptions_insufficient"
    assert "hbar, omega" in result.message and '"omega": ["positive"]' in result.message
    declared = run(
        system={"kind": "level_family", "energy": "hbar*omega*(n + 1/2)", "index": "n"},
        assumptions={"hbar": ["positive"], "omega": ["positive"]},
        action="analyze",
    )
    assert declared.reason != "assumptions_insufficient"
