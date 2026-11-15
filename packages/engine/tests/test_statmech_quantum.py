"""Finite-dimensional quantum systems: Tr exp(-beta H), thermal averages, noncommuting operators.

References are plain arithmetic or ``mpmath.expm`` computed in the test, never the code under test.
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
from pydantic import ValidationError

LIMITS = Limits(timeout_seconds=120)
SZ = [["1", "0"], ["0", "-1"]]
SX = [["0", "1"], ["1", "0"]]
# H = -h sigma_z - d sigma_x
H2 = [["-h", "-d"], ["-d", "h"]]
POS = {"h": ["positive"], "d": ["positive"]}


def run(**kwargs: object) -> StatMechResult:
    return statmech(StatMechInput(**kwargs), LIMITS)  # type: ignore[arg-type]


def value(result: StatMechResult, name: str, **symbols: float) -> float:
    q = next(q for q in result.quantities if q.name == name)
    names = {n: sp.Symbol(n) for n in (*symbols, "k_B", "beta")}
    expr = sp.sympify(q.text, locals=names)
    return float(expr.subs({sp.Symbol(k): v for k, v in symbols.items()}))


def passed(result: StatMechResult, name: str) -> dict[str, bool | None]:
    return {c.name: c.passed for q in result.quantities if q.name == name for c in q.checks}


def thermal_average(matrix: list[list[float]], op: list[list[float]], beta: float) -> float:
    """Tr(A exp(-beta H))/Tr exp(-beta H) with mpmath's matrix exponential."""
    mp.mp.dps = 30
    h, a = mp.matrix(matrix), mp.matrix(op)
    e = mp.expm(-beta * h)
    return float(
        mp.re(
            sum((a * e)[i, i] for i in range(len(matrix)))
            / sum(e[i, i] for i in range(len(matrix)))
        )
    )


def test_transverse_field_thermodynamics_matches_arithmetic() -> None:
    r = run(
        system={"kind": "quantum", "hamiltonian": H2},
        assumptions=POS,
        action="derive",
        quantities=["Z", "U", "S", "Var_E"],
    )
    assert r.status is Status.OK, r.message
    b, h, d, kB = 0.8, 0.6, 0.9, 1.5
    w = math.hypot(h, d)
    x = b * w
    kw = {"beta": b, "h": h, "d": d, "k_B": kB}
    assert value(r, "Z", **kw) == pytest.approx(2 * math.cosh(x))
    assert value(r, "U", **kw) == pytest.approx(-w * math.tanh(x))
    assert value(r, "S", **kw) == pytest.approx(
        kB * (math.log(2 * math.cosh(x)) - x * math.tanh(x))
    )
    assert value(r, "Var_E", **kw) == pytest.approx(w**2 / math.cosh(x) ** 2)
    # the matrix-exponential oracle and the symbolic eigenvalues are two different routes
    assert passed(r, "U")["enumeration"] is True and passed(r, "S")["finite_difference"] is True


def test_a_complex_hermitian_matrix_is_accepted() -> None:
    r = run(
        system={"kind": "quantum", "hamiltonian": [["0", "-I*h"], ["I*h", "0"]]},
        assumptions={"h": ["positive"]},
        action="derive",
        quantities=["Z"],
    )
    assert r.status is Status.OK, r.message
    assert value(r, "Z", beta=0.7, h=1.3) == pytest.approx(2 * math.cosh(0.7 * 1.3))


def test_thermal_average_of_an_operator_that_does_not_commute_with_h() -> None:
    r = run(
        system={"kind": "quantum", "hamiltonian": H2, "observables": {"sz": SZ}},
        assumptions=POS,
        action="expectation",
        observable="sz",
        statistic="mean",
    )
    assert r.status is Status.OK, r.message
    b, h, d = 0.8, 0.6, 0.9
    want = thermal_average([[-h, -d], [-d, h]], [[1, 0], [0, -1]], b)
    assert value(r, "mean_sz", beta=b, h=h, d=d) == pytest.approx(want)
    conditions = {c.latex: c for c in r.conditions}
    assert conditions["[sz, H] = 0"].established is False  # reported, never assumed
    assert conditions["sz = sz^{\\dagger}"].established is True


def test_variance_of_a_noncommuting_operator_is_computed_directly() -> None:
    r = run(
        system={"kind": "quantum", "hamiltonian": H2, "observables": {"sz": SZ}},
        assumptions=POS,
        action="expectation",
        observable="sz",
        statistic="variance",
    )
    assert r.status is Status.OK
    b, h, d = 0.8, 0.6, 0.9
    mean = thermal_average([[-h, -d], [-d, h]], [[1, 0], [0, -1]], b)
    assert value(r, "variance_sz", beta=b, h=h, d=d) == pytest.approx(1 - mean**2)  # sz^2 = 1
    assert "fluctuation_response" not in passed(r, "variance_sz")  # no response relation claimed


def test_first_derivative_response_holds_even_without_commutation() -> None:
    r = run(
        system={"kind": "quantum", "hamiltonian": H2, "observables": {"sz": SZ}},
        assumptions=POS,
        parameters=["h"],
        coupling={"parameter": "h", "observable": "sz"},
        action="expectation",
        observable="sz",
        statistic="mean",
    )
    assert r.status is Status.OK, r.message
    assert (
        passed(r, "mean_sz")["fluctuation_response"] is True
    )  # d ln Z / d h = beta <sz>, by cyclicity


def test_second_derivative_response_is_refused_when_operators_do_not_commute() -> None:
    r = run(
        system={"kind": "quantum", "hamiltonian": H2, "observables": {"sz": SZ}},
        assumptions=POS,
        parameters=["h"],
        coupling={"parameter": "h", "observable": "sz"},
        action="expectation",
        observable="sz",
        statistic="variance",
    )
    assert r.status is Status.CONDITIONS_NOT_SATISFIED and r.reason == "operators_do_not_commute"
    assert "Kubo" in r.message and not r.quantities


def test_second_derivative_response_holds_when_the_operator_commutes() -> None:
    r = run(
        system={
            "kind": "quantum",
            "hamiltonian": [["-h", "0"], ["0", "h"]],
            "observables": {"sz": SZ},
        },
        assumptions={"h": ["positive"]},
        parameters=["h"],
        coupling={"parameter": "h", "observable": "sz"},
        action="expectation",
        observable="sz",
        statistic="variance",
    )
    assert r.status is Status.OK, r.message
    assert passed(r, "variance_sz")["fluctuation_response"] is True
    assert value(r, "variance_sz", beta=0.8, h=0.6) == pytest.approx(1 / math.cosh(0.8 * 0.6) ** 2)


def test_covariance_of_noncommuting_operators_is_refused() -> None:
    r = run(
        system={"kind": "quantum", "hamiltonian": H2, "observables": {"sz": SZ, "sx": SX}},
        assumptions=POS,
        action="expectation",
        observable="sz",
        other_observable="sx",
        statistic="covariance",
    )
    assert r.reason == "operators_do_not_commute"


def test_a_matrix_that_is_not_hermitian_is_refused() -> None:
    r = run(
        system={"kind": "quantum", "hamiltonian": [["0", "a"], ["b", "0"]]},
        assumptions={"a": ["real"], "b": ["real"]},
        action="partition",
    )
    assert r.reason == "not_hermitian" and r.status is Status.CONDITIONS_NOT_SATISFIED
    undeclared = run(
        system={"kind": "quantum", "hamiltonian": [["0", "a"], ["a", "0"]]}, action="partition"
    )
    assert undeclared.reason == "not_hermitian"  # a could be complex: declare it real


def test_malformed_matrices_are_refused() -> None:
    assert (
        run(system={"kind": "quantum", "hamiltonian": [["0", "1"]]}, action="partition").reason
        == "parse_error"
    )
    big = [["0"] * 5 for _ in range(5)]
    with pytest.raises(ValidationError):  # the model caps the size: the API answers 422
        StatMechInput(system={"kind": "quantum", "hamiltonian": big})  # type: ignore[arg-type]
    wrong_size = run(
        system={"kind": "quantum", "hamiltonian": SX, "observables": {"A": [["1"]]}},
        action="expectation",
        observable="A",
    )
    assert wrong_size.reason == "parse_error"
    unknown = run(
        system={"kind": "quantum", "hamiltonian": SX, "observables": {"A": SZ}},
        action="expectation",
        observable="B",
    )
    assert unknown.reason == "unknown_observable"


def test_degenerate_levels_use_multiplicities() -> None:
    h3 = [["0", "0", "0"], ["0", "0", "0"], ["0", "0", "e1"]]
    r = run(
        system={
            "kind": "quantum",
            "hamiltonian": h3,
            "observables": {"A": [["0", "1", "0"], ["1", "0", "0"], ["0", "0", "0"]]},
        },
        assumptions={"e1": ["positive"]},
        action="expectation",
        observable="A",
    )
    assert r.status is Status.OK, r.message
    assert value(r, "mean_A", beta=0.9, e1=1.4) == pytest.approx(0.0, abs=1e-12)
    z = run(
        system={"kind": "quantum", "hamiltonian": h3},
        assumptions={"e1": ["positive"]},
        action="derive",
        quantities=["Z"],
    )
    assert value(z, "Z", beta=0.9, e1=1.4) == pytest.approx(2 + math.exp(-0.9 * 1.4))


def test_three_level_average_agrees_with_the_matrix_exponential() -> None:
    h3 = [["0", "J", "0"], ["J", "0", "J"], ["0", "J", "0"]]
    a = [["1", "0", "0"], ["0", "2", "0"], ["0", "0", "3"]]
    r = run(
        system={"kind": "quantum", "hamiltonian": h3, "observables": {"A": a}},
        assumptions={"J": ["positive"]},
        action="expectation",
        observable="A",
    )
    assert r.status is Status.OK, r.message
    want = thermal_average(
        [[0, 0.7, 0], [0.7, 0, 0.7], [0, 0.7, 0]], [[1, 0, 0], [0, 2, 0], [0, 0, 3]], 0.9
    )
    assert value(r, "mean_A", beta=0.9, J=0.7) == pytest.approx(want)
    assert any(
        "distinct" in c.how or "coincide" in c.how for c in r.conditions
    )  # Sylvester's condition is shown


def test_independent_spins_are_distinguishable_copies() -> None:
    r = run(
        system={"kind": "quantum", "hamiltonian": [["-h", "0"], ["0", "h"]]},
        assumptions={"h": ["positive"]},
        copies={"count": "3", "identical": False},
        action="derive",
        quantities=["Z", "U"],
    )
    assert r.status is Status.OK, r.message
    assert value(r, "Z", beta=0.5, h=1.1) == pytest.approx((2 * math.cosh(0.55)) ** 3)


PROFILE = settings(
    max_examples=10, deadline=None, derandomize=True, suppress_health_check=[HealthCheck.too_slow]
)


@PROFILE
@given(st.integers(-3, 3), st.integers(-3, 3), st.integers(-3, 3))
def test_random_symmetric_matrices_agree_with_the_matrix_exponential(
    a: int, b: int, c: int
) -> None:
    r = run(
        system={"kind": "quantum", "hamiltonian": [[str(a), str(b)], [str(b), str(c)]]},
        action="derive",
        quantities=["Z", "U", "Var_E"],
    )
    assert r.status is Status.OK, r.message
    beta = 0.8
    mp.mp.dps = 30
    h = mp.matrix([[a, b], [b, c]])
    e = mp.expm(-beta * h)
    z = e[0, 0] + e[1, 1]
    u = ((h * e)[0, 0] + (h * e)[1, 1]) / z
    var = ((h * h * e)[0, 0] + (h * h * e)[1, 1]) / z - u * u
    assert value(r, "Z", beta=beta) == pytest.approx(float(z))
    assert value(r, "U", beta=beta) == pytest.approx(float(u), abs=1e-10)
    assert value(r, "Var_E", beta=beta) == pytest.approx(float(var), abs=1e-10)
    assert float(var) >= -1e-12
