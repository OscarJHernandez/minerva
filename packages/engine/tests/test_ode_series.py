"""Local Taylor-series solutions: coefficients against independent Taylor expansions, wrong series
rejected, and the points where nothing may be shown."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mpmath as mp
import pytest
import sympy as sp
from minerva_engine import Engine, OdeSolveInput, Status
from minerva_engine.ode import OdeSolution
from minerva_engine.ode.approximation import _check_series, solve_series
from minerva_engine.ode.models import OdeNumericInput
from minerva_engine.ode.numerical import solve_numerical
from minerva_engine.ode.reading import read_ode
from minerva_engine.testing import budget_limits

L = budget_limits()


def series(*equations: str, order: int = 6, fmt: str = "latex", **kw: Any) -> OdeSolution:
    request = OdeSolveInput(
        equations=list(equations),
        format=fmt,
        method="series",
        order=order,
        **kw,  # type: ignore[arg-type]
    )
    return solve_series(request, L)


def coefficients(
    solution: OdeSolution, variable: str, point: float = 0.0, index: int = 0
) -> list[float]:
    """Taylor coefficients of the shown polynomial about `point`, read back from its text."""
    x = sp.Symbol(variable, real=True)
    poly = sp.sympify(solution.solutions[0].rhs_text[index], locals={variable: x})
    shifted = sp.expand(poly.subs(x, x + point))
    return [float(shifted.coeff(x, k)) for k in range(sp.degree(shifted, x) + 1)]


def reference(function: Callable[[Any], Any], at: float, order: int) -> list[float]:
    mp.mp.dps = 40
    return [float(c) for c in mp.taylor(function, at, order)]


KNOWN = [
    # (equations, conditions, variable, point, exact solution)
    (["y' = y"], ["y(0) = 1"], "x", 0.0, lambda x: mp.exp(x)),
    (["y'' + y = 0"], ["y(0) = 0", "y'(0) = 1"], "x", 0.0, lambda x: mp.sin(x)),
    (["y' = y^2"], ["y(0) = 1"], "x", 0.0, lambda x: 1 / (1 - x)),
    ([r"y' = y \cdot (1 - y)"], ["y(0) = 1/2"], "x", 0.0, lambda x: 1 / (1 + mp.exp(-x))),
    (["y' = y"], ["y(1) = 2"], "x", 1.0, lambda x: 2 * mp.exp(x - 1)),
    (["y' = -2 x y"], ["y(0) = 1"], "x", 0.0, lambda x: mp.exp(-(x**2))),
    (["y'' = -y"], ["y(0) = 1", "y'(0) = 0"], "x", 0.0, lambda x: mp.cos(x)),
    (["y'' - y = 0"], ["y(2) = 1", "y'(2) = 1"], "x", 2.0, lambda x: mp.exp(x - 2)),
    ([r"y' = \frac{1}{1 + x^2}"], ["y(0) = 0"], "x", 0.0, lambda x: mp.atan(x)),
]


@pytest.mark.parametrize(("equations", "conditions", "variable", "point", "exact"), KNOWN)
def test_coefficients_match_the_taylor_expansion_of_the_known_solution(
    equations: list[str],
    conditions: list[str],
    variable: str,
    point: float,
    exact: Callable[[Any], Any],
) -> None:
    order = 7
    r = series(*equations, order=order, independent=variable, conditions=conditions)
    assert r.status is Status.OK, r.message
    (item,) = r.solutions
    assert item.kind == "series" and item.family == "particular"
    assert item.expansion is not None and item.expansion.kind == "taylor"
    assert item.expansion.order == order and "local" in item.expansion.validity.lower()
    got = coefficients(r, variable, point)
    want = reference(exact, point, order)
    assert len(got) <= order + 1
    for k, value in enumerate(want):
        have = got[k] if k < len(got) else 0.0
        assert have == pytest.approx(value, rel=1e-9, abs=1e-12), (k, have, value)
    v = item.verification
    assert v.passed and v.points_checked >= 4 and v.points_agreed == v.points_checked
    assert all(c.passed for c in item.expansion.checks)
    assert all(c.satisfied for c in item.conditions)
    assert "O\\left(" in item.equations_latex[0] and f"^{{{order + 1}}}" in item.equations_latex[0]


def test_symbolic_parameters_stay_symbolic() -> None:
    r = series("y' = a y", order=5, independent="x", conditions=["y(0) = y_0"])
    assert r.status is Status.OK, r.message
    x, a, y0 = sp.symbols("x a y_0")
    poly = sp.sympify(r.solutions[0].rhs_text[0], locals={"x": x, "a": a, "y_0": y0})
    assert sp.simplify(poly - y0 * sum((a * x) ** k / sp.factorial(k) for k in range(6))) == 0


def test_a_system_is_expanded_together() -> None:
    r = series(r"\dot{x} = -y", r"\dot{y} = x", order=7, conditions=["x(0) = 1", "y(0) = 0"])
    assert r.status is Status.OK, r.message
    assert coefficients(r, "t", 0.0, 0)[:7:2] == pytest.approx([1, -0.5, 1 / 24, -1 / 720])
    assert coefficients(r, "t", 0.0, 1)[1:8:2] == pytest.approx([1, -1 / 6, 1 / 120, -1 / 5040])


def test_a_higher_order_equation_and_its_first_order_system_give_the_same_series() -> None:
    higher = series("y'' + 3 y' + 2 y = 0", independent="t", conditions=["y(0) = 1", "y'(0) = 0"])
    system = series(
        "y' = v", "v' = -3 v - 2 y", independent="t", conditions=["y(0) = 1", "v(0) = 0"]
    )
    assert coefficients(higher, "t") == pytest.approx(coefficients(system, "t"), abs=1e-12)


def test_the_series_agrees_with_a_numerical_solution_to_the_claimed_order() -> None:
    order = 5
    r = series(
        "y'' + 0.2 y' + y + y^3 = 0",
        order=order,
        independent="t",
        conditions=["y(0) = 1", "y'(0) = 0"],
    )
    assert r.status is Status.OK
    coeffs = coefficients(r, "t")
    errors = []
    for h in (0.2, 0.1):
        numeric = solve_numerical(
            OdeNumericInput(
                equations=["y'' + 0.2 y' + y + y^3 = 0"],
                independent="t",
                conditions=["y(0) = 1", "y'(0) = 0"],
                t_end=h,
                rtol=1e-12,
                atol=1e-14,
            ),
            L,
        )
        assert numeric.status is Status.OK
        exact = numeric.series[0].values[-1]
        errors.append(abs(sum(c * h**k for k, c in enumerate(coeffs)) - exact))
    assert errors[0] / errors[1] > 2 ** (order + 1) * 0.6  # the error falls like h^(order + 1)


# --- refusals --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "status", "fragment"),
    [
        (
            {"equations": ["y' = 1/x"], "conditions": ["y(0) = 1"]},
            Status.UNSUPPORTED,
            "singular point",
        ),
        (
            {"equations": ["y'' = -y"], "conditions": ["y(0) = 1"]},
            Status.PARSE_ERROR,
            "initial value",
        ),
        ({"equations": ["y'' = -y"], "conditions": []}, Status.PARSE_ERROR, "does not choose"),
        (
            {"equations": ["y'' = -y"], "conditions": ["y(0) = 0", "y(1) = 1"]},
            Status.PARSE_ERROR,
            "all at one point",
        ),
        (
            {"equations": ["(y')^2 = y"], "conditions": ["y(0) = 1"]},
            Status.UNSUPPORTED,
            "highest derivative",
        ),
        ({"equations": ["y' = 1/y"], "conditions": ["y(0) = 0"]}, Status.UNSUPPORTED, "singular"),
    ],
)
def test_nothing_is_shown_without_every_initial_value_or_at_a_singular_point(
    kwargs: dict[str, Any], status: Status, fragment: str
) -> None:
    r = solve_series(OdeSolveInput(method="series", order=6, independent="x", **kwargs), L)
    assert r.status is status and fragment in r.message
    assert r.solutions == []


def test_a_series_that_is_not_a_solution_on_both_sides_is_not_shown() -> None:
    # y' = sqrt(y), y(0) = 0 is not unique; the polynomial x^2/4 is a solution only for x >= 0.
    r = series("y' = sqrt(y)", fmt="text", independent="x", conditions=["y(0) = 0"], order=4)
    assert r.status is Status.NOT_VERIFIED and r.solutions == []


@pytest.mark.parametrize("order", [0, 13, -1])
def test_the_truncation_order_is_bounded(order: int) -> None:
    with pytest.raises(ValueError, match="validation error"):
        OdeSolveInput(equations=["y' = y"], independent="x", method="series", order=order)


# --- the checker rejects wrong series --------------------------------------------------------


def _check(coeffs: list[sp.Expr], order: int) -> bool:
    problem = read_ode(
        OdeSolveInput(equations=["y' = y"], independent="x", conditions=["y(0) = 1"]), L
    )
    x = problem.variable
    poly = sum((c * x**k for k, c in enumerate(coeffs)), sp.Integer(0))
    verification, _checks, conditions = _check_series(
        problem, [poly], sp.Integer(0), order, [("y", 0, sp.Integer(0), sp.Integer(1))]
    )
    return verification.passed and all(c.satisfied for c in conditions)


def test_a_wrong_coefficient_of_any_retained_order_is_caught() -> None:
    order = 6
    right = [sp.Rational(1, sp.factorial(k)) for k in range(order + 1)]
    assert _check(right, order)
    for k in range(order + 1):
        wrong = list(right)
        wrong[k] = wrong[k] * (1 + sp.Rational(1, 1000))
        assert not _check(wrong, order), k
    assert not _check(right[:5], order)  # truncated lower than claimed
    assert not _check([sp.Integer(1)] + [sp.Integer(0)] * order, order)


def test_wrong_initial_values_are_caught_even_if_the_residual_is_small() -> None:
    problem = read_ode(OdeSolveInput(equations=["y' = y"], independent="x"), L)
    x = problem.variable
    poly = 2 * sum(
        (x**k / sp.factorial(k) for k in range(7)), sp.Integer(0)
    )  # solves y' = y, y(0) = 2
    verification, _, conditions = _check_series(
        problem, [poly], sp.Integer(0), 6, [("y", 0, sp.Integer(0), sp.Integer(1))]
    )
    assert verification.passed  # it does solve the equation ...
    assert not all(c.satisfied for c in conditions)  # ... but not with y(0) = 1


# --- through the sandbox ---------------------------------------------------------------------


def test_the_engine_dispatches_the_series_method(engine: Engine) -> None:
    r = engine.ode_solve_symbolic(
        OdeSolveInput(
            equations=["y' = y"], independent="x", conditions=["y(0) = 1"], method="series", order=4
        )
    )
    assert r.status is Status.OK and r.method == "series"
    assert r.solutions[0].kind == "series" and r.diagnostics.method == "series"
    closed = engine.ode_solve_symbolic(
        OdeSolveInput(equations=["y' = y"], independent="x", conditions=["y(0) = 1"])
    )
    assert closed.method == "closed_form" and closed.solutions[0].kind == "explicit"
