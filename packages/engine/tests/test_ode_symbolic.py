"""Closed-form ODE solutions: correct answers against references that do not come from Minerva, wrong
candidates that must be rejected, and the cases where nothing may be claimed."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mpmath as mp
import pytest
import sympy as sp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from minerva_engine import Engine, Limits, OdeInput, Status
from minerva_engine.ode import OdeSolution
from minerva_engine.ode.reading import read_ode
from minerva_engine.ode.symbolic import solve_closed_form
from minerva_engine.ode.verification import (
    implicit_check,
    independent_constants,
    residual_check,
)

L = Limits(timeout_seconds=30)


def solve(*equations: str, fmt: str = "latex", **kw: Any) -> OdeSolution:
    return solve_closed_form(OdeInput(equations=list(equations), format=fmt, **kw), L)


def evaluate(text: str, variable: str, at: float, **values: float) -> complex:
    expr = sp.sympify(text, locals={variable: sp.Symbol(variable, real=True)})
    substitution: dict[sp.Symbol, Any] = {sp.Symbol(variable, real=True): at}
    for name, value in values.items():
        substitution[sp.Symbol(name)] = value
    return complex(sp.N(expr.xreplace(substitution), 30))


# --- particular solutions against independently known formulas -------------------------------

Reference = Callable[[float], complex]

IVPS: list[tuple[str, list[str], list[str], str, Reference]] = [
    # (equations, conditions, unknown names, variable, reference for the first unknown)
    ("y' = y", [], ["y(0) = 1"], "x", lambda x: mp.exp(x)),
    ("y' = x y", [], ["y(0) = 2"], "x", lambda x: 2 * mp.exp(x**2 / 2)),
    (
        "y' + 2 y = \\sin x",
        [],
        ["y(0) = 1"],
        "x",
        lambda x: 1.2 * mp.exp(-2 * x) + (2 * mp.sin(x) - mp.cos(x)) / 5,
    ),
    ("y'' + y = 0", [], ["y(0) = 0", "y'(0) = 1"], "x", lambda x: mp.sin(x)),
    (
        "y'' + 3y' + 2y = 0",
        [],
        ["y(0) = 1", "y'(0) = 0"],
        "x",
        lambda x: 2 * mp.exp(-x) - mp.exp(-2 * x),
    ),
    ("y'' + 2y' + y = 0", [], ["y(0) = 0", "y'(0) = 1"], "x", lambda x: x * mp.exp(-x)),
    (
        "y''' = y",
        [],
        ["y(0) = 1", "y'(0) = 0", "y''(0) = 0"],
        "x",
        lambda x: (mp.exp(x) + 2 * mp.exp(-x / 2) * mp.cos(mp.sqrt(3) * x / 2)) / 3,
    ),
    ("y' + y = y^2", [], ["y(0) = 1/2"], "x", lambda x: 1 / (1 + mp.exp(x))),
    (r"y' = y \cdot (1 - y)", [], ["y(0) = 1/2"], "x", lambda x: 1 / (1 + mp.exp(-x))),
    ("y' = -2 x y^2", [], ["y(0) = 1"], "x", lambda x: 1 / (1 + x**2)),
    ("2 x y + (x^2 + 1) y' = 0", [], ["y(0) = 3"], "x", lambda x: 3 / (x**2 + 1)),
    ("x^2 y'' + x y' - y = 0", [], ["y(1) = 1", "y'(1) = 0"], "x", lambda x: (x + 1 / x) / 2),
    (
        "y'' + y = \\sin x",
        [],
        ["y(0) = 0", "y'(0) = 0"],
        "x",
        lambda x: (mp.sin(x) - x * mp.cos(x)) / 2,
    ),
]


@pytest.mark.parametrize(("equation", "_unused", "conditions", "variable", "reference"), IVPS)
def test_initial_value_problems_match_known_solutions(
    equation: str, _unused: list[str], conditions: list[str], variable: str, reference: Reference
) -> None:
    r = solve(equation, independent=variable, conditions=conditions)
    assert r.status is Status.OK, r.message
    assert r.method == "closed_form" and r.solutions
    for item in r.solutions:
        v = item.verification
        assert v.passed and v.points_checked >= 4 and v.points_agreed == v.points_checked
        assert v.max_residual is not None and v.max_residual < 1e-9
        assert item.family == "particular" and item.constants == []
        assert all(c.satisfied for c in item.conditions) and len(item.conditions) == len(conditions)
        assert item.evaluated and item.kind == "explicit"
    # Exactly one of the proposed branches is the solution of the IVP: compare with the reference.
    for x in (0.3, 0.9, 1.7):
        want = complex(reference(x))
        got = [evaluate(i.rhs_text[0], variable, x) for i in r.solutions]
        assert any(abs(g - want) <= 1e-9 * max(1, abs(want)) for g in got), (x, want, got)


def test_a_system_is_solved_and_checked_together() -> None:
    r = solve(r"\dot{x} = -y", r"\dot{y} = x", conditions=["x(0) = 1", "y(0) = 0"])
    assert r.status is Status.OK
    (item,) = r.solutions
    assert len(item.equations_latex) == 2
    for t in (0.4, 1.3):
        assert abs(evaluate(item.rhs_text[0], "t", t) - float(mp.cos(t))) < 1e-12
        assert abs(evaluate(item.rhs_text[1], "t", t) - float(mp.sin(t))) < 1e-12
    r = solve(r"\dot{x} = 2 x", r"\dot{y} = x + y", conditions=["x(0) = 1", "y(0) = 0"])
    (item,) = r.solutions
    for t in (0.2, 0.9):
        assert abs(evaluate(item.rhs_text[1], "t", t) - float(mp.exp(2 * t) - mp.exp(t))) < 1e-12


def test_conditions_may_depend_on_parameters() -> None:
    r = solve(r"y' = a y", independent="x", conditions=["y(0) = y_0"])
    assert r.status is Status.OK
    (item,) = r.solutions
    assert (
        abs(evaluate(item.rhs_text[0], "x", 0.5, a=0.8, y_0=3.0) - 3 * float(mp.exp(0.4))) < 1e-12
    )


# --- general solutions ---------------------------------------------------------------------


def test_a_general_solution_reports_independent_constants() -> None:
    r = solve(r"y'' + 3y' + 2y = 0", independent="t")
    (item,) = r.solutions
    assert item.family == "general" and item.constants == ["C1", "C2"]
    assert item.verification.independent_constants == 2
    assert item.verification.symbolic_zero is True


def test_a_first_order_family_has_one_constant_and_branches_are_all_checked() -> None:
    r = solve(r"y' + y = x y^3", independent="x")
    assert r.status is Status.OK and len(r.solutions) == 2  # the two signs of the square root
    assert all(s.family == "general" and s.constants == ["C1"] for s in r.solutions)


def test_an_implicit_solution_is_checked_without_implicit_differentiation() -> None:
    r = solve(r"y' = \frac{x + y}{x - y}", independent="x")
    assert r.status is Status.OK
    (item,) = r.solutions
    assert item.kind == "implicit" and item.evaluated
    assert item.rhs_text == []
    assert item.verification.passed and item.verification.points_checked >= 4
    assert item.family == "general"


def test_restrictions_are_reported() -> None:
    r = solve(r"x^2 y'' + x y' - y = 0", independent="x")
    assert r.solutions[0].restrictions_latex == ["x \\neq 0"]


# --- nothing may be claimed ----------------------------------------------------------------


def test_no_closed_form_is_unsupported_and_says_it_proves_nothing() -> None:
    r = solve(r"y' = y^2 + x", independent="x")
    assert r.status is Status.UNSUPPORTED
    assert "does not show that none exists" in r.message
    assert r.solutions == []


def test_inconsistent_conditions_show_nothing() -> None:
    r = solve(r"y' = y", independent="x", conditions=["y(0) = 1", "y(0) = 2"])
    assert r.status is Status.UNSUPPORTED and r.solutions == []
    assert "satisfies the conditions" in r.message


def test_conditions_where_the_solution_is_undefined_show_nothing() -> None:
    r = solve(r"y' = -y/x", independent="x", conditions=["y(0) = 1"])
    assert r.status is Status.UNSUPPORTED and r.solutions == []


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"equations": [r"y'(t) = y(t - 1)"], "independent": "t"}, Status.UNSUPPORTED),
        ({"equations": [r"y' = y"]}, Status.PARSE_ERROR),
        (
            {"equations": ["x' = y", "y = x"], "independent": "t", "unknowns": ["x", "y"]},
            Status.UNSUPPORTED,
        ),
        ({"equations": ["x' = y"], "independent": "t", "unknowns": ["x", "y"]}, Status.UNSUPPORTED),
        ({"equations": [r"y' = C_1 y"], "independent": "t"}, Status.UNSUPPORTED),
    ],
)
def test_unreadable_or_unsupported_input_has_an_explicit_status(
    kwargs: dict[str, Any], status: Status
) -> None:
    r = solve_closed_form(OdeInput(**kwargs), L)
    assert r.status is status and r.solutions == [] and r.message


# --- the checker rejects wrong candidates ---------------------------------------------------


def problem(*equations: str, **kw: Any) -> Any:
    return read_ode(OdeInput(equations=list(equations), **kw), L)


def test_a_candidate_is_rejected_if_it_is_wrong_by_one_part_in_a_million() -> None:
    p = problem("y' = y", independent="x")
    x = p.variable
    assert residual_check(p, [sp.exp(x)]).passed
    assert residual_check(p, [3 * sp.exp(x)]).passed  # a multiple is still a solution
    assert not residual_check(p, [sp.exp(sp.Rational(1000001, 1000000) * x)]).passed
    assert not residual_check(p, [sp.exp(-x)]).passed
    assert not residual_check(p, [sp.exp(x) + sp.Rational(1, 10**6)]).passed  # an offset of 1e-6


def test_sign_errors_missing_terms_and_wrong_exponents_are_rejected() -> None:
    p = problem(r"y'' + 3y' + 2y = 0", independent="x")
    x = p.variable
    assert residual_check(p, [sp.exp(-x)]).passed
    assert not residual_check(p, [sp.exp(x)]).passed
    assert not residual_check(p, [sp.exp(-x) + sp.exp(-3 * x)]).passed
    assert not residual_check(p, [sp.exp(-x) * (1 + x)]).passed
    forced = problem(r"y' + 2y = \sin x", independent="x")
    right = sp.Rational(2, 5) * sp.sin(x) - sp.Rational(1, 5) * sp.cos(x)
    assert residual_check(forced, [right]).passed
    assert not residual_check(forced, [-right]).passed
    assert not residual_check(forced, [sp.Rational(2, 5) * sp.sin(x)]).passed


def test_a_wrong_candidate_of_a_system_is_rejected() -> None:
    p = problem(r"\dot{x} = -y", r"\dot{y} = x")
    t = p.variable
    assert residual_check(p, [sp.cos(t), sp.sin(t)]).passed
    assert not residual_check(p, [sp.cos(t), -sp.sin(t)]).passed
    assert not residual_check(p, [sp.cos(2 * t), sp.sin(2 * t)]).passed


def test_dependent_constants_are_not_a_general_solution() -> None:
    p = problem(r"y'' - y = 0", independent="x")
    x = p.variable
    c1, c2 = sp.symbols("C1 C2")
    independent = c1 * sp.exp(x) + c2 * sp.exp(-x)
    dependent = c1 * sp.exp(x) + c2 * sp.exp(x)
    assert residual_check(p, [dependent], [c1, c2]).passed  # it solves the equation ...
    assert independent_constants(p, [independent], [c1, c2]) == 2
    assert independent_constants(p, [dependent], [c1, c2]) == 1  # ... but is not a family of two


def test_a_wrong_implicit_relation_is_rejected() -> None:
    p = problem(r"y' = \frac{x}{y}", independent="x")
    x, y = p.variable, p.jets[("y", 0)]
    c = sp.Symbol("C1")
    assert implicit_check(p, y**2 - x**2 - c, [c]).passed
    assert not implicit_check(p, y**2 - x**2 + x - c, [c]).passed
    assert not implicit_check(p, y**2 - 2 * x**2 - c, [c]).passed
    assert not implicit_check(p, y**3 - x**3 - c, [c]).passed


def test_an_unevaluated_integral_is_checked_by_quadrature() -> None:
    p = problem(r"y' - y = \frac{1}{\ln x}", independent="x")
    x = p.variable
    c = sp.Symbol("C1")
    good = sp.exp(x) * (c + sp.Integral(sp.exp(-x) / sp.log(x), x))
    bad = sp.exp(x) * (c - sp.Integral(sp.exp(-x) / sp.log(x), x))
    assert residual_check(p, [good], [c], unevaluated=True).passed
    assert not residual_check(p, [bad], [c], unevaluated=True).passed


def test_unevaluable_or_missing_candidates_are_never_accepted() -> None:
    p = problem("y' = y", independent="x")
    r = residual_check(p, [sp.zoo * p.variable])
    assert not r.passed and r.checked == 0
    assert not residual_check(p, []).passed


def test_a_candidate_defined_nowhere_real_is_not_accepted_on_too_few_points() -> None:
    p = problem(r"y' = \frac{1}{2 y}", independent="x")
    x = p.variable
    assert residual_check(p, [sp.sqrt(x)]).passed  # analytic continuation: checked also for x < 0
    assert not residual_check(p, [sp.sqrt(x) + 1]).passed


# --- metamorphic: the same dynamics written differently ----------------------------------------


def test_a_second_order_equation_and_its_first_order_system_give_the_same_solution() -> None:
    higher = solve(r"y'' + 3y' + 2y = 0", independent="t", conditions=["y(0) = 1", "y'(0) = 0"])
    system = solve(
        r"y' = v", r"v' = -3 v - 2 y", independent="t", conditions=["y(0) = 1", "v(0) = 0"]
    )
    assert higher.status is Status.OK and system.status is Status.OK
    for t in (0.2, 0.8, 1.5):
        a = evaluate(higher.solutions[0].rhs_text[0], "t", t)
        b = evaluate(system.solutions[0].rhs_text[0], "t", t)
        v = evaluate(system.solutions[0].rhs_text[1], "t", t)
        assert abs(a - b) < 1e-12
        h = 1e-6  # v is the derivative of y
        slope = (
            evaluate(higher.solutions[0].rhs_text[0], "t", t + h)
            - evaluate(higher.solutions[0].rhs_text[0], "t", t - h)
        ) / (2 * h)
        assert abs(v - slope) < 1e-6


def test_scaling_the_unknown_scales_a_linear_solution() -> None:
    one = solve(r"y' = x y", independent="x", conditions=["y(0) = 1"])
    three = solve(r"y' = x y", independent="x", conditions=["y(0) = 3"])
    for x in (0.5, 1.2):
        assert (
            abs(
                3 * evaluate(one.solutions[0].rhs_text[0], "x", x)
                - evaluate(three.solutions[0].rhs_text[0], "x", x)
            )
            < 1e-12
        )


def test_shifting_the_independent_variable_shifts_the_solution() -> None:
    original = solve(r"y' = x y", independent="x", conditions=["y(1) = 1"])
    shifted = solve(r"w' = (s + 1) w", independent="s", conditions=["w(0) = 1"])
    for s in (0.0, 0.7, 1.9):
        assert (
            abs(
                evaluate(original.solutions[0].rhs_text[0], "x", s + 1)
                - evaluate(shifted.solutions[0].rhs_text[0], "s", s)
            )
            < 1e-12
        )


def test_equivalent_notations_give_the_same_solution() -> None:
    forms = [
        (r"\frac{d^2 y}{dt^2} + 4 y = 0", "latex"),
        (r"\ddot{y} + 4 y = 0", "latex"),
        ("diff(y, t, 2) + 4*y = 0", "text"),
    ]
    values = []
    for source, fmt in forms:
        r = solve(
            source,
            fmt=fmt,
            independent="t",
            conditions=["y(0) = 0", "y'(0) = 2"],
        )
        assert r.status is Status.OK, r.message
        values.append(evaluate(r.solutions[0].rhs_text[0], "t", 0.9))
    assert max(abs(v - values[0]) for v in values) < 1e-12
    assert abs(values[0] - float(mp.sin(1.8))) < 1e-12


# --- property: against an integrator that shares no code with the solver ------------------------


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    a=st.integers(-3, 3),
    b=st.integers(-3, 3),
    y0=st.integers(-3, 3),
    v0=st.integers(-3, 3),
)
def test_constant_coefficient_ivps_agree_with_a_taylor_integrator(
    a: int, b: int, y0: int, v0: int
) -> None:
    r = solve(
        f"y'' + ({a}) y' + ({b}) y = 0",
        independent="t",
        conditions=[f"y(0) = {y0}", f"y'(0) = {v0}"],
    )
    assert r.status is Status.OK, r.message
    reference = mp.odefun(lambda t, state: [state[1], -a * state[1] - b * state[0]], 0, [y0, v0])
    for t in (0.35, 1.1):
        want = complex(reference(t)[0])
        got = evaluate(r.solutions[0].rhs_text[0], "t", t)
        assert abs(got - want) <= 1e-9 * max(1, abs(want))


# --- through the sandbox ------------------------------------------------------------------


def test_the_engine_runs_the_solver_in_the_sandbox(engine: Engine) -> None:
    r = engine.ode_solve_symbolic(
        OdeInput(equations=["y' = y"], independent="x", conditions=["y(0) = 1"])
    )
    assert r.status is Status.OK and r.duration_ms >= 0
    assert r.diagnostics.verified == 1 and r.diagnostics.dropped == 0
    assert [s.name for s in r.diagnostics.stages][-1] == "total"
    refused = engine.ode_solve_symbolic(OdeInput(equations=["y' = y"]))
    assert refused.status is Status.PARSE_ERROR


def test_diagnostics_hold_no_equation_name_or_value() -> None:
    secret = "Zqmark7919"
    r = solve(
        f"{secret}' = kappa_4242*{secret}",
        fmt="text",
        independent="s",
        conditions=[f"{secret}(31337) = 2"],
    )
    assert r.status is Status.OK, r.message
    dump = r.diagnostics.model_dump_json()
    for fragment in (secret, "4242", "31337", "kappa"):
        assert fragment not in dump
