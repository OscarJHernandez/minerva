"""Reading differential equations: notation, what is refused, and what is never guessed."""

from __future__ import annotations

from typing import Any

import pytest
import sympy as sp
from minerva_engine import Limits, MinervaMathError, ParseError
from minerva_engine.errors import UnsupportedInput
from minerva_engine.ode import OdeCondition, OdeInput
from minerva_engine.ode.reading import OdeProblem, read_ode

L = Limits()


def read(*equations: str, fmt: str = "latex", **kw: Any) -> OdeProblem:
    return read_ode(OdeInput(equations=list(equations), format=fmt, **kw), L)


def residual(problem: OdeProblem, index: int = 0) -> sp.Expr:
    return problem.residuals[index]


def test_every_notation_for_the_same_equation_reads_the_same() -> None:
    forms = [
        (r"y'' + 3 y' + 2 y = 0", "latex"),
        (r"\ddot{y} + 3 \dot{y} + 2 y = 0", "latex"),
        (r"\frac{d^2 y}{d t^2} + 3 \frac{d y}{d t} + 2 y = 0", "latex"),
        (
            r"\frac{\mathrm{d}^{2}y}{\mathrm{d}t^{2}} + 3\frac{\mathrm{d}y}{\mathrm{d}t} + 2y = 0",
            "latex",
        ),
        (r"y^{(2)} + 3 y^{(1)} + 2 y = 0", "latex"),
        (r"\frac{d^2}{dt^2} y + 3 \frac{d}{dt} y + 2 y = 0", "latex"),
        (r"y''(t) + 3 y'(t) + 2 y(t) = 0", "latex"),
        ("y'' + 3*y' + 2*y = 0", "text"),
        ("diff(y, t, 2) + 3*diff(y, t) + 2*y = 0", "text"),
        ("y_ddot + 3*y_dot + 2*y = 0", "text"),
    ]
    reference: sp.Expr | None = None
    for source, fmt in forms:
        problem = read(source, fmt=fmt, independent="t")
        assert problem.unknowns == ["y"], source
        assert problem.orders == {"y": 2}, source
        assert problem.variable.name == "t", source
        current = residual(problem).xreplace(
            {problem.jets[("y", k)]: sp.Symbol(f"j{k}") for k in range(3)}
        )
        if reference is None:
            reference = current
        assert sp.expand(current - reference) == 0, source


def test_dfrac_dt_is_a_derivative_not_a_ratio() -> None:
    # The general parser reads \frac{dy}{dt} as y/t (the d cancels); here it is a derivative.
    problem = read(r"\frac{dy}{dt} = y")
    assert problem.orders == {"y": 1}
    assert problem.variable.name == "t"
    assert sp.expand(residual(problem) - (problem.jets[("y", 1)] - problem.jets[("y", 0)])) == 0


def test_the_independent_variable_is_read_from_the_derivative_when_it_is_named() -> None:
    assert read(r"\frac{dy}{dx} = x y").variable.name == "x"
    assert read(r"y(x) + y'(x) = 0", independent=None).variable.name == "x"
    assert read(r"\dot{x} = -x").variable.name == "t"


def test_an_unnamed_independent_variable_is_never_guessed() -> None:
    with pytest.raises(ParseError, match="independent"):
        read("y' = y")
    with pytest.raises(ParseError, match="independent"):
        read("y' + y = 0", fmt="text")


def test_conflicting_variables_are_an_error() -> None:
    with pytest.raises(ParseError):
        read(r"\frac{dy}{dx} + \frac{dz}{dt} = 0")
    with pytest.raises(ParseError, match="independent variable"):
        read(r"\frac{dy}{dx} = y", independent="t")


def test_a_system_keeps_the_order_of_first_appearance() -> None:
    problem = read(r"\dot{x} = -y", r"\dot{y} = x")
    assert problem.unknowns == ["x", "y"]
    assert problem.orders == {"x": 1, "y": 1}
    assert [s.name for s in problem.parameters] == []


def test_parameters_are_the_remaining_symbols() -> None:
    problem = read(r"m \ddot{x} + c \dot{x} + k x = F", independent="t")
    assert [s.name for s in problem.parameters] == ["F", "c", "k", "m"]


def test_assumptions_on_an_unknown_reach_its_symbol_and_derivatives() -> None:
    problem = read(r"y'' = -y", independent="t", assumptions={"y": ["positive"]})
    assert problem.jets[("y", 0)].is_positive is True
    assert problem.jets[("y", 1)].is_real is True
    assert problem.variable.is_real is True


def test_greek_and_subscripted_unknowns() -> None:
    problem = read(r"\theta'' + \sin\theta = 0", independent="t")
    assert problem.unknowns == ["theta"]
    problem = read(r"\dot{x}_1 = x_2", r"\dot{x}_2 = -x_1")
    assert problem.unknowns == ["x_1", "x_2"]
    assert problem.latex(problem.jets[("x_1", 1)]) == "x_{1}'"


def test_conditions_in_several_forms() -> None:
    problem = read(
        r"y'' + y = 0",
        independent="t",
        conditions=[
            "y(0) = 1",
            "y'(0) = 0",
            OdeCondition(unknown="y", order=2, at="0", value="-1"),
        ],
    )
    assert [(c.unknown, c.order) for c in problem.conditions] == [("y", 0), ("y", 1), ("y", 2)]
    assert [c.value for c in problem.conditions] == [1, 0, -1]
    dotted = read(r"\ddot{x} = -x", conditions=[r"\dot{x}(0) = 2", r"x(\pi) = 1"])
    assert [(c.unknown, c.order) for c in dotted.conditions] == [("x", 1), ("x", 0)]
    assert dotted.conditions[1].at == sp.pi


@pytest.mark.parametrize(
    "condition",
    ["y(0)", "= 1", "y = 1", "z(0) = 1", "y'(0) = 1 = 2", "y' + y(0) = 1"],
)
def test_malformed_conditions_are_rejected(condition: str) -> None:
    with pytest.raises(ParseError):
        read("y'' + y = 0", independent="t", conditions=[condition])


@pytest.mark.parametrize(
    "source",
    [
        r"y'(t) = y(t - 1)",  # delayed
        r"y'(t) = y(2 t)",  # composed
        r"\partial_t u = u",  # partial
        r"\frac{\partial u}{\partial t} = u",
        r"\frac{d}{dt}(t y) = 1",  # operator on an expression
    ],
)
def test_unsupported_forms_are_refused_not_reinterpreted(source: str) -> None:
    with pytest.raises(UnsupportedInput):
        read(source, independent="t")


def test_an_undeclared_function_call_is_ambiguous_not_a_product() -> None:
    with pytest.raises(ParseError, match="ambiguous"):
        read(r"y' = f(t) y", independent="t")


def test_equation_shape_errors() -> None:
    with pytest.raises(ParseError, match="equals"):
        read(r"y' + y", independent="t")
    with pytest.raises(ParseError, match="exactly one"):
        read(r"y' = y = 0", independent="t")
    with pytest.raises(ParseError, match="empty"):
        read(r"y' = ", independent="t")
    with pytest.raises(ParseError, match="no unknown"):
        read(r"x + 1 = 0", independent="t")


def test_reserved_placeholder_names_cannot_be_smuggled_in() -> None:
    with pytest.raises(ParseError, match="reserved"):
        read(r"y' = Q_{a1}", independent="t")
    with pytest.raises(ParseError, match="reserved"):
        read("y' = Q_a1", fmt="text", independent="t")


def test_a_parameter_cannot_shadow_a_derivative_symbol() -> None:
    with pytest.raises(ParseError, match="reserved"):
        read(r"y' = y_{d1}", independent="t")
    with pytest.raises(ParseError, match="reserved"):
        read("y' = y_d1", fmt="text", independent="t")


def test_limits_and_hostile_input() -> None:
    with pytest.raises(MinervaMathError):
        read_ode(OdeInput(equations=["y' = " + "y+" * 5000 + "1"], independent="t"), Limits())
    with pytest.raises(ValueError, match="at most 8"):
        read(*[f"x_{i}' = x_{i}" for i in range(9)], independent="t")
    with pytest.raises(UnsupportedInput, match="order"):
        read("y^{(11)} = y", independent="t")
    with pytest.raises(ParseError):
        read("y' = __import__('os')", fmt="text", independent="t")


def test_text_diff_forms() -> None:
    assert read("diff(y, t) = y", fmt="text").orders == {"y": 1}
    assert read("Derivative(y(t), t, 3) = y", fmt="text").orders == {"y": 3}
    assert read("diff(y, t, t) = y", fmt="text").orders == {"y": 2}
    with pytest.raises(UnsupportedInput):
        read("diff(y*y, t) = 1", fmt="text")
    with pytest.raises(UnsupportedInput):
        read("diff(y, t, s) = 1", fmt="text")


def test_a_symbol_that_only_a_condition_mentions_is_a_parameter() -> None:
    problem = read(r"y' = a y", independent="x", conditions=["y(0) = y_0"])
    assert [s.name for s in problem.parameters] == ["a", "y_0"]
    with pytest.raises(ParseError, match="variable"):
        read(r"y' = a y", independent="x", conditions=["y(x) = 1"])


# --- first-order systems written with names such as y1 ---------------------------------------


def test_leibniz_fractions_are_read_in_text_for_declared_unknowns_only() -> None:
    problem = read(
        "dy1/dt = y2", "dy2/dt = -y1", fmt="text", unknowns=["y1", "y2"], independent="t"
    )
    assert problem.unknowns == ["y1", "y2"] and problem.orders == {"y1": 1, "y2": 1}
    assert problem.parameters == []
    second = read("d^2y/dt^2 = -y", fmt="text", unknowns=["y"], independent="t")
    assert second.orders == {"y": 2}
    # Not declared: the fraction is a product of letters and is never read as a derivative.
    with pytest.raises(ParseError, match="listed in `unknowns`"):
        read("dy1/dt = y2", fmt="text")
    # The orders above and below must agree.
    mismatch = read("dy/dt^2 = y", fmt="text", unknowns=["y"], independent="t")
    assert "dy" in {s.name for s in mismatch.parameters}


def test_a_name_such_as_y1_in_latex_is_refused_with_the_fix() -> None:
    with pytest.raises(ParseError, match=r"y_1.*format to text"):
        read("y1' = y2", unknowns=["y1", "y2"], independent="t")
    assert read("y_1' = y_2", unknowns=["y_1", "y_2"], independent="t").unknowns == ["y_1", "y_2"]
