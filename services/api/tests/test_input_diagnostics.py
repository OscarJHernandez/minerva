"""A tool reports how its input was read, and a refused input as a typed diagnostic the loop can act on.

Regression: velocities written `Xdot`, `Ydot`, `sdot` made `analyze_lagrangian` fail and the model retried. The
spelling is now read (and said so); an input that cannot be read carries a closed code, the field, the user's
symbols and the spellings that would work, in one fixed JSON line, and only the code reaches the log.
"""

import json

import pytest
from minerva_api.agent.routing import plan_route
from minerva_api.agent.tools import (
    AnalyzeLagrangianArgs,
    ToolContext,
    execute,
    resolve_syntax,
    run_lagrangian,
)
from minerva_api.agent.tracing import REASON_CODES
from minerva_engine import Engine, Limits
from minerva_engine.notation.models import InputCode
from minerva_knowledge import KnowledgeBase

WEDGE = (
    "1/2*(M+m)*(Xdot^2 + Ydot^2) + 1/2*m*sdot^2 + m*sdot*(Xdot*cos(alpha) + Ydot*sin(alpha))"
    " - (M+m)*g*Y - m*g*s*sin(alpha)"
)


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def _lagrangian(ctx: ToolContext, source: str, **extra: str):  # type: ignore[no-untyped-def]
    return run_lagrangian(
        AnalyzeLagrangianArgs(lagrangian=source, coordinates=["X", "Y", "s"], **extra), ctx
    )


def test_every_input_code_can_be_logged() -> None:
    assert {code.value for code in InputCode} <= REASON_CODES


def test_a_spelled_velocity_is_analysed_and_the_reading_is_said(ctx: ToolContext) -> None:
    out = _lagrangian(ctx, WEDGE)
    assert out.status == "ok", out.for_model[:300]
    (line,) = [x for x in out.for_model.splitlines() if x.startswith("input:")]
    for spelled, canonical in (("Xdot", "X_dot"), ("Ydot", "Y_dot"), ("sdot", "s_dot")):
        assert f"`{spelled}` as `{canonical}`" in line
    assert out.reason_code is None
    assert r"\ddot{Y} = - g" in out.for_model


def test_the_card_carries_the_reading_for_the_user(ctx: ToolContext) -> None:
    out = _lagrangian(ctx, WEDGE)
    reading = out.card.data["report"]["reading"]
    assert reading["format"] == "text"
    assert {n["source"] for n in reading["normalizations"]} == {"Xdot", "Ydot", "sdot"}


def test_canonical_input_has_no_input_line(ctx: ToolContext) -> None:
    out = _lagrangian(ctx, WEDGE.replace("dot", "_dot"))
    assert out.status == "ok"
    assert not any(x.startswith("input:") for x in out.for_model.splitlines())


def test_a_refused_input_has_one_machine_readable_line_and_a_logged_code(ctx: ToolContext) -> None:
    out = _lagrangian(ctx, WEDGE.replace("*", r"\*"), format="latex")
    assert out.status == "parse_error"
    (line,) = [x for x in out.for_model.splitlines() if x.startswith("diagnostic: ")]
    payload = json.loads(line.removeprefix("diagnostic: "))
    assert payload["status"] == "invalid_input"
    assert payload["code"] == out.reason_code == "unrecognized_derivative"
    assert payload["field"] == "lagrangian" and payload["recoverable"] is True
    assert payload["symbols"] and any("dot" in s for s in payload["suggestions"])


def test_a_forgotten_coordinate_is_a_typed_diagnostic_through_execute(ctx: ToolContext) -> None:
    out = execute(
        "analyze_lagrangian",
        {"lagrangian": "1/2*m*(x_dot^2 + y_dot^2)", "coordinates": ["x"]},
        ctx,
    )
    assert out.status == "parse_error"
    assert out.reason_code == "undeclared_dependent"
    assert out.error_category == "parse_error"


def test_text_is_chosen_for_a_spelled_velocity_without_other_markers() -> None:
    assert resolve_syntax("m*Xdot^2", "auto") == "text"
    assert resolve_syntax(r"\frac12 m \dot{x}^2", "auto") == "latex"
    assert resolve_syntax("m x^2", "auto") == "latex"


@pytest.mark.parametrize(
    ("message", "coordinates"),
    [
        ("What are the equations of motion for L = " + WEDGE, ["X", "Y", "s"]),
        ("Find the equations of motion for $L = 1/2*m*Xdot^2 - 1/2*k*X^2$", ["X"]),
        ("equations of motion of L = 1/2*m*(qdot^2 + q1dot^2) - V0*q*q1", ["q", "q1"]),
    ],
)
def test_the_router_reads_spelled_velocities(message: str, coordinates: list[str]) -> None:
    route = plan_route(message)
    assert route is not None and route.arguments["coordinates"] == coordinates


@pytest.mark.parametrize(
    "message",
    [
        # `xddot` is the acceleration of x or the velocity of xd: two readings, left to the model.
        "What are the equations of motion for L = 1/2*m*xddot^2 - xdot",
        # An acceleration is not a first-order Lagrangian.
        "What are the equations of motion for L = 1/2*m*xdot^2 + x*xddot",
    ],
)
def test_the_router_declines_an_ambiguous_or_second_order_spelling(message: str) -> None:
    assert plan_route(message) is None
