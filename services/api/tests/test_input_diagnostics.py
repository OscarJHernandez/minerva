"""A tool reports how its input was read, and a refused input as a typed diagnostic the loop can act on.

Regression: velocities written `Xdot`, `Ydot`, `sdot` made `analyze_lagrangian` fail and the model retried. The
spelling is now read (and said so); an input that cannot be read carries a closed code, the field, the user's
symbols and the spellings that would work, in one fixed JSON line, and only the code reaches the log.
"""

import json

import pytest
from minerva_api.agent.routing import plan_route
from minerva_api.agent.tools import (
    ToolContext,
    execute,
    resolve_syntax,
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
    return execute(
        "analyze_lagrangian", {"lagrangian": source, "coordinates": ["X", "Y", "s"], **extra}, ctx
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


# -- find_conserved_quantities: `format`, and the coordinate limit (#131) ---------------------------------------

# A one-dimensional oscillator: these tests check how `format` is read, not the search, and a Kepler search takes
# several seconds per call, which ran out of the discovery time budget on shared CI runners.
OSCILLATOR_TEXT = "1/2*m*x_dot^2 - 1/2*k*x^2"


def test_find_conserved_quantities_takes_a_format_like_its_siblings(ctx: ToolContext) -> None:
    for syntax in ("text", "auto"):
        out = execute(
            "find_conserved_quantities",
            {"lagrangian": OSCILLATOR_TEXT, "coordinates": ["x"], "format": syntax},
            ctx,
        )
        assert out.status == "ok", out.for_model[:300]
        assert "constants of motion" in out.for_model
    latex = execute(
        "find_conserved_quantities",
        {"lagrangian": OSCILLATOR_TEXT, "coordinates": ["x"], "format": "latex"},
        ctx,
    )
    assert latex.status == "parse_error" and latex.reason_code == "unrecognized_derivative"


def test_text_in_the_default_format_of_find_conserved_quantities_is_read_as_text(
    ctx: ToolContext,
) -> None:
    out = execute(
        "find_conserved_quantities", {"lagrangian": OSCILLATOR_TEXT, "coordinates": ["x"]}, ctx
    )
    assert out.status == "ok"


def test_four_coordinates_are_searched(ctx: ToolContext) -> None:
    lagrangian = "1/2*(a_dot^2 + b_dot^2 + c_dot^2 + d_dot^2) - 1/2*(a^2 + b^2 + c^2 + d^2)"
    arguments = {"lagrangian": lagrangian, "coordinates": ["a", "b", "c", "d"], "budget": "quick"}
    out = execute("find_conserved_quantities", arguments, ctx)
    assert out.status == "ok" and out.reason_code != "too_many_coordinates"


def test_five_coordinates_are_refused_with_the_limit_and_what_to_do(ctx: ToolContext) -> None:
    lagrangian = "1/2*(a_dot^2 + b_dot^2 + c_dot^2 + d_dot^2 + e_dot^2)"
    arguments = {"lagrangian": lagrangian, "coordinates": ["a", "b", "c", "d", "e"]}
    out = execute("find_conserved_quantities", arguments, ctx)
    assert out.status == "unsupported"
    assert out.reason_code == "too_many_coordinates"
    assert "at most 4 coordinates" in out.for_model and "not available" in out.for_model
    assert "List should have" not in out.for_model
    (line,) = [x for x in out.for_model.splitlines() if x.startswith("diagnostic: ")]
    payload = json.loads(line.removeprefix("diagnostic: "))
    assert payload["code"] == "too_many_coordinates" and payload["recoverable"] is False
    assert not out.known_latex  # nothing was searched and no coordinate was dropped
    assert "#" not in out.for_model  # no issue numbers in model-facing text


def test_a_hamiltonian_in_text_syntax_is_searched(ctx: ToolContext) -> None:
    out = execute(
        "find_conserved_quantities",
        {
            "hamiltonian": "p_x^2/(2*m) + k*x^2/2",
            "coordinates": ["x"],
            "format": "text",
        },
        ctx,
    )
    assert out.status == "ok", out.for_model[:300]
