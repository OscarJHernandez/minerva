"""The `time_derivative` tool: what the model is told, what the card holds, and the marks it earns."""

from __future__ import annotations

import pytest
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

KEPLER = r"\frac12 \mu (\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{k}{r}"
ASSUME = {"mu": ["positive"], "k": ["positive"], "r": ["positive"], "theta": ["real"]}


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def test_differentiate_by_t_points_to_the_total_derivative(ctx: ToolContext) -> None:
    out = execute(
        "differentiate",
        {"latex": r"\mu r^2 \dot{\theta}", "variables": [{"name": "t"}], "assumptions": ASSUME},
        ctx,
    )
    assert out.status == "unsupported" and "time_derivative" in out.for_model
    assert out.evidence is not None and not out.evidence.accepted


def test_total_derivative_of_the_report_expression_is_verified(ctx: ToolContext) -> None:
    out = execute(
        "time_derivative",
        {"latex": r"\mu r^2 \dot{\theta}", "coordinates": ["r", "theta"], "assumptions": ASSUME},
        ctx,
    )
    assert out.status == "ok"
    assert out.evidence is not None and out.evidence.accepted
    assert "total time derivative=" in out.for_model and r"\ddot{\theta}" in out.for_model
    assert any(r"\ddot{\theta}" in latex for latex in out.known_latex)  # a verified formula
    assert out.card.type == "differentiate"


def test_with_a_lagrangian_a_constant_is_reported_zero_on_the_motion(ctx: ToolContext) -> None:
    out = execute(
        "time_derivative",
        {
            "latex": r"\mu r^2 \dot{\theta}",
            "coordinates": ["r", "theta"],
            "lagrangian": KEPLER,
            "assumptions": ASSUME,
        },
        ctx,
    )
    assert out.status == "ok" and "zero on the equations of motion" in out.for_model
    assert "0" in out.known_latex


def test_a_non_constant_is_quoted_not_marked_verified(ctx: ToolContext) -> None:
    out = execute(
        "time_derivative",
        {
            "latex": r"r \dot{r}",
            "coordinates": ["r", "theta"],
            "lagrangian": KEPLER,
            "assumptions": ASSUME,
        },
        ctx,
    )
    assert out.status == "ok" and "NOT zero" in out.for_model
    assert out.quoted_latex  # the substituted formula is shown but never `known_latex`


def test_an_equation_for_the_lagrangian_is_refused(ctx: ToolContext) -> None:
    out = execute(
        "time_derivative",
        {"latex": r"r \dot{r}", "coordinates": ["r"], "lagrangian": "L = " + KEPLER},
        ctx,
    )
    assert out.status == "parse_error" and "after the last" in out.for_model


def test_scalar_guard_applies_to_time_derivative(ctx: ToolContext) -> None:
    guarded = ToolContext(engine=ctx.engine, kb=ctx.kb, matrices=frozenset({"r"}))
    out = execute("time_derivative", {"latex": r"r \dot{r}", "coordinates": ["r"]}, guarded)
    assert out.status == "error" and "declared as a matrix" in out.for_model
