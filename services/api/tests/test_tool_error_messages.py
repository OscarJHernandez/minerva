"""Every refusal tells the model what to change, for every tool (no dead-end errors).

A small model that is told only "invalid" or "failed" repeats the call. These tests run over the whole
registry, so a new tool inherits them.
"""

import re

import pytest
from minerva_api.agent.tools import TOOLS, ToolContext, execute
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

NO_LEAKS = re.compile(
    r"Traceback|pydantic|errors\.pydantic\.dev|File \"|site-packages|\.py\", line"
)


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def _required(name: str) -> list[str]:
    return [f for f, info in TOOLS[name].args_model.model_fields.items() if info.is_required()]


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_missing_required_fields_are_named(name: str, ctx: ToolContext) -> None:
    required = _required(name)
    if not required:
        pytest.skip("no required fields")
    out = execute(name, {}, ctx)
    assert out.status == "error" and out.arguments_valid is False
    assert required[0] in out.for_model
    assert not NO_LEAKS.search(out.for_model)


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_a_wrong_type_names_the_field(name: str, ctx: ToolContext) -> None:
    required = _required(name)
    if not required:
        pytest.skip("no required fields")
    arguments = {f: {"unexpected": [1]} for f in required}
    out = execute(name, arguments, ctx)
    assert out.status == "error"
    assert any(f in out.for_model for f in required)
    assert not NO_LEAKS.search(out.for_model)


@pytest.mark.parametrize("name", ["bogus", "mechanics.bogus", "Analyze Lagrangian", ""])
def test_an_unknown_tool_says_how_to_call_one(name: str, ctx: ToolContext) -> None:
    out = execute(name, {}, ctx)
    assert out.status == "error"
    assert "exactly its name" in out.for_model
    assert "simplify" in out.for_model


@pytest.mark.parametrize(
    ("tool", "field", "bad", "fragment"),
    [
        ("simplify", "latex", "sin(x)^2 + cos(x)^2", r"\sin("),
        ("differentiate", "latex", "theta_dot^2", "_{dot}"),
        ("integrate", "latex", "exp(x)", r"\exp("),
        ("series", "latex", "omega x", r"\omega"),
        ("analyze_lagrangian", "lagrangian", "x_dot^2", ""),
    ],
)
def test_text_syntax_in_a_latex_field_says_how_to_write_it(
    tool: str, field: str, bad: str, fragment: str, ctx: ToolContext
) -> None:
    arguments: dict[str, object] = {field: bad}
    if tool in ("integrate", "series"):
        arguments["variable"] = "x"
    if tool == "differentiate":
        arguments["variables"] = [{"name": "x", "order": 1}]
    if tool == "analyze_lagrangian":
        arguments.update(coordinates=["x"], format="latex")
    out = execute(tool, arguments, ctx)
    assert out.status == "parse_error", out.for_model
    assert fragment in out.for_model
    assert "format text" in out.for_model or "write" in out.for_model
    assert not NO_LEAKS.search(out.for_model)


# --- the two chats of 2026-11-26: every problem named in the first answer -----------------------------------


def test_a_mixed_syntax_metric_is_refused_with_every_fix(ctx: ToolContext) -> None:
    metric = [
        ["-(1-2M/r)", "0", "0", "0"],
        ["0", "(1-2M/r)^{-1}", "0", "0"],
        ["0", "0", "r^2", "0"],
        ["0", "0", "0", "r^2*sin(theta)^2"],
    ]
    arguments = {
        "coordinates": ["t", "r", "theta", "phi"],
        "metric": metric,
        "quantities": ["christoffel"],
        "format": "latex",
    }
    out = execute("geometry", arguments, ctx)
    assert out.status == "parse_error"
    assert r"\\sin(" in out.for_model and r"\\theta" in out.for_model, out.for_model
    out = execute("geometry", {**arguments, "format": "text"}, ctx)
    assert out.status == "parse_error"
    assert "2*M" in out.for_model and "implicit_multiplication" in out.for_model, out.for_model


def test_a_text_lagrangian_without_stars_names_each_product(ctx: ToolContext) -> None:
    out = execute(
        "analyze_lagrangian",
        {
            "lagrangian": "1/2 m1 (x1_dot^2) - k/2 * [ (x1 - a)^2 ]",
            "coordinates": ["x1"],
            "format": "text",
        },
        ctx,
    )
    assert out.status == "parse_error"
    for fix in ("2*m1", "m1*(", "instead of [ ]"):
        assert fix in out.for_model, out.for_model


def test_a_misspelt_coordinate_gets_the_name_the_lagrangian_uses(ctx: ToolContext) -> None:
    out = execute(
        "analyze_lagrangian",
        {
            "lagrangian": r"\frac12 m \dot{x}_1^2 - k x_1^2",
            "coordinates": ["x1"],
            "format": "latex",
        },
        ctx,
    )
    assert out.status == "parse_error"
    assert "x1 → x_1" in out.for_model and '"suggestions":["x_1"]' in out.for_model, out.for_model


def test_a_refused_equilibrium_names_the_accelerations_there(ctx: ToolContext) -> None:
    out = execute(
        "expand_motion",
        {
            "lagrangian": r"\frac12 m \dot{x}^2 - \frac12 k (x - a)^2",
            "coordinates": ["x"],
            "equilibrium": {"x": "0"},
            "order": 1,
        },
        ctx,
    )
    assert out.status == "unsupported"
    assert "accelerations at this point" in out.for_model, out.for_model
    assert r"x: $\frac{a k}{m}$" in out.for_model, out.for_model
    assert all(latex not in out.known_latex for latex, _ in out.quoted_latex)
