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
