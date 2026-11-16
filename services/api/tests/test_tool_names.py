"""Tool names: the prompt never writes a group next to a tool with a period, and only closed aliases resolve.

Regression for a model calling `mechanics.analyze_lagrangian` (the prompt had shown "mechanics. ...
`analyze_lagrangian`") and being refused with a list of every tool.
"""

import asyncio
from typing import Any

import pytest
from minerva_api.agent import toolsets
from minerva_api.agent.loop import run_agent
from minerva_api.agent.prompts import GROUP_TOOLS, assemble
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import TOOLS, LoadToolsArgs, ToolContext, run_load_tools
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def play(llm: ReplayLLM, ctx: ToolContext, text: str) -> list[dict[str, Any]]:
    async def go() -> list[dict[str, Any]]:
        return [e async for e in run_agent(llm, [{"role": "user", "content": text}], ctx)]

    return asyncio.run(go())


def test_no_prompt_puts_a_group_name_and_a_period_before_a_tool() -> None:
    for group, tools in GROUP_TOOLS.items():
        text = assemble([group])
        for tool in tools:
            assert f"{group}.{tool}" not in text
            assert f"{group}. `{tool}`" not in text
            assert f"{group}.\n`{tool}`" not in text


def test_the_load_tools_result_lists_bare_names(ctx: ToolContext) -> None:
    out = run_load_tools(LoadToolsArgs(groups=["mechanics"]), ctx)
    assert "mechanics." not in out.for_model
    assert "analyze_lagrangian" in out.for_model


@pytest.mark.parametrize(
    ("requested", "name", "kind"),
    [
        ("analyze_lagrangian", "analyze_lagrangian", None),
        ("mechanics.analyze_lagrangian", "analyze_lagrangian", "group_prefix"),
        ("Mechanics/analyze_lagrangian", "analyze_lagrangian", "group_prefix"),
        ("functions.simplify", "simplify", "functions_prefix"),
        ("  Simplify ", "simplify", "case"),
    ],
)
def test_closed_aliases_resolve(requested: str, name: str, kind: str | None) -> None:
    assert toolsets.resolve_tool_name(requested, TOOLS) == (name, kind)


@pytest.mark.parametrize(
    "requested",
    [
        "mechanics.integrate",
        "ode.analyze_lagrangian",
        "foo.bar",
        "analyse_lagrangian",
        "mechanics.",
        "",
    ],
)
def test_genuinely_unknown_names_stay_unknown(requested: str) -> None:
    assert toolsets.resolve_tool_name(requested, TOOLS) == (requested, None)


def test_a_dotted_call_runs_and_the_trace_keeps_both_names(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call(
                "mechanics.analyze_lagrangian",
                {"lagrangian": r"\frac12 m \dot{x}^2 - \frac12 k x^2", "coordinates": ["x"]},
            ),
            say("Done."),
        ]
    )
    events = play(llm, ctx, "Equations of motion for this Lagrangian")
    step = next(e["data"] for e in events if e["event"] == "message")["trace"][0]
    assert step["name"] == "analyze_lagrangian"
    assert step["requested_name"] == "mechanics.analyze_lagrangian"
    assert step["alias_kind"] == "group_prefix"
    assert step["status"] == "ok"


def test_an_unknown_tool_is_refused_without_a_redirect(ctx: ToolContext) -> None:
    llm = ReplayLLM([call("mechanics.integrate", {"expression": "x"}), say("Sorry.")])
    events = play(llm, ctx, "hello")
    step = next(e["data"] for e in events if e["event"] == "message")["trace"][0]
    assert step["name"] == "mechanics.integrate"
    assert step["status"] == "error"
    assert step["requested_name"] is None
    assert "exactly its name" in step["for_model"]
