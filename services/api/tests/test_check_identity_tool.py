"""`check_identity` in the chat: the verdict the model reads, what may be repeated as verified, and what is refused."""

import asyncio
from typing import Any

import pytest
from minerva_api.agent.llm import Chunk, ToolCall
from minerva_api.agent.loop import AgentLimits, run_agent
from minerva_api.agent.replay import ReplayLLM
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

FAST = AgentLimits(retry_delay_seconds=0)


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def marks(ctx: ToolContext, call: dict[str, Any], reply: str) -> dict[str, str]:
    llm = ReplayLLM([[Chunk(tool_calls=[ToolCall("check_identity", call)])], [Chunk(text=reply)]])

    async def go() -> list[dict[str, Any]]:
        history = [{"role": "user", "content": "is it true?"}]
        return [e async for e in run_agent(llm, history, ctx, FAST)]

    message = next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")
    return {
        s["value"]: s["status"]
        for b in message["blocks"]
        if b["kind"] == "text"
        for s in b["segments"]
        if s["type"] != "text"
    }


def test_a_false_identity_reaches_the_model_with_its_point_and_is_not_marked_verified(
    ctx: ToolContext,
) -> None:
    out = execute("check_identity", {"latex": r"(x+y)^2 = x^2 + y^2"}, ctx)
    assert out.status == "ok" and out.arguments_valid and out.card.type == "identity"
    assert "verdict=not_identity" in out.for_model and "counterexample at" in out.for_model
    assert out.evidence is not None and out.evidence.check == "numeric" and out.evidence.accepted
    # the user's own equation was not verified true by this result: it is quoted, never from_tool
    said = marks(ctx, {"latex": r"(x+y)^2 = x^2 + y^2"}, r"No: $(x+y)^2 = x^2 + y^2$ is false.")
    assert all(status != "from_tool" for value, status in said.items() if "=" in value)


def test_a_true_identity_may_be_repeated_as_verified_and_a_wrong_twin_may_not(
    ctx: ToolContext,
) -> None:
    call = {"latex": r"\cosh^2 x - \sinh^2 x = 1"}
    out = execute("check_identity", call, ctx)
    assert "verdict=proved_equal" in out.for_model
    assert out.known_latex and "=" in out.known_latex[0]
    said = marks(ctx, call, r"Yes: $\cosh^{2}{\left(x \right)} - \sinh^{2}{\left(x \right)} = 1$.")
    assert "from_tool" in said.values()
    wrong = marks(ctx, call, r"Yes, and also $\cosh^2 x - \sinh^2 x = 2$.")
    assert wrong[r"\cosh^2 x - \sinh^2 x = 2"] != "from_tool"


def test_the_counterexample_values_are_the_engines_and_other_numbers_are_not(
    ctx: ToolContext,
) -> None:
    out = execute("check_identity", {"latex": r"(x+y)^2 = x^2 + y^2"}, ctx)
    assert out.card.data["counterexample"]["substitution"] == {"x": "1", "y": "1"}
    said = marks(
        ctx,
        {"latex": r"(x+y)^2 = x^2 + y^2"},
        r"At $x = 1$ and $y = 1$ the sides are $4.0$ and $2.0$, not $7$.",
    )
    assert said["x = 1"] == "from_tool" and said["7"] != "from_tool"


def test_a_relation_and_a_matrix_statement_are_refused_not_called_false(ctx: ToolContext) -> None:
    for call in (
        {"latex": "F = ma"},
        {"latex": "AB = BA", "assumptions": {"A": ["noncommutative"], "B": ["noncommutative"]}},
    ):
        out = execute("check_identity", call, ctx)
        assert out.status == "unsupported", call
        assert "verdict=" not in out.for_model and "counterexample" not in out.for_model
        assert out.evidence is not None and not out.evidence.accepted


def test_bad_input_is_a_typed_refusal(ctx: ToolContext) -> None:
    assert execute("check_identity", {"latex": "x + 1"}, ctx).status == "parse_error"
    out = execute("check_identity", {}, ctx)
    assert out.status == "error" and not out.arguments_valid


def test_the_tool_is_in_the_core_set_and_its_formula_field_states_its_syntax() -> None:
    from minerva_api.agent.tools import TOOLS
    from minerva_api.agent.toolsets import CORE_TOOLS

    assert "check_identity" in CORE_TOOLS
    description = TOOLS["check_identity"].args_model.model_json_schema()["properties"]["latex"][
        "description"
    ]
    assert "LaTeX" in description
