"""The `routh_reduce` tool (issue #153): its schema, its card, its numbers and the handle of the Routhian.

The constant held fixed is the verified momentum, never a value or a name the model picks: the schema has no
field for one. R is the card's single result (one number, a result id that carries the restrictions); what was
held fixed and the rebuilt velocities are numbered list rows (names only). A later tool that reads R through
`@rN` inherits the restrictions.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from minerva_api.agent.card_equations import listed_equations, result_formula, routh
from minerva_api.agent.equations import EquationTable, number_blocks
from minerva_api.agent.events import Block
from minerva_api.agent.loop import AgentLimits, run_agent
from minerva_api.agent.refs import COMPONENT_SUFFIX, REFERENCEABLE_TOOLS, ResultStore
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import TOOLS, ToolContext, ToolOutput, execute
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_equation_rows_contract import FIXTURE

SCHWARZSCHILD = (
    r"\frac{1}{2}\left(-\left(1-\frac{2M}{r}\right)\dot{t}^2+\frac{\dot{r}^2}{1-\frac{2M}{r}}"
    r"+r^2\dot{\theta}^2+r^2\sin^2(\theta)\dot{\phi}^2\right)"
)
ARGS: dict[str, Any] = {
    "lagrangian": SCHWARZSCHILD,
    "coordinates": ["t", "r", "theta", "phi"],
    "time": "s",
    "assumptions": {"M": ["positive"], "r": ["positive"]},
    "cyclic": ["t"],
}
FAST = AgentLimits(retry_delay_seconds=0)


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


@pytest.fixture(scope="module")
def reduced(ctx: ToolContext) -> ToolOutput:
    out = execute("routh_reduce", ARGS, ctx)
    assert out.status == "ok", out.for_model
    return out


def play(ctx: ToolContext, replies: list[Any]) -> dict[str, Any]:
    llm = ReplayLLM(replies)

    async def go() -> list[dict[str, Any]]:
        return [e async for e in run_agent(llm, [{"role": "user", "content": "q"}], ctx, FAST)]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")


def cards(msg: dict[str, Any]) -> list[dict[str, Any]]:
    return [b["card"] for b in msg["blocks"] if b["kind"] == "card"]


def test_the_schema_has_no_field_for_a_constant_a_value_or_a_name() -> None:
    properties = TOOLS["routh_reduce"].schema()["function"]["parameters"]["properties"]
    assert set(properties) == {
        "lagrangian",
        "coordinates",
        "time",
        "format",
        "assumptions",
        "cyclic",
    }


def test_the_card_holds_the_routhian_what_was_held_fixed_and_the_velocity(
    reduced: ToolOutput,
) -> None:
    data = reduced.card.data
    assert reduced.card.type == "routh" and reduced.card.tool == "routh_reduce"
    assert data["status"] == "ok" and data["result_latex"]
    assert [h["symbol_latex"] for h in data["held_fixed"]] == ["p_{t}"]
    assert data["reconstruction"][0]["latex"].startswith(r"\dot{t} =")
    assert data["restrictions_latex"]
    assert data["request"]["cyclic"] == ["t"]
    text = reduced.for_model
    assert f"$R = {data['result_latex']}$" in text
    assert "held fixed: $p_{t} = " in text and "verified" in text
    assert "never rename it yourself" in text and "substitute" in text
    assert data["result_latex"] in reduced.known_latex
    assert reduced.evidence is not None and reduced.evidence.accepted


def test_the_routhian_is_the_single_result_and_the_rows_are_list_rows(
    reduced: ToolOutput,
) -> None:
    data = reduced.card.data
    assert result_formula("routh", data) == data["result_latex"]
    rows = listed_equations("routh", data)
    assert [k for k, _ in rows] == ["held:t", "velocity:t"]
    assert dict(rows)["held:t"] == data["held_fixed"][0]["latex"]
    card = reduced.card.model_copy(deep=True)
    table = EquationTable()
    number_blocks([Block(kind="card", card=card)], table)
    assert card.number == 1 and card.numbers == {"held:t": 2, "velocity:t": 3}
    # a row is a name: reused by number it is exact LaTeX and says it is one component of a result
    store = ResultStore.from_prior({}, {}, table.items, unverified={})
    resolved = store.resolve({"latex": "@e3"})
    assert resolved.error is None and resolved.arguments["latex"] == table.items[3].latex
    assert resolved.unverified == [f"equation (3){COMPONENT_SUFFIX}"]


def test_a_card_that_was_not_accepted_lists_nothing() -> None:
    assert (
        routh({"status": "not_verified", "held_fixed": [{"coordinate": "t", "latex": "x"}]}) == []
    )


def test_the_result_id_names_the_routhian_with_its_restrictions(reduced: ToolOutput) -> None:
    assert "routh_reduce" in REFERENCEABLE_TOOLS
    store = ResultStore()
    assert store.register("routh_reduce", reduced) == "r1"
    assert store.results["r1"] == reduced.card.data["result_latex"]
    assert store.restrictions["r1"] == reduced.card.data["restrictions_latex"]


def test_a_refusal_has_no_handle(ctx: ToolContext) -> None:
    out = execute("routh_reduce", {**ARGS, "cyclic": ["r"]}, ctx)
    assert out.status == "unsupported" and "reason: not_cyclic" in out.for_model
    assert ResultStore().register("routh_reduce", out) is None
    assert out.known_latex == []


# The block d^2L/dt_dot^2 = r - 1 vanishes at r = 1, but R = r_dot^2/2 - p_t^2/(2(r - 1)) - r^2/2 has a constant mass
# matrix: a tool that reads R reports no restriction of its own, so only the inherited one says where R holds.
COUPLED: dict[str, Any] = {
    "lagrangian": r"\frac{1}{2}\dot{r}^2 + \frac{1}{2}(r - 1)\dot{t}^2 - \frac{1}{2} r^2",
    "coordinates": ["r", "t"],
    "time": "s",
    "assumptions": {"r": ["real"]},
    "cyclic": ["t"],
}


@pytest.mark.parametrize("tool", ["analyze_lagrangian", "find_conserved_quantities"])
def test_a_tool_that_reads_the_routhian_by_its_id_inherits_the_restrictions(
    ctx: ToolContext, tool: str
) -> None:
    msg = play(
        ctx,
        [
            call("routh_reduce", COUPLED),
            call(
                tool,
                {
                    "lagrangian": "@r1",
                    "coordinates": ["r"],
                    "time": "s",
                    "assumptions": COUPLED["assumptions"],
                },
            ),
            say("done"),
        ],
    )
    first, second = msg["trace"][0], msg["trace"][1]
    restrictions = cards(msg)[0]["data"]["restrictions_latex"]
    assert first["status"] == "ok" and first["result_id"] == "r1"
    assert restrictions == [r"r \neq 1"]
    assert second["status"] == "ok", second["for_model"]
    assert "Inherited from @r1" in second["for_model"]
    assert r"r \neq 1" in second["for_model"]
    assert cards(msg)[1]["data"]["inherited_restrictions_latex"] == restrictions


def test_the_schwarzschild_routhian_read_back_by_its_id_keeps_its_restriction(
    ctx: ToolContext, reduced: ToolOutput
) -> None:
    """Here the analysis of R finds the horizon again itself (its mass matrix divides by 1 - 2M/r): the
    restriction is stated once, as the tool's own, and not repeated as inherited."""
    msg = play(
        ctx,
        [
            call("routh_reduce", ARGS),
            call(
                "analyze_lagrangian",
                {
                    "lagrangian": "@r1",
                    "coordinates": ["r", "theta", "phi"],
                    "time": "s",
                    "assumptions": ARGS["assumptions"],
                },
            ),
            say("done"),
        ],
    )
    second = cards(msg)[1]["data"]
    assert msg["trace"][1]["status"] == "ok"
    shown = set(second["restrictions_latex"]) | set(
        second.get("inherited_restrictions_latex") or []
    )
    assert set(reduced.card.data["restrictions_latex"]) <= shown


def test_a_momentum_name_clash_is_a_typed_refusal(ctx: ToolContext) -> None:
    out = execute(
        "routh_reduce",
        {
            "lagrangian": r"\frac{1}{2}\dot{x}^2 + \frac{1}{2}\dot{y}^2 + p_{x} y",
            "coordinates": ["x", "y"],
            "cyclic": ["x"],
        },
        ctx,
    )
    assert out.status == "parse_error"
    assert out.reason_code == "momentum_symbol_clash"
    assert '"code":"momentum_symbol_clash"' in out.for_model


def test_the_fixture_the_browser_renders_is_the_reader_and_a_live_card() -> None:
    """`packages/ui/fixtures/equation-rows.json` (`routh`, real engine output) is what `equation-rows.test.tsx`
    renders: the server's keys and the rows the panel marks cannot drift apart."""
    entry = json.loads(FIXTURE.read_text())["routh"]
    rows = listed_equations("routh", entry["data"])
    assert [k for k, _ in rows] == entry["keys"] == ["held:t", "velocity:t"]
    assert dict(rows) == entry["rows"]
    assert result_formula("routh", entry["data"]) == entry["data"]["result_latex"]


def test_a_live_card_is_read_like_the_fixture(reduced: ToolOutput) -> None:
    entry = json.loads(FIXTURE.read_text())["routh"]
    assert [k for k, _ in listed_equations("routh", reduced.card.data)] == entry["keys"]


# --- the tool group (maintainer's decision: a group of its own, so the mechanics group is unchanged) -----------
def test_routh_reduce_is_in_its_own_group_and_not_in_mechanics() -> None:
    from minerva_api.agent.prompts import GROUP_TOOLS
    from minerva_api.agent.toolsets import group_of

    assert group_of("routh_reduce") == "reduction"
    assert GROUP_TOOLS["reduction"] == ("routh_reduce", "effective_potential")
    assert "routh_reduce" not in GROUP_TOOLS["mechanics"]


@pytest.mark.parametrize(
    "text",
    [
        "Reduce this Lagrangian by the cyclic coordinate t (Routh reduction).",
        "What is the Routhian of the Kepler problem?",
        "Remove the cyclic coordinates phi and t.",
    ],
)
def test_a_keyword_loads_the_group(text: str) -> None:
    from minerva_api.agent.toolsets import groups_in_text

    assert "reduction" in groups_in_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "Find the equations of motion of the pendulum Lagrangian.",
        "Which quantities are conserved along the geodesics? The coordinates are t, r, theta, phi.",
        "Is the angle coordinate cyclic here?",
    ],
)
def test_an_unrelated_mechanics_message_does_not(text: str) -> None:
    from minerva_api.agent.toolsets import groups_in_text

    assert "reduction" not in groups_in_text(text)


def test_load_tools_and_a_direct_call_load_it(ctx: ToolContext) -> None:
    """Without a keyword, the model reaches the tool through `load_tools` (the group is listed by name) or by
    calling it directly; the call runs and is answered."""
    from minerva_api.agent.toolsets import requested_groups

    assert requested_groups({"groups": ["reduction"]}) == ["reduction"]
    msg = play(ctx, [call("routh_reduce", ARGS), say("done")])
    assert msg["trace"][0]["status"] == "ok"
