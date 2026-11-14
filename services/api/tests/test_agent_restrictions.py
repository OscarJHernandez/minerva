"""Restrictions travel with a verified result (V-015).

`simplify` cancelling (x-1) reports x != 1. A later step that reuses the result as `@r1` computes something
about the simplified expression, which is defined at x = 1; the original expression was not. The new result must
say it is valid only where the earlier one was, and that must survive a chain of steps and a new turn.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from minerva_api.agent.loop import AgentLimits, run_agent
from minerva_api.agent.refs import ResultStore, inherited_note
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_api.routers_chat import _history
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

FAST = AgentLimits(retry_delay_seconds=0)
RESTRICTION = r"x \neq 1"


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def play(llm: Any, ctx: ToolContext, **kw: Any) -> dict[str, Any]:
    async def go() -> list[dict[str, Any]]:
        return [
            e async for e in run_agent(llm, [{"role": "user", "content": "q"}], ctx, FAST, **kw)
        ]

    events = asyncio.run(go())
    return next(e["data"] for e in events if e["event"] == "message")


def cards(msg: dict[str, Any]) -> list[dict[str, Any]]:
    return [b["card"] for b in msg["blocks"] if b["kind"] == "card"]


def test_the_first_step_reports_its_own_restriction(ctx: ToolContext) -> None:
    msg = play(ReplayLLM([call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}), say("ok")]), ctx)
    assert cards(msg)[0]["data"]["restrictions_latex"] == [RESTRICTION]
    assert msg["trace"][0]["result_id"] == "r1"


def test_a_step_that_reuses_the_result_inherits_its_restriction(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}),
            call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}),
            say("done"),
        ]
    )
    msg = play(llm, ctx)
    second = msg["trace"][1]
    assert second["status"] == "ok"
    assert "Inherited from @r1" in second["for_model"] and RESTRICTION in second["for_model"]
    assert cards(msg)[1]["data"]["inherited_restrictions_latex"] == [RESTRICTION]
    # the model sees it too, in the message that follows the tool call
    assert RESTRICTION in llm.seen[2][-1]["content"]


def test_the_restriction_survives_a_chain_of_steps(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}),
            call(
                "differentiate", {"latex": "@r1", "variables": [{"name": "x"}], "assumptions": {}}
            ),
            call("integrate", {"latex": "@r2 + x", "variable": "x"}),
            say("done"),
        ]
    )
    msg = play(llm, ctx)
    third = msg["trace"][2]
    assert third["status"] == "ok", third["for_model"]
    assert "Inherited from @r2" in third["for_model"] and RESTRICTION in third["for_model"]


def test_a_result_with_no_restriction_adds_nothing(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call("integrate", {"latex": r"x \cos x", "variable": "x"}),
            call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}),
            say("done"),
        ]
    )
    msg = play(llm, ctx)
    assert "Inherited" not in msg["trace"][1]["for_model"]
    assert "inherited_restrictions_latex" not in cards(msg)[1]["data"]


def test_a_restriction_the_new_result_reports_itself_is_not_repeated(ctx: ToolContext) -> None:
    store = ResultStore(results={"r1": "x"}, restrictions={"r1": [RESTRICTION]})
    resolution = store.resolve({"latex": "@r1"})
    assert resolution.inherited == [RESTRICTION] and resolution.inherited_from == ["@r1"]
    assert store.resolve({"latex": "x"}).inherited == []  # no reference, nothing inherited
    both = ResultStore(
        results={"r1": "a", "r2": "b"},
        restrictions={"r1": [RESTRICTION], "r2": [RESTRICTION, "y > 0"]},
    ).resolve({"latex": "@r1 + @r2"})
    assert both.inherited == [RESTRICTION, "y > 0"]  # the union, in order, without repeats
    assert both.inherited_from == ["@r1", "@r2"]


def test_restrictions_carry_into_the_next_turn(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}), say("done")]
    )
    msg = play(llm, ctx, prior_results={"r1": "x + 1"}, prior_restrictions={"r1": [RESTRICTION]})
    assert RESTRICTION in msg["trace"][0]["for_model"]
    plain = play(
        ReplayLLM(
            [call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}), say("ok")]
        ),
        ctx,
        prior_results={"r1": "x + 1"},
    )
    assert (
        "Inherited" not in plain["trace"][0]["for_model"]
    )  # none were recorded, none are invented


def test_the_stored_conversation_rebuilds_the_restrictions() -> None:
    """`_history` reads them back from the cards of earlier messages (own and inherited)."""
    stored = SimpleNamespace(
        role="assistant",
        content={
            "text": "done",
            "blocks": [
                {
                    "kind": "card",
                    "card": {
                        "id": "a1",
                        "data": {"result_latex": "x + 1", "restrictions_latex": [RESTRICTION]},
                    },
                },
                {
                    "kind": "card",
                    "card": {
                        "id": "a2",
                        "data": {
                            "result_latex": "1",
                            "inherited_restrictions_latex": [RESTRICTION],
                        },
                    },
                },
            ],
            "trace": [
                {
                    "id": "a1",
                    "name": "simplify",
                    "arguments": {},
                    "for_model": "r",
                    "result_id": "r1",
                },
                {
                    "id": "a2",
                    "name": "differentiate",
                    "arguments": {},
                    "for_model": "r",
                    "result_id": "r2",
                },
            ],
        },
    )
    *_, results, restrictions = _history([stored])  # type: ignore[arg-type]
    assert results == {"r1": "x + 1", "r2": "1"}
    assert restrictions == {"r1": [RESTRICTION], "r2": [RESTRICTION]}


def test_the_note_names_its_sources_and_every_restriction() -> None:
    note = inherited_note([RESTRICTION, "y > 0"], ["@r1", "@r2"])
    assert "@r1, @r2" in note and f"${RESTRICTION}$" in note and "$y > 0$" in note
