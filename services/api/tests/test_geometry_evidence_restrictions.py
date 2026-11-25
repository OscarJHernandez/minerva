"""A restated geometry row carries the card's restrictions, in its own turn and in the next one.

The geometry card keeps its engine result, with `restrictions_latex` (zeros of `det g` and of denominators: where
the coordinate expressions hold), under `data["result"]`. `card_equations.result_restrictions` read only the top
level and the nested results of numbered cards, so the trace step of a geometry call recorded no restrictions and
a Christoffel row linked by `same_indexed_label`, or a Killing charge linked by `named_quantity`, was marked
supported without "It holds where ...". AGENTS.md: the result's restrictions travel with the match. These tests run
the real tool; the claim evidence unit tests build their records by hand and could not see it.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from minerva_api.agent.card_equations import result_formula, result_restrictions
from minerva_api.agent.loop import run_agent
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_api.agent.turn_context import turn_context
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_geometry_indexed_rows_agent import GOOD, SCHWARZSCHILD, WRONG_VALUE

# What the engine reports for Schwarzschild (written here, not read from the code under test).
HORIZON = r"- 2 M + r \neq 0"
AXIS = r"\sin{\left(\theta \right)} \neq 0"
SPHERE = {
    "coordinates": [r"\theta", r"\phi"],
    "metric": [["1", "0"], ["0", r"\sin^2\theta"]],
    "quantities": ["geodesic_constants"],
}
CHARGE = r"Q_{1} = \sin^2\theta \, \dot{\phi}"  # the cyclic-coordinate charge of the 2-sphere, factors reordered
WRONG_CHARGE = r"Q_{1} = \sin\theta \, \dot{\phi}"


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def run(
    ctx: ToolContext, history: list[dict[str, Any]], script: list[Any], **kwargs: Any
) -> dict[str, Any]:
    llm = ReplayLLM(script)

    async def go() -> list[dict[str, Any]]:
        return [e async for e in run_agent(llm, history, ctx, **kwargs)]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")


def notes(message: dict[str, Any]) -> dict[str, tuple[str, str | None]]:
    return {
        s["value"]: (s["status"], s.get("note"))
        for b in message["blocks"]
        if b["kind"] == "text"
        for s in b["segments"]
        if s["type"] != "text"
    }


CHRISTOFFEL_PROMPT = "What are the Christoffel symbols of the Schwarzschild metric?"


@pytest.fixture(scope="module")
def christoffel(ctx: ToolContext) -> dict[str, Any]:
    history = [{"role": "user", "content": CHRISTOFFEL_PROMPT}]
    return run(
        ctx, history, [call("geometry", SCHWARZSCHILD), say(f"So ${GOOD}$ and ${WRONG_VALUE}$.")]
    )


def test_the_trace_step_records_the_geometry_restrictions(christoffel: dict[str, Any]) -> None:
    (step,) = [s for s in christoffel["trace"] if s["name"] == "geometry"]
    assert step["status"] == "ok"
    assert HORIZON in step["restrictions"] and AXIS in step["restrictions"], step["restrictions"]


def test_a_restated_christoffel_row_says_where_it_holds(christoffel: dict[str, Any]) -> None:
    status, note = notes(christoffel)[GOOD]
    assert status == "from_tool"
    assert note is not None and "It holds where" in note and HORIZON in note and AXIS in note, note
    # the wrong twin is still unchecked: the restrictions widen nothing
    assert notes(christoffel)[WRONG_VALUE][0] == "unchecked"


def test_the_next_turn_carries_the_restrictions_through_the_stored_trace(
    ctx: ToolContext, christoffel: dict[str, Any]
) -> None:
    stored = [
        SimpleNamespace(role="user", content={"text": CHRISTOFFEL_PROMPT}),
        SimpleNamespace(role="assistant", content=christoffel),
    ]
    prompt = "Repeat the radial one."
    turn = turn_context(stored, prompt)
    rows = [r for r in turn.prior_evidence if r.get("restrictions")]
    assert rows and HORIZON in rows[0]["restrictions"]
    message = run(ctx, turn.history, [say(f"It is ${GOOD}$.")], **turn.agent_kwargs())
    status, note = notes(message)[GOOD]
    assert status == "from_tool"
    assert note is not None and HORIZON in note and AXIS in note, note


def test_a_restated_killing_charge_says_where_it_holds(ctx: ToolContext) -> None:
    history = [{"role": "user", "content": "What is conserved along geodesics of the unit sphere?"}]
    message = run(
        ctx, history, [call("geometry", SPHERE), say(f"One is ${CHARGE}$, not ${WRONG_CHARGE}$.")]
    )
    found = notes(message)
    status, note = found[CHARGE]
    assert status == "from_tool", found
    assert note is not None and "It holds where" in note and AXIS in note, note
    assert found[WRONG_CHARGE][0] == "unchecked"


def test_the_reader_takes_the_nested_restrictions_of_geometry_and_vector_cards_only() -> None:
    nested = {"result": {"status": "ok", "restrictions_latex": [AXIS], "result_latex": "x"}}
    assert result_restrictions("geometry", nested) == [AXIS]
    assert result_restrictions("vector", nested) == [AXIS]
    # a card type that does not keep its result there is not read from it
    assert result_restrictions("lagrangian", nested) == []
    # the top level still wins, and the numbering is unchanged: neither card is numbered as a whole
    assert result_restrictions("geometry", {**nested, "restrictions_latex": [HORIZON]}) == [HORIZON]
    assert result_formula("geometry", nested) is None
    assert result_formula("vector", nested) is None
