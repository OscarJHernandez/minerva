"""A conserved charge of the geodesics that the model restates under its own name is linked to the verified row.

`geometry` with `geodesic_constants` gives each charge the role `conserved` and the coordinates as the system's
variables, so claim-evidence rule 3 (a named conserved quantity) applies; a wrong twin stays unchecked.
"""

import asyncio
from typing import Any

import pytest
from minerva_api.agent.loop import is_claim, run_agent
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext, run_geometry
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

POLAR = {
    "coordinates": ["r", r"\phi"],
    "metric": [["1", "0"], ["0", "r^2"]],
    "assumptions": {"r": ["positive"]},
    "quantities": ["killing_vectors", "geodesic_constants"],
}


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def turn(ctx: ToolContext, answer: str) -> dict[str, Any]:
    llm = ReplayLLM([call("geometry", POLAR), say(answer)])
    text = "What is conserved along the geodesics of the plane in polar coordinates?"

    async def go() -> list[dict[str, Any]]:
        return [e async for e in run_agent(llm, [{"role": "user", "content": text}], ctx)]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")


def marks(message: dict[str, Any]) -> dict[str, str]:
    return {
        s["value"]: s["status"]
        for b in message["blocks"]
        if b["kind"] == "text"
        for s in b["segments"]
        if s["type"] != "text" and is_claim(s["value"])
    }


def test_the_charges_are_conserved_roles_over_the_coordinates(ctx: ToolContext) -> None:
    from minerva_api.agent.tools import GeometryArgs

    out = run_geometry(GeometryArgs(**POLAR), ctx)
    assert out.status == "ok"
    assert out.variables == ("r", "phi")
    conserved = [f for f, role in out.formula_roles.items() if role == "conserved"]
    assert any(f.startswith("Q_{1} = ") for f in conserved)
    assert all(f in out.known_latex for f in conserved)
    assert "(cyclic coordinate)" in out.for_model and "(normalisation)" in out.for_model


def test_a_restated_charge_is_linked_by_its_name(ctx: ToolContext) -> None:
    message = turn(
        ctx, "The angular momentum $L = r^2\\dot\\phi$ is conserved along every geodesic."
    )
    status = marks(message)
    assert status == {"L = r^2\\dot\\phi": "from_tool"}, status
    assert {c["evidence_rule"] for c in message["debug"]["claims"]} == {"named_quantity"}


def test_a_wrong_charge_stays_unchecked(ctx: ToolContext) -> None:
    message = turn(ctx, "The angular momentum $L = r\\dot\\phi$ is conserved along every geodesic.")
    assert marks(message) == {"L = r\\dot\\phi": "unchecked"}
