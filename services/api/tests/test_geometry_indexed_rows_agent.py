"""A Christoffel symbol the model restates by its label is linked to the verified geometry row (#146 part 4).

The geometry card lists `\\Gamma^{r}{}_{t t} = ...`; the reply writes `\\Gamma^r_{tt} = \\frac{M}{r^2}(1 - 2M/r)`. The
label is compared exactly (kernel and indices in order) and the value by the claim checker's equivalence test, so the
reply's formula is marked `from_tool` with `evidence_rule: "same_indexed_label"`. A wrong value, a swapped index pair
and a label no row lists stay unchecked (never contradicted, never unparsed).
"""

import asyncio
from typing import Any

import pytest
from minerva_api.agent.loop import is_claim, run_agent
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

SCHWARZSCHILD = {
    "coordinates": ["t", "r", r"\theta", r"\phi"],
    "metric": [
        ["-(1-2M/r)", "0", "0", "0"],
        ["0", "1/(1-2M/r)", "0", "0"],
        ["0", "0", "r^2", "0"],
        ["0", "0", "0", r"r^2 \sin^2\theta"],
    ],
    "assumptions": {"M": ["positive"], "r": ["positive"]},
    "quantities": ["christoffel"],
}
GOOD = r"\Gamma^r_{tt} = \frac{M}{r^2}\left(1 - \frac{2M}{r}\right)"
ALSO_GOOD = r"\Gamma^\phi_{\theta\phi} = \frac{\cos\theta}{\sin\theta}"
WRONG_VALUE = r"\Gamma^r_{tt} = \frac{M}{r^2}"
SWAPPED = r"\Gamma^\theta_{\theta r} = \frac{1}{r}"
NO_ROW = r"\Gamma^t_{\theta\theta} = 0"


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


@pytest.fixture(scope="module")
def message(ctx: ToolContext) -> dict[str, Any]:
    answer = (
        f"The radial component is ${GOOD}$ and ${ALSO_GOOD}$. "
        f"Some say ${WRONG_VALUE}$, others ${SWAPPED}$, and ${NO_ROW}$."
    )
    llm = ReplayLLM([call("geometry", SCHWARZSCHILD), say(answer)])
    text = "What are the Christoffel symbols of the Schwarzschild metric?"

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


def claims(message: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["value"]: c for c in message["debug"]["claims"]}


def test_a_restated_christoffel_row_is_from_tool(message: dict[str, Any]) -> None:
    status = marks(message)
    assert status[GOOD] == "from_tool", status
    assert status[ALSO_GOOD] == "from_tool", status
    debug = claims(message)
    for good in (GOOD, ALSO_GOOD):
        assert debug[good]["status"] == "supported"
        assert debug[good]["evidence_rule"] == "same_indexed_label"
        # geometry results have no `@rN` id, so the link names no source


def test_wrong_rows_stay_unchecked_never_contradicted_or_unparsed(message: dict[str, Any]) -> None:
    status = marks(message)
    debug = claims(message)
    for wrong in (WRONG_VALUE, SWAPPED, NO_ROW):
        assert status[wrong] == "unchecked", status
        assert debug[wrong]["status"] == "unverified", debug[wrong]
        assert debug[wrong]["evidence_rule"] is None
