"""A Lagrangian plus a request for its equations of motion runs `analyze_lagrangian` without a model call.

Both signals are required; a failing route leaves the turn exactly as if no rule existed; the model still
writes the answer and every formula in it is checked like any other.
"""

import asyncio
from typing import Any

import pytest
from minerva_api.agent.loop import AgentLimits, TurnInfo, run_agent
from minerva_api.agent.replay import ReplayLLM, say
from minerva_api.agent.routing import plan_route
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

PENDULUM_LATEX = (
    r"\frac12 m R^2 (\dot{\theta}^2 + \sin^2(\theta) \dot{\phi}^2) - m g R \cos(\theta)"
)
PENDULUM_TEXT = "1/2*m*R^2*(theta_dot^2 + sin(theta)^2*phi_dot^2) - m*g*R*cos(theta)"


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


@pytest.mark.parametrize(
    ("message", "coordinates"),
    [
        (f"Find the equations of motion for $L = {PENDULUM_LATEX}$", ["theta", "phi"]),
        (f"What are the equations of motion for L = {PENDULUM_TEXT}", ["theta", "phi"]),
        (f"Equations of motion for\n\nL = {PENDULUM_TEXT}\n", ["theta", "phi"]),
        (r"Derive the Euler-Lagrange equations for $\frac12 m\dot x^2 - \frac12 k x^2$", ["x"]),
        ("equation of motion for $L=\\frac12 m \\dot{q}^2 - V(q)$", ["q"]),
    ],
)
def test_the_rule_reads_every_spelling_of_the_request(message: str, coordinates: list[str]) -> None:
    route = plan_route(message)
    assert route is not None and route.tool == "analyze_lagrangian"
    assert route.arguments["coordinates"] == coordinates


@pytest.mark.parametrize(
    "message",
    [
        f"Simplify $L = {PENDULUM_LATEX}$",  # a Lagrangian but no request for equations of motion
        "What are the equations of motion of a double pendulum?",  # a request but no formula
        r"Find the equations of motion for $\frac12 m \dot{q}_1^2 + \frac12 m\dot{q}_2^2$",  # indexed
        r"Derive the Hamiltonian and the equations of motion for $L=\frac12 m\dot x^2$",
        r"Equations of motion for $L=\frac12m\dot x^2$ and also for $L=\frac12 k\dot y^2$",  # two
        r"Equations of motion for $\ddot{x}^2 + x$",  # accelerations
        r"Equations of motion for $x^2 + 1$",  # no velocity
    ],
)
def test_one_signal_alone_never_routes(message: str) -> None:
    assert plan_route(message) is None


def run(llm: ReplayLLM, ctx: ToolContext, message: str, routing: bool = True) -> dict[str, Any]:
    async def go() -> list[dict[str, Any]]:
        return [
            e
            async for e in run_agent(
                llm,
                [{"role": "user", "content": message}],
                ctx,
                AgentLimits(retry_delay_seconds=0),
                info=TurnInfo(routing=routing),
            )
        ]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")


ANSWER = r"The equations are $\ddot\phi = -2\cot\theta\,\dot\theta\dot\phi$ and the rest follows."


def test_a_routed_turn_has_one_tool_call_and_one_model_call(ctx: ToolContext) -> None:
    llm = ReplayLLM([say(ANSWER)])
    message = run(llm, ctx, f"Find the equations of motion for L = {PENDULUM_TEXT}")
    trace = message["trace"]
    assert [s["name"] for s in trace] == ["analyze_lagrangian"]
    assert trace[0]["routed"] is True and trace[0]["status"] == "ok"
    assert len(message["debug"]["llm_calls"]) == 1  # the model only wrote the answer
    assert message["debug"]["routing"] == {
        "decision": "routed",
        "reason": "lagrangian_equations",
        "tool": "analyze_lagrangian",
    }
    # the model saw its own (routed) call and the verified result before writing
    roles = [m["role"] for m in llm.seen[0]]
    assert roles[-2:] == ["assistant", "tool"]
    assert "equation of motion" in llm.seen[0][-1]["content"]
    # and the sentence it wrote is linked to the engine's result
    claim = next(
        s
        for b in message["blocks"]
        if b["kind"] == "text"
        for s in b["segments"]
        if s["type"] != "text"
    )
    assert claim["status"] == "from_tool"


def test_routing_off_leaves_the_model_to_plan(ctx: ToolContext) -> None:
    llm = ReplayLLM([say("I would need to look at that.")])
    message = run(llm, ctx, f"Find the equations of motion for L = {PENDULUM_TEXT}", routing=False)
    assert message["trace"] == [] and message["debug"]["routing"] == {}


def test_a_route_the_engine_declines_is_dropped_and_the_model_plans(ctx: ToolContext) -> None:
    llm = ReplayLLM([say("That function is not supported here.")])
    message = run(
        llm,
        ctx,
        r"Find the equations of motion for $L = \frac12 m \dot{x}^2 + \operatorname{erf}(x)$",
    )
    assert message["trace"] == []  # the refused call is not shown as the model's
    assert message["debug"]["routing"]["decision"] == "declined"
    assert [m["role"] for m in llm.seen[0]][-1] == "user"  # the model plans from the user's message
    assert len(message["debug"]["llm_calls"]) == 1
