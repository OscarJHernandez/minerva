"""Defects found by the reliability audit, each pinned by a test that failed before its fix (A-001 to A-005 in
`docs/verification-defects.md`): what a routed call costs, what a recovered call registers as seen, which
context a tool reads, what a stopped turn keeps."""

import asyncio
from typing import Any

import pytest
from minerva_api.agent import loop as loop_module
from minerva_api.agent.loop import AgentLimits, TurnInfo, run_agent
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

from .test_agent_reliability import final, play
from .test_routing import PENDULUM_TEXT, run

WEDGE = "1/2*m*(Xdot^2 + Ydot^2) - m*g*Y"
ARGS = {"lagrangian": WEDGE, "coordinates": ["X", "Y"]}
ANSWER = "They fall freely."


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def events(message: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [e for e in message["debug"]["events"] if e["event"] == name]


# -- D1: a routed call computed before the first step, and its cost went missing -------------------------


def test_a_routed_call_reports_what_it_cost(ctx: ToolContext) -> None:
    llm = ReplayLLM([say(ANSWER)])
    message = run(llm, ctx, f"Find the equations of motion for L = {PENDULUM_TEXT}")
    (step,) = message["trace"]
    assert step["routed"] is True
    assert step["duration_ms"] > 0 and step["engine_ms"] is not None and step["engine_ms"] > 0
    (completed,) = events(message, "tool.completed")
    assert completed["duration_ms"] > 0 and completed["sandbox_calls"] > 0
    (done,) = events(message, "agent.completed")
    assert done["tool_ms"] > 0 and done["engine_ms"] > 0  # the totals include the routed call


def test_a_declined_route_has_a_duration_and_uses_the_tool_call_budget(ctx: ToolContext) -> None:
    declined = r"Find the equations of motion for $L = \frac12 m \dot{x}^2 + \operatorname{erf}(x)$"
    llm = ReplayLLM([say("That function is not supported here.")])
    message = run(llm, ctx, declined)
    assert message["debug"]["routing"]["decision"] == "declined"
    assert message["debug"]["routing"]["duration_ms"] > 0

    async def capped() -> dict[str, Any]:
        replies = [call("simplify", {"latex": "x+x"}), say("Done.")]
        async for e in run_agent(
            ReplayLLM(replies),
            [{"role": "user", "content": declined}],
            ctx,
            AgentLimits(retry_delay_seconds=0, max_tool_calls=1),
            info=TurnInfo(routing=True),
        ):
            if e["event"] == "message":
                return e["data"]  # type: ignore[no-any-return]
        raise AssertionError("no message")

    # the declined route already ran one call, so the model's own call is over the limit of one
    assert asyncio.run(capped())["stop_reason"] == "tool_call_limit"


def test_an_unknown_route_reason_cannot_end_the_turn(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minerva_api.agent import routing

    real = routing.plan_route

    def odd(text: str) -> Any:
        route = real(text)
        if route is not None:
            object.__setattr__(route, "reason", "not-in-the-vocabulary")
        return route

    monkeypatch.setattr(loop_module, "plan_route", odd)
    llm = ReplayLLM([say(ANSWER)])
    message = run(llm, ctx, f"Find the equations of motion for L = {PENDULUM_TEXT}")
    assert message["stop_reason"] == "completed"
    assert message["debug"]["routing"]["decision"] == "declined"


# -- D2: a recovered call registers both spellings ---------------------------------------------------------


def test_the_corrected_call_the_model_then_sends_is_not_run_again(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[dict[str, Any]] = []
    real = loop_module.execute

    def counting(name: str, arguments: dict[str, Any], c: ToolContext) -> Any:
        runs.append(arguments)
        return real(name, arguments, c)

    monkeypatch.setattr(loop_module, "execute", counting)
    llm = ReplayLLM(
        [
            call("analyze_lagrangian", {**ARGS, "format": "latex"}),  # recovered by the loop
            call(
                "analyze_lagrangian", {**ARGS, "format": "text"}
            ),  # the model sends the fix itself
            say(ANSWER),
        ]
    )
    message = final(play(llm, ctx))
    first, second = message["trace"]
    assert first["policy"] == "format_switch" and first["status"] == "ok"
    assert second["repeated"] is True and second["status"] == "skipped"
    assert len(runs) == 2  # the original and the recovery; never a third run


# -- D3: a tool that timed out must not leave its child process running -----------------------------------


def _never_answers(conn: Any, op: str, payload: dict[str, Any], cpu: int, mem: int) -> None:
    import time

    time.sleep(30)


def test_a_tool_that_times_out_frees_the_sandbox_slot_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from minerva_engine import sandbox

    monkeypatch.setattr(sandbox, "_child_main", _never_answers)
    slow = ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=1)), kb=KnowledgeBase.load()
    )
    limits = AgentLimits(retry_delay_seconds=0, max_seconds=2)
    message = final(play(ReplayLLM([call("simplify", {"latex": "x+x"})]), slow, limits=limits))
    assert message["stop_reason"] == "time_limit"
    assert message["trace"][0]["error_category"] == "timeout"
    # the abandoned worker notices within a slice; it did not wait out the sandbox's 30 s limit
    freed = False
    for _ in range(150):
        if slow.engine.sandbox._slots.acquire(blocking=False):
            slow.engine.sandbox._slots.release()
            freed = True
            break
        time.sleep(0.1)
    assert freed


# -- D4: each call reads its own copy of the context -------------------------------------------------------


def test_a_tool_gets_its_own_context_and_the_shared_one_is_never_marked(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[ToolContext] = []
    real = loop_module.execute

    def spy(name: str, arguments: dict[str, Any], c: ToolContext) -> Any:
        seen.append(c)
        return real(name, arguments, c)

    monkeypatch.setattr(loop_module, "execute", spy)
    llm = ReplayLLM(
        [call("simplify", {"latex": "x+x"}), call("simplify", {"latex": "y+y"}), say("Ok.")]
    )
    final(play(llm, ctx, text="Let A and B be matrices. Simplify x+x."))
    assert len(seen) == 2 and seen[0] is not ctx and seen[0] is not seen[1]
    assert (
        ctx.matrices == frozenset() and ctx.matrix_sources == {}
    )  # the shared context is untouched


# -- an empty model call is asked again once -------------------------------------------------------------


def test_a_model_call_with_nothing_visible_is_asked_again_once_and_the_answer_is_kept(
    ctx: ToolContext,
) -> None:
    """Reasoning that never closes, or an empty stop, left the user an empty reply after a tool had run."""
    llm = ReplayLLM([call("simplify", {"latex": "x+x"}), say(""), say("The result is 2x.")])
    message = final(play(llm, ctx, limits=AgentLimits(retry_delay_seconds=0)))
    assert message["text"].strip() == "The result is 2x."
    retries = events(message, "agent.retry")
    assert [r["reason"] for r in retries] == ["empty_reply"]
    assert message["stop_reason"] == "completed"
    assert len(message["trace"]) == 1  # the tool was not run again


def test_a_second_empty_call_ends_the_turn_and_nothing_is_retried_again(ctx: ToolContext) -> None:
    llm = ReplayLLM([say(""), say(""), say("never reached")])
    message = final(play(llm, ctx, limits=AgentLimits(retry_delay_seconds=0)))
    assert message["text"].strip() == ""
    assert [r["reason"] for r in events(message, "agent.retry")] == ["empty_reply"]
    assert message["trace"] == []
