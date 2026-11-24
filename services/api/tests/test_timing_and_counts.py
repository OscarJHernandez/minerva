"""A-007 (issue #171): what the trace says about time, queueing, counts and a failed claim check.

Timings are measurements, so these tests assert relations (a sum, a bound, `None` where nothing was
measured), never a wall-clock value.
"""

import asyncio
import json
from typing import Any

import pytest
from minerva_api.agent import loop as loop_module
from minerva_api.agent.evidence import measured_verify_ms
from minerva_api.agent.llm import LLMError
from minerva_api.agent.loop import AgentLimits, TurnInfo, run_agent
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext, execute
from minerva_api.agent.tracing import Tracer
from minerva_engine import Engine, SandboxCall
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

FAST = AgentLimits(retry_delay_seconds=0)
SIMPLIFY = {"latex": r"\frac{x^2-1}{x-1}"}
CLAIM = "It is $x + y = y + x$."
SECRET = "zzSECRETzz"


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def play(llm: Any, ctx: ToolContext, text: str = "q", **kw: Any) -> dict[str, Any]:
    async def go() -> list[dict[str, Any]]:
        return [
            e async for e in run_agent(llm, [{"role": "user", "content": text}], ctx, FAST, **kw)
        ]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")


def events(msg: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [e for e in msg["debug"]["events"] if e["event"] == name]


# ------------------------------------------------------------------ the claim check's failure site
def test_a_claim_check_that_raises_records_where_and_never_why(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: Any) -> Any:
        raise ValueError(f"cannot read {SECRET}")  # the message holds the user's formula

    monkeypatch.setattr(loop_module, "check_model_claims", broken)
    msg = play(ReplayLLM([say(CLAIM)]), ctx)
    debug = msg["debug"]
    assert debug["claim_check"] == "error"
    site = debug["claim_check_site"]
    assert site["error_type"] in {"ValueError", "other"}  # builtins are not a root we vouch for
    assert site.get("error_module", "").startswith("minerva_")
    claims = next(e for e in events(msg, "verification.completed") if e["scope"] == "claims")
    assert claims["reason"] == "error" and claims["accepted"] is False
    assert claims["error_type"] == site["error_type"] and claims["error_line"] == site["error_line"]
    assert SECRET not in json.dumps(debug["events"]) and SECRET not in json.dumps(site)
    assert msg["stop_reason"] == "completed"  # the turn is not turned into a failure by it


def test_a_claim_check_that_runs_out_of_time_is_a_timeout_not_an_error(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def too_slow(*_: Any) -> Any:
        raise TimeoutError

    monkeypatch.setattr(loop_module, "check_model_claims", too_slow)
    msg = play(ReplayLLM([say(CLAIM)]), ctx)
    assert msg["debug"]["claim_check"] == "timeout" and msg["debug"]["claim_check_site"] == {}
    claims = next(e for e in events(msg, "verification.completed") if e["scope"] == "claims")
    assert claims["reason"] == "timeout" and claims["error_type"] is None


def test_a_claim_check_that_works_has_no_site(ctx: ToolContext) -> None:
    msg = play(ReplayLLM([say(CLAIM)]), ctx)
    assert msg["debug"]["claim_check"] == "ok" and msg["debug"]["claim_check_site"] == {}


# ------------------------------------------------------------------------------------- verify_ms
def test_verify_ms_is_the_engine_measured_time_or_none(ctx: ToolContext) -> None:
    out = execute("substitute", {"latex": "x^2 + y^2 + z", "relations": ["y^2 = 1 - x^2"]}, ctx)
    assert out.status == "ok"
    measured = out.card.data["result"]["diagnostics"]["verify_ms"]
    assert measured_verify_ms(out.card.data) == measured  # the engine's own number, not engine_ms


def test_an_operation_that_measures_no_verification_has_verify_ms_none(ctx: ToolContext) -> None:
    msg = play(ReplayLLM([call("simplify", SIMPLIFY), say("It is $x + 1$.")]), ctx)
    tool_v = next(e for e in events(msg, "verification.completed") if e["scope"] == "tool")
    assert tool_v["verify_ms"] is None  # unavailable is said as None, never filled with engine_ms
    assert msg["trace"][0]["engine_ms"] is not None


def test_not_run_diagnostics_are_not_a_zero_millisecond_verification() -> None:
    data = {"diagnostics": {"verify_ms": 0, "verification_outcome": "not_run"}}
    assert measured_verify_ms(data) is None
    ran = {"diagnostics": {"verify_ms": 7, "verification_outcome": "passed"}}
    assert measured_verify_ms(ran) == 7
    assert measured_verify_ms({"result": ran}) == 7  # a card that wraps the engine result


# ---------------------------------------------------------------------------------------- queueing
def test_the_tracer_sums_queue_and_run_time_of_a_turns_sandbox_calls() -> None:
    tracer = Tracer("t")
    tracer.on_sandbox(SandboxCall("simplify", "ok", 12, "t", wait_ms=5, run_ms=7))
    tracer.on_sandbox(SandboxCall("simplify", "busy", 9, "t", wait_ms=9, run_ms=0))
    calls = tracer.sandbox_since(0)
    assert sum(c.ms for c in calls) == 21 and sum(c.wait_ms for c in calls) == 14
    assert all(c.ms == c.wait_ms + c.run_ms for c in calls)


def test_queue_time_is_part_of_engine_time_in_every_summary(ctx: ToolContext) -> None:
    msg = play(ReplayLLM([call("simplify", SIMPLIFY), say("It is $x + 1$.")]), ctx)
    done = events(msg, "tool.completed")[0]
    assert done["queue_ms"] is not None and 0 <= done["queue_ms"] <= done["engine_ms"]
    total = events(msg, "agent.completed")[0]
    assert total["queue_ms"] == done["queue_ms"] and total["queue_ms"] <= total["engine_ms"]


def test_a_tool_that_never_reached_the_sandbox_has_no_queue_time(ctx: ToolContext) -> None:
    msg = play(ReplayLLM([call("simplify", {}), say("Sorry.")]), ctx)  # refused before any call
    assert events(msg, "tool.completed")[0]["queue_ms"] is None
    assert events(msg, "agent.completed")[0]["queue_ms"] == 0


# ------------------------------------------------------------------------------------- the counts
def test_steps_are_model_calls_and_a_routed_call_is_counted_apart(ctx: ToolContext) -> None:
    pendulum = "1/2*m*R^2*(theta_dot^2 + sin(theta)^2*phi_dot^2) - m*g*R*cos(theta)"
    msg = play(
        ReplayLLM([say("Those are the equations.")]),
        ctx,
        f"Find the equations of motion for L = {pendulum}",
        info=TurnInfo(routing=True),
    )
    stopped = events(msg, "agent.stopped")[0]
    done = events(msg, "agent.completed")[0]
    assert stopped["steps"] == done["steps"] == len(msg["debug"]["llm_calls"]) == 1
    assert stopped["tool_calls"] == done["tool_calls"] == len(msg["trace"]) == 1
    assert stopped["routed_calls"] == done["routed_calls"] == msg["debug"]["routed_calls"] == 1
    assert msg["trace"][0]["routed"] is True


def test_an_unrouted_turn_counts_no_routed_calls(ctx: ToolContext) -> None:
    msg = play(ReplayLLM([call("simplify", SIMPLIFY), say("It is $x + 1$.")]), ctx)
    done = events(msg, "agent.completed")[0]
    assert done["steps"] == len(msg["debug"]["llm_calls"]) == 2 and done["tool_calls"] == 1
    assert done["routed_calls"] == 0


def test_a_failed_wrap_up_call_is_recorded_like_any_other_model_call(ctx: ToolContext) -> None:
    """Three identical calls stop the turn; the one call without tools that asks for a summary fails
    (and is retried once). Both attempts are in `llm_calls` and counted in `steps`."""
    llm = ReplayLLM([call("simplify", SIMPLIFY)] * 3)
    llm.fail_on = {3: LLMError("host down"), 4: LLMError("still down")}
    msg = play(llm, ctx)
    assert msg["stop_reason"] == "repeated_calls"
    wrap_up = msg["debug"]["llm_calls"][3:]
    assert wrap_up, "the failed wrap-up call is not recorded"
    assert all(c["finish_reason"] == "error" for c in wrap_up)
    assert events(msg, "agent.completed")[0]["steps"] == len(msg["debug"]["llm_calls"])
