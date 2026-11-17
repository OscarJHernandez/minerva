"""Faults injected below the loop: a model that stalls, a host that answers 5xx, a sandbox child that dies or
hangs, a client that leaves during a tool, two turns at once. Each must end a bounded turn with one stop
reason, free what it held, and never mark a formula verified that no tool verified."""

import asyncio
import os
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from minerva_api.agent.llm import Chunk, LLMError
from minerva_api.agent.loop import AgentLimits, run_agent
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine, Limits, sandbox
from minerva_knowledge import KnowledgeBase

from .test_agent_reliability import cards, final, play, segments

STOP_REASONS = {
    "completed",
    "step_limit",
    "tool_call_limit",
    "time_limit",
    "repeated_calls",
    "output_limit",
    "model_error",
    "malformed_output",
    "internal_error",
}
FAST = AgentLimits(retry_delay_seconds=0)


def _nothing_verified_was_claimed(message: dict[str, Any]) -> None:
    marks = {s["status"] for s in segments(message) if s["type"] != "text"}
    assert "from_tool" not in marks and "verified" not in marks


def _slow_child(conn: Any, op: str, payload: dict[str, Any], cpu: int, mem: int) -> None:
    time.sleep(30)


def _dying_child(conn: Any, op: str, payload: dict[str, Any], cpu: int, mem: int) -> None:
    os._exit(1)  # the child vanishes without answering


def _slot_is_free(engine: Engine, wait: float = 4.0) -> bool:
    end = time.monotonic() + wait
    while time.monotonic() < end:
        if engine.sandbox._slots.acquire(blocking=False):
            engine.sandbox._slots.release()
            return True
        time.sleep(0.1)
    return False


def _context(timeout: float = 30) -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=timeout, max_concurrency=1)), kb=KnowledgeBase.load()
    )


# -- the model host -------------------------------------------------------------------------------------


class _Stalls(ReplayLLM):
    """Streams a few words and then never produces another chunk."""

    async def chat(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AsyncIterator[Chunk]:
        self.seen.append([dict(m) for m in messages])
        yield Chunk(text="The derivative")
        await asyncio.sleep(60)
        yield Chunk(text=" is 2x.")


def test_a_model_that_stalls_mid_stream_ends_at_the_time_limit() -> None:
    started = time.monotonic()
    message = final(
        play(_Stalls(), _context(), limits=AgentLimits(retry_delay_seconds=0, max_seconds=2))
    )
    assert time.monotonic() - started < 15
    assert message["stop_reason"] in STOP_REASONS
    assert message["stop_reason"] == "time_limit"
    _nothing_verified_was_claimed(message)


@pytest.mark.parametrize("kind", ["unavailable", "rate_limit", "timeout", "bad_reply"])
def test_a_host_that_keeps_failing_ends_the_turn_with_no_result(kind: str) -> None:
    llm = ReplayLLM()
    llm.error = LLMError("The model host did not answer.", kind=kind)
    message = final(play(llm, _context()))
    assert message["stop_reason"] == "model_error"
    assert len(llm.seen) == 2  # the call and its one retry, never a third
    assert cards(message) == []
    _nothing_verified_was_claimed(message)


def test_a_host_failure_after_a_verified_result_keeps_the_result_and_marks_nothing_else() -> None:
    llm = ReplayLLM([call("simplify", {"latex": r"\frac{x^2-1}{x-1}"})])
    down = LLMError("The model host is not reachable.", kind="unavailable")
    llm.fail_on = {1: down, 2: down}  # the next call and its one retry
    ctx = _context()
    message = final(play(llm, ctx))
    assert message["stop_reason"] == "model_error"
    assert len(cards(message)) == 1


# -- the sandbox ----------------------------------------------------------------------------------------


def test_a_sandbox_child_that_dies_is_an_error_card_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox, "_child_main", _dying_child)
    ctx = _context()
    llm = ReplayLLM([call("simplify", {"latex": "x+x"}), say("It could not be computed.")])
    message = final(play(llm, ctx))
    (step,) = message["trace"]
    assert step["status"] != "ok" and message["stop_reason"] in STOP_REASONS
    assert cards(message)[0]["type"] == "error" or step["status"] in {"error", "resource_limit"}
    assert _slot_is_free(ctx.engine)
    _nothing_verified_was_claimed(message)


def test_a_child_that_hangs_is_killed_at_the_sandbox_limit_and_the_turn_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox, "_child_main", _slow_child)
    ctx = _context(timeout=1)
    llm = ReplayLLM([call("simplify", {"latex": "x+x"}), say("That took too long.")])
    started = time.monotonic()
    message = final(play(llm, ctx))
    assert time.monotonic() - started < 20
    (step,) = message["trace"]
    assert step["status"] == "resource_limit" and step["error_category"] == "timeout"
    assert message["stop_reason"] == "completed"
    assert _slot_is_free(ctx.engine)
    _nothing_verified_was_claimed(message)


# -- the client -----------------------------------------------------------------------------------------


def test_a_client_that_leaves_during_a_tool_leaves_no_child_and_no_slot_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox, "_child_main", _slow_child)
    ctx = _context(timeout=30)

    async def go() -> None:
        llm = ReplayLLM([call("simplify", {"latex": "x+x"}), say("Done.")])
        stream = run_agent(llm, [{"role": "user", "content": "q"}], ctx, FAST)

        async def consume() -> None:
            async for _ in stream:
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(1.5)  # the tool is running in its child
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    started = time.monotonic()
    asyncio.run(go())
    assert _slot_is_free(ctx.engine, wait=6)
    assert time.monotonic() - started < 20  # not the child's 30 s


# -- two turns at once ----------------------------------------------------------------------------------


def test_two_turns_at_once_do_not_share_declarations_or_results() -> None:
    ctx = ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )

    async def turn(text: str, expr: str) -> dict[str, Any]:
        llm = ReplayLLM([call("simplify", {"latex": expr}), say("Done.")])
        events = [e async for e in run_agent(llm, [{"role": "user", "content": text}], ctx, FAST)]
        return final(events)

    async def both() -> tuple[dict[str, Any], dict[str, Any]]:
        return await asyncio.gather(
            turn("Let A and B be matrices. Simplify.", "A+A"),
            turn("Simplify.", r"\frac{x^2-1}{x-1}"),
        )

    first, second = asyncio.run(both())
    assert first["trace"][0]["status"] != "ok"  # A is a matrix: the scalar tool refuses it
    assert second["trace"][0]["status"] == "ok"
    assert ctx.matrices == frozenset()
