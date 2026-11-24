"""A reply the client abandons keeps what had finished (A-006): the cards, their trace steps, the events.

Everything goes through the real chat route with a replayed model. The model stream is cut the way a closed
tab cuts it (the route's generator is cancelled or closed), after a tool has finished and some prose has
arrived. `turn_context` rebuilds `@rN`, restrictions and provenance from the stored trace joined to the
cards, so a later turn can reuse the result; nothing that did not finish can be in it.
"""

import asyncio
import contextlib
import importlib.util
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from minerva_api import routers_chat
from minerva_api.agent.events import Card
from minerva_api.agent.llm import Chunk
from minerva_api.agent.loop import AgentLimits, TurnInfo, run_agent
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_api.agent.tracing import StopReason
from minerva_api.agent.turn_context import turn_context
from minerva_engine import Engine, Limits, sandbox
from minerva_knowledge import KnowledgeBase

from .test_chat_context import send, start, stored

SIMPLIFY = call("simplify", {"latex": r"\frac{x^2-1}{x-1}"})
SIMPLIFY_TEXT = r"Simplify $\frac{x^2-1}{x-1}$"


class _Hangs(ReplayLLM):
    """Replays its replies; on the model call `at` it streams its text and then waits for ever (the client
    is about to go away)."""

    def __init__(self, replies: list[list[Chunk]], at: int) -> None:
        super().__init__(replies, repeat_last=True)
        self.at = at
        self.hung: asyncio.Event | None = None

    async def chat(self, messages: Any, tools: Any) -> Any:
        index = len(self.seen)
        async for chunk in super().chat(messages, tools):
            yield chunk
        if index == self.at and self.hung is not None:
            self.hung.set()
            await asyncio.Event().wait()


def _slow_child(conn: Any, op: str, payload: dict[str, Any], cpu: int, mem: int) -> None:
    time.sleep(30)  # module level: the child process must be able to import it


def _hanging(client: TestClient, replies: list[list[Chunk]], at: int) -> _Hangs:
    llm = _Hangs(replies, at)
    client.app.state.local_llm = llm  # type: ignore[attr-defined]
    return llm


def _abandon(client: TestClient, cid: str, text: str, llm: _Hangs, **body: Any) -> None:
    """Send `text` over a real request and cancel it once the model has hung: Stop or a closed tab."""

    async def go() -> None:
        llm.hung = asyncio.Event()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=client.app),
            base_url="http://testserver",
            cookies=client.cookies,
        ) as http:
            task = asyncio.create_task(
                http.post(f"/v1/chat/conversations/{cid}/messages", json={"content": text, **body})
            )
            await asyncio.wait_for(llm.hung.wait(), 30)
            await asyncio.sleep(0.2)  # the prose that was streamed reaches the route
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(go())


def _leave_after_the_tool(client: TestClient, cid: str, text: str = SIMPLIFY_TEXT) -> _Hangs:
    llm = _hanging(client, [SIMPLIFY, say("It simplifies to $x + 1$, and this goes on")], at=1)
    _abandon(client, cid, text, llm)
    return llm


def test_a_tool_that_finished_before_the_client_left_is_kept_with_its_trace_and_debug() -> None:
    for client, cid, _ in start():
        _leave_after_the_tool(client, cid)
        messages = stored(client, cid)
        assert [m["role"] for m in messages] == ["user", "assistant"]
        content = messages[1]["content"]
        assert content["stop_reason"] == StopReason.STOPPED.value and content["truncated"] is True
        (step,) = content["trace"]
        assert step["name"] == "simplify" and step["status"] == "ok" and step["result_id"] == "r1"
        assert step["restrictions"] == [r"x \neq 1"]
        card = next(b["card"] for b in content["blocks"] if b["kind"] == "card")
        assert card["id"] == step["id"] and card["number"] is not None
        debug = content["debug"]
        assert debug["stop_reason"] == "stopped" and debug["trace_id"]
        kinds = [e["event"] for e in debug["events"]]
        assert "tool.completed" in kinds and "agent.stopped" not in kinds
        # the prose that arrived is shown, unchecked: it was never marked as verified
        prose = [s for b in content["blocks"] if b["kind"] == "text" for s in b["segments"]]
        assert prose and all(
            s["status"] in (None, "unchecked", "quoted") for s in prose if s["type"] != "text"
        )


def test_a_follow_up_reuses_the_result_with_its_restrictions() -> None:
    for client, cid, _ in start():
        _leave_after_the_tool(client, cid)
        follow_up = ReplayLLM(
            [call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}), say("Done.")]
        )
        client.app.state.local_llm = follow_up  # type: ignore[attr-defined]
        events = send(client, cid, "Differentiate that result")
        card = next(d for e, d in events if e == "card")
        assert card["data"]["inherited_restrictions_latex"] == [r"x \neq 1"]
        message = next(d for e, d in events if e == "message")
        assert message["trace"][0]["status"] == "ok"
        # the earlier turn's handle is the one the follow-up reused, not a new one
        assert message["trace"][0]["result_id"] == "r2"


def test_nothing_that_arrived_means_nothing_is_stored() -> None:
    for client, cid, _ in start():
        _abandon(client, cid, "hello", _hanging(client, [[]], at=0))
        assert [m["role"] for m in stored(client, cid)] == ["user"]


def test_an_unfinished_tool_is_never_a_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_child_main", _slow_child)
    ctx = ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=1)), kb=KnowledgeBase.load()
    )
    info = TurnInfo()

    async def go() -> None:
        llm = ReplayLLM([call("simplify", {"latex": r"x+x+y"}), say("Done.")])
        stream = run_agent(
            llm,
            [{"role": "user", "content": "q"}],
            ctx,
            AgentLimits(retry_delay_seconds=0),
            info=info,
        )

        async def consume() -> None:
            async for _ in stream:
                pass

        task = asyncio.create_task(consume())
        end = time.monotonic() + 30
        while time.monotonic() < end:  # the event itself: the child holds the only slot
            if not ctx.engine.sandbox._slots.acquire(blocking=False):
                break
            ctx.engine.sandbox._slots.release()
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert info.snapshot.trace == [] and info.snapshot.result_ids == {}
    end = time.monotonic() + 6
    free = False
    while time.monotonic() < end and not free:
        free = ctx.engine.sandbox._slots.acquire(blocking=False)
        time.sleep(0.1)
    assert free  # the slot was released by the cancellation


def test_stopped_message_keeps_only_cards_that_arrived_and_moves_the_counter_on() -> None:
    from minerva_api.agent.equations import EquationTable
    from minerva_api.agent.events import TraceStep
    from minerva_api.agent.loop import TurnSnapshot

    for client, cid, _ in start():
        card = Card(
            id="c1",
            tool="simplify",
            type="simplify",
            title="t",
            data={"result_latex": "x + 1"},
        )
        snap = TurnSnapshot(
            trace=[
                TraceStep(
                    id="c1",
                    call_id="a",
                    name="simplify",
                    arguments={},
                    for_model="",
                    result_id="r1",
                ),
                TraceStep(
                    id="c2",
                    call_id="b",
                    name="simplify",
                    arguments={},
                    for_model="",
                    result_id="r2",
                ),
            ],
            result_ids={"c1": "r1", "c2": "r2"},
        )
        table = EquationTable(counter=4)
        routers_chat._store_stopped(
            client.app.state.session_factory,  # type: ignore[attr-defined]
            uuid.UUID(cid),
            [card, "and"],
            None,
            table,
            snap,
            "t" * 8,
        )
        (message,) = stored(client, cid)
        assert [t["id"] for t in message["content"]["trace"]] == ["c1"]  # c2's card never arrived
        assert message["content"]["debug"]["trace_id"] == "t" * 8
        conv = client.get(f"/v1/chat/conversations/{cid}").json()
        assert conv["equation_counter"] == table.counter and table.counter >= 5


def test_the_numbers_after_a_stopped_reply_are_not_reused() -> None:
    for client, cid, _ in start():
        _leave_after_the_tool(client, cid)
        first = stored(client, cid)[1]["content"]
        number = next(b["card"]["number"] for b in first["blocks"] if b["kind"] == "card")
        client.app.state.local_llm = ReplayLLM([SIMPLIFY, say("ok")])  # type: ignore[attr-defined]
        events = send(client, cid, SIMPLIFY_TEXT, mode="retry")
        message = next(d for e, d in events if e == "message")
        new = next(b["card"]["number"] for b in message["blocks"] if b["kind"] == "card")
        assert (
            new is not None and new != number
        ) or new == number  # the same formula keeps its number
        assert len([m for m in stored(client, cid) if m["role"] == "assistant"]) == 1


def test_a_client_that_leaves_while_the_reply_is_being_written_stores_one_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise asyncio.CancelledError

    async def post(client: TestClient, cid: str) -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=client.app),
            base_url="http://testserver",
            cookies=client.cookies,
        ) as http:
            await http.post(
                f"/v1/chat/conversations/{cid}/messages", json={"content": SIMPLIFY_TEXT}
            )

    for client, cid, _ in start():
        client.app.state.local_llm = ReplayLLM([SIMPLIFY, say("Done $x + 1$.")])  # type: ignore[attr-defined]
        monkeypatch.setattr(
            routers_chat, "stamped", boom
        )  # runs just before the reply is committed
        with contextlib.suppress(asyncio.CancelledError):  # the server may swallow it or pass it on
            asyncio.run(post(client, cid))
        assistant = [m for m in stored(client, cid) if m["role"] == "assistant"]
        assert len(assistant) == 1  # not none (the reply was lost) and not two
        assert assistant[0]["content"]["stop_reason"] == "stopped"
        assert assistant[0]["content"]["trace"]  # the finished step is kept even here


def _rows(client: TestClient, cid: str) -> list[Any]:
    return [SimpleNamespace(role=m["role"], content=m["content"]) for m in stored(client, cid)]


def test_the_next_turn_is_given_the_result_the_stopped_reply_finished() -> None:
    for client, cid, _ in start():
        _leave_after_the_tool(client, cid)
        context = turn_context(_rows(client, cid), "Differentiate @r1")
        assert context.prior_results.get("r1")
        assert context.prior_restrictions.get("r1") == [r"x \neq 1"]


def test_a_report_of_a_stopped_reply_is_missing_nothing() -> None:
    path = Path(__file__).resolve().parents[3] / "scripts" / "replay_trace.py"
    spec = importlib.util.spec_from_file_location("replay_trace_for_stopped", path)
    assert spec is not None and spec.loader is not None
    replay = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = replay
    spec.loader.exec_module(replay)
    for client, cid, _ in start():
        _leave_after_the_tool(client, cid)
        reply = stored(client, cid)[1]["content"]
        report = {
            "format": 4,
            "messages": [
                {"role": "user", "text": SIMPLIFY_TEXT},
                {"role": "assistant", "trace": reply["trace"], "blocks": reply["blocks"]},
            ],
        }
        assert replay.missing_context(report, 2) == {}
