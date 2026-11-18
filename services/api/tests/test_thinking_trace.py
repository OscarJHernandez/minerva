"""The model's reasoning is trace material: stored with the reply for debugging, never part of it.

It is kept in the stored assistant message (`debug.llm_calls[i].thinking`, under the conversation's own access
and deletion rules), never shown as reply text, never replayed to the model, never checked or numbered, and
only its length reaches a trace event or the log.
"""

import asyncio
import json
from typing import Any

import httpx
import pytest
from minerva_api.agent.llm import Chunk, OllamaClient, ToolCall
from minerva_api.agent.loop import THINKING_CAP, TurnInfo, run_agent
from minerva_api.agent.replay import ReplayLLM
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

SECRET = "PRIVATE-REASONING-MARKER"


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def run(
    llm: ReplayLLM, ctx: ToolContext, info: TurnInfo | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    async def go() -> list[dict[str, Any]]:
        history = [{"role": "user", "content": "simplify (x^2-1)/(x-1)"}]
        return [e async for e in run_agent(llm, history, ctx, info=info)]

    events = asyncio.run(go())
    return next(e["data"] for e in events if e["event"] == "message"), events


def _turn(thought: str) -> ReplayLLM:
    return ReplayLLM(
        [
            [
                Chunk(thinking=thought),
                Chunk(tool_calls=[ToolCall("simplify", {"latex": r"\frac{x^2-1}{x-1}"})]),
            ],
            [Chunk(text="Done.")],
        ]
    )


def test_thinking_is_stored_per_model_call_and_is_not_part_of_the_reply(ctx: ToolContext) -> None:
    message, events = run(_turn(f"I should use simplify. {SECRET}"), ctx)
    first, second = message["debug"]["llm_calls"]
    assert SECRET in first["thinking"] and first["thinking_chars"] == len(first["thinking"])
    assert first["thinking_truncated"] is False
    assert second["thinking"] is None and second["thinking_chars"] == 0
    assert SECRET not in message["text"]
    streamed = "".join(e["data"].get("delta", "") for e in events if e["event"] == "text")
    assert SECRET not in streamed
    blocks = json.dumps(message["blocks"])
    assert SECRET not in blocks and SECRET not in json.dumps(message["trace"])


def test_thinking_is_never_replayed_to_the_model(ctx: ToolContext) -> None:
    llm = _turn(f"private {SECRET}")
    run(llm, ctx)
    assert len(llm.seen) == 2
    assert SECRET not in json.dumps(llm.seen)


def test_thinking_is_cut_at_the_cap_and_says_so(ctx: ToolContext) -> None:
    long = "x" * (THINKING_CAP + 5000)
    message, _ = run(_turn(long), ctx)
    call = message["debug"]["llm_calls"][0]
    assert len(call["thinking"]) == THINKING_CAP
    assert call["thinking_chars"] == len(long) and call["thinking_truncated"] is True


def test_a_model_that_did_not_think_has_nothing_stored(ctx: ToolContext) -> None:
    llm = ReplayLLM([[Chunk(text="Hello.")]])
    message, _ = run(llm, ctx)
    call = message["debug"]["llm_calls"][0]
    assert call["thinking"] is None and call["thinking_chars"] == 0


def test_a_trace_event_carries_the_length_never_the_text(ctx: ToolContext) -> None:
    seen: list[dict[str, Any]] = []
    run(_turn(f"private {SECRET}"), ctx, TurnInfo(sink=seen.append))
    responses = [e for e in seen if e["event"] == "llm.response"]
    assert responses[0]["thinking_chars"] == len(f"private {SECRET}")
    assert responses[1]["thinking_chars"] == 0
    assert SECRET not in json.dumps(seen)


def _ollama(lines: list[dict[str, Any]], **kw: Any) -> OllamaClient:
    body = ("\n".join(json.dumps(x) for x in lines) + "\n").encode()
    return OllamaClient(
        "http://h.test",
        "m",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)),
        **kw,
    )


def test_the_ollama_client_forwards_thinking_separately_from_the_reply() -> None:
    client = _ollama(
        [
            {"message": {"thinking": "step one. "}},
            {"message": {"thinking": "step two."}},
            {"message": {"content": "The answer."}},
            {"message": {"content": ""}, "done": True, "done_reason": "stop"},
        ],
        think=True,
    )

    async def collect() -> list[Chunk]:
        return [c async for c in client.chat([{"role": "user", "content": "hi"}], [])]

    chunks = asyncio.run(collect())
    assert "".join(c.thinking for c in chunks) == "step one. step two."
    assert "".join(c.text for c in chunks) == "The answer."
