"""Agent loop, tools and segment marking, with a deterministic fake model."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from minerva_api.agent.llm import Chunk, LLMError, ToolCall
from minerva_api.agent.loop import AgentLimits, run_agent
from minerva_api.agent.segments import normalize, split_segments
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase


class FakeLLM:
    """Replays prepared replies, one list of chunks per model call, and records what it was sent."""

    name = "fake"

    def __init__(self, replies: list[list[Chunk]] | None = None, repeat_last: bool = False) -> None:
        self.replies = replies or []
        self.repeat_last = repeat_last
        self.seen: list[list[dict[str, Any]]] = []
        self.error: LLMError | None = None

    async def chat(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AsyncIterator[Chunk]:
        self.seen.append([dict(m) for m in messages])
        if self.error:
            raise self.error
        index = len(self.seen) - 1
        if index >= len(self.replies):
            if not self.repeat_last:
                return
            index = len(self.replies) - 1
        for chunk in self.replies[index]:
            yield chunk


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def collect(
    llm: FakeLLM, ctx: ToolContext, history: list[dict[str, Any]], **kw: Any
) -> list[dict[str, Any]]:
    async def go() -> list[dict[str, Any]]:
        return [e async for e in run_agent(llm, history, ctx, **kw)]

    return asyncio.run(go())


def final(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(e["data"] for e in events if e["event"] == "message")


def cards_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    return [b["card"] for b in message["blocks"] if b["kind"] == "card"]


def segments_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for b in message["blocks"] if b["kind"] == "text" for s in b["segments"]]


def test_text_only_answer_has_no_cards(ctx: ToolContext) -> None:
    llm = FakeLLM([[Chunk(text="Which variables are positive?")]])
    events = collect(llm, ctx, [{"role": "user", "content": "help"}])
    msg = final(events)
    assert cards_of(msg) == [] and msg["text"] == "Which variables are positive?"
    assert [e["event"] for e in events] == ["text", "message"]


def test_tool_result_is_a_card_and_formula_is_verified(ctx: ToolContext) -> None:
    llm = FakeLLM(
        [
            [Chunk(tool_calls=[ToolCall("simplify", {"latex": r"\frac{x^2-1}{x-1}"})])],
            [Chunk(text="The result is $x + 1$.")],
        ]
    )
    events = collect(llm, ctx, [{"role": "user", "content": "simplify"}])
    msg = final(events)
    assert cards_of(msg)[0]["type"] == "simplify"
    assert cards_of(msg)[0]["data"]["status"] == "ok"
    math = [s for s in segments_of(msg) if s["type"] == "math"]
    assert math and math[0]["verified"] is True
    # the model saw the tool's compact result, not the raw card
    assert any(m["role"] == "tool" and "result=$x + 1$" in m["content"] for m in llm.seen[1])


def test_formula_not_from_a_tool_is_marked_unverified(ctx: ToolContext) -> None:
    llm = FakeLLM([[Chunk(text=r"It equals $\frac{\sqrt{\pi}}{2}$ and also $$x^2+1$$.")]])
    msg = final(collect(llm, ctx, [{"role": "user", "content": "q"}]))
    formulas = [s for s in segments_of(msg) if s["type"] != "text"]
    assert len(formulas) == 2 and all(s["verified"] is False for s in formulas)


def test_step_cap_stops_a_looping_model(ctx: ToolContext) -> None:
    call = [Chunk(tool_calls=[ToolCall("search_handbook", {"query": "gaussian"})])]
    llm = FakeLLM([call], repeat_last=True)
    msg = final(
        collect(llm, ctx, [{"role": "user", "content": "q"}], limits=AgentLimits(max_steps=3))
    )
    assert msg["truncated"] is True
    assert len(cards_of(msg)) == 3
    assert "limit" in msg["text"]


def test_time_cap_stops_the_turn(ctx: ToolContext) -> None:
    llm = FakeLLM([[Chunk(text="x")]], repeat_last=True)
    msg = final(
        collect(llm, ctx, [{"role": "user", "content": "q"}], limits=AgentLimits(max_seconds=-1))
    )
    assert msg["truncated"] is True and llm.seen == []


def test_unknown_tool_and_bad_arguments_become_errors_the_model_can_read(ctx: ToolContext) -> None:
    llm = FakeLLM(
        [
            [
                Chunk(
                    tool_calls=[
                        ToolCall("rm_rf", {"path": "/"}),
                        ToolCall("simplify", {"latex": 5}),
                        ToolCall("simplify", {"latex": "x", "assumptions": {"x": ["evil"]}}),
                        ToolCall("simplify", {}),
                    ]
                )
            ],
            [Chunk(text="I could not run that.")],
        ]
    )
    msg = final(collect(llm, ctx, [{"role": "user", "content": "q"}]))
    assert [c["type"] for c in cards_of(msg)] == ["error"] * 4
    tool_msgs = [m["content"] for m in llm.seen[1] if m["role"] == "tool"]
    assert tool_msgs[0].startswith("error: unknown tool")
    assert all(t.startswith("error:") for t in tool_msgs)


def test_tool_arguments_are_bounded(ctx: ToolContext) -> None:
    out = execute("simplify", {"latex": "x" * 5000}, ctx)
    assert out.card.type == "error"
    out = execute("search_handbook", {"query": "a", "limit": 500}, ctx)
    assert out.card.type == "error"


def test_model_host_failure_is_an_error_event_without_a_message(ctx: ToolContext) -> None:
    llm = FakeLLM()
    llm.error = LLMError("The model host is not reachable.")
    events = collect(llm, ctx, [{"role": "user", "content": "q"}])
    assert [e["event"] for e in events] == ["error"]


def test_tools_have_no_side_effects_surface(ctx: ToolContext) -> None:
    from minerva_api.agent.tools import TOOLS

    assert set(TOOLS) == {
        "search_handbook",
        "get_entry",
        "simplify",
        "find_identities",
        "tensor_canonicalize",
    }
    for tool in TOOLS.values():
        assert tool.schema()["function"]["parameters"]["type"] == "object"


def test_identities_tool_reports_conditions_honestly(ctx: ToolContext) -> None:
    out = execute("find_identities", {"latex": r"\log(xy)"}, ctx)
    assert "log.product" in out.for_model and "conditions_not_satisfied" in out.for_model
    ok = execute(
        "find_identities",
        {"latex": r"\log(xy)", "assumptions": {"x": ["positive"], "y": ["positive"]}},
        ctx,
    )
    assert "applicable" in ok.for_model


def test_segments_parse_all_math_delimiters() -> None:
    known = {normalize("x^{2}")}
    segs = split_segments(r"A $x^2$, B $$y$$, C \(z\), D \[w\], cost \$5.", known)
    kinds = [(s.type, s.value) for s in segs if s.type != "text"]
    assert kinds == [("math", "x^2"), ("display_math", "y"), ("math", "z"), ("display_math", "w")]
    assert segs[1].verified is True  # x^2 equals x^{2} after normalisation
    assert all(s.verified is False for s in segs if s.type != "text" and s.value != "x^2")
    assert "".join(s.value for s in segs if s.type == "text").endswith(r"cost \$5.")


def test_normalize_ignores_spacing_and_size_commands() -> None:
    assert normalize(r"\left( a + b \right)^{2}") == normalize("(a+b)^2")


def test_prose_and_cards_keep_the_order_they_happened_in(ctx: ToolContext) -> None:
    llm = FakeLLM(
        [
            [
                Chunk(text="Let me check. "),
                Chunk(tool_calls=[ToolCall("simplify", {"latex": "x+x"})]),
            ],
            [Chunk(text="Done: $2x$.")],
        ]
    )
    msg = final(collect(llm, ctx, [{"role": "user", "content": "q"}]))
    assert [b["kind"] for b in msg["blocks"]] == ["text", "card", "text"]
