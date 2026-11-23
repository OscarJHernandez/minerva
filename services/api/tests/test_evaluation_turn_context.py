"""The evaluation harness builds a turn's context with `turn_context`, the function the chat route calls (issue 150).

Before, a case with plain-text history got a hand-assembled subset of it. These tests pin what the harness now hands
the loop for such a case, so a later change to it is deliberate: the numbered formulas of the case prompt and the
reference and handle notes appended to it (as in the chat), consecutive user messages merged into one, the user's
messages as a list on the tool context, and a copy of the tool context per turn (the caller's is never edited).
Live evaluation scores recorded before this change are therefore not comparable with scores after it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from minerva_api.agent.evaluation import runner
from minerva_api.agent.evaluation.cases import EvalCase
from minerva_api.agent.loop import AgentLimits
from minerva_api.agent.replay import ReplayLLM, say
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def case(prompt: str, history: list[dict[str, Any]]) -> EvalCase:
    return EvalCase(
        id="c",
        category="symbolic",
        prompt=prompt,
        history=history,
        expected_tool="simplify",
        source="a test",
    )


def observed(
    monkeypatch: pytest.MonkeyPatch, ctx: ToolContext, c: EvalCase
) -> tuple[list[dict[str, Any]], dict[str, Any], ToolContext]:
    """What the loop was given: its history, its keyword arguments and the context it ran with."""
    seen: dict[str, Any] = {}
    real = runner.run_agent

    async def spy(llm: Any, history: Any, call_ctx: Any, limits: Any = None, **kwargs: Any) -> Any:
        seen.update(history=[dict(m) for m in history], kwargs=kwargs, ctx=call_ctx)
        async for event in real(llm, history, call_ctx, limits, **kwargs):
            yield event

    monkeypatch.setattr(runner, "run_agent", spy)
    obs = asyncio.run(
        runner.observe(ReplayLLM([say("ok")]), c, ctx, AgentLimits(retry_delay_seconds=0))
    )
    assert not obs.error
    return seen["history"], seen["kwargs"], seen["ctx"]


def test_a_plain_history_case_gets_the_chat_notes_numbers_and_merged_user_messages(
    monkeypatch: pytest.MonkeyPatch, ctx: ToolContext
) -> None:
    history = [
        {"role": "user", "content": "Let A and B be matrices."},
        {"role": "user", "content": "I mean 2x2 ones. See $$a = 1$$."},
        {"role": "assistant", "content": "Understood."},
    ]
    prompt = "Compare equation 1 with equation 7, and take $$w = x + 1$$."
    sent, kwargs, call_ctx = observed(monkeypatch, ctx, case(prompt, history))
    # consecutive user messages are one message, as the model sees them in the chat
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[0]["content"] == "Let A and B be matrices.\n\nI mean 2x2 ones. See $$a = 1$$."
    # the prompt carries the notes the chat appends: plain-text history has no equation numbers, so (1) and (7)
    # do not exist yet when the prompt is read, and its own displayed formula gets the handle @e1
    assert sent[-1]["content"].startswith(prompt + "\n\n[notice from Minerva")
    assert "(1) does not exist in this conversation (equations: none yet)" in sent[-1]["content"]
    assert "(7) does not exist in this conversation" in sent[-1]["content"]
    assert "@e1" in sent[-1]["content"]
    # the numbered table, declarations and the user's messages come from the same function as the route's
    assert sorted(kwargs["equations"].items) == [1]
    assert kwargs["equations"].items[1].latex == "w = x + 1"
    assert kwargs["prior_matrices"] == ["A", "B"]
    assert kwargs["prior_unverified"] == {} and kwargs["prior_evidence"] == []
    assert call_ctx.user_messages == [
        "Let A and B be matrices.",
        "I mean 2x2 ones. See $$a = 1$$.",
        prompt,
    ]


def test_each_turn_runs_in_its_own_copy_of_the_tool_context(
    monkeypatch: pytest.MonkeyPatch, ctx: ToolContext
) -> None:
    """The messages and constants of one case never leak into another through a shared context.

    `dataclasses.replace` is shallow: its mutable fields (`matrices`, `matrix_sources`, `notation`, `numeric`,
    `statmech_answers`) are the same objects as the caller's. The harness only ever assigns whole fields of the
    copy (`user_messages`, `constants`), which is what this protects; code that mutated one of those containers in
    place would still reach the caller's.
    """
    _, _, first = observed(monkeypatch, ctx, case("Simplify $x + x$.", []))
    assert first is not ctx
    assert first.user_messages == ["Simplify $x + x$."] and ctx.user_messages is None
    assert ctx.constants == () and first.constants == ()
