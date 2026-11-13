"""A model that replays prepared replies: the deterministic stand-in for tests, benchmarks and replay.

Unlike `ScriptedLLM` (keyword rules, one tool call per turn) this plays any sequence: each entry of
`replies` is the chunk stream of one model call. It never contacts a host and is safe to ship; it is
not selectable through `MINERVA_LLM_MODE`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from .llm import Chunk, LLMError, ToolCall


class ReplayLLM:
    """Replays prepared replies, one list of chunks per model call, and records what it was sent."""

    name = "replay"
    provider = "replay"

    def __init__(self, replies: list[list[Chunk]] | None = None, repeat_last: bool = False) -> None:
        self.replies = replies or []
        self.repeat_last = repeat_last
        self.seen: list[list[dict[str, Any]]] = []
        self.tools_seen: list[list[dict[str, Any]]] = []  # the tool schemas offered with each call
        self.error: LLMError | None = None  # every call fails with this
        self.fail_on: dict[int, LLMError] = {}  # only these calls (0-based) fail

    async def chat(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AsyncIterator[Chunk]:
        self.seen.append([dict(m) for m in messages])
        self.tools_seen.append(list(tools))
        index = len(self.seen) - 1
        if self.error:
            raise self.error
        if index in self.fail_on:
            raise self.fail_on[index]
        if index >= len(self.replies):
            if not self.repeat_last or not self.replies:
                return
            index = len(self.replies) - 1
        for chunk in self.replies[index]:
            yield chunk


def call(name: str, arguments: dict[str, Any]) -> list[Chunk]:
    """One model reply that makes a single tool call."""
    return [Chunk(tool_calls=[ToolCall(name, arguments)])]


def say(text: str) -> list[Chunk]:
    """One model reply that is plain prose."""
    return [Chunk(text=text)]
