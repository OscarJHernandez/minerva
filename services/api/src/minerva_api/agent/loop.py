"""The agent loop: model <-> tools, with step and time limits, streaming events as it goes."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from fastapi.concurrency import run_in_threadpool

from .events import AssistantMessage, Card, TraceStep
from .llm import LLMClient, LLMError
from .prompts import SYSTEM_PROMPT
from .segments import split_segments
from .tools import ToolContext, ToolOutput, execute, known_formulas, schemas


@dataclass
class AgentLimits:
    max_steps: int = 6
    max_seconds: float = 180.0


async def run_agent(
    llm: LLMClient,
    history: Sequence[dict[str, Any]],
    ctx: ToolContext,
    limits: AgentLimits | None = None,
    *,
    prior_known: set[str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield events: {"event": "text"|"tool_call"|"card"|"message"|"error", "data": {...}}."""
    limits = limits or AgentLimits()
    started = time.monotonic()
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    outputs: list[ToolOutput] = []
    cards: list[Card] = []
    trace: list[TraceStep] = []
    full_text: list[str] = []
    truncated = False

    try:
        for _ in range(limits.max_steps):
            if time.monotonic() - started > limits.max_seconds:
                truncated = True
                break
            step_text: list[str] = []
            calls = []
            async for chunk in llm.chat(messages, schemas()):
                if chunk.text:
                    step_text.append(chunk.text)
                    yield {"event": "text", "data": {"delta": chunk.text}}
                calls.extend(chunk.tool_calls)
            text = "".join(step_text)
            if text:
                full_text.append(text)
            if not calls:
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": [
                        {"function": {"name": c.name, "arguments": c.arguments}} for c in calls
                    ],
                }
            )
            for call in calls:
                call_id = uuid.uuid4().hex[:8]
                yield {
                    "event": "tool_call",
                    "data": {"id": call_id, "name": call.name, "arguments": call.arguments},
                }
                output = await run_in_threadpool(execute, call.name, call.arguments, ctx)
                output.card.id = call_id
                outputs.append(output)
                cards.append(output.card)
                trace.append(
                    TraceStep(
                        id=call_id,
                        name=call.name,
                        arguments=call.arguments,
                        for_model=output.for_model,
                    )
                )
                yield {"event": "card", "data": output.card.model_dump(mode="json")}
                messages.append(
                    {"role": "tool", "tool_name": call.name, "content": output.for_model}
                )
        else:
            truncated = True
    except LLMError as exc:
        yield {"event": "error", "data": {"message": str(exc)}}
        return

    raw = "\n\n".join(full_text).strip()
    if truncated:
        raw = (
            raw + "\n\n" if raw else ""
        ) + "I stopped here because I reached the step or time limit for one answer."
    known = known_formulas(outputs) | (prior_known or set())
    final = AssistantMessage(
        text=raw,
        segments=split_segments(raw, known),
        cards=cards,
        trace=trace,
        truncated=truncated,
    )
    yield {"event": "message", "data": final.model_dump(mode="json")}
