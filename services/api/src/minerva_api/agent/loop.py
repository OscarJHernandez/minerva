"""The agent loop: model <-> tools, with step and time limits, streaming events as it goes."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from fastapi.concurrency import run_in_threadpool

from .events import (
    AssistantMessage,
    Block,
    Card,
    ClaimDebug,
    LLMCallDebug,
    TraceStep,
    TurnDebug,
)
from .llm import LLMClient, LLMError
from .prompts import SYSTEM_PROMPT
from .segments import split_segments
from .tools import ToolContext, ToolOutput, execute, known_formulas, schemas


@dataclass
class AgentLimits:
    max_steps: int = 6
    max_seconds: float = 180.0


@dataclass
class TurnInfo:
    """Facts about the request that the loop cannot know itself, copied into the debug record."""

    request_id: str = ""
    mode: str = ""
    api_version: str = ""


def prompt_sha() -> str:
    """Short fingerprint of the system prompt and tool schemas: which version produced a turn."""
    blob = SYSTEM_PROMPT + json.dumps(schemas(), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


async def run_agent(
    llm: LLMClient,
    history: Sequence[dict[str, Any]],
    ctx: ToolContext,
    limits: AgentLimits | None = None,
    *,
    prior_known: set[str] | None = None,
    prior_relations: Sequence[str] = (),
    prior_assumptions: dict[str, list[str]] | None = None,
    info: TurnInfo | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield events: {"event": "text"|"tool_call"|"card"|"message"|"error", "data": {...}}."""
    limits = limits or AgentLimits()
    started = time.monotonic()
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    outputs: list[ToolOutput] = []
    order: list[str | Card] = []  # prose and cards in the order they happened
    trace: list[TraceStep] = []
    full_text: list[str] = []
    truncated = False
    info = info or TurnInfo()
    debug = TurnDebug(
        request_id=info.request_id,
        api_version=info.api_version,
        mode=info.mode,
        model=llm.name,
        prompt_sha=prompt_sha(),
        max_steps=limits.max_steps,
        max_seconds=limits.max_seconds,
    )

    try:
        for _ in range(limits.max_steps):
            if time.monotonic() - started > limits.max_seconds:
                truncated = True
                break
            step_text: list[str] = []
            calls = []
            call_started = time.monotonic()
            async for chunk in llm.chat(messages, schemas()):
                if chunk.text:
                    step_text.append(chunk.text)
                    yield {"event": "text", "data": {"delta": chunk.text}}
                calls.extend(chunk.tool_calls)
            text = "".join(step_text)
            debug.llm_calls.append(
                LLMCallDebug(ms=_ms(call_started), text_chars=len(text), tool_calls=len(calls))
            )
            if text:
                full_text.append(text)
                order.append(text)
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
                tool_started = time.monotonic()
                output = await run_in_threadpool(execute, call.name, call.arguments, ctx)
                output.card.id = call_id
                outputs.append(output)
                order.append(output.card)
                trace.append(
                    TraceStep(
                        id=call_id,
                        name=call.name,
                        arguments=call.arguments,
                        for_model=output.for_model,
                        duration_ms=_ms(tool_started),
                        status=output.status,
                    )
                )
                yield {"event": "card", "data": output.card.model_dump(mode="json")}
                messages.append(
                    {"role": "tool", "tool_name": call.name, "content": output.for_model}
                )
        else:
            truncated = True
    except LLMError as exc:
        debug.error = str(exc)
        debug.duration_ms = _ms(started)
        yield {
            "event": "error",
            "data": {
                "message": str(exc),
                "debug": debug.model_dump(mode="json"),
                "trace": [t.model_dump(mode="json") for t in trace],
            },
        }
        return

    raw = "\n\n".join(full_text).strip()
    if truncated:
        raw = (
            raw + "\n\n" if raw else ""
        ) + "I stopped here because I reached the step or time limit for one answer."
    known = known_formulas(outputs) | (prior_known or set())
    blocks = [
        Block(kind="card", card=item)
        if isinstance(item, Card)
        else Block(kind="text", segments=split_segments(item, known))
        for item in order
    ]
    debug.claims = await check_model_claims(blocks, ctx, trace, prior_relations, prior_assumptions)
    debug.duration_ms = _ms(started)
    final = AssistantMessage(
        text=raw,
        blocks=blocks,
        trace=trace,
        truncated=truncated,
        debug=debug,
    )
    yield {"event": "message", "data": final.model_dump(mode="json")}


MAX_CLAIMS = 12


async def check_model_claims(
    blocks: list[Block],
    ctx: ToolContext,
    trace: list[TraceStep],
    prior_relations: Sequence[str] = (),
    prior_assumptions: dict[str, list[str]] | None = None,
) -> list[ClaimDebug]:
    """Test the equations the model wrote itself; mark them checked, contradicted or unchecked.

    Assumptions the model declared in its own tool calls apply to its claims too (an equation about
    a positive x is checked for positive x), as do those of earlier turns. An equation with a symbol
    on one side only is a relation, not an identity: it is never called false by sampling, only
    checked against definitions stated earlier (`prior_relations`, then the model's own formulas in
    order), and the result names them.
    """
    pending = [
        s
        for b in blocks
        if b.kind == "text"
        for s in b.segments
        if s.type != "text" and s.status == "unchecked" and len(s.value) <= 500
    ][:MAX_CLAIMS]
    if not pending:
        return []
    assumptions: dict[str, list[str]] = dict(prior_assumptions or {})
    for step in trace:
        declared = step.arguments.get("assumptions")
        if isinstance(declared, dict):
            for name, items in declared.items():
                if isinstance(items, list):
                    assumptions[str(name)] = [str(i) for i in items]
    results = await run_in_threadpool(
        ctx.engine.check_claims, [s.value for s in pending], assumptions, list(prior_relations)
    )
    report: list[ClaimDebug] = []
    for segment, result in zip(pending, results, strict=True):
        report.append(
            ClaimDebug(
                value=segment.value,
                status=str(result.status),
                detail=result.detail,
                kind=result.kind,
                given=list(result.given),
            )
        )
        if result.status == "holds":
            segment.status = "checked"
        elif result.status == "conditional":
            segment.status = "conditional"
            segment.note = result.detail
        elif result.status == "fails":
            segment.status = "contradicted"
            segment.note = result.detail
        elif result.kind == "relation":
            segment.note = result.detail
    return report
