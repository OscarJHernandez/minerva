"""Run benchmark cases through the real agent loop and score them.

The model client is injected: an Ollama host for a real evaluation, a `ReplayLLM` for the fast
deterministic tests. Tools, verification and claim checking are always the real ones.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Any

from ..llm import LLMClient
from ..loop import AgentLimits, run_agent
from ..tools import ToolContext
from .cases import EvalCase
from .score import CaseResult, Observation, score_case

ProgressFn = Callable[[CaseResult], None]


async def observe(
    llm: LLMClient,
    case: EvalCase,
    ctx: ToolContext,
    limits: AgentLimits,
    grace_seconds: float = 30.0,
) -> Observation:
    """Drive one turn and record what happened. Never raises: a crashed turn is an observation too."""
    history = [*case.history, {"role": "user", "content": case.prompt}]
    obs = Observation()
    started = time.monotonic()

    async def drive() -> None:
        async for event in run_agent(llm, history, ctx, limits):
            kind, data = event["event"], event["data"]
            if kind == "tool_call":
                obs.calls.append((data["name"], data["arguments"]))
                obs.call_ids.append(data["id"])
            elif kind == "card":
                obs.cards[data["id"]] = data
            elif kind == "message":
                obs.final = data
            elif kind == "error":
                obs.error = str(data.get("message", ""))
                obs.partial_trace = list(data.get("trace") or [])
                obs.partial_debug = data.get("debug")

    try:
        # The loop enforces its own deadline; this guard only catches a loop that fails to.
        await asyncio.wait_for(drive(), timeout=limits.max_seconds + grace_seconds)
    except TimeoutError:
        obs.error = "the evaluation guard timeout fired"
    except Exception as exc:  # a crash is a result, not a reason to lose the other cases
        obs.error = f"{type(exc).__name__} in the agent loop"
    obs.seconds = time.monotonic() - started
    return obs


async def run_case(
    llm: LLMClient, ctx: ToolContext, case: EvalCase, limits: AgentLimits, repeat: int = 0
) -> CaseResult:
    return score_case(case, await observe(llm, case, ctx, limits), repeat)


async def run_suite(
    llm: LLMClient,
    ctx: ToolContext,
    cases: Sequence[EvalCase],
    limits: AgentLimits,
    *,
    repeats: int = 1,
    first_repeat: int = 0,
    parallel: int = 1,
    on_result: ProgressFn | None = None,
) -> list[CaseResult]:
    """Every case `repeats` times; results come back in case order, then repeat order."""
    gate = asyncio.Semaphore(max(1, parallel))

    async def one(case: EvalCase, repeat: int) -> CaseResult:
        async with gate:
            result = await run_case(llm, ctx, case, limits, repeat)
        if on_result is not None:
            on_result(result)
        return result

    jobs = [one(case, first_repeat + repeat) for case in cases for repeat in range(max(1, repeats))]
    return list(await asyncio.gather(*jobs))


def filter_cases(
    cases: Sequence[EvalCase],
    *,
    ids: Sequence[str] = (),
    categories: Sequence[str] = (),
) -> list[EvalCase]:
    """Select cases by id or category; unknown names are an error, not a silently smaller suite."""
    known_ids = {c.id for c in cases}
    known_categories = {c.category for c in cases}
    if unknown := sorted(set(ids) - known_ids):
        raise ValueError(f"unknown case ids: {unknown}")
    if unknown := sorted(set(categories) - known_categories):
        raise ValueError(f"unknown categories: {unknown}")
    chosen = [
        c
        for c in cases
        if (not ids or c.id in ids) and (not categories or c.category in categories)
    ]
    if not chosen:
        raise ValueError("no case matches the selection")
    return chosen


def describe_limits(limits: AgentLimits) -> dict[str, Any]:
    return {
        "max_steps": limits.max_steps,
        "max_seconds": limits.max_seconds,
        "max_tool_calls": limits.max_tool_calls,
    }
