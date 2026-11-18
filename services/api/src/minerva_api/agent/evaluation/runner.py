"""Run benchmark cases through the real agent loop and score them.

The model client is injected: an Ollama host for a real evaluation, a `ReplayLLM` for the fast
deterministic tests. Tools, verification and claim checking are always the real ones.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

from ...routers_chat import (
    _claim_context,
    _conversation_events,
    _history,
    _prior_matrices,
)
from ..llm import LLMClient
from ..loop import AgentLimits, TurnInfo, run_agent
from ..params import EffectiveParams
from ..segments import split_segments
from ..tools import ToolContext
from ..toolsets import conversation_groups
from .cases import EvalCase
from .score import CaseResult, FailureClass, Observation, combine, score_case

ProgressFn = Callable[[CaseResult], None]
# Called with every live turn: case id, repeat, turn number (1-based), the prompt, what happened, whether it passed.
TurnFn = Callable[[str, int, int, str, Observation, bool], None]
_CONVERSATION_ENDERS = frozenset({FailureClass.MODEL_HOST, FailureClass.NO_REPLY})


async def observe(
    llm: LLMClient,
    case: EvalCase,
    ctx: ToolContext,
    limits: AgentLimits,
    grace_seconds: float = 30.0,
    stored: Sequence[Any] = (),
    params: EffectiveParams | None = None,
    routing: bool = False,
) -> Observation:
    """Drive one turn and record what happened. Never raises: a crashed turn is an observation too.

    `stored` are the earlier messages of a live conversation (as the chat route stores them: the user's
    text, and the agent's answer with its trace and cards). The loop then gets exactly what the route would
    hand it for the next turn: the rebuilt history, verified results (`@r1`) and their restrictions,
    declared matrices and assumptions. Without it the case's own plain-text `history` is used.
    """
    history = [*case.history, {"role": "user", "content": case.prompt}]
    carried: dict[str, Any] = {}
    if stored:
        messages = list(stored)
        rebuilt, known, quoted, results, restrictions = _history(
            messages,
            max_messages=params.history_messages if params else 20,
            max_tokens=params.history_tokens if params else None,
        )
        prior_relations, prior_assumptions = _claim_context(messages, case.prompt)
        history = [*rebuilt, {"role": "user", "content": case.prompt}]
        carried = {
            "prior_known": known,
            "prior_quoted": quoted,
            "prior_results": results,
            "prior_restrictions": restrictions,
            "prior_assumptions": prior_assumptions,
            "prior_matrices": _prior_matrices(messages, case.prompt),
            "prior_groups": conversation_groups(_conversation_events(messages, case.prompt)),
        }
    else:
        # What the chat route passes the loop from earlier turns: the formulas written so far, which the
        # claim checker uses as definitions. (Case histories are plain text: no earlier tool calls.)
        prior_relations = [
            seg.value
            for message in history
            for seg in split_segments(str(message.get("content", "")), set())
            if seg.type != "text" and len(seg.value) <= 500
        ][-60:]
    obs = Observation()
    started = time.monotonic()

    async def drive() -> None:
        async for event in run_agent(
            llm,
            history,
            ctx,
            limits,
            prior_relations=prior_relations,
            params=params,
            info=TurnInfo(routing=routing),
            **carried,
        ):
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


def _stored(prompt: str, final: dict[str, Any]) -> list[Any]:
    """A user message and the agent's answer, shaped like the chat route's stored messages."""
    return [
        SimpleNamespace(role="user", content={"text": prompt}),
        SimpleNamespace(role="assistant", content=final),
    ]


async def run_case(
    llm: LLMClient,
    ctx: ToolContext,
    case: EvalCase,
    limits: AgentLimits,
    repeat: int = 0,
    params: EffectiveParams | None = None,
    on_turn: TurnFn | None = None,
    routing: bool = False,
) -> CaseResult:
    """One case: its first turn, then each follow-up turn answered live on top of the earlier answers.

    A conversation that cannot continue (the earlier turn produced no answer) ends there and fails.
    """
    obs = await observe(llm, case, ctx, limits, params=params, routing=routing)
    results = [score_case(case, obs, repeat)]
    if on_turn is not None:
        on_turn(case.id, repeat, 1, case.prompt, obs, results[0].passed)
    stored: list[Any] = []
    previous_prompt, previous = case.prompt, obs
    for number, turn in enumerate(case.followups, start=2):
        # A turn the model host or the loop could not answer leaves nothing to build the next turn on: the
        # conversation ends there, and the case has already failed (its first turn is scored as such).
        if previous.final is None or results[-1].failure_class in _CONVERSATION_ENDERS:
            break
        stored += _stored(previous_prompt, previous.final)
        turn_case = turn.as_case(case, number)
        previous = await observe(
            llm, turn_case, ctx, limits, stored=stored, params=params, routing=routing
        )
        previous_prompt = turn_case.prompt
        results.append(score_case(turn_case, previous, repeat))
        if on_turn is not None:
            on_turn(case.id, repeat, number, turn_case.prompt, previous, results[-1].passed)
    return combine(results)


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
    params: EffectiveParams | None = None,
    on_turn: TurnFn | None = None,
    routing: bool = False,
) -> list[CaseResult]:
    """Every case `repeats` times; results come back in case order, then repeat order."""
    gate = asyncio.Semaphore(max(1, parallel))

    async def one(case: EvalCase, repeat: int) -> CaseResult:
        async with gate:
            result = await run_case(llm, ctx, case, limits, repeat, params, on_turn, routing)
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
