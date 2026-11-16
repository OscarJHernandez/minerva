"""The agent loop: model <-> tools, with limits, tracing and honest stopping, streaming events as it goes.

The model only plans, asks and explains. Every turn is bounded (steps, tool calls, seconds), every stop
has a reason, and every step is recorded as a content-free trace event (see tracing.py). What the loop
refuses to do: re-run a call that already ran, keep calling tools that cannot help, retry a mathematical
result into a different answer, or show a formula as verified without engine evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fastapi.concurrency import run_in_threadpool
from minerva_engine import ClaimEvidence, trace_scope
from minerva_engine.notation.models import Normalization

from ..logging import error_fields
from .budget import estimate_tokens, fit_history, message_chars, request_chars
from .choice import delegation_active, names_a_method, numbers_in
from .declarations import declared_matrices, declared_vectors
from .equations import EquationTable, number_blocks, strip_equation_tags
from .events import (
    AssistantMessage,
    Block,
    Card,
    ClaimDebug,
    LLMCallDebug,
    TraceStep,
    TurnDebug,
)
from .llm import LLMClient, LLMError, ToolCall, Usage
from .mathinput import MATH_FIELDS, prepare_arguments
from .params import EffectiveParams
from .prompts import SYSTEM_PROMPT
from .reasoning import ReasoningGate, clean_reply
from .refs import UNVERIFIED_INPUT, ResultStore, inherited_note, unverified_note
from .routing import plan_route
from .segments import normalize, split_segments
from .tools import (
    NEEDS_CHOICE,
    ODE_SOLVERS,
    TOOLS,
    ToolContext,
    ToolOutput,
    _error,
    check_arguments,
    execute,
    known_formulas,
    quoted_formulas,
    schemas,
)
from .toolsets import (
    LOAD_TOOLS,
    conversation_groups,
    group_of,
    grow,
    prompt_for,
    requested_groups,
    resolve_tool_name,
    tool_names,
)
from .tracing import (
    AgentCompleted,
    AgentRetry,
    AgentRouted,
    AgentStarted,
    AgentStopped,
    ErrorCategory,
    LLMRequest,
    LLMResponse,
    OutcomeKind,
    RetryReason,
    RouteReason,
    Sink,
    StopReason,
    ToolCompleted,
    ToolDeclined,
    ToolFailed,
    ToolStarted,
    Tracer,
    VerificationCompleted,
    activate,
    category_for,
    deactivate,
    outcome_kind,
    tool_status,
)


@dataclass
class AgentLimits:
    max_steps: int = 6
    max_seconds: float = 180.0
    max_tool_calls: int = 12  # across the whole turn, whatever the steps
    retry_delay_seconds: float = 0.5  # before the one retry of a model call that failed outright


@dataclass
class TurnInfo:
    """Facts about the request that the loop cannot know itself, copied into the debug record."""

    request_id: str = ""
    mode: str = ""
    api_version: str = ""
    trace_id: str = ""  # one per agent request; generated here when the caller has none
    sink: Sink | None = None  # where trace events go as they happen (the server log); may be None
    # Run a tool without asking the model when the request leaves no doubt which (see routing.py). Off by
    # default so scripted tests see only what they scripted; the chat route turns it on from settings.
    routing: bool = False


# What the user reads when a turn ends for a reason other than a finished answer. Fixed text: it names
# the cause and never repeats anything the user or the model wrote.
STOP_SENTENCES: dict[StopReason, str] = {
    StopReason.STEP_LIMIT: "I stopped here because I reached the step limit for one answer.",
    StopReason.TOOL_CALL_LIMIT: "I stopped here because I reached the limit on tool calls for one answer.",
    StopReason.TIME_LIMIT: "I stopped here because I reached the time limit for one answer.",
    StopReason.REPEATED_CALLS: "I stopped here because further tool calls were not producing new results.",
    StopReason.OUTPUT_LIMIT: (
        "I stopped here because my reply reached the model's length limit before it was finished. "
        "Ask me to continue, or to answer more briefly."
    ),
    StopReason.MODEL_ERROR: "I stopped here because the model host stopped answering.",
    StopReason.MALFORMED_OUTPUT: "I stopped here because the model's tool calls could not be used.",
    StopReason.INTERNAL_ERROR: "I stopped here because of an internal error.",
}
REPEAT_MESSAGE = (
    "error: this exact call already ran in this turn; its result is above. Do not repeat it: use that "
    "result, or tell the user what could not be established."
)
REPEAT_INVALID_MESSAGE = (
    "These are the same arguments as the call above, which was refused before anything ran: change "
    "the arguments as described, or tell the user what could not be established."
)
CAP_MESSAGE = "error: not run, the limit on tool calls or time for one answer was reached"
NOTICE_PREFIX = "[notice from Minerva, not from the user] "
TEXT_TOOL_NOTICE = (
    NOTICE_PREFIX + "Your last message wrote a tool call as text. Call the tool through the tool "
    "interface, or answer in words."
)
WRAP_UP_NOTICE = (
    NOTICE_PREFIX
    + "Stop calling tools. In at most 80 words say which results above were verified, what "
    "could not be established, and what the user could do next. Write no formula that is not in a tool "
    "result above."
)
MALFORMED_TEXT = {
    "missing_name": "the call had no tool name",
    "arguments_not_json": "the arguments were not valid JSON",
    "arguments_not_object": "the arguments were not a JSON object",
}
# Statuses that are a definite answer from the engine, however unwelcome.
ANSWER_STATUSES = frozenset({"unchanged", "unsupported", "conditions_not_satisfied"})
DROPPED_ASSUMPTIONS_NOTE = (
    "note: {symbols} was declared earlier in this conversation with assumptions, but this call did "
    "not repeat them, so the result holds for general complex symbols. If the earlier assumptions "
    "were meant to apply, call again with them."
)
MAX_UNPRODUCTIVE = 3  # consecutive executed calls that established nothing
MAX_REPEATS = 2  # identical calls in one turn before the turn is wrapped up
MIN_WRAP_UP_SECONDS = 3.0


def prompt_sha() -> str:
    """Short fingerprint of the system prompt and tool schemas: which version produced a turn."""
    blob = SYSTEM_PROMPT + json.dumps(schemas(), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


_SCHEMA_BY_NAME: dict[str, dict[str, Any]] = {s["function"]["name"]: s for s in schemas()}


def _schemas_for(names: Sequence[str]) -> list[dict[str, Any]]:
    return [_SCHEMA_BY_NAME[n] for n in names if n in _SCHEMA_BY_NAME]


def _events(history: Sequence[dict[str, Any]]) -> list[tuple[str, Any]]:
    """The (user text / tool call) events of a neutral history, for `conversation_groups`."""
    out: list[tuple[str, Any]] = []
    for m in history:
        if m.get("role") == "user":
            out.append(("user", str(m.get("content") or "")))
        for call in m.get("tool_calls") or []:
            out.append(("tool", (str(call.get("name", "")), call.get("arguments", {}))))
    return out


def _latest_user_message(history: Sequence[dict[str, Any]]) -> tuple[str, bool]:
    """The user's current message, and whether it answers a list of ODE methods the assistant just
    offered (an `ode_analyze` call sits between the previous user message and this one)."""
    users = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if not users:
        return "", False
    last = users[-1]
    text = str(history[last].get("content") or "")
    before = history[(users[-2] if len(users) > 1 else 0) : last]
    replying = any(
        m.get("name") == "ode_analyze"
        or any(c.get("name") == "ode_analyze" for c in m.get("tool_calls") or [])
        for m in before
    )
    return text, replying


def _ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


@dataclass
class _Routed:
    """A call decided by rule (routing.py) that already ran, its result waiting for the loop to record."""

    call: ToolCall
    output: ToolOutput


async def _run_route(
    route: Any, ctx: ToolContext, tracer: Tracer, debug: TurnDebug, remaining: Any
) -> _Routed | None:
    """Run the routed call like any other. Anything but a verified result drops the route, and the model
    plans the turn as if there had been none (the decision is recorded either way)."""
    call = ToolCall(route.tool, dict(route.arguments))
    reason = RouteReason(route.reason)
    try:
        output = await asyncio.wait_for(
            run_in_threadpool(execute, call.name, call.arguments, ctx),
            timeout=max(remaining(), 1.0),
        )
    except Exception:
        output = None
    if output is None or not output.arguments_valid or output.status != "ok":
        tracer.emit(AgentRouted, decision="declined", reason=RouteReason.DECLINED, tool=route.tool)
        debug.routing = {
            "decision": "declined",
            "reason": RouteReason.DECLINED.value,
            "tool": route.tool,
        }
        return None
    tracer.emit(AgentRouted, decision="routed", reason=reason, tool=route.tool)
    debug.routing = {"decision": "routed", "reason": reason.value, "tool": route.tool}
    return _Routed(call, output)


def _already_run(output: ToolOutput) -> Any:
    """Stands in for `execute` for the routed call, which ran before the first step."""

    def run(*_args: Any) -> ToolOutput:
        return output

    return run


def _resolve_calls(calls: Sequence[ToolCall]) -> tuple[list[ToolCall], dict[str, tuple[str, str]]]:
    """Calls with accepted aliases replaced by the registered name; `{call id: (requested, kind)}`."""
    resolved: list[ToolCall] = []
    aliases: dict[str, tuple[str, str]] = {}
    for call in calls:
        name, kind = resolve_tool_name(call.name, TOOLS) if call.name else (call.name, None)
        if kind is not None:
            aliases[call.id] = (call.name, kind)
            call = ToolCall(
                name=name, arguments=call.arguments, id=call.id, malformed=call.malformed
            )
        resolved.append(call)
    return resolved, aliases


def _sanitized(name: str) -> str:
    """A tool name that is safe to put in an event: one of the registry's, or `unknown`."""
    return name if name in TOOLS else "unknown"


Outcomes = list[tuple[str, ErrorCategory | None, OutcomeKind | None]]


def _arg_fields(name: str, arguments: Mapping[str, Any]) -> list[str]:
    """The argument names the model filled that belong to the tool's own schema, sorted. A name the
    schema does not have is dropped, so what is logged is never chosen by the model."""
    tool = TOOLS.get(name)
    if tool is None:
        return []
    fields = tool.args_model.model_fields
    return sorted(k for k in arguments if isinstance(k, str) and k in fields)


def _emit_outcome(
    tracer: Tracer,
    outcomes: Outcomes,
    step: int,
    *,
    name: str,
    log_name: str,
    call_id: str,
    status: str,
    category: ErrorCategory | None,
    arguments: Mapping[str, Any],
    valid: bool,
    repeated: bool = False,
    references: int = 0,
    duration_ms: int = 0,
    engine_ms: int | None = None,
    output: ToolOutput | None = None,
) -> None:
    """One line, at the default log setting, for a call that did not deliver a verified result:
    `tool.failed` for something to fix, `tool.declined` for an honest answer or a call not run."""
    kind = outcome_kind(status, category)
    outcomes.append((log_name, category, kind))
    if kind is None:
        return
    extra: dict[str, Any] = {}
    if output is not None:
        extra = {
            "reason_code": output.reason_code,
            "invalid_fields": list(output.invalid_fields),
            **output.error_site,
        }
    tracer.emit(
        ToolFailed if kind == "failure" else ToolDeclined,
        step,
        tool=log_name,
        call_id=call_id,
        status=tool_status(status),
        error_category=category,
        outcome_kind=kind,
        arguments_valid=valid,
        repeated=repeated,
        references=references,
        duration_ms=duration_ms,
        engine_ms=engine_ms,
        arg_fields=_arg_fields(name, arguments),
        **extra,
    )


_TOOL_JSON = re.compile(r"^\s*(```(?:json)?)?\s*(\{.*\})\s*(```)?\s*$", re.DOTALL)


def looks_like_tool_call(text: str) -> bool:
    """Did the model write a tool call as prose (`{"name": "simplify", "arguments": {...}}`)?"""
    match = _TOOL_JSON.match(text)
    if not match:
        return False
    try:
        data = json.loads(match.group(2))
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    name = data.get("name")
    function = data.get("function")
    if name is None and isinstance(function, dict):
        name = function.get("name")
    return isinstance(name, str) and name in TOOLS


def _call_key(name: str, arguments: dict[str, Any]) -> str:
    return name + "\u0000" + json.dumps(arguments, sort_keys=True, default=str)


def _dropped_assumptions(arguments: dict[str, Any], declared: Mapping[str, list[str]]) -> list[str]:
    """Symbols declared earlier that this call uses in its formula but does not declare again."""
    formula = " ".join(str(arguments[k]) for k in MATH_FIELDS if k in arguments)
    given = arguments.get("assumptions")
    given = given if isinstance(given, dict) else {}
    return [
        symbol
        for symbol in declared
        if symbol not in given
        and re.search(rf"(?<![A-Za-z\\]){re.escape(symbol)}(?![A-Za-z])", formula)
    ]


@dataclass
class _Step:
    """What one model call produced."""

    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None
    latency_ms: int = 0
    first_token_ms: int | None = None
    timed_out: bool = False
    error: LLMError | None = None


async def _model_step(
    llm: LLMClient,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    seconds: float,
) -> AsyncIterator[dict[str, Any]]:
    """Stream one model call: yields text events, then a final `_step` record (not for the client)."""
    step = _Step()
    parts: list[str] = []  # the raw text of the call
    gate = ReasoningGate()  # what may be shown while it streams: no leaked reasoning
    shown: list[str] = []
    started = time.monotonic()
    # Text that starts like JSON or a code fence may be a tool call written as prose: it is held back until
    # the call ends, and shown only if it turns out to be prose. (The stored reply never contained it; the
    # live view used to show it and then lose it.) Once the text starts like anything else it streams as usual.
    holding = True

    def visible(delta: str) -> list[dict[str, Any]]:
        nonlocal holding
        if not delta:
            return []
        shown.append(delta)
        if holding:
            so_far = "".join(shown).lstrip()
            if not so_far or so_far.startswith(("{", "`")):
                return []
            holding = False
            return [{"event": "text", "data": {"delta": "".join(shown)}}]
        return [{"event": "text", "data": {"delta": delta}}]

    try:
        # The model host's own timeout is per read: a stream that keeps trickling is cut here.
        async with asyncio.timeout(seconds):
            async for chunk in llm.chat(messages, tools):
                if chunk.text:
                    if step.first_token_ms is None:
                        step.first_token_ms = _ms(started)
                    parts.append(chunk.text)
                    for event in visible(gate.feed(chunk.text)):
                        yield event
                step.calls.extend(chunk.tool_calls)
                if chunk.usage is not None:
                    step.usage = chunk.usage
    except TimeoutError:
        step.timed_out = True
    except LLMError as exc:
        step.error = exc
    for event in visible(gate.finish()):
        yield event
    # What is stored is the raw text without leaked reasoning, wherever in it the reasoning sits.
    step.text = clean_reply("".join(parts))
    if holding and step.text.strip() and not looks_like_tool_call(step.text):
        yield {"event": "text", "data": {"delta": "".join(shown) or step.text}}
    step.latency_ms = _ms(started)
    yield {"event": "_step", "data": step}


def _finish(step: _Step) -> str:
    if step.timed_out:
        return "timeout"
    if step.error is not None:
        return "error"
    if step.usage is not None and step.usage.finish_reason == "length":
        return "length"
    if step.calls:
        return "tool_calls"
    return (step.usage.finish_reason if step.usage else None) or "stop"


def _tool_failure(name: str, why: str, category: ErrorCategory) -> ToolOutput:
    # Fixed text: no expression, no exception message (conversations are never logged or leaked).
    return _error(name, f"the tool could not finish: {why}", category.value)


async def run_agent(
    llm: LLMClient,
    history: Sequence[dict[str, Any]],
    ctx: ToolContext,
    limits: AgentLimits | None = None,
    *,
    prior_known: set[str] | None = None,
    prior_quoted: Mapping[str, str] | None = None,
    prior_relations: Sequence[str] = (),
    prior_assumptions: dict[str, list[str]] | None = None,
    prior_matrices: Sequence[str] = (),
    prior_results: Mapping[str, str] | None = None,
    prior_restrictions: Mapping[str, Sequence[str]] | None = None,
    prior_groups: Sequence[str] = (),
    params: EffectiveParams | None = None,
    equations: EquationTable | None = None,
    prior_unverified: Mapping[str, Sequence[str]] | None = None,
    info: TurnInfo | None = None,
    declared_for_tools: Mapping[str, str] | None = None,
    prior_evidence: Sequence[Mapping[str, Any]] = (),
) -> AsyncIterator[dict[str, Any]]:
    """Yield events: {"event": "text"|"tool_call"|"card"|"message"|"error", "data": {...}}.

    A failed turn yields `error` and then still a `message` holding what was done before the failure,
    so the caller can store the trace.
    """
    limits = limits or AgentLimits()
    info = info or TurnInfo()
    tracer = Tracer(info.trace_id or uuid.uuid4().hex, info.sink)
    activate(tracer, ctx.engine)
    try:
        with trace_scope(tracer.trace_id):
            async for event in _run_turn(
                llm,
                history,
                ctx,
                limits,
                info,
                tracer,
                prior_known or set(),
                prior_quoted or {},
                prior_relations,
                prior_assumptions,
                prior_results,
                prior_matrices,
                prior_restrictions,
                prior_groups,
                params,
                equations,
                prior_unverified,
                declared_for_tools,
                prior_evidence,
            ):
                yield event
    finally:
        deactivate(tracer)


async def _run_turn(
    llm: LLMClient,
    history: Sequence[dict[str, Any]],
    ctx: ToolContext,
    limits: AgentLimits,
    info: TurnInfo,
    tracer: Tracer,
    prior_known: set[str],
    prior_quoted: Mapping[str, str],
    prior_relations: Sequence[str],
    prior_assumptions: dict[str, list[str]] | None,
    prior_results: Mapping[str, str] | None,
    prior_matrices: Sequence[str] = (),
    prior_restrictions: Mapping[str, Sequence[str]] | None = None,
    prior_groups: Sequence[str] = (),
    params: EffectiveParams | None = None,
    equations: EquationTable | None = None,
    prior_unverified: Mapping[str, Sequence[str]] | None = None,
    declared_for_tools: Mapping[str, str] | None = None,
    prior_evidence: Sequence[Mapping[str, Any]] = (),
) -> AsyncIterator[dict[str, Any]]:
    started = time.monotonic()
    # Without parameters (tests, scripts that predate them) everything is sent and nothing is trimmed.
    toolset = params.toolset if params is not None else "all"
    active: list[str] = grow(prior_groups, conversation_groups(_events(history)))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt_for(active, toolset)},
        *history,
    ]
    kept_history = len(history)
    store = ResultStore.from_prior(
        prior_results,
        prior_restrictions,
        equations.items if equations is not None else None,
        prior_unverified,
    )
    result_ids: dict[str, str] = {}  # card id -> the id of the verified result it is
    outputs: list[ToolOutput] = []
    order: list[str | Card] = []  # prose and cards in the order they happened
    trace: list[TraceStep] = []
    outcomes: Outcomes = []
    full_text: list[str] = []
    debug = TurnDebug(
        request_id=info.request_id,
        api_version=info.api_version,
        mode=info.mode,
        provider=getattr(llm, "provider", ""),
        model=llm.name,
        prompt_sha=prompt_sha(),
        max_steps=limits.max_steps,
        max_seconds=limits.max_seconds,
        max_tool_calls=limits.max_tool_calls,
        trace_id=tracer.trace_id,
        params=params.model_dump(mode="json") if params is not None else {},
        context_window=params.num_ctx if params is not None else 0,
    )
    declared: dict[str, list[str]] = dict(prior_assumptions or {})
    seen: dict[str, str] = {}
    # A call whose arguments failed validation, with what the model was told: sending the same
    # arguments again is told the same thing, not "its result is above" (there is no result).
    invalid_seen: dict[str, str] = {}
    stop = StopReason.COMPLETED
    failure: str | None = None
    failure_kind: str | None = None  # LLMError.kind, or `internal`
    failure_site: dict[str, Any] = {}  # class and code location of an unexpected exception
    failure_category: ErrorCategory | None = None
    tool_calls_made = repeats = unproductive = malformed_streak = 0
    # Differential equations: when the analysis offers several methods and the user's message named
    # none, a solver call is refused and the model asks which one (`needs_user_choice`).
    user_text, replying = _latest_user_message(history)
    method_named = names_a_method(user_text, replying=replying)
    # The user's messages of the whole conversation (the route passes them; the history window may have
    # dropped old ones): did they hand the numbers over, and which numbers did they write?
    user_messages = (
        list(ctx.user_messages)
        if ctx.user_messages is not None
        else [str(m.get("content") or "") for m in history if m.get("role") == "user"]
    )
    numbers_delegated = delegation_active(user_messages)
    user_numbers = frozenset(x for text in user_messages for x in numbers_in(text))
    offered: tuple[str, ...] = ()
    host_retried = False
    step = 0

    def remaining() -> float:
        return limits.max_seconds - (time.monotonic() - started)

    tracer.emit(
        AgentStarted,
        request_id=info.request_id,
        model=llm.name,
        prompt_sha=debug.prompt_sha,
        mode=info.mode,
        api_version=info.api_version,
        max_steps=limits.max_steps,
        max_seconds=limits.max_seconds,
        max_tool_calls=limits.max_tool_calls,
        history_messages=len(history),
    )
    pre: _Routed | None = None
    try:
        if info.routing and not replying and (route := plan_route(user_text)) is not None:
            pre = await _run_route(route, ctx, tracer, debug, remaining)
            if pre is not None and toolset == "auto":
                needed = group_of(pre.call.name)
                active = grow(active, [needed] if needed else [])
        while step < limits.max_steps:
            if remaining() <= 0:
                stop = StopReason.TIME_LIMIT
                break
            step += 1
            routed_now = pre if step == 1 else None
            tools_now = _schemas_for(tool_names(active, toolset))
            system_now = prompt_for(active, toolset)
            trimmed = 0
            if params is not None:
                reserve = min(params.num_predict or 4096, params.num_ctx // 4)
                current = messages[1 + kept_history :]
                fixed = (
                    len(system_now)
                    + len(json.dumps(tools_now))
                    + sum(message_chars(m) for m in current)
                )
                limit = params.num_ctx - reserve
                kept, _ = fit_history(history, fixed_chars=fixed, limit_tokens=limit)
                if params.history_tokens:
                    kept, _ = fit_history(kept, fixed_chars=0, limit_tokens=params.history_tokens)
                trimmed = len(history) - len(kept)
                messages = [{"role": "system", "content": system_now}, *kept, *current]
                kept_history = len(kept)
                debug.history_dropped = max(debug.history_dropped, trimmed)
            else:
                messages[0] = {"role": "system", "content": system_now}
            estimate = estimate_tokens(request_chars(system_now, tools_now, messages[1:]))
            breakdown = {
                "est_system": estimate_tokens(len(system_now)),
                "est_tools": estimate_tokens(len(json.dumps(tools_now))),
                "est_history": estimate_tokens(
                    sum(message_chars(m) for m in messages[1 : 1 + kept_history])
                ),
                "est_turn": estimate_tokens(
                    sum(message_chars(m) for m in messages[1 + kept_history :])
                ),
            }
            debug.prompt_estimate_tokens = max(debug.prompt_estimate_tokens, estimate)
            debug.groups = list(active)
            if routed_now is None:
                tracer.emit(
                    LLMRequest,
                    step,
                    message_count=len(messages),
                    tool_count=len(tools_now),
                    prompt_estimate_tokens=estimate,
                    window=params.num_ctx if params is not None else 0,
                    dropped_messages=trimmed,
                    groups_loaded=len(active),
                )
            result = _Step()
            if (
                routed_now is not None
            ):  # the call was decided by rule and already ran: no model call
                result = _Step(calls=[routed_now.call])
            else:
                async for ev in _model_step(llm, messages, tools_now, remaining()):
                    if ev["event"] == "_step":
                        result = ev["data"]
                    else:
                        yield ev
            usage = result.usage
            if routed_now is None:
                debug.llm_calls.append(
                    LLMCallDebug(
                        ms=result.latency_ms,
                        text_chars=len(result.text),
                        tool_calls=len(result.calls),
                        first_token_ms=result.first_token_ms,
                        finish_reason=_finish(result),
                        prompt_tokens=usage.prompt_tokens if usage else None,
                        completion_tokens=usage.completion_tokens if usage else None,
                        cost_usd=usage.cost_usd if usage else None,
                        **breakdown,
                    )
                )
                tracer.emit(
                    LLMResponse,
                    step,
                    latency_ms=result.latency_ms,
                    first_token_ms=result.first_token_ms,
                    finish_reason=_finish(result),
                    tool_calls=len(result.calls),
                    proposed_tool=_sanitized(result.calls[0].name) if result.calls else None,
                    text_chars=len(result.text),
                    tokens_in=usage.prompt_tokens if usage else None,
                    tokens_out=usage.completion_tokens if usage else None,
                )
            text = result.text
            calls, aliases = _resolve_calls(result.calls)

            if result.timed_out:
                stop = StopReason.TIME_LIMIT
                if text:
                    full_text.append(text)
                    order.append(text)
                break
            if result.error is not None:
                if (
                    not (text or calls)
                    and not host_retried
                    and remaining() > limits.retry_delay_seconds + 1
                ):
                    # Nothing arrived: one retry is safe, nothing mathematical is repeated.
                    host_retried = True
                    tracer.emit(AgentRetry, step, reason=RetryReason.MODEL_HOST, attempt=1)
                    await asyncio.sleep(limits.retry_delay_seconds)
                    step -= 1  # a retry is not a new step
                    continue
                stop = StopReason.MODEL_ERROR
                failure = str(result.error)
                failure_kind = result.error.kind
                failure_category = ErrorCategory.MODEL_HOST
                if text:
                    full_text.append(text)
                    order.append(text)
                break

            if not calls and text and looks_like_tool_call(text):
                # A tool call written as prose: never shown as an answer; one correction, then stop.
                malformed_streak += 1
                if malformed_streak >= 2:
                    stop, failure_category = (
                        StopReason.MALFORMED_OUTPUT,
                        ErrorCategory.MALFORMED_MODEL_OUTPUT,
                    )
                    break
                tracer.emit(
                    AgentRetry,
                    step,
                    reason=RetryReason.MALFORMED_MODEL_OUTPUT,
                    attempt=malformed_streak,
                )
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": TEXT_TOOL_NOTICE},
                ]
                continue
            if text:
                full_text.append(text)
                order.append(text)
            if not calls:
                if _finish(result) == "length":
                    # Cut off by the model's own length limit, mid-sentence or mid-reasoning: not an
                    # answer. Said plainly, so the user (and the next turn) know what happened.
                    stop = StopReason.OUTPUT_LIMIT
                break  # an answer without more tool calls: the turn is complete

            messages.append(
                {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls
                    ],
                }
            )
            all_malformed = all(c.malformed or not c.name for c in calls)
            for call in calls:
                log_name = _sanitized(call.name)
                if tool_calls_made >= limits.max_tool_calls or remaining() <= 0:
                    capped = tool_calls_made >= limits.max_tool_calls
                    if stop is StopReason.COMPLETED:
                        stop = StopReason.TOOL_CALL_LIMIT if capped else StopReason.TIME_LIMIT
                    call_id = uuid.uuid4().hex[:8]
                    trace.append(
                        TraceStep(
                            id=call_id,
                            call_id=call.id,
                            name=call.name or "unknown",
                            arguments=call.arguments,
                            for_model=CAP_MESSAGE,
                            status="skipped",
                            error_category=ErrorCategory.LIMIT_REACHED.value,
                        )
                    )
                    tracer.emit(
                        ToolCompleted,
                        step,
                        tool=log_name,
                        call_id=call_id,
                        status="skipped",
                        error_category=ErrorCategory.LIMIT_REACHED,
                    )
                    _emit_outcome(
                        tracer,
                        outcomes,
                        step,
                        name=call.name,
                        log_name=log_name,
                        call_id=call_id,
                        status="skipped",
                        category=ErrorCategory.LIMIT_REACHED,
                        arguments=call.arguments,
                        valid=True,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": CAP_MESSAGE,
                        }
                    )
                    continue
                tool_calls_made += 1
                call_id = uuid.uuid4().hex[:8]
                if toolset == "auto" and (needed := group_of(call.name)) is not None:
                    active = grow(active, [needed])  # calling a tool of a group loads the group
                arguments, references, refusal = call.arguments, 0, None
                inherited: list[str] = []
                inherited_from: list[str] = []
                unverified_inputs: list[str] = []
                input_notes: tuple[Normalization, ...] = ()
                if call.malformed or not call.name:
                    defect = MALFORMED_TEXT.get(call.malformed or "missing_name", "it was unusable")
                    refusal = _error(
                        call.name or "unknown",
                        f"the tool call could not be used: {defect}. Send a valid call with a JSON "
                        "object as arguments.",
                        ErrorCategory.MALFORMED_MODEL_OUTPUT.value,
                        arguments_valid=False,
                    )
                else:
                    resolution = store.resolve(call.arguments)
                    if resolution.error:
                        refusal = _error(
                            call.name,
                            resolution.error,
                            ErrorCategory.INVALID_ARGUMENTS.value,
                            arguments_valid=False,
                        )
                    else:
                        arguments, references = resolution.arguments, resolution.count
                        inherited, inherited_from = resolution.inherited, resolution.inherited_from
                        unverified_inputs = resolution.unverified
                        # Transport damage (markdown escapes, a `$` wrapper, `L =`) is removed before the
                        # call is keyed, so a retry that differs only by it is the same call.
                        arguments, input_notes = prepare_arguments(call.name, arguments)
                if (
                    refusal is None
                    and call.name in ODE_SOLVERS
                    and len(offered) >= 2
                    and not method_named
                ):
                    refusal = _error(call.name, NEEDS_CHOICE, ErrorCategory.NEEDS_USER_CHOICE.value)
                key = _call_key(call.name, arguments)
                if refusal is None and key in seen:
                    repeats += 1
                    earlier_error = invalid_seen.get(key)
                    repeat_valid = earlier_error is None
                    repeat_category = (
                        ErrorCategory.REPEATED_CALL
                        if repeat_valid
                        else ErrorCategory.INVALID_ARGUMENTS
                    )
                    repeat_text = (
                        REPEAT_MESSAGE
                        if earlier_error is None
                        else f"{earlier_error}\n{REPEAT_INVALID_MESSAGE}"
                    )
                    trace.append(
                        TraceStep(
                            id=call_id,
                            call_id=call.id,
                            name=call.name,
                            arguments=arguments,
                            for_model=repeat_text,
                            status="skipped",
                            repeated=True,
                            arguments_valid=repeat_valid,
                            error_category=repeat_category.value,
                        )
                    )
                    tracer.emit(
                        ToolStarted,
                        step,
                        tool=log_name,
                        call_id=call_id,
                        arguments_valid=repeat_valid,
                        repeated=True,
                    )
                    tracer.emit(
                        ToolCompleted,
                        step,
                        tool=log_name,
                        call_id=call_id,
                        status="skipped",
                        error_category=repeat_category,
                    )
                    _emit_outcome(
                        tracer,
                        outcomes,
                        step,
                        name=call.name,
                        log_name=log_name,
                        call_id=call_id,
                        status="skipped",
                        category=repeat_category,
                        arguments=arguments,
                        valid=repeat_valid,
                        repeated=True,
                        references=references,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": repeat_text,
                        }
                    )
                    continue

                valid = refusal is None and check_arguments(call.name, arguments)
                tracer.emit(
                    ToolStarted,
                    step,
                    tool=log_name,
                    call_id=call_id,
                    arguments_valid=valid,
                    references=references,
                )
                yield {
                    "event": "tool_call",
                    "data": {"id": call_id, "name": call.name, "arguments": arguments},
                }
                tool_started = time.monotonic()
                mark = tracer.sandbox_mark()
                timed_out = False
                if refusal is not None:
                    output = refusal
                else:
                    # What refuses a scalar tool is the strict reading of earlier messages (a name a
                    # sentence declares, a user sentence can clear it); `prior_matrices`, the generous
                    # reading, only leaves claims unchecked.
                    gate = prior_matrices if declared_for_tools is None else declared_for_tools
                    ctx.matrices = (
                        frozenset(matrix_names(trace))
                        | frozenset(vector_names(trace))
                        | frozenset(gate)
                    )
                    ctx.matrix_sources = dict(declared_for_tools or {})
                    ctx.numbers_delegated, ctx.user_numbers = numbers_delegated, user_numbers
                    ctx.input_notes = input_notes
                    run_tool: Any = execute
                    if routed_now is not None and call.id == routed_now.call.id:
                        run_tool = _already_run(routed_now.output)
                    try:
                        output = await asyncio.wait_for(
                            run_in_threadpool(run_tool, call.name, arguments, ctx),
                            timeout=max(remaining(), 1.0),
                        )
                    except TimeoutError:
                        output = _tool_failure(
                            call.name,
                            "it did not finish within the time limit",
                            ErrorCategory.TIMEOUT,
                        )
                        timed_out = True
                        stop = StopReason.TIME_LIMIT
                    except Exception as exc:
                        output = _tool_failure(
                            call.name, "it failed unexpectedly", ErrorCategory.TOOL_FAILURE
                        )
                        output.error_site = error_fields(
                            exc
                        )  # class and location, never the message
                    finally:  # only this call is the model's: the context is not left marked
                        ctx.numbers_delegated, ctx.user_numbers = False, None
                        ctx.input_notes = ()
                    seen[key] = call_id
                    if not output.arguments_valid:
                        invalid_seen[key] = output.for_model
                if (
                    call.name == LOAD_TOOLS
                    and refusal is None
                    and output.status == "ok"
                    and toolset == "auto"
                ):
                    active = grow(active, requested_groups(arguments))
                if refusal is None and call.name == "ode_analyze" and output.status == "ok":
                    offered = output.choices
                duration = _ms(tool_started)
                sandbox = tracer.sandbox_since(mark)
                engine_ms = sum(c.ms for c in sandbox) if sandbox else None
                bad_outcome = next((c.outcome for c in sandbox if c.outcome != "ok"), None)
                site = next((c for c in sandbox if c.error_type), None)
                if site is not None and not output.error_site:
                    # the sandbox child's own exception: class and where our code raised it
                    output.error_site = {
                        "error_type": site.error_type,
                        "error_module": site.error_module,
                        "error_line": site.error_line,
                    }
                category = category_for(output.status, bad_outcome)
                if timed_out:
                    category = ErrorCategory.TIMEOUT
                error_category = category.value if category else output.error_category
                if refusal is None and inherited and output.status == "ok":
                    own = output.card.data.get("restrictions_latex") or []
                    new = [r for r in inherited if r not in own]
                    if new:
                        # The new result is valid only where the one it was computed from was.
                        output.card.data["inherited_restrictions_latex"] = new
                        output.for_model += "\n" + inherited_note(new, inherited_from)
                if refusal is None and unverified_inputs and output.status == "ok":
                    # A verified computation on an input nobody verified: say so, and never let the
                    # result's formulas count as verified (they are quoted, with the reason).
                    output.card.data["unverified_inputs"] = list(unverified_inputs)
                    output.for_model += "\n" + unverified_note(unverified_inputs)
                    output.quoted_latex = [
                        *output.quoted_latex,
                        *((normalize(f), UNVERIFIED_INPUT) for f in output.known_latex if f),
                    ]
                    output.known_latex = []
                result_id = (
                    store.register(
                        call.name,
                        output,
                        inherited if output.status == "ok" else (),
                        unverified_inputs if output.status == "ok" else (),
                    )
                    if refusal is None
                    else None
                )
                if result_id:
                    result_ids[call_id] = result_id

                output.card.id = call_id
                outputs.append(output)
                order.append(output.card)
                dropped = _dropped_assumptions(arguments, declared) if refusal is None else []
                given = arguments.get("assumptions")
                if isinstance(given, dict):
                    for name, items in given.items():
                        if isinstance(items, list):
                            declared[str(name)] = [str(i) for i in items]
                trace.append(
                    TraceStep(
                        id=call_id,
                        call_id=call.id,
                        name=call.name,
                        arguments=arguments,
                        for_model=output.for_model,
                        duration_ms=duration,
                        status=output.status,
                        evidence=output.evidence,
                        arguments_valid=output.arguments_valid and valid,
                        error_category=error_category,
                        engine_ms=engine_ms,
                        result_id=result_id,
                        model_arguments=call.arguments if arguments != call.arguments else None,
                        routed=routed_now is not None and call.id == routed_now.call.id,
                        formula_roles={
                            f: r for f, r in output.formula_roles.items() if f in output.known_latex
                        },
                        variables=list(output.variables),
                        restrictions=[
                            r
                            for key in ("restrictions_latex", "inherited_restrictions_latex")
                            for r in (output.card.data.get(key) or [])
                            if isinstance(r, str)
                        ],
                        requested_name=aliases[call.id][0] if call.id in aliases else None,
                        alias_kind=aliases[call.id][1] if call.id in aliases else None,
                        assumptions_dropped=dropped,
                        verified_latex=list(output.known_latex),
                        quoted=dict(output.quoted_latex),
                    )
                )
                tracer.emit(
                    ToolCompleted,
                    step,
                    tool=log_name,
                    call_id=call_id,
                    status=tool_status(output.status),
                    duration_ms=duration,
                    engine_ms=engine_ms,
                    sandbox_calls=len(sandbox),
                    error_category=ErrorCategory(error_category) if error_category else None,
                    result_id=result_id,
                )
                _emit_outcome(
                    tracer,
                    outcomes,
                    step,
                    name=call.name,
                    log_name=log_name,
                    call_id=call_id,
                    status=output.status,
                    category=ErrorCategory(error_category) if error_category else None,
                    arguments=arguments,
                    valid=output.arguments_valid and valid,
                    references=references,
                    duration_ms=duration,
                    engine_ms=engine_ms,
                    output=output,
                )
                evidence = output.evidence
                if evidence is not None and refusal is None:
                    tracer.emit(
                        VerificationCompleted,
                        step,
                        scope="tool",
                        tool=log_name,
                        check=evidence.check,
                        accepted=evidence.accepted,
                        reason=evidence.reason,
                        points_checked=evidence.points_checked,
                        points_agreed=evidence.points_agreed,
                        symbolic_zero=evidence.symbolic_zero,
                        restrictions=evidence.restrictions,
                        restrictions_complete=evidence.restrictions_complete,
                        verify_ms=engine_ms,
                    )
                accepted = evidence.accepted if evidence else False
                # An answer is not a stuck loop: "unchanged", "unsupported" and "conditions not
                # satisfied" are what the engine established. Only errors, refused or invalid calls
                # and unverified candidates count towards the limit.
                answered = output.status in ANSWER_STATUSES
                unproductive = 0 if (accepted or answered) else unproductive + 1
                yield {"event": "card", "data": output.card.model_dump(mode="json")}
                content = output.for_model + (f"\n{store.note_for(result_id)}" if result_id else "")
                if dropped:
                    content += "\n" + DROPPED_ASSUMPTIONS_NOTE.format(symbols=", ".join(dropped))
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}
                )

            if all_malformed:
                malformed_streak += 1
                if malformed_streak >= 2:
                    stop, failure_category = (
                        StopReason.MALFORMED_OUTPUT,
                        ErrorCategory.MALFORMED_MODEL_OUTPUT,
                    )
                    break
                tracer.emit(
                    AgentRetry,
                    step,
                    reason=RetryReason.MALFORMED_MODEL_OUTPUT,
                    attempt=malformed_streak,
                )
            else:
                malformed_streak = 0
            if stop is not StopReason.COMPLETED:
                break
            if repeats >= MAX_REPEATS or unproductive >= MAX_UNPRODUCTIVE:
                stop = StopReason.REPEATED_CALLS
                failure_category = ErrorCategory.REPEATED_CALL
                break
        else:
            stop = StopReason.STEP_LIMIT

        if stop is StopReason.REPEATED_CALLS and remaining() > MIN_WRAP_UP_SECONDS:
            # One last call without tools, so the user gets what was and was not established.
            messages.append({"role": "user", "content": WRAP_UP_NOTICE})
            step += 1
            tracer.emit(LLMRequest, step, message_count=len(messages), tool_count=0)
            summary = _Step()
            async for ev in _model_step(llm, messages, [], remaining()):
                if ev["event"] == "_step":
                    summary = ev["data"]
                else:
                    yield ev
            debug.llm_calls.append(
                LLMCallDebug(
                    ms=summary.latency_ms,
                    text_chars=len(summary.text),
                    tool_calls=0,
                    first_token_ms=summary.first_token_ms,
                    finish_reason=_finish(summary)
                    if summary.error or summary.timed_out
                    else "stop",
                    prompt_tokens=summary.usage.prompt_tokens if summary.usage else None,
                    completion_tokens=summary.usage.completion_tokens if summary.usage else None,
                    cost_usd=summary.usage.cost_usd if summary.usage else None,
                )
            )
            tracer.emit(
                LLMResponse,
                step,
                latency_ms=summary.latency_ms,
                first_token_ms=summary.first_token_ms,
                finish_reason=_finish(summary),
                text_chars=len(summary.text),
                tokens_in=summary.usage.prompt_tokens if summary.usage else None,
                tokens_out=summary.usage.completion_tokens if summary.usage else None,
            )
            if summary.text and not looks_like_tool_call(summary.text):
                full_text.append(summary.text)
                order.append(summary.text)
    except Exception as exc:
        stop = StopReason.INTERNAL_ERROR
        failure = "The assistant hit an internal error."
        failure_kind = "internal"
        failure_category = ErrorCategory.TOOL_FAILURE
        failure_site = error_fields(exc)  # class and location only, never the message

    # ---- the end of the turn: one place builds the message, whatever way the turn ended ----------
    raw = strip_equation_tags("\n\n".join(full_text).strip())
    order = [strip_equation_tags(i) if isinstance(i, str) else i for i in order]
    if stop is not StopReason.COMPLETED:
        sentence = STOP_SENTENCES[stop]
        raw = (raw + "\n\n" if raw else "") + sentence
        order.append(sentence)
    known = known_formulas(outputs) | prior_known
    quoted = {**prior_quoted, **quoted_formulas(outputs)}
    blocks = [
        Block(kind="card", card=item)
        if isinstance(item, Card)
        else Block(kind="text", segments=split_segments(item, known, quoted))
        for item in order
    ]
    if _pending_claims(blocks):  # formulas the model wrote are about to be tested: tell the user
        yield {"event": "phase", "data": {"phase": "checking"}}
    claims_started = time.monotonic()
    debug.claims, debug.claim_check = await _check_claims_in_budget(
        blocks,
        ctx,
        trace,
        remaining(),
        prior_relations,
        prior_assumptions,
        prior_matrices,
        prior_evidence,
    )
    claims_ms = _ms(claims_started)
    if equations is not None:  # numbers are assigned here, in reading order, never by the model
        number_blocks(blocks, equations, result_ids)
    # A lone symbol (`\\theta`) asserts nothing: it is not counted as a formula that lacks evidence. An
    # expression that is not a relation still counts, and is not sent to the claim checker.
    formulas = [
        s
        for b in blocks
        if b.kind == "text"
        for s in b.segments
        if s.type != "text" and not is_lone_symbol(s.value)
    ]
    backed = sum(1 for s in formulas if s.status in ("from_tool", "checked", "conditional"))
    debug.formulas, debug.formulas_with_evidence = len(formulas), backed
    debug.evidence_pct = round(100 * backed / len(formulas), 1) if formulas else None
    statuses = [c.status for c in debug.claims]
    if debug.claim_check != "none":
        tracer.emit(
            VerificationCompleted,
            scope="claims",
            check="claims",
            accepted=debug.claim_check == "ok",
            reason="verified" if debug.claim_check == "ok" else "error",
            verify_ms=claims_ms,
            claims=len(debug.claims),
            claims_holding=statuses.count("holds") + statuses.count("supported"),
            claims_conditional=statuses.count("conditional"),
            claims_contradicted=statuses.count("fails"),
            claims_unchecked=len(debug.claims)
            - sum(statuses.count(s) for s in ("holds", "supported", "conditional", "fails")),
        )
    debug.duration_ms = _ms(started)
    debug.error = failure
    debug.error_kind = failure_kind
    debug.error_site = failure_site
    debug.stop_reason = stop.value
    in_tokens = [c.prompt_tokens for c in debug.llm_calls if c.prompt_tokens is not None]
    out_tokens = [c.completion_tokens for c in debug.llm_calls if c.completion_tokens is not None]
    tracer.emit(
        AgentStopped,
        reason=stop,
        steps=len(debug.llm_calls),
        tool_calls=len(trace),
        error_category=failure_category,
        **failure_site,
    )
    tracer.emit(
        AgentCompleted,
        status="ok" if failure is None else "error",
        total_ms=debug.duration_ms,
        steps=len(debug.llm_calls),
        tool_calls=len(trace),
        tool_errors=sum(1 for t in trace if t.status == "error"),
        invalid_calls=sum(1 for t in trace if t.arguments_valid is False),
        repeated_calls=repeats,
        llm_ms=sum(c.ms for c in debug.llm_calls),
        tool_ms=sum(t.duration_ms or 0 for t in trace),
        engine_ms=sum(t.engine_ms or 0 for t in trace),
        verify_ms=claims_ms,
        tokens_in=sum(in_tokens) if in_tokens else None,
        tokens_out=sum(out_tokens) if out_tokens else None,
        formulas=debug.formulas,
        formulas_with_evidence=debug.formulas_with_evidence,
        evidence_pct=debug.evidence_pct,
        stop_reason=stop,
        error_category=failure_category,
        events_dropped=tracer.dropped,
        tools_called=list(dict.fromkeys(name for name, _, _ in outcomes)),
        failed_tools=list(dict.fromkeys(n for n, _, kind in outcomes if kind == "failure")),
        declined_tools=list(dict.fromkeys(n for n, _, kind in outcomes if kind == "answer")),
        first_failure_code=next((c for _, c, kind in outcomes if kind == "failure" and c), None),
        **failure_site,
    )
    debug.events = list(tracer.events)
    debug.events_dropped = tracer.dropped
    final = AssistantMessage(
        text=raw,
        blocks=blocks,
        trace=trace,
        truncated=stop is not StopReason.COMPLETED,
        debug=debug,
        stop_reason=stop.value,
        context_trimmed=debug.history_dropped,
    )
    if failure is not None:
        yield {
            "event": "error",
            "data": {
                "message": failure,
                "debug": debug.model_dump(mode="json"),
                "trace": [t.model_dump(mode="json") for t in trace],
            },
        }
    yield {"event": "message", "data": final.model_dump(mode="json")}


def matrix_names(trace: Sequence[TraceStep | dict[str, Any]]) -> set[str]:
    """Names this conversation declared as matrices: in `matrix_calculus` calls, or as
    `noncommutative` in the assumptions of any tool call (a refused or failed call counts: the
    cautious reading)."""
    names: set[str] = set()
    for step in trace:
        name = step.name if isinstance(step, TraceStep) else step.get("name")
        arguments = step.arguments if isinstance(step, TraceStep) else step.get("arguments", {})
        if not isinstance(arguments, dict):
            continue
        if name == "matrix_calculus":
            for decl in arguments.get("matrices", []) or []:
                if isinstance(decl, dict) and isinstance(decl.get("name"), str):
                    names.add(decl["name"])
        declared = arguments.get("assumptions")
        if isinstance(declared, dict):
            names.update(
                str(n)
                for n, items in declared.items()
                if isinstance(items, list) and "noncommutative" in items
            )
    return names


def vector_names(trace: Sequence[TraceStep | dict[str, Any]]) -> set[str]:
    """Names this conversation declared as vector fields in `vector_calculus` calls (a refused or
    failed call counts: the cautious reading)."""
    names: set[str] = set()
    for step in trace:
        name = step.name if isinstance(step, TraceStep) else step.get("name")
        arguments = step.arguments if isinstance(step, TraceStep) else step.get("arguments", {})
        if name != "vector_calculus" or not isinstance(arguments, dict):
            continue
        for decl in arguments.get("fields", []) or []:
            if (
                isinstance(decl, dict)
                and decl.get("kind") == "vector"
                and isinstance(decl.get("name"), str)
            ):
                names.add(decl["name"])
    return names


def _prose_of(blocks: list[Block]) -> list[str]:
    """The model's own text of this turn, formulas put back in `$...$`, to read declarations from."""
    texts: list[str] = []
    for b in blocks:
        if b.kind == "text":
            texts.append(
                "".join(s.value if s.type == "text" else f"${s.value}$" for s in b.segments)
            )
    return texts


MAX_CLAIMS = 12
MAX_EVIDENCE = 40
# An equation or inequality is a claim Minerva can test; a lone symbol such as `r` or `\\theta` is not
# (same rule as `isClaim` in apps/web/src/components/chat/rich-text.tsx).
_RELATION = re.compile(r"=|<|>|\\(?:le|ge|leq|geq|neq|ne|approx|equiv|sim|propto|to|implies)\b")


def is_claim(latex: str) -> bool:
    return _RELATION.search(latex) is not None


# A symbol on its own: a letter or Greek command, optionally accented, subscripted or primed (`\\theta`,
# `m`, `x_1`, `\\dot{\\phi}`, `p_{\\theta}`). Naming it asserts nothing, so there is nothing to verify.
_LONE_SYMBOL = re.compile(
    r"\s*(?:\\(?:dot|ddot|bar|hat|vec|tilde)\s*\{\s*)?(?:\\[A-Za-z]+|[A-Za-z])\s*\}?"
    r"(?:_\s*(?:\{[^{}]*\}|\\[A-Za-z]+|\w))?(?:'+)?\s*"
)


def is_lone_symbol(latex: str) -> bool:
    return _LONE_SYMBOL.fullmatch(latex) is not None


def claim_evidence(
    trace: Sequence[TraceStep], prior: Sequence[Mapping[str, Any]] = ()
) -> list[ClaimEvidence]:
    """The verified equations of this conversation, newest last, for linking the model's equations to them.

    Only formulas a verified result produced (`verified_latex`) and that are equations. How each was
    established comes from the engine's own result; a result without a symbolic zero is `numeric`.
    """
    rows: list[Mapping[str, Any]] = [*prior, *(step.model_dump() for step in trace)]
    found: list[ClaimEvidence] = []
    for row in rows:
        if row.get("status") not in ("ok", None) or row.get("repeated"):
            continue
        evidence = row.get("evidence") or {}
        symbolic = isinstance(evidence, Mapping) and evidence.get("check") == "symbolic_zero"
        roles = row.get("formula_roles") or {}
        for latex in row.get("verified_latex") or []:
            if "=" not in latex:
                continue
            found.append(
                ClaimEvidence(
                    latex=latex,
                    source=str(row.get("result_id") or ""),
                    basis="symbolic" if symbolic else "numeric",
                    role=str(roles.get(latex, "")),
                    variables=[str(v) for v in row.get("variables") or []],
                    restrictions=[str(r) for r in row.get("restrictions") or []],
                )
            )
    return found[-MAX_EVIDENCE:]


def _supported_note(result: Any) -> str:
    where = (
        " It holds where " + ", ".join(f"${r}$" for r in result.restrictions)
        if result.restrictions
        else ""
    )
    basis = (
        "equivalent by a symbolic zero"
        if result.evidence_basis == "symbolic"
        else "equivalent when sampled"
    )
    return f"{result.detail}; {basis}.{where}"


CLAIM_CHECK_MIN_SECONDS = 2.0  # the claim check may run a little past the deadline, never unbounded


async def _check_claims_in_budget(
    blocks: list[Block],
    ctx: ToolContext,
    trace: list[TraceStep],
    seconds_left: float,
    prior_relations: Sequence[str] = (),
    prior_assumptions: dict[str, list[str]] | None = None,
    prior_matrices: Sequence[str] = (),
    prior_evidence: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[ClaimDebug], str]:
    """Run the claim check inside what is left of the turn's time.

    On any failure the claims stay unchecked, and the reason is recorded: `none` (nothing to check),
    `ok`, `timeout` or `error`. "Could not check" is never the same as "nothing to check".
    """
    if not _pending_claims(blocks):
        return [], "none"
    try:
        report = await asyncio.wait_for(
            check_model_claims(
                blocks,
                ctx,
                trace,
                prior_relations,
                prior_assumptions,
                prior_matrices,
                prior_evidence,
            ),
            timeout=max(seconds_left, CLAIM_CHECK_MIN_SECONDS),
        )
        return report, "ok"
    except TimeoutError:
        return [], "timeout"
    except Exception:
        return [], "error"


def _pending_claims(blocks: list[Block]) -> list[Any]:
    return [
        s
        for b in blocks
        if b.kind == "text"
        for s in b.segments
        if s.type != "text"
        and s.status == "unchecked"
        and len(s.value) <= 500
        and is_claim(s.value)
    ][:MAX_CLAIMS]


async def check_model_claims(
    blocks: list[Block],
    ctx: ToolContext,
    trace: list[TraceStep],
    prior_relations: Sequence[str] = (),
    prior_assumptions: dict[str, list[str]] | None = None,
    prior_matrices: Sequence[str] = (),
    prior_evidence: Sequence[Mapping[str, Any]] = (),
) -> list[ClaimDebug]:
    """Test the equations the model wrote itself; mark them checked, contradicted or unchecked.

    Assumptions the model declared in its own tool calls apply to its claims too (an equation about
    a positive x is checked for positive x), as do those of earlier turns. An equation with a symbol
    on one side only is a relation, not an identity: it is never called false by sampling, only
    checked against definitions stated earlier (`prior_relations`, then the model's own formulas in
    order), and the result names them.
    """
    pending = _pending_claims(blocks)
    if not pending:
        return []
    assumptions: dict[str, list[str]] = dict(prior_assumptions or {})
    for step in trace:
        declared = step.arguments.get("assumptions")
        if isinstance(declared, dict):
            for name, items in declared.items():
                if isinstance(items, list):
                    assumptions[str(name)] = [str(i) for i in items]
    prose = _prose_of(blocks)
    matrices = matrix_names(trace) | set(prior_matrices) | declared_matrices(prose)
    results = await run_in_threadpool(
        ctx.engine.check_claims,
        [s.value for s in pending],
        assumptions,
        list(prior_relations),
        sorted(matrices),
        sorted(declared_vectors(prose) | vector_names(trace)),
        claim_evidence(trace, prior_evidence),
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
                evidence_source=result.evidence_source,
                evidence_rule=result.evidence_rule,
                evidence_basis=result.evidence_basis,
            )
        )
        if result.status == "supported":
            # The same statement as a result a tool verified, found by mathematics, not by spelling.
            segment.status = "from_tool"
            segment.note = _supported_note(result)
        elif result.status == "holds":
            segment.status = "checked"
        elif result.status == "conditional":
            segment.status = "conditional"
            segment.note = result.detail
        elif result.status == "fails":
            segment.status = "contradicted"
            segment.note = result.detail
        elif result.status == "not_identity" or result.kind == "relation":
            segment.note = result.detail  # stays unchecked; the reason is shown, never "false"
    return report
