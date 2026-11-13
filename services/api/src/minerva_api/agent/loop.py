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
from minerva_engine import trace_scope

from .declarations import declared_matrices
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
from .prompts import SYSTEM_PROMPT
from .refs import ResultStore
from .segments import split_segments
from .tools import (
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
from .tracing import (
    AgentCompleted,
    AgentRetry,
    AgentStarted,
    AgentStopped,
    ErrorCategory,
    LLMRequest,
    LLMResponse,
    RetryReason,
    Sink,
    StopReason,
    ToolCompleted,
    ToolStarted,
    Tracer,
    VerificationCompleted,
    activate,
    category_for,
    deactivate,
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


# What the user reads when a turn ends for a reason other than a finished answer. Fixed text: it names
# the cause and never repeats anything the user or the model wrote.
STOP_SENTENCES: dict[StopReason, str] = {
    StopReason.STEP_LIMIT: "I stopped here because I reached the step limit for one answer.",
    StopReason.TOOL_CALL_LIMIT: "I stopped here because I reached the limit on tool calls for one answer.",
    StopReason.TIME_LIMIT: "I stopped here because I reached the time limit for one answer.",
    StopReason.REPEATED_CALLS: "I stopped here because further tool calls were not producing new results.",
    StopReason.MODEL_ERROR: "I stopped here because the model host stopped answering.",
    StopReason.MALFORMED_OUTPUT: "I stopped here because the model's tool calls could not be used.",
    StopReason.INTERNAL_ERROR: "I stopped here because of an internal error.",
}
REPEAT_MESSAGE = (
    "error: this exact call already ran in this turn; its result is above. Do not repeat it: use that "
    "result, or tell the user what could not be established."
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
MAX_UNPRODUCTIVE = 3  # consecutive executed calls that established nothing
MAX_REPEATS = 2  # identical calls in one turn before the turn is wrapped up
MIN_WRAP_UP_SECONDS = 3.0


def prompt_sha() -> str:
    """Short fingerprint of the system prompt and tool schemas: which version produced a turn."""
    blob = SYSTEM_PROMPT + json.dumps(schemas(), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


def _sanitized(name: str) -> str:
    """A tool name that is safe to put in an event: one of the registry's, or `unknown`."""
    return name if name in TOOLS else "unknown"


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
    formula = " ".join(
        str(arguments[k])
        for k in ("latex", "expression", "equation", "lagrangian")
        if k in arguments
    )
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
    parts: list[str] = []
    started = time.monotonic()
    try:
        # The model host's own timeout is per read: a stream that keeps trickling is cut here.
        async with asyncio.timeout(seconds):
            async for chunk in llm.chat(messages, tools):
                if chunk.text:
                    if step.first_token_ms is None:
                        step.first_token_ms = _ms(started)
                    parts.append(chunk.text)
                    yield {"event": "text", "data": {"delta": chunk.text}}
                step.calls.extend(chunk.tool_calls)
                if chunk.usage is not None:
                    step.usage = chunk.usage
    except TimeoutError:
        step.timed_out = True
    except LLMError as exc:
        step.error = exc
    step.text = "".join(parts)
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
    info: TurnInfo | None = None,
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
) -> AsyncIterator[dict[str, Any]]:
    started = time.monotonic()
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    store = ResultStore.from_prior(prior_results)
    outputs: list[ToolOutput] = []
    order: list[str | Card] = []  # prose and cards in the order they happened
    trace: list[TraceStep] = []
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
    )
    declared: dict[str, list[str]] = dict(prior_assumptions or {})
    seen: dict[str, str] = {}
    stop = StopReason.COMPLETED
    failure: str | None = None
    failure_kind: str | None = None  # LLMError.kind, or `internal`
    failure_category: ErrorCategory | None = None
    tool_calls_made = repeats = unproductive = malformed_streak = 0
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
    try:
        while step < limits.max_steps:
            if remaining() <= 0:
                stop = StopReason.TIME_LIMIT
                break
            step += 1
            tracer.emit(LLMRequest, step, message_count=len(messages), tool_count=len(TOOLS))
            result = _Step()
            async for ev in _model_step(llm, messages, schemas(), remaining()):
                if ev["event"] == "_step":
                    result = ev["data"]
                else:
                    yield ev
            usage = result.usage
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
            text, calls = result.text, result.calls

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
                arguments, references, refusal = call.arguments, 0, None
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
                key = _call_key(call.name, arguments)
                if refusal is None and key in seen:
                    repeats += 1
                    trace.append(
                        TraceStep(
                            id=call_id,
                            call_id=call.id,
                            name=call.name,
                            arguments=arguments,
                            for_model=REPEAT_MESSAGE,
                            status="skipped",
                            repeated=True,
                            arguments_valid=True,
                            error_category=ErrorCategory.REPEATED_CALL.value,
                        )
                    )
                    tracer.emit(
                        ToolStarted,
                        step,
                        tool=log_name,
                        call_id=call_id,
                        arguments_valid=True,
                        repeated=True,
                    )
                    tracer.emit(
                        ToolCompleted,
                        step,
                        tool=log_name,
                        call_id=call_id,
                        status="skipped",
                        error_category=ErrorCategory.REPEATED_CALL,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": REPEAT_MESSAGE,
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
                    ctx.matrices = frozenset(matrix_names(trace)) | frozenset(prior_matrices)
                    try:
                        output = await asyncio.wait_for(
                            run_in_threadpool(execute, call.name, arguments, ctx),
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
                    except Exception:
                        output = _tool_failure(
                            call.name, "it failed unexpectedly", ErrorCategory.TOOL_FAILURE
                        )
                    seen[key] = call_id
                duration = _ms(tool_started)
                sandbox = tracer.sandbox_since(mark)
                engine_ms = sum(c.ms for c in sandbox) if sandbox else None
                bad_outcome = next((c.outcome for c in sandbox if c.outcome != "ok"), None)
                category = category_for(output.status, bad_outcome)
                if timed_out:
                    category = ErrorCategory.TIMEOUT
                error_category = category.value if category else output.error_category
                result_id = store.register(call.name, output) if refusal is None else None

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
                unproductive = 0 if accepted else unproductive + 1
                yield {"event": "card", "data": output.card.model_dump(mode="json")}
                content = output.for_model + (f"\n{store.note_for(result_id)}" if result_id else "")
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
    except Exception:
        stop = StopReason.INTERNAL_ERROR
        failure = "The assistant hit an internal error."
        failure_kind = "internal"
        failure_category = ErrorCategory.TOOL_FAILURE

    # ---- the end of the turn: one place builds the message, whatever way the turn ended ----------
    raw = "\n\n".join(full_text).strip()
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
    claims_started = time.monotonic()
    debug.claims, debug.claim_check = await _check_claims_in_budget(
        blocks, ctx, trace, remaining(), prior_relations, prior_assumptions, prior_matrices
    )
    claims_ms = _ms(claims_started)
    formulas = [s for b in blocks if b.kind == "text" for s in b.segments if s.type != "text"]
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
            claims_holding=statuses.count("holds"),
            claims_conditional=statuses.count("conditional"),
            claims_contradicted=statuses.count("fails"),
            claims_unchecked=len(debug.claims)
            - sum(statuses.count(s) for s in ("holds", "conditional", "fails")),
        )
    debug.duration_ms = _ms(started)
    debug.error = failure
    debug.error_kind = failure_kind
    debug.stop_reason = stop.value
    in_tokens = [c.prompt_tokens for c in debug.llm_calls if c.prompt_tokens is not None]
    out_tokens = [c.completion_tokens for c in debug.llm_calls if c.completion_tokens is not None]
    tracer.emit(
        AgentStopped,
        reason=stop,
        steps=len(debug.llm_calls),
        tool_calls=len(trace),
        error_category=failure_category,
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
CLAIM_CHECK_MIN_SECONDS = 2.0  # the claim check may run a little past the deadline, never unbounded


async def _check_claims_in_budget(
    blocks: list[Block],
    ctx: ToolContext,
    trace: list[TraceStep],
    seconds_left: float,
    prior_relations: Sequence[str] = (),
    prior_assumptions: dict[str, list[str]] | None = None,
    prior_matrices: Sequence[str] = (),
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
                blocks, ctx, trace, prior_relations, prior_assumptions, prior_matrices
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
        if s.type != "text" and s.status == "unchecked" and len(s.value) <= 500
    ][:MAX_CLAIMS]


async def check_model_claims(
    blocks: list[Block],
    ctx: ToolContext,
    trace: list[TraceStep],
    prior_relations: Sequence[str] = (),
    prior_assumptions: dict[str, list[str]] | None = None,
    prior_matrices: Sequence[str] = (),
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
    matrices = matrix_names(trace) | set(prior_matrices) | declared_matrices(_prose_of(blocks))
    results = await run_in_threadpool(
        ctx.engine.check_claims,
        [s.value for s in pending],
        assumptions,
        list(prior_relations),
        sorted(matrices),
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
        elif result.status == "not_identity" or result.kind == "relation":
            segment.note = result.detail  # stays unchecked; the reason is shown, never "false"
    return report
