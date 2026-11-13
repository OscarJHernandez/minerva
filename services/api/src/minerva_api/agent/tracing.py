"""Structured, content-free tracing of one agent request.

Every request gets a `trace_id`. The agent loop, the tools and the engine's sandbox report to one
`Tracer`, which records typed events in order (kept with the turn for debug mode) and may pass each to a
sink (the server log). The design rule that matters: **an event cannot hold content.** Its fields are
identifiers, counts, durations, booleans and members of fixed enums; there is no field for an
expression, a tool argument, a tool result, model prose or an exception message. A test fails if a free
text field is added, and the log allow-list (`logging.py`) is checked against these models.

Detailed content (arguments and results) lives only in the user's own stored message, next to the
conversation it belongs to, under the same access, deletion and export rules. It is never written here.

Observing must never change a computation: `Tracer.emit` swallows its own failures and counts them.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Literal

from minerva_engine import Engine, SandboxCall
from pydantic import BaseModel

from .events import CheckKind, EvidenceReason

MAX_EVENTS = 400  # per turn; the rest are counted, not kept


class ErrorCategory(StrEnum):
    """Why a step went wrong, in the vocabulary of the benchmark's failure classes."""

    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    PARSE_ERROR = "parse_error"
    CONDITIONS_NOT_SATISFIED = "conditions_not_satisfied"
    UNSUPPORTED = "unsupported"
    NOT_VERIFIED = "not_verified"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    SANDBOX_KILLED = "sandbox_killed"
    BUSY = "busy"
    TOOL_FAILURE = "tool_failure"
    MODEL_HOST = "model_host"
    MALFORMED_MODEL_OUTPUT = "malformed_model_output"
    REPEATED_CALL = "repeated_call"
    LIMIT_REACHED = "limit_reached"


class StopReason(StrEnum):
    """Why a turn ended."""

    COMPLETED = "completed"  # the model answered without asking for more tools
    STEP_LIMIT = "step_limit"
    TOOL_CALL_LIMIT = "tool_call_limit"
    TIME_LIMIT = "time_limit"
    REPEATED_CALLS = "repeated_calls"  # the model kept making calls that could not help
    MODEL_ERROR = "model_error"  # the model host failed
    MALFORMED_OUTPUT = "malformed_output"  # the model kept producing unusable tool calls
    INTERNAL_ERROR = "internal_error"


class RetryReason(StrEnum):
    MODEL_HOST = "model_host"
    MALFORMED_MODEL_OUTPUT = "malformed_model_output"


FinishReason = Literal["stop", "length", "tool_calls", "other", "error", "timeout"]
ToolStatus = Literal[
    "ok",
    "unchanged",
    "not_verified",
    "conditions_not_satisfied",
    "unsupported",
    "parse_error",
    "resource_limit",
    "internal_error",
    "error",
    "no_results",
    "skipped",
    "other",
]
_TOOL_STATUSES = set(ToolStatus.__args__)  # type: ignore[attr-defined]

_STATUS_CATEGORY: dict[str, ErrorCategory] = {
    "parse_error": ErrorCategory.PARSE_ERROR,
    "conditions_not_satisfied": ErrorCategory.CONDITIONS_NOT_SATISFIED,
    "unsupported": ErrorCategory.UNSUPPORTED,
    "not_verified": ErrorCategory.NOT_VERIFIED,
    "resource_limit": ErrorCategory.RESOURCE_LIMIT,
    "internal_error": ErrorCategory.TOOL_FAILURE,
}
_OUTCOME_CATEGORY: dict[str, ErrorCategory] = {
    "timeout": ErrorCategory.TIMEOUT,
    "killed": ErrorCategory.SANDBOX_KILLED,
    "busy": ErrorCategory.BUSY,
    "error": ErrorCategory.TOOL_FAILURE,
}


def tool_status(status: str | None) -> ToolStatus:
    """A tool's status as a member of the fixed set (anything else is reported as `other`)."""
    return status if status in _TOOL_STATUSES else "other"  # type: ignore[return-value]


def category_for(status: str | None, sandbox_outcome: str | None = None) -> ErrorCategory | None:
    """The error category of a tool result, if it is a failure to deliver a verified result.

    `unchanged` and `no_results` are honest answers, not errors. A resource limit is refined by the
    sandbox's own outcome (timeout, killed, busy) when the engine reported it.
    """
    if status == "resource_limit" and sandbox_outcome in _OUTCOME_CATEGORY:
        return _OUTCOME_CATEGORY[sandbox_outcome]
    return _STATUS_CATEGORY.get(status or "")


# ------------------------------------------------------------------------------------------ events
class TraceEvent(BaseModel):
    trace_id: str
    seq: int
    t_ms: int  # monotonic milliseconds since the turn started
    step: int | None = None  # the model step (1-based) the event belongs to


class AgentStarted(TraceEvent):
    event: Literal["agent.started"] = "agent.started"
    request_id: str = ""
    model: str = ""
    prompt_sha: str = ""
    mode: str = ""
    api_version: str = ""
    max_steps: int = 0
    max_seconds: float = 0
    max_tool_calls: int = 0
    history_messages: int = 0


class LLMRequest(TraceEvent):
    event: Literal["llm.request"] = "llm.request"
    message_count: int = 0
    tool_count: int = 0


class LLMResponse(TraceEvent):
    event: Literal["llm.response"] = "llm.response"
    latency_ms: int = 0
    first_token_ms: int | None = None
    finish_reason: FinishReason = "other"
    tool_calls: int = 0
    proposed_tool: str | None = (
        None  # a tool of the registry, or `unknown`: never the model's own text
    )
    text_chars: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None


class ToolStarted(TraceEvent):
    event: Literal["tool.started"] = "tool.started"
    tool: str = ""
    call_id: str = ""
    arguments_valid: bool = True
    repeated: bool = False
    references: int = 0  # how many arguments referred to an earlier result


class ToolCompleted(TraceEvent):
    event: Literal["tool.completed"] = "tool.completed"
    tool: str = ""
    call_id: str = ""
    status: ToolStatus = "other"
    duration_ms: int = 0
    engine_ms: int | None = None
    sandbox_calls: int = 0
    error_category: ErrorCategory | None = None
    result_id: str | None = None


class VerificationCompleted(TraceEvent):
    event: Literal["verification.completed"] = "verification.completed"
    scope: Literal["tool", "claims"] = "tool"
    tool: str | None = None
    check: CheckKind | Literal["claims"] = "none"  # "claims": the check of the model's own formulas
    accepted: bool = False
    reason: EvidenceReason = "error"
    points_checked: int | None = None
    points_agreed: int | None = None
    symbolic_zero: bool | None = None
    restrictions: int | None = None
    restrictions_complete: bool | None = None
    verify_ms: int | None = None
    claims: int | None = None
    claims_holding: int | None = None
    claims_conditional: int | None = None
    claims_contradicted: int | None = None
    claims_unchecked: int | None = None


class AgentRetry(TraceEvent):
    event: Literal["agent.retry"] = "agent.retry"
    reason: RetryReason = RetryReason.MODEL_HOST
    attempt: int = 1


class AgentStopped(TraceEvent):
    event: Literal["agent.stopped"] = "agent.stopped"
    reason: StopReason = StopReason.COMPLETED
    steps: int = 0
    tool_calls: int = 0
    error_category: ErrorCategory | None = None


class AgentCompleted(TraceEvent):
    event: Literal["agent.completed"] = "agent.completed"
    status: Literal["ok", "error"] = "ok"
    total_ms: int = 0
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    invalid_calls: int = 0
    repeated_calls: int = 0
    llm_ms: int = 0
    tool_ms: int = 0
    engine_ms: int = 0  # time inside the sandbox for tools (slot waits included)
    verify_ms: int = 0  # time spent checking the model's own formulas
    tokens_in: int | None = None
    tokens_out: int | None = None
    formulas: int = 0
    formulas_with_evidence: int = 0
    evidence_pct: float | None = None
    stop_reason: StopReason = StopReason.COMPLETED
    error_category: ErrorCategory | None = None
    events_dropped: int = 0


EVENT_MODELS: tuple[type[TraceEvent], ...] = (
    AgentStarted,
    LLMRequest,
    LLMResponse,
    ToolStarted,
    ToolCompleted,
    VerificationCompleted,
    AgentRetry,
    AgentStopped,
    AgentCompleted,
)
# The only plain-text fields an event may have: identifiers and configuration names, never content.
IDENTIFIER_FIELDS = frozenset(
    {
        "trace_id",
        "request_id",
        "model",
        "prompt_sha",
        "mode",
        "api_version",
        "tool",
        "proposed_tool",
        "call_id",
        "result_id",
    }
)


def event_field_names() -> set[str]:
    """Every name an event can carry (what the log allow-list must cover)."""
    names: set[str] = {"event"}
    for model in EVENT_MODELS:
        names |= set(model.model_fields)
    return names


# --------------------------------------------------------------------------------------- the tracer
Sink = Callable[[dict[str, Any]], None]


class Tracer:
    """Collects the events of one turn, in order, and the sandbox calls made on its behalf."""

    def __init__(
        self, trace_id: str, sink: Sink | None = None, max_events: int = MAX_EVENTS
    ) -> None:
        self.trace_id = trace_id
        self._sink = sink
        self._max = max_events
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._seq = 0
        self.events: list[dict[str, Any]] = []
        self.dropped = 0  # events lost to the cap or to a failure; reported, never raised
        self.sandbox: list[SandboxCall] = []

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._started) * 1000)

    def emit(self, event_type: type[TraceEvent], step: int | None = None, /, **fields: Any) -> None:
        """Record an event. Never raises: tracing must not interrupt the mathematics."""
        try:
            with self._lock:
                self._seq += 1
                record = event_type(
                    trace_id=self.trace_id,
                    seq=self._seq,
                    t_ms=self.elapsed_ms(),
                    step=step,
                    **fields,
                ).model_dump(mode="json")
                if len(self.events) < self._max:
                    self.events.append(record)
                else:
                    self.dropped += 1
            if self._sink is not None:
                self._sink(record)
        except Exception:
            self.dropped += 1

    # The engine reports its sandbox calls here (see register / route_sandbox_call).
    def on_sandbox(self, call: SandboxCall) -> None:
        with self._lock:
            self.sandbox.append(call)

    def sandbox_mark(self) -> int:
        with self._lock:
            return len(self.sandbox)

    def sandbox_since(self, mark: int) -> list[SandboxCall]:
        with self._lock:
            return list(self.sandbox[mark:])


# One engine serves many requests; its observer routes each sandbox call to the tracer whose trace id
# was bound when the call was made.
_ACTIVE: dict[str, Tracer] = {}
_ACTIVE_LOCK = threading.Lock()


def route_sandbox_call(call: SandboxCall) -> None:
    if call.trace_id is None:
        return
    with _ACTIVE_LOCK:
        tracer = _ACTIVE.get(call.trace_id)
    if tracer is not None:
        tracer.on_sandbox(call)


def activate(tracer: Tracer, engine: Engine) -> None:
    """Start routing the engine's sandbox calls for this trace to its tracer."""
    with _ACTIVE_LOCK:
        _ACTIVE[tracer.trace_id] = tracer
    if engine.observer is not route_sandbox_call:
        engine.observer = route_sandbox_call


def deactivate(tracer: Tracer) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.pop(tracer.trace_id, None)
