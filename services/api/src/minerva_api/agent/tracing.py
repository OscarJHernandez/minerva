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

import re
import threading
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Literal, get_args

from minerva_engine import Engine, SandboxCall
from minerva_engine.notation.models import InputCode
from pydantic import BaseModel, field_validator
from pydantic_core import core_schema

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
    NEEDS_USER_CHOICE = "needs_user_choice"  # several methods apply and the user named none


class StopReason(StrEnum):
    """Why a turn ended."""

    COMPLETED = "completed"  # the model answered without asking for more tools
    STEP_LIMIT = "step_limit"
    TOOL_CALL_LIMIT = "tool_call_limit"
    TIME_LIMIT = "time_limit"
    REPEATED_CALLS = "repeated_calls"  # the model kept making calls that could not help
    OUTPUT_LIMIT = "output_limit"  # the last model call hit its length limit with no tool call
    MODEL_ERROR = "model_error"  # the model host failed
    MALFORMED_OUTPUT = "malformed_output"  # the model kept producing unusable tool calls
    INTERNAL_ERROR = "internal_error"


class RouteReason(StrEnum):
    """Why a rule chose the first call (a fixed vocabulary), or why its result was not used."""

    LAGRANGIAN_EQUATIONS = "lagrangian_equations"
    FOLLOWUP_REUSE = "followup_reuse"
    LAGRANGIAN_FOLLOWUP = "lagrangian_followup"
    ODE_ANALYZE = "ode_analyze"
    MATRIX_SIMPLIFY = "matrix_simplify"
    EXPLICIT_OPERATION = "explicit_operation"
    DECLINED = "not_verified_or_refused"


class RetryReason(StrEnum):
    MODEL_HOST = "model_host"
    MALFORMED_MODEL_OUTPUT = "malformed_model_output"
    EMPTY_REPLY = "empty_reply"


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


OutcomeKind = Literal["failure", "answer"]

# What a non-ok tool result is. A `failure` is something to fix in Minerva or in how the model called
# it; an `answer` is what the engine or the handbook established (no entry, nothing changed, not
# supported, a condition not provable, the user must choose) or a call that was not run because of a
# limit. Total over `ErrorCategory`: a test fails when a category is added without a decision here.
OUTCOME_OF_CATEGORY: dict[ErrorCategory, OutcomeKind] = {
    ErrorCategory.INVALID_ARGUMENTS: "failure",
    ErrorCategory.UNKNOWN_TOOL: "failure",
    ErrorCategory.PARSE_ERROR: "failure",
    ErrorCategory.NOT_VERIFIED: "failure",
    ErrorCategory.TIMEOUT: "failure",
    ErrorCategory.RESOURCE_LIMIT: "failure",
    ErrorCategory.SANDBOX_KILLED: "failure",
    ErrorCategory.BUSY: "failure",
    ErrorCategory.TOOL_FAILURE: "failure",
    ErrorCategory.MODEL_HOST: "failure",
    ErrorCategory.MALFORMED_MODEL_OUTPUT: "failure",
    ErrorCategory.CONDITIONS_NOT_SATISFIED: "answer",
    ErrorCategory.UNSUPPORTED: "answer",
    ErrorCategory.LIMIT_REACHED: "answer",
    ErrorCategory.NEEDS_USER_CHOICE: "answer",
    ErrorCategory.REPEATED_CALL: "answer",
}
_ANSWER_STATUSES_WITHOUT_CATEGORY = frozenset({"unchanged", "no_results"})


def outcome_kind(status: str | None, category: ErrorCategory | str | None) -> OutcomeKind | None:
    """`None` for a verified result, otherwise whether the call failed or was an honest answer."""
    if status == "ok":
        return None
    if category:
        try:
            return OUTCOME_OF_CATEGORY[ErrorCategory(category)]
        except (ValueError, KeyError):
            return "failure"
    if status in _ANSWER_STATUSES_WITHOUT_CATEGORY:
        return "answer"
    return "failure"


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
    prompt_estimate_tokens: int = 0
    window: int = 0
    dropped_messages: int = 0
    groups_loaded: int = 0
    request_sha: str = ""  # what this call was actually sent: system prompt and tool schemas


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
    thinking_chars: int = 0  # a count: the reasoning itself is in the stored message only


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


# The closed vocabulary of `reason_code`: the error types of argument validation (pydantic-core's own
# list) and the engine's input codes (`minerva_engine.notation.models.InputCode`, why a mathematical input
# could not be read). Nothing else can be logged as a code.
REASON_CODES: frozenset[str] = frozenset(get_args(core_schema.ErrorType)) | frozenset(
    code.value for code in InputCode
)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}")
_DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,119}")
_PATH = re.compile(r"[A-Za-z_*][A-Za-z0-9_.*]{0,119}")


class ErrorSite(BaseModel):
    """Where an unexpected exception was raised: its class, and the module and line of our own code.
    Never its message, which can hold the user's expression. Whatever is not a name of that kind is
    dropped by the validators, whoever built the value."""

    error_type: str | None = None
    error_module: str | None = None
    error_line: int | None = None

    @field_validator("error_type")
    @classmethod
    def _class_name(cls, value: str | None) -> str | None:
        return value if value is not None and _NAME.fullmatch(value) else None

    @field_validator("error_module")
    @classmethod
    def _module_name(cls, value: str | None) -> str | None:
        ok = value is not None and value.startswith("minerva_") and _DOTTED.fullmatch(value)
        return value if ok else None


class ToolOutcome(TraceEvent, ErrorSite):
    """One line for a tool call that did not deliver a verified result: which tool, what kind of call,
    and the coded reason. `arg_fields` are the argument names the model filled that belong to the
    tool's own schema, `invalid_fields` the schema paths that failed validation (list indices and
    dictionary keys, which can be the user's symbols, replaced by `*`). No value of any argument."""

    tool: str = ""
    call_id: str = ""
    status: ToolStatus = "other"
    error_category: ErrorCategory | None = None
    outcome_kind: OutcomeKind = "failure"
    reason_code: str | None = None  # a fixed code (a validation error type), never text
    arguments_valid: bool = True
    repeated: bool = False
    references: int = 0
    duration_ms: int = 0
    engine_ms: int | None = None
    arg_fields: list[str] = []
    invalid_fields: list[str] = []

    # Second line of defence behind the code that builds these values: whatever does not look like a
    # name from our own code or schema (an e-mail address, an expression) is dropped, never logged.
    @field_validator("reason_code")
    @classmethod
    def _known_code(cls, value: str | None) -> str | None:
        return value if value in REASON_CODES else None

    @field_validator("arg_fields")
    @classmethod
    def _field_names(cls, value: list[str]) -> list[str]:
        return [v for v in value if _NAME.fullmatch(v)]

    @field_validator("invalid_fields")
    @classmethod
    def _field_paths(cls, value: list[str]) -> list[str]:
        return [v for v in value if _PATH.fullmatch(v)]


class ToolFailed(ToolOutcome):
    event: Literal["tool.failed"] = "tool.failed"


class ToolDeclined(ToolOutcome):
    event: Literal["tool.declined"] = "tool.declined"
    outcome_kind: OutcomeKind = "answer"


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


class AgentRouted(TraceEvent):
    event: Literal["agent.routed"] = "agent.routed"
    decision: Literal["routed", "declined"] = "routed"
    reason: RouteReason = RouteReason.LAGRANGIAN_EQUATIONS
    tool: str = ""


class AgentStopped(TraceEvent, ErrorSite):
    event: Literal["agent.stopped"] = "agent.stopped"
    reason: StopReason = StopReason.COMPLETED
    steps: int = 0
    tool_calls: int = 0
    error_category: ErrorCategory | None = None


class AgentCompleted(TraceEvent, ErrorSite):
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
    # Equation numbering (counts only): labels the model wrote that were removed, formulas that kept the
    # number they had, references to a number that is an alias of a lower one.
    equation_labels_removed: int = 0
    equations_reused: int = 0
    equation_aliases_resolved: int = 0
    stop_reason: StopReason = StopReason.COMPLETED
    error_category: ErrorCategory | None = None
    events_dropped: int = 0
    # Which tools the turn called, and which of them failed or were declined (registry names, in the
    # order first called; a name the model invented is `unknown`).
    tools_called: list[str] = []
    failed_tools: list[str] = []
    declined_tools: list[str] = []
    first_failure_code: ErrorCategory | None = None

    @field_validator("tools_called", "failed_tools", "declined_tools")
    @classmethod
    def _tool_names(cls, value: list[str]) -> list[str]:
        return [v for v in value if _NAME.fullmatch(v)]


EVENT_MODELS: tuple[type[TraceEvent], ...] = (
    AgentStarted,
    LLMRequest,
    LLMResponse,
    ToolStarted,
    ToolCompleted,
    ToolFailed,
    ToolDeclined,
    VerificationCompleted,
    AgentRetry,
    AgentRouted,
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
        "request_sha",
        "mode",
        "api_version",
        "tool",
        "proposed_tool",
        "call_id",
        "result_id",
        # Closed vocabularies enforced by validators on the event (`ToolOutcome`): a validation error
        # type, the class and module of our own code that raised.
        "reason_code",
        "error_type",
        "error_module",
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
