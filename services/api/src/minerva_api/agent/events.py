"""Events streamed to the browser (server-sent events) and stored with each assistant message."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """A piece of model prose: plain text, or a formula with its verification mark."""

    type: Literal["text", "math", "display_math"]
    value: str
    # Formulas only. "from_tool": appears in a tool result or handbook entry of this conversation.
    # "checked": an equation the model wrote that Minerva tested numerically and found true.
    # "conditional": false in general but true under extra assumptions (the condition is in `note`).
    # "contradicted": an equation the model wrote that Minerva tested, as an identity or against
    #   definitions stated earlier in the conversation (named in `note`), and found not to hold.
    # "unchecked": written by the model, and Minerva could not check it. A definition, coordinate
    #   change or equation to solve is not an identity: it stays unchecked, with the reason in `note`.
    # "quoted": appears in a tool result without being a verified result (the user's own input as
    #   read, or a handbook statement that is only stated): shown, but not marked as verified.
    status: (
        Literal["from_tool", "quoted", "checked", "conditional", "contradicted", "unchecked"] | None
    ) = None
    note: str | None = None  # e.g. "false in general, but holds if x is positive"


class Card(BaseModel):
    """A structured result rendered by Minerva (never by the model) from a tool's own output."""

    id: str
    tool: str
    type: Literal[
        "simplify",
        "differentiate",
        "series",
        "integrate",
        "matrix",
        "geometry",
        "distribution",
        "identities",
        "entries",
        "entry",
        "tensor",
        "solve",
        "lagrangian",
        "expansion",
        "perturbation",
        "invariants",
        "motion",
        "dimensional",
        "units",
        "error",
    ]
    title: str
    data: dict[str, Any] = Field(default_factory=dict)


CheckKind = Literal[
    "symbolic_zero",  # a symbolic zero test passed (the only thing called "proved")
    "numeric",  # independent numeric agreement at sample points
    "quadrature",  # independent adaptive quadrature
    "series_checks",  # coefficients and remainder checked without sympy.series
    "structural",  # canonical form of a tensor expression
    "dimensional",  # dimensions only
    "lookup",  # a handbook record: retrieved, not computed
    "none",  # nothing was checked
]
EvidenceReason = Literal[
    "verified",
    "lookup",
    "unchanged",
    "not_verified",
    "conditions_not_satisfied",
    "unsupported",
    "parse_error",
    "resource_limit",
    "internal_error",
    "no_results",
    "error",
]


class Evidence(BaseModel):
    """What the engine established for one tool result, and why it was accepted or not.

    Numbers and flags only, read from the engine's own result: no expression, so it is safe in events
    and logs. `accepted` means the result may be shown as an engine result; `unchanged`, `not_verified`,
    `unsupported` and the like are not accepted, whatever the model says about them.
    """

    check: CheckKind
    accepted: bool
    reason: EvidenceReason
    points_checked: int | None = None
    points_agreed: int | None = None
    symbolic_zero: bool | None = None
    restrictions: int = 0  # how many validity restrictions the result carries
    restrictions_complete: bool | None = None


class TraceStep(BaseModel):
    """Tool call and the compact text the model saw, kept so later turns have the same context."""

    id: str
    name: str
    arguments: dict[str, Any]
    for_model: str
    # Debugging aids, absent on messages stored before they existed.
    duration_ms: int | None = None
    call_id: str | None = None  # the provider's tool-call id, replayed when history is rebuilt
    status: str | None = None  # the tool's own status (`ok`, `parse_error`, ...) or `error`
    evidence: Evidence | None = None
    arguments_valid: bool | None = None
    error_category: str | None = None  # see tracing.ErrorCategory
    engine_ms: int | None = None  # time inside the sandbox (waiting for a slot included)
    repeated: bool = False  # the same call had already run in this turn
    result_id: str | None = None  # "r1", "r2", ...: how a later call may refer to this result
    # What the model wrote before `@rN` references were replaced; present only when it differs.
    model_arguments: dict[str, Any] | None = None
    # Symbols declared earlier in the conversation that this call uses without declaring them again.
    assumptions_dropped: list[str] = Field(default_factory=list)
    # Formulas this result may be quoted for (verified) and formulas it only quoted, with the reason:
    # kept so later turns mark formulas the same way, instead of guessing from the text.
    verified_latex: list[str] = Field(default_factory=list)
    quoted: dict[str, str] = Field(default_factory=dict)


class Block(BaseModel):
    """One item of an assistant turn, in the order it happened: prose, or a tool result card."""

    kind: Literal["text", "card"]
    segments: list[Segment] = Field(default_factory=list)  # kind == "text"
    card: Card | None = None  # kind == "card"


class LLMCallDebug(BaseModel):
    """One request to the model within a turn. Sizes and timings only, never content."""

    ms: int
    text_chars: int
    tool_calls: int
    first_token_ms: int | None = None
    finish_reason: str | None = None  # stop | length | tool_calls | other | error | timeout
    prompt_tokens: int | None = None  # as reported by the provider, when it reports them
    completion_tokens: int | None = None
    cost_usd: float | None = None


class ClaimDebug(BaseModel):
    """How an equation the model wrote itself was tested, and why it ended up with its mark."""

    value: str
    status: str  # holds | conditional | fails | unverified
    detail: str = ""
    kind: str | None = None  # identity | relation (a symbol on one side only)
    given: list[str] = Field(default_factory=list)  # earlier definitions the result used


class TurnDebug(BaseModel):
    """Everything needed to reproduce a turn, shown only when the user switches debug mode on."""

    request_id: str = ""
    api_version: str = ""
    mode: str = ""
    provider: str = ""  # ollama | openrouter | scripted (never a key or host)
    model: str = ""
    error_kind: str | None = None
    prompt_sha: str = ""  # system prompt + tool schemas; changes when either is edited
    max_steps: int = 0
    max_seconds: float = 0
    duration_ms: int = 0
    llm_calls: list[LLMCallDebug] = Field(default_factory=list)
    claims: list[ClaimDebug] = Field(default_factory=list)
    error: str | None = None
    # Added with tracing; absent on messages stored before it existed.
    trace_id: str = (
        ""  # one per agent request; joins the stored trace, the events and the log lines
    )
    stop_reason: str | None = None
    max_tool_calls: int = 0
    events: list[dict[str, Any]] = Field(default_factory=list)  # tracing.py event records, in order
    events_dropped: int = 0
    claim_check: str | None = (
        None  # ok | none | timeout | error: why claims were or were not checked
    )
    formulas: int = 0  # formulas the model wrote in this turn
    formulas_with_evidence: int = 0  # ... that are backed by engine evidence
    evidence_pct: float | None = None


class AssistantMessage(BaseModel):
    text: str  # raw model text, used as conversation history
    blocks: list[Block]
    trace: list[TraceStep]
    truncated: bool = False  # a limit or a failure ended the turn early (see stop_reason)
    debug: TurnDebug | None = None
    stop_reason: str | None = (
        None  # tracing.StopReason; absent on messages stored before it existed
    )
