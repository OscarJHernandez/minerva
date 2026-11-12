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
    # "contradicted": an equation the model wrote that Minerva tested and found false.
    # "unchecked": written by the model, and Minerva could not check it.
    status: Literal["from_tool", "checked", "conditional", "contradicted", "unchecked"] | None = (
        None
    )
    note: str | None = None  # e.g. "false in general, but holds if x is positive"


class Card(BaseModel):
    """A structured result rendered by Minerva (never by the model) from a tool's own output."""

    id: str
    tool: str
    type: Literal[
        "simplify",
        "differentiate",
        "identities",
        "entries",
        "entry",
        "tensor",
        "solve",
        "lagrangian",
        "invariants",
        "motion",
        "dimensional",
        "units",
        "error",
    ]
    title: str
    data: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    """Tool call and the compact text the model saw, kept so later turns have the same context."""

    id: str
    name: str
    arguments: dict[str, Any]
    for_model: str
    # Debugging aids, absent on messages stored before they existed.
    duration_ms: int | None = None
    status: str | None = None  # the tool's own status (`ok`, `parse_error`, ...) or `error`


class Block(BaseModel):
    """One item of an assistant turn, in the order it happened: prose, or a tool result card."""

    kind: Literal["text", "card"]
    segments: list[Segment] = Field(default_factory=list)  # kind == "text"
    card: Card | None = None  # kind == "card"


class LLMCallDebug(BaseModel):
    """One request to the model within a turn. Sizes only, never content."""

    ms: int
    text_chars: int
    tool_calls: int


class ClaimDebug(BaseModel):
    """How an equation the model wrote itself was tested, and why it ended up with its mark."""

    value: str
    status: str  # holds | conditional | fails | unverified
    detail: str = ""


class TurnDebug(BaseModel):
    """Everything needed to reproduce a turn, shown only when the user switches debug mode on."""

    request_id: str = ""
    api_version: str = ""
    mode: str = ""
    model: str = ""
    prompt_sha: str = ""  # system prompt + tool schemas; changes when either is edited
    max_steps: int = 0
    max_seconds: float = 0
    duration_ms: int = 0
    llm_calls: list[LLMCallDebug] = Field(default_factory=list)
    claims: list[ClaimDebug] = Field(default_factory=list)
    error: str | None = None


class AssistantMessage(BaseModel):
    text: str  # raw model text, used as conversation history
    blocks: list[Block]
    trace: list[TraceStep]
    truncated: bool = False  # the step or time limit ended the turn early
    debug: TurnDebug | None = None
