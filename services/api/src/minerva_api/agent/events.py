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
        "simplify", "identities", "entries", "entry", "tensor", "solve", "lagrangian", "error"
    ]
    title: str
    data: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    """Tool call and the compact text the model saw, kept so later turns have the same context."""

    id: str
    name: str
    arguments: dict[str, Any]
    for_model: str


class Block(BaseModel):
    """One item of an assistant turn, in the order it happened: prose, or a tool result card."""

    kind: Literal["text", "card"]
    segments: list[Segment] = Field(default_factory=list)  # kind == "text"
    card: Card | None = None  # kind == "card"


class AssistantMessage(BaseModel):
    text: str  # raw model text, used as conversation history
    blocks: list[Block]
    trace: list[TraceStep]
    truncated: bool = False  # the step or time limit ended the turn early
