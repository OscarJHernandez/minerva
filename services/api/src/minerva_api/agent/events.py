"""Events streamed to the browser (server-sent events) and stored with each assistant message."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """A piece of model prose: plain text, or a formula with its verification mark."""

    type: Literal["text", "math", "display_math"]
    value: str
    # For formulas only: True when the formula appears in a tool result or handbook entry of this
    # conversation, i.e. Minerva itself produced or contains it. False means "model-written, unchecked".
    verified: bool | None = None


class Card(BaseModel):
    """A structured result rendered by Minerva (never by the model) from a tool's own output."""

    id: str
    tool: str
    type: Literal["simplify", "identities", "entries", "entry", "tensor", "error"]
    title: str
    data: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    """Tool call and the compact text the model saw, kept so later turns have the same context."""

    id: str
    name: str
    arguments: dict[str, Any]
    for_model: str


class AssistantMessage(BaseModel):
    text: str  # raw model text, used as conversation history
    segments: list[Segment]
    cards: list[Card]
    trace: list[TraceStep]
    truncated: bool = False  # the step or time limit ended the turn early
