"""Request/response models specific to the HTTP API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from minerva_engine import DiffVariable, ExpressionInput
from pydantic import BaseModel, EmailStr, Field, model_validator


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)


class UserOut(BaseModel):
    id: uuid.UUID
    email: str


class Linked(ExpressionInput):
    """An expression request that may be saved as a step of a Workbench derivation."""

    save: bool = True
    derivation_id: uuid.UUID | None = None
    parent_id: uuid.UUID | None = None


class IntegrateRequest(Linked):
    variable: str = Field(min_length=1, max_length=32)
    lower: str | None = Field(default=None, min_length=1, max_length=200)
    upper: str | None = Field(default=None, min_length=1, max_length=200)


class IdentitiesRequest(Linked):
    query: str = Field(default="", max_length=200)


class SimplifyRequest(Linked):
    """`goal` ranks the results: simplify (default), expand, factor, or rewrite in terms of `target`."""

    goal: Literal["simplify", "expand", "factor", "rewrite"] = "simplify"
    target: list[Annotated[str, Field(min_length=1, max_length=32)]] = Field(
        default_factory=list, max_length=8
    )


class DifferentiateRequest(Linked):
    variables: list[DiffVariable] = Field(min_length=1, max_length=4)


class SeriesRequest(Linked):
    variable: str = Field(min_length=1, max_length=32)
    point: str = Field(default="0", min_length=1, max_length=64)
    order: int = Field(default=6, ge=1, le=12)


class ApplyIdentityRequest(Linked):
    rule_id: str = Field(min_length=1, max_length=128)


class Selection(BaseModel):
    """A part chosen from `subterms`: its path, optional subset of terms/factors, and the LaTeX it was
    chosen as (so a stale choice is refused instead of applied somewhere else)."""

    path: list[Annotated[int, Field(ge=0, le=10_000)]] = Field(max_length=100)
    args: list[Annotated[int, Field(ge=0, le=10_000)]] | None = Field(default=None, max_length=100)
    expected_latex: str = Field(max_length=2_000)


class SubtermsRequest(ExpressionInput):
    """List the parts of an expression: all of them, where `find` occurs, or where `rule_id` matches."""

    find: str | None = Field(default=None, min_length=1, max_length=400)
    rule_id: str | None = Field(default=None, min_length=1, max_length=128)
    reverse: bool = False


class RewriteRequest(Linked):
    """Apply one move to one part. The part is a `selection` from `subterms`, or typed (`part`, with
    `occurrence` if it appears more than once); neither means the whole expression."""

    action: Literal["simplify", "expand", "factor", "cancel", "collect", "rule"]
    rule_id: str | None = Field(default=None, min_length=1, max_length=128)
    reverse: bool = False
    selection: Selection | None = None
    part: str | None = Field(default=None, max_length=400)
    occurrence: int | None = Field(default=None, ge=1, le=1_000)
    by: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=4
    )

    @model_validator(mode="after")
    def _consistent(self) -> RewriteRequest:
        if (self.action == "rule") != (self.rule_id is not None):
            raise ValueError("name the identity (rule_id) exactly when the action is `rule`")
        if self.selection is not None and self.part is not None:
            raise ValueError("choose the part with `selection` or `part`, not both")
        if self.action == "collect" and not self.by:
            raise ValueError("collect needs something to collect by")
        return self


class SubstituteRequest(Linked):
    """Replace the left side of each relation by its right side. Relations are premises, written in
    the same syntax (`format`) as the input."""

    relations: list[Annotated[str, Field(min_length=1, max_length=400)]] = Field(
        min_length=1, max_length=4
    )
    order: Literal["given", "simultaneous"] | None = None


class IsolateRequest(Linked):
    """`input` is an equation `lhs = rhs`; `variable` is the unknown to solve for."""

    variable: str = Field(min_length=1, max_length=32)


class HistoryItem(BaseModel):
    id: uuid.UUID
    operation: str
    input: str
    format: str
    assumptions: dict[str, Any]
    status: str
    result: dict[str, Any]
    created_at: datetime
    derivation_id: uuid.UUID | None = None
    parent_id: uuid.UUID | None = None


class DerivationSummary(BaseModel):
    id: uuid.UUID
    updated_at: datetime
    steps: int
    first_input: str | None = None
    first_format: str | None = None


class DerivationOut(BaseModel):
    id: uuid.UUID
    updated_at: datetime
    steps: list[HistoryItem]


class EntryOut(BaseModel):
    id: str
    title: str
    kind: str
    chapter: str
    section: str | None
    statement_latex: str
    validity: str
    verification: str
    tags: list[str]
    keywords: list[str]
    reference: dict[str, Any] | None
    symbols: list[dict[str, Any]]
    constant: dict[str, Any] | None
    notes: str
    applicable_as_rule: bool
    conditions: list[dict[str, str]]
