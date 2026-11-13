"""Request/response models specific to the HTTP API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from minerva_engine import DiffVariable, ExpressionInput, MatrixDecl
from minerva_engine.matrix.models import MAX_MATRICES, Action
from pydantic import BaseModel, EmailStr, Field


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


class MatrixCalculusRequest(Linked):
    """An expression in declared matrices (`input` uses `\\exp(tA)` for exponentials)."""

    matrices: list[MatrixDecl] = Field(min_length=1, max_length=MAX_MATRICES)
    commuting: list[tuple[str, str]] = Field(default_factory=list, max_length=16)
    invertible: list[str] = Field(default_factory=list, max_length=MAX_MATRICES)
    action: Action = "simplify"
    variable: str | None = Field(default=None, min_length=1, max_length=32)
    lower: str | None = Field(default=None, min_length=1, max_length=200)
    upper: str | None = Field(default=None, min_length=1, max_length=200)


class DistributionRequest(Linked):
    """An expression with `DiracDelta` / `Heaviside` (text or `\\operatorname{...}` LaTeX)."""

    action: Literal["simplify", "integrate", "differentiate"] = "simplify"
    variable: str = Field(min_length=1, max_length=32)
    lower: str | None = Field(default=None, min_length=1, max_length=200)
    upper: str | None = Field(default=None, min_length=1, max_length=200)
    # How a delta exactly on a limit counts. Chosen by the user, never by Minerva.
    endpoint: Literal["half", "include", "exclude"] | None = None


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
