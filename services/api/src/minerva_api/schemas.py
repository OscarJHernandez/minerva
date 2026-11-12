"""Request/response models specific to the HTTP API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from minerva_engine import DiffVariable, ExpressionInput
from pydantic import BaseModel, EmailStr, Field


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)


class UserOut(BaseModel):
    id: uuid.UUID
    email: str


class IntegrateRequest(ExpressionInput):
    variable: str = Field(min_length=1, max_length=32)
    lower: str | None = Field(default=None, min_length=1, max_length=200)
    upper: str | None = Field(default=None, min_length=1, max_length=200)
    save: bool = True


class IdentitiesRequest(ExpressionInput):
    query: str = Field(default="", max_length=200)
    save: bool = True


class SimplifyRequest(ExpressionInput):
    """`goal` ranks the results: simplify (default), expand, factor, or rewrite in terms of `target`."""

    goal: Literal["simplify", "expand", "factor", "rewrite"] = "simplify"
    target: list[Annotated[str, Field(min_length=1, max_length=32)]] = Field(
        default_factory=list, max_length=8
    )
    save: bool = True


class DifferentiateRequest(ExpressionInput):
    variables: list[DiffVariable] = Field(min_length=1, max_length=4)
    save: bool = True


class SeriesRequest(ExpressionInput):
    variable: str = Field(min_length=1, max_length=32)
    point: str = Field(default="0", min_length=1, max_length=64)
    order: int = Field(default=6, ge=1, le=12)
    save: bool = True


class HistoryItem(BaseModel):
    id: uuid.UUID
    operation: str
    input: str
    format: str
    assumptions: dict[str, Any]
    status: str
    result: dict[str, Any]
    created_at: datetime


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
