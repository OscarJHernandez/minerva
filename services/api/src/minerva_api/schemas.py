"""Request/response models specific to the HTTP API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

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
    save: bool = True


class IdentitiesRequest(ExpressionInput):
    query: str = Field(default="", max_length=200)
    save: bool = True


class SimplifyRequest(ExpressionInput):
    save: bool = True


class DifferentiateRequest(ExpressionInput):
    variables: list[DiffVariable] = Field(min_length=1, max_length=4)
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
