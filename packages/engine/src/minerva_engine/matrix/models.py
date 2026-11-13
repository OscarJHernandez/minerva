"""Typed input and output of the matrix calculus operation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..models import SYMBOL_NAME_MAX, ExpressionInput, Status

MAX_MATRICES = 6
MAX_ABSTRACT_SIZE = 8
MAX_EXPLICIT_SIZE = 4

Action = Literal["evaluate", "simplify", "differentiate", "integrate"]
ResultKind = Literal["explicit", "expression", "equation", "representation"]


class MatrixDecl(BaseModel):
    """A declared matrix: abstract (a size only) or explicit (entries, possibly symbolic)."""

    name: str = Field(min_length=1, max_length=SYMBOL_NAME_MAX, description="Matrix name, e.g. A")
    rows: int | None = Field(
        default=None,
        ge=1,
        le=MAX_ABSTRACT_SIZE,
        description="Number of rows of an abstract matrix (use entries for an explicit one)",
    )
    cols: int | None = Field(
        default=None,
        ge=1,
        le=MAX_ABSTRACT_SIZE,
        description="Number of columns of an abstract matrix; defaults to rows (square)",
    )
    entries: list[list[str]] | None = Field(
        default=None,
        description="Entries of an explicit matrix as LaTeX expressions, row by row, up to 4x4",
    )

    @field_validator("entries")
    @classmethod
    def _entries(cls, value: list[list[str]] | None) -> list[list[str]] | None:
        if value is None:
            return None
        if not value or len(value) > MAX_EXPLICIT_SIZE or len(value[0]) > MAX_EXPLICIT_SIZE:
            raise ValueError(
                f"explicit matrices have between 1 and {MAX_EXPLICIT_SIZE} rows/columns"
            )
        if any(len(row) != len(value[0]) or not row for row in value):
            raise ValueError("the rows of an explicit matrix must have the same length")
        if any(not e.strip() or len(e) > 100 for row in value for e in row):
            raise ValueError("each entry must be a non-empty expression of at most 100 characters")
        return value

    @model_validator(mode="after")
    def _one_form(self) -> MatrixDecl:
        if self.entries is not None:
            if self.rows is not None or self.cols is not None:
                raise ValueError("give either entries or a size, not both")
        elif self.rows is None:
            raise ValueError("give the entries, or at least the number of rows")
        return self

    @property
    def shape(self) -> tuple[int, int]:
        if self.entries is not None:
            return len(self.entries), len(self.entries[0])
        rows = self.rows or 1  # the validator guarantees rows is set for an abstract matrix
        return rows, self.cols if self.cols is not None else rows


class MatrixCalculusInput(BaseModel):
    """An expression in declared matrices, and what to do with it."""

    expression: ExpressionInput
    matrices: list[MatrixDecl] = Field(min_length=1, max_length=MAX_MATRICES)
    commuting: list[tuple[str, str]] = Field(default_factory=list, max_length=16)
    invertible: list[str] = Field(default_factory=list, max_length=MAX_MATRICES)
    action: Action = "simplify"
    variable: str | None = Field(default=None, min_length=1, max_length=SYMBOL_NAME_MAX)
    lower: str | None = Field(default=None, min_length=1, max_length=200)
    upper: str | None = Field(default=None, min_length=1, max_length=200)


class MatrixStep(BaseModel):
    """One transformation: which method, what it read, what it produced."""

    method: str
    title: str
    before_latex: str | None = None
    after_latex: str | None = None
    note: str = ""


class MatrixBlocked(BaseModel):
    """A transformation that was not applied because its conditions are not established."""

    method: str
    title: str
    needs: str
    latex: str | None = None


class MatrixVerification(BaseModel):
    """Two separate legs. `passed` is decided by the numeric leg; the symbolic leg is extra evidence."""

    method: str = ""
    points_checked: int = 0
    points_agreed: int = 0
    passed: bool = False
    symbolic_zero: bool | None = None  # None: no exact test was run (see symbolic_method)
    symbolic_method: str = ""
    note: str = ""


class MatrixCalculusResult(BaseModel):
    status: Status
    action: Action | None = None
    input_latex: str | None = None
    structure: str = ""  # the recognised structure, in words
    features: list[str] = Field(default_factory=list)  # retrieval keys for handbook methods
    method: str | None = None  # id of the method that produced the result
    method_title: str | None = None
    result_kind: ResultKind | None = None
    result_latex: str | None = None
    result_entries: list[list[str]] | None = None  # explicit results: one text per entry
    equation_latex: str | None = None  # the verified equation, for reductions
    evaluated: bool = False  # False for an equation or a representation: nothing was evaluated
    assumptions_used: list[str] = Field(default_factory=list)
    steps: list[MatrixStep] = Field(default_factory=list)
    blocked: list[MatrixBlocked] = Field(default_factory=list)
    verification: MatrixVerification | None = None
    notes: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0
