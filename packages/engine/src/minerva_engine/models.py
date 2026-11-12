"""Typed request/response models shared by the engine, knowledge base and API."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ALLOWED_ASSUMPTIONS: frozenset[str] = frozenset(
    {
        "real",
        "positive",
        "negative",
        "nonnegative",
        "nonpositive",
        "nonzero",
        "integer",
        "rational",
    }
)

SYMBOL_NAME_MAX = 32


class Status(StrEnum):
    """Outcome of a computation. Anything other than OK/UNCHANGED is never a result."""

    OK = "ok"  # a different expression was found and verified
    UNCHANGED = "unchanged"  # no simpler form was found; the input is returned as is
    NOT_VERIFIED = "not_verified"  # a candidate was produced but could not be verified
    CONDITIONS_NOT_SATISFIED = "conditions_not_satisfied"  # needs assumptions not declared
    UNSUPPORTED = "unsupported"
    PARSE_ERROR = "parse_error"
    RESOURCE_LIMIT = "resource_limit"
    INTERNAL_ERROR = "internal_error"


class ExpressionInput(BaseModel):
    """A user expression plus declared assumptions about its symbols."""

    input: str = Field(min_length=1, max_length=20_000)
    format: Literal["latex", "text"] = "latex"
    assumptions: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("assumptions")
    @classmethod
    def _check_assumptions(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        if len(value) > 32:
            raise ValueError("too many symbols with assumptions")
        for name, items in value.items():
            if not name or len(name) > SYMBOL_NAME_MAX:
                raise ValueError("invalid symbol name in assumptions")
            unknown = set(items) - ALLOWED_ASSUMPTIONS
            if unknown:
                raise ValueError(f"unknown assumption(s): {sorted(unknown)}")
        return value


class Verification(BaseModel):
    """How a transformation was checked. `passed` is only true if a check actually ran."""

    symbolic_zero: bool = False  # simplify(original - result) == 0
    numeric_points_checked: int = 0
    numeric_points_agreed: int = 0
    passed: bool = False
    note: str = ""


class SimplifyResult(BaseModel):
    status: Status
    input_latex: str | None = None
    result_latex: str | None = None
    result_text: str | None = None
    verification: Verification | None = None
    message: str = ""
    duration_ms: int = 0


class RuleCondition(BaseModel):
    """Requirement on what a pattern variable is bound to, e.g. x positive."""

    variable: str
    assumption: str

    @field_validator("assumption")
    @classmethod
    def _known(cls, value: str) -> str:
        if value not in ALLOWED_ASSUMPTIONS:
            raise ValueError(f"unknown assumption {value!r}")
        return value


class Rule(BaseModel):
    """A rewrite rule `lhs -> rhs` (text syntax) over declared pattern variables."""

    id: str
    lhs: str
    rhs: str
    variables: list[str]
    conditions: list[RuleCondition] = Field(default_factory=list)

    @model_validator(mode="after")
    def _conditions_use_declared_variables(self) -> Rule:
        unknown = {c.variable for c in self.conditions} - set(self.variables)
        if unknown:
            raise ValueError(f"conditions refer to undeclared variable(s) {sorted(unknown)}")
        return self


class Applicability(StrEnum):
    APPLICABLE = "applicable"  # matched, conditions proven, rewrite verified
    CONDITIONS_NOT_SATISFIED = "conditions_not_satisfied"  # matched, but not provably valid
    NOT_VERIFIED = "not_verified"  # matched and conditions hold, rewrite failed verification
    NO_MATCH = "no_match"


class RuleApplication(BaseModel):
    rule_id: str
    applicability: Applicability
    matched_subexpression_latex: str | None = None
    rewritten_latex: str | None = None
    unmet_conditions: list[RuleCondition] = Field(default_factory=list)
    verification: Verification | None = None
    message: str = ""


class IdentitiesResult(BaseModel):
    status: Status
    input_latex: str | None = None
    applications: list[RuleApplication] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0


class FormulaReading(BaseModel):
    """How Minerva's strict parser reads one transcribed formula. Nothing is simplified or evaluated."""

    status: Status
    message: str = ""
    read_as: str | None = None  # the parsed sides re-rendered as LaTeX, joined by " = "


class Limits(BaseModel):
    """Static (pre-parse) and dynamic (sandbox) resource limits."""

    max_input_chars: int = 2_000
    max_tree_nodes: int = 2_000
    max_tree_depth: int = 60
    max_int_digits: int = 40
    max_pow_exponent: int = 10_000
    timeout_seconds: float = 10.0
    memory_mb: int = 1_024
    max_concurrency: int = 4


class TensorSpec(BaseModel):
    """Declared index structure and symmetry of a tensor head."""

    rank: int = Field(ge=1, le=6)
    symmetry: Literal["none", "symmetric", "antisymmetric", "riemann"] = "none"


class TensorInput(BaseModel):
    """A tensor polynomial such as `S(a,b)*A(-a,-b)`; a leading `-` marks a lowered index."""

    input: str = Field(min_length=1, max_length=2_000)
    tensors: dict[str, TensorSpec]
    dimension: int = Field(default=4, ge=1, le=64)

    @field_validator("tensors")
    @classmethod
    def _names(cls, value: dict[str, TensorSpec]) -> dict[str, TensorSpec]:
        if not value or len(value) > 16:
            raise ValueError("declare between 1 and 16 tensors")
        for name, spec in value.items():
            if not name.isidentifier() or len(name) > SYMBOL_NAME_MAX:
                raise ValueError("invalid tensor name")
            if spec.symmetry == "riemann" and spec.rank != 4:
                raise ValueError("riemann symmetry requires rank 4")
        return value


class TensorResult(BaseModel):
    status: Status
    input_text: str | None = None
    result_text: str | None = None
    result_latex: str | None = None
    independent_verification: bool = False
    message: str = ""
    duration_ms: int = 0
