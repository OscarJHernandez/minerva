"""Input and result types of creative telescoping.

Every refusal carries one of the enumerated `Reason` codes. Results hold formulas the engine
printed, counts and fixed vocabulary; nothing here is free text from a model.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import sympy as sp
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from ..distributions.models import Condition
from ..errors import MinervaMathError
from ..models import SYMBOL_DECLARATIONS, SYMBOL_NAME_MAX, Status

MAX_ORDER = 5
MAX_DEGREE = 12
MAX_UNKNOWNS = 48

Reason = Literal[
    "not_hyperexponential",  # the integrand is outside the supported class
    "variable_missing",
    "parameter_missing",
    "parameter_domain_unknown",  # the parameter must be declared real (and ideally signed)
    "bounds_required",
    "bound_invalid",
    "no_telescoper_found",  # nothing within the searched families; proves nothing outside them
    "certificate_rejected",  # a candidate failed verification and was dropped
    "singularity_not_excluded",  # the integrand may have a pole inside the interval
    "branch_not_established",  # a non-integer power of something not provably positive
    "convergence_not_established",
    "divergent",
    "differentiation_not_justified",
    "boundary_not_established",
    "singular_point_in_domain",  # the leading coefficient may vanish inside the parameter domain
    "ode_not_solved",
    "initial_value_unavailable",
    "value_not_verified",
    "unsupported_input",
    "resource_limit",
]

Outcome = Literal["evaluated", "equation", "certificate", "none"]
Verdict = Literal["proved", "numerically_checked", "not_run", "failed"]


class Refusal(MinervaMathError):
    """The problem cannot be treated; `reason` is one of the enumerated codes."""

    def __init__(self, reason: Reason, message: str) -> None:
        super().__init__(message)
        self.reason: Reason = reason


class TelescopingInput(BaseModel):
    """A parameterised integral `I(a) = int_lower^upper f(x, a) dx`."""

    input: str = Field(min_length=1, max_length=2000, description="The integrand f(x, a)")
    format: Literal["latex", "text"] = "latex"
    assumptions: dict[str, list[str]] = Field(default_factory=dict)
    variable: str = Field(min_length=1, max_length=SYMBOL_NAME_MAX)
    parameter: str = Field(min_length=1, max_length=SYMBOL_NAME_MAX)
    lower: str | None = Field(default=None, max_length=64)
    upper: str | None = Field(default=None, max_length=64)
    max_order: int = Field(default=3, ge=1, le=MAX_ORDER)
    max_degree: int = Field(default=6, ge=0, le=MAX_DEGREE)

    @field_validator("assumptions")
    @classmethod
    def _check_assumptions(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        if len(value) > 32:
            raise ValueError("too many symbols with assumptions")
        for name, items in value.items():
            if not name or len(name) > SYMBOL_NAME_MAX:
                raise ValueError("invalid symbol name in assumptions")
            unknown = set(items) - SYMBOL_DECLARATIONS
            if unknown:
                raise ValueError(f"unknown assumption(s): {sorted(unknown)}")
            if "noncommutative" in items:
                raise ValueError("noncommutative symbols are not supported here")
        return value


class Family(BaseModel):
    """One family of the search: operator order and degree of the certificate's numerator."""

    order: int
    degree: int
    unknowns: int
    outcome: Literal["found", "none", "rejected"]


class OperatorOut(BaseModel):
    order: int
    coefficients_latex: list[str]  # p_0 ... p_N, polynomials in the parameter
    coefficients_text: list[str] = Field(default_factory=list)  # the same, in text syntax
    latex: str  # L = sum p_k d^k/da^k


class CertificateOut(BaseModel):
    latex: str  # G(x, a) with L f = dG/dx
    latex_ratio: str  # G / f, a rational function of x: what the search solved for


class CertificateCheck(BaseModel):
    """Symbolic proof of `L f = dG/dx` and two independent numerical legs, reported apart."""

    symbolic_zero: bool = False
    numeric_points_checked: int = 0
    numeric_points_agreed: int = 0
    derivative_points_checked: int = 0
    derivative_points_agreed: int = 0
    passed: bool = False
    note: str = ""


class BoundaryOut(BaseModel):
    lower_latex: str  # G at the lower end (a limit when the end is infinite)
    upper_latex: str
    contribution_latex: str  # G(upper) - G(lower)
    vanishes: bool


class EquationOut(BaseModel):
    latex: str  # sum p_k I^(k)(a) = boundary
    homogeneous: bool
    rhs_latex: str
    rhs_text: str = ""  # the right-hand side in text syntax (the benchmark re-reads it)
    leading_nonzero: bool | None  # is p_N provably nonzero on the whole parameter domain?


class SolutionOut(BaseModel):
    general_latex: str
    initial_point_latex: str
    initial_values_latex: list[str]
    constants_latex: list[str]


class NumericPoint(BaseModel):
    """The accepted value at one parameter value, as decimal text (for readers that cannot parse
    special functions: the benchmark compares these with its own quadrature)."""

    parameter: str
    value: str


class ValueVerification(BaseModel):
    ode_symbolic_zero: bool = False  # the solution satisfies the ODE, exactly
    initial_values_hold: bool = False
    quadrature_points_checked: int = 0
    quadrature_points_agreed: int = 0
    passed: bool = False
    note: str = ""
    samples: list[NumericPoint] = Field(default_factory=list)


StageName = Literal["analyse", "search", "verify", "boundary", "equation", "solve", "value"]


class Stage(BaseModel):
    name: StageName
    status: Literal["ok", "failed", "skipped"]
    duration_ms: int = 0
    reason: Reason | None = None


class Diagnostics(BaseModel):
    """Counters and enumerated labels only: never an expression, a bound or a symbol name."""

    stages: list[Stage] = Field(default_factory=list)
    order: int | None = None
    degree: int | None = None
    families_tried: int = 0
    candidates_rejected: int = 0
    reason: Reason | None = None

    def log_fields(self) -> dict[str, object]:
        return {
            "stages": [f"{s.name}:{s.status}" for s in self.stages],
            "highest_order": self.order,
            "points_checked": self.families_tried,
            "reason": self.reason,
        }


class TelescopingResult(BaseModel):
    status: Status
    outcome: Outcome = "none"
    reason: Reason | None = None
    message: str = ""
    input_latex: str | None = None
    variable: str = ""
    parameter: str = ""
    lower_latex: str | None = None
    upper_latex: str | None = None
    parameter_domain_latex: str | None = None
    operator: OperatorOut | None = None
    certificate: CertificateOut | None = None
    certificate_check: CertificateCheck | None = None
    families: list[Family] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    boundary: BoundaryOut | None = None
    equation: EquationOut | None = None
    solution: SolutionOut | None = None
    result_latex: str | None = None
    result_text: str | None = None
    value_verification: ValueVerification | None = None
    unresolved: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)
    duration_ms: int = 0
    # The accepted value as an expression, for in-process callers (the integration strategies). Not
    # part of the model's data: it is dropped by `model_dump`, so it never crosses the sandbox.
    _value: sp.Expr | None = PrivateAttr(default=None)

    @property
    def value_expr(self) -> sp.Expr | None:
        return self._value


@contextmanager
def timed(stages: list[Stage], name: StageName) -> Iterator[Stage]:
    """Record a stage with its duration; the caller sets `status` / `reason` before leaving."""
    stage = Stage(name=name, status="ok")
    started = time.monotonic()
    try:
        yield stage
    finally:
        stage.duration_ms = int((time.monotonic() - started) * 1000)
        stages.append(stage)
