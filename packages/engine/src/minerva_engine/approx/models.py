"""Typed request and result for the approximation operation.

One input model with a `method` and one report. Every number that sets the regime of an
approximation (the interval, the period, the tolerance, the degree caps) is a field the USER fills:
there are no defaults, and a missing one gives `domain_unspecified` with the question to ask.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..models import ExpressionInput, Status
from ..series import MeasuredError

Method = Literal[
    "auto",
    "taylor",
    "perturbation",
    "laurent",
    "asymptotic",
    "multivariable",
    "leading",
    "limit",
    "pade",
    "chebyshev",
    "legendre",
    "fourier",
    "piecewise",
    "continued_fraction",
    "euler_maclaurin",
    "integral_expansion",
    "root_expansion",
    "reversion",
    "laplace",
    "accelerate",
    "compare",
]

# Fixed vocabulary: the reason a request was refused, or what a warning is about. Logs and traces
# carry these words and counts, never an expression.
Reason = Literal[
    "domain_unspecified",
    "variable_missing",
    "parameter_missing",
    "point_invalid",
    "order_out_of_range",
    "not_analytic",
    "not_laurent",
    "essential_singularity",
    "small_parameter_not_isolable",
    "direction_required",
    "limit_undecided",
    "verification_failed",
    "not_implemented",
    "unsupported_input",
    "resource_limit",
    "tolerance_not_met",
    "degenerate_system",
    "domain_invalid",
]

Warning = Literal[
    "removable_singularity",
    "pole_in_interval",
    "spurious_pole",
    "jump_discontinuity",
    "gibbs",
    "slow_coefficient_decay",
    "asymptotic_not_convergent",
    "no_extrapolation",
    "local_only",
    "error_estimate_unstable",
    "one_sided_limits_differ",
    "expansion_not_uniform",
]

# What the checks established. `series_checked` and `numeric_estimate` are measurements and never a
# proof; `bound_established` is used only where an actual inequality was verified.
Evidence = Literal["series_checked", "numeric_estimate", "bound_established", "none"]


class ApproximationInput(ExpressionInput):
    """`input` is the expression (or the summand, for a sum). Fields not used by `method` are ignored."""

    method: Method = "taylor"
    variable: str = Field(default="x", min_length=1, max_length=32)
    # Local expansions
    point: str | None = Field(default=None, max_length=64)  # a finite real number, or oo / -oo
    direction: Literal["+", "-"] | None = None
    order: int | None = Field(default=None, ge=0, le=64)
    small: str | None = Field(default=None, max_length=64)  # e.g. epsilon, or v/c
    variables: list[str] = Field(default_factory=list, max_length=4)  # multivariable
    points: list[str] = Field(default_factory=list, max_length=4)
    # Interval and degree based
    lower: str | None = Field(default=None, max_length=64)
    upper: str | None = Field(default=None, max_length=64)
    degree: int | None = Field(default=None, ge=0, le=64)
    denominator_degree: int | None = Field(default=None, ge=0, le=32)
    period: str | None = Field(default=None, max_length=64)
    modes: int | None = Field(default=None, ge=0, le=128)
    segments: int | None = Field(default=None, ge=1, le=512)
    kind: str | None = Field(default=None, max_length=24)
    # Accuracy targets and compare
    tolerance: str | None = Field(default=None, max_length=32)
    tolerance_kind: Literal["absolute", "relative"] | None = None
    max_terms: int | None = Field(default=None, ge=1, le=128)
    methods: list[str] = Field(default_factory=list, max_length=8)


class OrderTerm(BaseModel):
    """One order of an expansion: the coefficient and the term it contributes, in the user's symbols."""

    order: int
    coefficient_latex: str
    term_latex: str


class Check(BaseModel):
    name: str
    passed: bool | None  # None: could not be decided (never a refutation)
    points: int = 0
    detail: str = ""


class Labeled(BaseModel):
    label: str
    latex: str


class ErrorSummary(BaseModel):
    """Errors measured on a finite sample. A measurement, never a bound."""

    samples: int = 0
    max_abs: str | None = None
    rms: str | None = None
    max_rel: str | None = None
    relative_note: str = ""
    at: str | None = None  # where the largest sampled error occurred


class Complexity(BaseModel):
    degree: int | None = None
    numerator_degree: int | None = None
    denominator_degree: int | None = None
    modes: int | None = None
    segments: int | None = None
    coefficients: int | None = None


class PlotPoint(BaseModel):
    x: float
    error: float


class Candidate(BaseModel):
    """One approximation in a comparison or a tolerance search."""

    method: str
    label: str
    status: Status
    reason: Reason | None = None
    result_latex: str | None = None
    complexity: Complexity = Field(default_factory=Complexity)
    error: ErrorSummary | None = None
    tolerance_met: bool | None = None
    warnings: list[Warning] = Field(default_factory=list)
    error_curve: list[PlotPoint] = Field(default_factory=list)
    chosen: bool = False


class Suggestion(BaseModel):
    method: str
    applies: bool
    why: str
    tried: bool = False


class Diagnostics(BaseModel):
    """Counts and fixed words only: never a name, an expression or a bound."""

    method: str = ""
    stages: int = 0
    checks: int = 0
    checks_passed: int = 0
    warnings: int = 0
    candidates: int = 0
    duration_ms: float = 0.0

    def log_fields(self) -> dict[str, object]:
        return self.model_dump()


class ApproximationReport(BaseModel):
    status: Status
    reason: Reason | None = None
    method: str = ""
    message: str = ""
    input_latex: str | None = None
    variable: str | None = None
    domain_latex: str | None = None  # the point or interval the approximation is for
    result_latex: str | None = None
    result_text: str | None = None
    remainder_latex: str | None = None
    statement_latex: str | None = None  # for limits and leading behaviour: the whole claim
    orders: list[OrderTerm] = Field(default_factory=list)
    extras: list[Labeled] = Field(default_factory=list)
    evidence: Evidence = "none"
    checks: list[Check] = Field(default_factory=list)
    complexity: Complexity = Field(default_factory=Complexity)
    error: ErrorSummary | None = None
    measured_errors: list[MeasuredError] = Field(default_factory=list)
    error_curve: list[PlotPoint] = Field(default_factory=list)
    tolerance_met: bool | None = None
    warnings: list[Warning] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    assumptions_stated: list[str] = Field(default_factory=list)
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)
    duration_ms: float = 0.0

    @property
    def shown(self) -> bool:
        return self.status in (Status.OK, Status.UNCHANGED)
