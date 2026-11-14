"""Typed input and output of the `variation` operation.

The request always carries the whole functional (nothing is stored on the server), so every call is
replayable. Results are plain data; LaTeX strings are produced by the engine's one printer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..geometry.models import FunctionDecl
from ..models import SYMBOL_DECLARATIONS, SYMBOL_NAME_MAX, Status, Verification

Action = Literal["analyze", "first", "evaluate", "verify", "solve", "second"]
End = Literal["lower", "upper"]
Cell = Field(min_length=1, max_length=2000)

MAX_CONDITIONS = 64


class EndpointCondition(BaseModel):
    """What the user says about one derivative of one function at one end.

    `fixed` means the admissible perturbations have `η^(order) = 0` there (and `value` is what the
    path takes, needed only by numerical actions). `free` means the perturbation is not restricted.
    A (function, order, end) that is not listed is *unspecified*: nothing is assumed, its boundary
    term is kept in the first variation and a numerical solve refuses to run.
    """

    function: str = Field(min_length=1, max_length=SYMBOL_NAME_MAX)
    order: int = Field(ge=0, le=7)
    end: End
    kind: Literal["fixed", "free"]
    value: str | None = Field(default=None, max_length=2000)


class FunctionalInput(BaseModel):
    """S[x] = ∫_lower^upper L dt + G_lower + G_upper for fixed bounds."""

    integrand: str = Field(min_length=1, max_length=2000)
    format: Literal["latex", "text"] = "latex"
    variable: str = Field(default="t", min_length=1, max_length=SYMBOL_NAME_MAX)
    functions: list[str] = Field(min_length=1, max_length=4)
    given_functions: list[FunctionDecl] = Field(default_factory=list, max_length=8)
    lower: str | None = Field(default=None, max_length=200)
    upper: str | None = Field(default=None, max_length=200)
    # Terms of the action that sit at an end (functions of the jets there), e.g. `\frac{k}{2}x^2`.
    lower_term: str | None = Field(default=None, max_length=2000)
    upper_term: str | None = Field(default=None, max_length=2000)
    conditions: list[EndpointCondition] = Field(default_factory=list, max_length=MAX_CONDITIONS)
    assumptions: dict[str, list[str]] = Field(default_factory=dict)

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
                raise ValueError("noncommutative symbols are not supported by `variation`")
        return value


class SampledFunction(BaseModel):
    """Samples of one function: values at increasing times, optionally with slopes."""

    t: list[float] = Field(min_length=4, max_length=400)
    values: list[float] = Field(min_length=4, max_length=400)
    slopes: list[float] | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def _finite(self) -> SampledFunction:
        import math

        data = [*self.t, *self.values, *(self.slopes or [])]
        if not all(math.isfinite(v) for v in data):
            raise ValueError("samples must be finite numbers")
        return self


class TrialPath(BaseModel):
    """A path given either by one expression per function in the variable, or by samples."""

    label: str | None = Field(default=None, max_length=40)
    analytic: dict[str, str] = Field(default_factory=dict)
    sampled: dict[str, SampledFunction] = Field(default_factory=dict)


class Sweep(BaseModel):
    """A trial family: the symbol `name` in the path takes each of 2 to 6 values."""

    name: str = Field(min_length=1, max_length=SYMBOL_NAME_MAX)
    values: list[str] = Field(min_length=2, max_length=6)


class PerturbationSpec(BaseModel):
    label: str | None = Field(default=None, max_length=40)
    expressions: dict[str, str]


class VariationInput(FunctionalInput):
    action: Action = "analyze"
    # Numerical actions (evaluate, verify, solve, second) only; the model must ask the user for these.
    values: dict[str, str] = Field(default_factory=dict, max_length=32)
    definitions: dict[str, str] = Field(default_factory=dict, max_length=8)
    paths: list[TrialPath] = Field(default_factory=list, max_length=6)
    sweep: Sweep | None = None
    perturbations: list[PerturbationSpec] = Field(default_factory=list, max_length=4)
    automatic_perturbations: bool = True
    with_second: bool = False

    @model_validator(mode="after")
    def _functions_distinct(self) -> VariationInput:
        if len(set(self.functions)) != len(self.functions):
            raise ValueError("the dependent functions must have different names")
        return self


# -- results ------------------------------------------------------------------------------
NextStepId = Literal[
    "first_variation",
    "euler_lagrange",
    "boundary_terms",
    "evaluate_action",
    "verify_variation",
    "solve_stationary",
    "second_variation",
    "declare_conditions",
]
Need = Literal["interval", "endpoint_conditions", "parameter_values", "trial_path", "initial_guess"]
Regularity = Literal["regular", "conditional", "degenerate"]
ConditionState = Literal["fixed", "free", "unspecified"]


class NextStep(BaseModel):
    """A suggested next action, computed from the structure of the problem (never by a model)."""

    id: NextStepId
    needs: list[Need] = Field(default_factory=list)


class FunctionInfo(BaseModel):
    name: str
    name_latex: str
    order: int  # highest derivative in the integrand, 0 if it only appears undifferentiated
    appears: bool


class ConditionStatus(BaseModel):
    function: str
    order: int
    end: End
    state: ConditionState
    value_latex: str | None = None


class Analysis(BaseModel):
    functional_latex: str
    variable: str
    lower_latex: str
    upper_latex: str
    functions: list[FunctionInfo]
    parameters: list[str]
    given_functions: list[str]
    highest_order: int
    explicit_in_variable: bool
    has_endpoint_terms: bool
    regularity: Regularity
    hessian_det_latex: str
    # (end, function, order < n_i): the places a boundary term can be removed or must stay.
    conditions: list[ConditionStatus]
    unspecified: int
    # Σ n_i per end: how many conditions each end needs for a regular boundary value problem.
    conditions_needed_per_end: int
    reading: list[str]
    next_steps: list[NextStep]


BoundaryEffect = Literal[
    "removed_by_fixed", "natural_condition", "kept_unspecified", "vanishes_identically"
]


class IbpStage(BaseModel):
    """One integration by parts: a derivative moved off the perturbation, nothing dropped."""

    index: int
    function: str | None = None
    moved_order: int | None = None
    integrand_latex: str
    boundary_latex: str  # the boundary function accumulated so far
    needs_derivatives: int  # the integrand now contains derivatives up to this order
    checked: bool | None = None


class EulerLagrangeEquation(BaseModel):
    function: str
    equation_latex: str  # EL_i = 0
    expression_latex: str
    order: int  # highest derivative that appears


class BoundaryEntry(BaseModel):
    end: End
    function: str
    order: int
    perturbation_latex: str  # η^(order) at the end
    coefficient_latex: str  # what multiplies it in δS (t replaced by the end)
    state: ConditionState
    effect: BoundaryEffect
    natural_condition_latex: str | None = None  # coefficient = 0, if the perturbation is free


class ComparisonRow(BaseModel):
    step: str
    quotient: str
    abs_error: float
    rel_error: float


class Comparison(BaseModel):
    kind: Literal["bubble", "generic"]
    function: str | None = None
    analytic: str
    scale: str
    rows: list[ComparisonRow]
    observed_order: float | None = None
    richardson_rel_error: float
    verdict: Literal["agree", "disagree", "inconclusive"]


class FirstVariationChecks(BaseModel):
    deciding: Literal["passed", "failed", "inconclusive"]
    comparisons: list[Comparison]
    pointwise_passed: bool | None = None
    stages_checked: int = 0
    stages_passed: bool | None = None
    sympy_agrees: bool | None = None
    seed: int = 0
    statement: str = (
        "Numerical evidence: finite differences of the action along polynomial perturbations "
        "agree with the claimed first variation. It is not a proof."
    )


class FirstVariation(BaseModel):
    varied_latex: list[str]
    delta_integrand_latex: str
    stages: list[IbpStage]
    equations: list[EulerLagrangeEquation]
    null_lagrangian: bool
    boundary_form_latex: str
    endpoint_terms_latex: str
    boundary: list[BoundaryEntry]
    first_variation_latex: str
    stationarity_latex: list[str]
    notes: list[str] = Field(default_factory=list)
    checks: FirstVariationChecks
    verification: Verification


class PathCondition(BaseModel):
    function: str
    order: int
    end: End
    required: str
    found: str
    met: bool


class TrialResult(BaseModel):
    label: str
    parameter_value: str | None = None
    action: str  # S[path]
    integral: str
    endpoint_terms: str
    quadrature_error: float  # disagreement of independent quadrature rules: an estimate
    route: Literal["analytic", "sampled"]
    conditions: list[PathCondition] = Field(default_factory=list)
    half_data_difference: float | None = (
        None  # sampled paths: S with half the samples, a measurement
    )


class Evaluation(BaseModel):
    lower: str
    upper: str
    parameters: dict[str, str]
    trials: list[TrialResult]
    ordering: list[str] | None = None
    notes: list[str] = Field(default_factory=list)
    statement: str = (
        "These values are for the paths given only. The smallest of them is not claimed to be a "
        "minimum of the functional, and a stationary path is not implied."
    )


class PerturbationResult(BaseModel):
    label: str
    kind: Literal["bubble", "free_end", "generic", "given"]
    admissible: bool | None = None  # satisfies every declared fixed condition
    interior_part: str
    boundary_part: str
    analytic: str
    comparison: Comparison


class VerificationReport(BaseModel):
    route: Literal["integrated", "direct"]
    perturbations: list[PerturbationResult]
    verdict: Literal["passed", "failed", "inconclusive"]
    notes: list[str] = Field(default_factory=list)
    statement: str = (
        "Finite differences of the action agree with the analytic first variation to the stated "
        "error. This is numerical evidence, not a proof."
    )


Fact = Literal[
    "not_a_weak_minimum",
    "not_a_weak_maximum",
    "legendre_violated_for_minimum",
    "legendre_violated_for_maximum",
    "no_negative_direction_found",
    "no_positive_direction_found",
]


class DirectionResult(BaseModel):
    label: str
    second_variation: str
    sign: Literal["negative", "zero", "positive"]
    certified: (
        bool  # quadrature rules agree, finite differences confirm, measured change has the sign
    )


class RitzResult(BaseModel):
    """The quadratic form restricted to a space of admissible polynomial perturbations.

    The counts are the inertia of the restriction: k negative eigenvalues prove that the second
    variation is negative on a k-dimensional space of admissible perturbations. Positive
    eigenvalues prove nothing about perturbations outside the chosen space.
    """

    dimension: int
    eigenvalues: list[float]
    negative: int
    zero: int
    positive: int


class LegendreSamples(BaseModel):
    points: int
    smallest_eigenvalue: float
    largest_eigenvalue: float


class SecondNumeric(BaseModel):
    stationary: Literal["consistent", "violated", "not_tested"]
    stationary_residual: float  # worst |first variation| relative to the size of its terms
    legendre: LegendreSamples | None = None
    directions: list[DirectionResult] = Field(default_factory=list)
    ritz: RitzResult | None = None
    facts: list[Fact] = Field(default_factory=list)
    not_established: list[str] = Field(default_factory=list)
    statement: str = (
        "Only facts that follow from the computed values are listed. The necessary condition "
        "(second variation not negative, Legendre) and the sufficient conditions (strengthened "
        "Legendre and no conjugate points) are different things, and a limited set of "
        "perturbations cannot establish a minimum."
    )


class SecondVariation(BaseModel):
    convention_latex: str
    quadratic_form_latex: str
    endpoint_form_latex: str
    legendre_matrix_latex: str
    checks: FirstVariationChecks
    numeric: SecondNumeric | None = None


class PathSamples(BaseModel):
    t: list[float]
    values: dict[str, list[float]]
    slopes: dict[str, list[float]]


class MethodReport(BaseModel):
    method: Literal["hermite_action", "chebyshev_euler_lagrange"]
    ran: bool
    converged: bool  # Newton converged on every level and the refinement check passed
    levels: list[int] = Field(default_factory=list)  # elements (Hermite) or nodes (Chebyshev)
    successive_differences: list[float] = Field(default_factory=list)
    newton_iterations: list[int] = Field(default_factory=list)
    note: str = ""


class StationarityCheck(BaseModel):
    residual: float  # worst |first variation| relative to the size of its terms
    trials: int
    tolerance: float
    verdict: Literal["stationary_within_tolerance", "not_stationary"]


class StationaryPath(BaseModel):
    method: Literal["hermite_action", "chebyshev_euler_lagrange"]
    samples: PathSamples
    action: str
    action_quadrature_error: float
    stationarity: StationarityCheck


class Solution(BaseModel):
    paths: list[StationaryPath]
    methods: list[MethodReport]
    agreement: float | None = None  # largest relative difference between the two methods' paths
    confirmed_by_both: bool = False
    distinct_paths: bool = False
    second: SecondNumeric | None = None
    notes: list[str] = Field(default_factory=list)
    statement: str = (
        "A stationary path found from this starting guess. A stationary path may be a minimum, "
        "a maximum or a saddle; it is not claimed to be unique, nor to minimise the action."
    )


class Diagnostics(BaseModel):
    """Counts, durations and enum codes only; never names, expressions or values."""

    functions: int = 0
    highest_order: int = 0
    stages: int = 0
    duration_ms: float = 0.0

    def log_fields(self) -> dict[str, int | float]:
        return {
            "functions": self.functions,
            "highest_order": self.highest_order,
            "stages": self.stages,
            "duration_ms": round(self.duration_ms, 3),
        }


class VariationReport(BaseModel):
    status: Status
    message: str = ""
    action: Action = "analyze"
    analysis: Analysis | None = None
    first: FirstVariation | None = None
    evaluation: Evaluation | None = None
    verification: VerificationReport | None = None
    second: SecondVariation | None = None
    solution: Solution | None = None
    notes: list[str] = Field(default_factory=list)
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)
    duration_ms: float = 0.0
