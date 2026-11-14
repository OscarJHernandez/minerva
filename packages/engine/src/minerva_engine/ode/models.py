"""Request and result models of the ODE package.

Everything that reaches a log (`OdeDiagnostics`) is an identifier, a count, a duration or a fixed
word: never an equation, a name or a value.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..models import SYMBOL_DECLARATIONS, SYMBOL_NAME_MAX, Status

MAX_EQUATIONS = 8
MAX_ORDER = 10
MAX_CONDITIONS = 40


class OdeCondition(BaseModel):
    """`unknown^(order)(at) = value`. `at` and `value` are written in the format of the input."""

    unknown: str = Field(min_length=1, max_length=SYMBOL_NAME_MAX)
    order: int = Field(default=0, ge=0, le=MAX_ORDER)
    at: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1, max_length=200)


class OdeInput(BaseModel):
    """One equation or a system, in the notation the user writes.

    `equations`: each `left = right`. Derivatives: `y'`, `y''`, `y^{(n)}`, `\\dot y`, `\\ddot y`,
    `\\frac{d^n y}{d t^n}` (LaTeX); `y'`, `diff(y, t, n)` (text). `unknowns` and `independent` are
    optional when the notation makes them unambiguous. `conditions`: structured, or written
    `y(0) = 1`, `y'(0) = 0`.
    """

    equations: list[str] = Field(
        min_length=1,
        max_length=MAX_EQUATIONS,
        description="Each equation as the user wrote it, `left = right`; several for a system. "
        "Derivatives: y', y'', \\dot y, \\frac{d^2 y}{d t^2}, y^{(3)}. Write a product with \\cdot "
        "when a parenthesis follows an unknown: y \\cdot (1 - y), never y(1 - y).",
    )
    format: Literal["latex", "text"] = "latex"
    unknowns: list[str] = Field(
        default_factory=list,
        max_length=MAX_EQUATIONS,
        description="Names of the unknown functions, only if the notation does not make them clear.",
    )
    independent: str | None = Field(
        default=None,
        max_length=SYMBOL_NAME_MAX,
        description="The independent variable (t, x, ...). Ask the user if the equation does not say; "
        "never guess it.",
    )
    assumptions: dict[str, list[str]] = Field(
        default_factory=dict,
        description='Properties of symbols the user stated, e.g. {"k": ["positive"]}. Never add any.',
    )
    conditions: list[OdeCondition | str] = Field(
        default_factory=list,
        max_length=MAX_CONDITIONS,
        description="Initial or boundary conditions the user stated, written like `y(0) = 1`, "
        "`y'(0) = 0`. Never invent them.",
    )

    @field_validator("equations")
    @classmethod
    def _equations(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 20_000 for item in value):
            raise ValueError("an equation is empty or too long")
        return value

    @field_validator("assumptions")
    @classmethod
    def _assumptions(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        if len(value) > 32:
            raise ValueError("too many symbols with assumptions")
        for name, items in value.items():
            if not name or len(name) > SYMBOL_NAME_MAX:
                raise ValueError("invalid symbol name in assumptions")
            if set(items) - SYMBOL_DECLARATIONS:
                raise ValueError("unknown assumption")
            if "noncommutative" in items:
                raise ValueError(
                    "noncommutative symbols are not supported for differential equations"
                )
        return value


SolveMethod = Literal["closed_form", "series", "perturbation"]
MAX_SERIES_ORDER = 12


class OdeSolveInput(OdeInput):
    """An equation and the analytical approach to try. `closed_form`: a formula, verified by
    substitution. `series`: a Taylor polynomial about the point of the initial values (needs them all).
    `perturbation`: an expansion in a small parameter the user names; with an `equilibrium` it is the
    Poincare-Lindstedt method for periodic motion about it, without one a regular expansion."""

    method: SolveMethod = "closed_form"
    order: int = Field(
        default=6,
        ge=1,
        le=MAX_SERIES_ORDER,
        description="series: the highest power of (x - x0) kept. perturbation: the highest power of "
        "the small parameter kept (at most 6 for Poincare-Lindstedt).",
    )
    small_parameter: str | None = Field(
        default=None,
        max_length=SYMBOL_NAME_MAX,
        description="perturbation only: the parameter that is small. Ask the user; never choose it.",
    )
    equilibrium: str | None = Field(
        default=None,
        max_length=200,
        description="perturbation only: the equilibrium position to oscillate about "
        "(Poincare-Lindstedt, one second-order equation y'' = a(y)). Ask the user.",
    )


# --- reading and structure ----------------------------------------------------------------


class OdeInterpretation(BaseModel):
    """How Minerva read the input. Shown to the user before anything is computed."""

    unknowns: list[str]
    independent: str
    parameters: list[str]
    orders: dict[str, int]
    equations_latex: list[str]
    conditions_latex: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


EquationClass = Literal[
    "separable",
    "linear_first_order",
    "exact",
    "bernoulli",
    "homogeneous_degree_zero",
    "linear_constant_coefficients",
    "euler_cauchy",
    "second_order_linear",
]

ProblemKind = Literal[
    "general",  # no conditions given
    "initial_value",  # all at one point and exactly the number needed
    "boundary_value",  # at two or more points and exactly the number needed
    "incomplete",  # fewer conditions than the order
    "overdetermined",  # more conditions than the order
]


class OdeStructure(BaseModel):
    kind: Literal["ode", "system"]
    n_unknowns: int
    n_equations: int
    orders: dict[str, int]
    order: int
    state_dimension: int  # the size of the equivalent first-order system
    linear: bool
    homogeneous: bool | None  # None when not linear
    constant_coefficients: bool | None  # None when not linear
    autonomous: bool
    solved_for_highest: bool  # the highest derivatives can be written as functions of the rest
    position_only: bool  # y'' = a(y, parameters): a single unknown, no y', no explicit variable


class MethodSuggestion(BaseModel):
    """One approach, whether it can be tried, and what it needs. "Applicable" means that Minerva will
    try it on request: never that it will succeed."""

    id: Literal[
        "symbolic_closed_form",
        "numerical_ivp",
        "numerical_bvp",
        "power_series",
        "regular_perturbation",
        "poincare_lindstedt",
        "multiple_scales",
        "equilibrium_analysis",
        "phase_plane",
    ]
    title: str
    applicable: bool
    available: bool = (
        True  # False: not implemented in Minerva (listed so that nobody assumes it is)
    )
    rank: int | None = None
    reason: str
    requires: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    tool: Literal["ode_solve_symbolic", "ode_solve_numerical", "ode_analyze"] | None = None
    action: str | None = None  # the `method` / `kind` / `detail` value that selects it


class OdeStage(BaseModel):
    name: Literal[
        "read",
        "structure",
        "classify",
        "methods",
        "solve",
        "verify",
        "conditions",
        "integrate",
        "refine",
        "total",
    ]
    ms: float = 0.0


class OdeDiagnostics(BaseModel):
    trace_id: str | None = None
    stages: list[OdeStage] = Field(default_factory=list)
    n_equations: int | None = None
    order: int | None = None
    state_dimension: int | None = None
    classes: list[str] = Field(default_factory=list)
    method: str | None = None
    methods_tried: list[str] = Field(default_factory=list)
    steps: int | None = None
    evaluations: int | None = None
    candidates: int | None = None
    verified: int | None = None
    dropped: int | None = None
    solution_kind: str | None = None
    max_residual: float | None = None
    failure_reason: str | None = None


class OdeAnalysis(BaseModel):
    status: Status
    message: str = ""
    interpretation: OdeInterpretation | None = None
    structure: OdeStructure | None = None
    problem_kind: ProblemKind | None = None
    conditions_needed: list[str] = Field(default_factory=list)
    classes: list[EquationClass] = Field(default_factory=list)
    solved_latex: list[str] = Field(default_factory=list)
    restrictions_latex: list[str] = Field(default_factory=list)
    methods: list[MethodSuggestion] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    diagnostics: OdeDiagnostics = Field(default_factory=OdeDiagnostics)
    duration_ms: int = 0


# --- closed-form solutions ------------------------------------------------------------------


class OdeVerification(BaseModel):
    """What was checked about a candidate solution, and how. The numeric residual check differentiates
    the candidate numerically (it does not use the derivative SymPy would give) and puts the values
    into the equation as written. The symbolic zero test is reported separately and never decides."""

    points_checked: int = 0
    points_agreed: int = 0
    points_skipped: int = 0
    max_residual: float | None = None  # largest residual relative to the size of the terms
    symbolic_zero: bool | None = None  # True: simplified to zero exactly; None: not decided
    independent_constants: int | None = None  # numerical rank of the family's derivatives
    passed: bool = False
    note: str = ""


class ConditionCheck(BaseModel):
    latex: str
    satisfied: bool
    residual: float | None = None


class ExpansionCheck(BaseModel):
    name: Literal["initial_conditions", "residual_order", "boundedness", "frequency_order"]
    passed: bool
    value: float | None = None
    note: str = ""


class OdeExpansion(BaseModel):
    """What a series or perturbation result is, and where it can be trusted."""

    kind: Literal["taylor", "regular_perturbation", "poincare_lindstedt"]
    order: int
    point_latex: str | None = None  # the expansion point (taylor)
    small_latex: str | None = None  # the small parameter (perturbation)
    secular_orders: list[int] = Field(default_factory=list)  # orders with terms that grow with x
    checks: list[ExpansionCheck] = Field(default_factory=list)
    validity: str = ""  # fixed text: local, small parameter, bounded interval


class OdeSolutionItem(BaseModel):
    """One verified solution (or, for a system, one verified solution of every unknown)."""

    kind: Literal["explicit", "implicit", "unevaluated", "series", "perturbation"]
    expansion: OdeExpansion | None = None
    equations_latex: list[str]
    # The right-hand side for each unknown as text the text parser reads (explicit results only).
    rhs_text: list[str] = Field(default_factory=list)
    constants: list[str] = Field(default_factory=list)
    # general: as many independent constants as the order, no conditions used;
    # particular: every constant fixed by the conditions;
    # partial: a family with fewer independent constants than the order, or only some fixed.
    family: Literal["general", "particular", "partial"]
    evaluated: bool = True  # False: an integral remains in the result (a representation)
    verification: OdeVerification
    conditions: list[ConditionCheck] = Field(default_factory=list)
    restrictions_latex: list[str] = Field(default_factory=list)


class OdeSolution(BaseModel):
    status: Status
    message: str = ""
    method: SolveMethod | None = None
    interpretation: OdeInterpretation | None = None
    solutions: list[OdeSolutionItem] = Field(default_factory=list)
    # The weakest of the shown solutions' checks (fewest points, symbolic only if every one proved it).
    verification: OdeVerification | None = None
    dropped: int = 0  # candidates that failed their checks and are not shown
    notes: list[str] = Field(default_factory=list)
    diagnostics: OdeDiagnostics = Field(default_factory=OdeDiagnostics)
    duration_ms: int = 0


# --- numerical initial-value problems -------------------------------------------------------

NumericMethod = Literal["auto", "RK45", "DOP853", "Radau", "BDF", "LSODA"]
NumericWarning = Literal[
    "stiffness_indicated", "step_collapse", "not_converged", "blow_up_suspected", "domain_left"
]
MAX_STATE = 12
MAX_PARAMETERS = 32


class OdeNumericInput(OdeInput):
    """An initial-value problem with every number supplied by the user: the initial values are the
    `conditions` (all at one point, which is the start of the interval), `parameters` gives each
    parameter's value, and `t_end` ends the interval. Nothing is defaulted that changes the problem."""

    model_config = ConfigDict(allow_inf_nan=False)

    parameters: dict[str, float] = Field(
        default_factory=dict,
        max_length=MAX_PARAMETERS,
        description="The numerical value of every parameter. Ask the user: never choose them.",
    )
    kind: Literal["ivp", "bvp"] = Field(
        default="ivp",
        description="`ivp`: initial values at one point (in `conditions`). `bvp`: conditions at two "
        "points (the ends of the interval).",
    )
    t_end: float | None = Field(
        default=None,
        description="ivp: where the interval ends (it starts at the point of the initial values). Ask "
        "the user: never choose it. bvp: not used, the two points of the conditions are the ends.",
    )
    method: NumericMethod = Field(
        default="auto",
        description="`auto` unless the user asked for a method: it starts explicit and switches to an "
        "implicit method when measured behaviour shows stiffness.",
    )
    rtol: float = Field(default=1e-8, ge=1e-12, le=1e-3)
    atol: float = Field(default=1e-10, ge=1e-14, le=1e-3)
    samples: int = Field(default=200, ge=20, le=400)


class NumericCheck(BaseModel):
    name: Literal[
        "initial_conditions",
        "independent_method",
        "tolerance_refinement",
        "defect",
        "first_integral",
        "boundary_conditions",
        "residual",
        "mesh_refinement",
        "shooting",
    ]
    status: Literal["passed", "failed", "skipped"]
    value: float | None = None
    tolerance: float | None = None
    note: str = ""


class NumericVerification(BaseModel):
    checks: list[NumericCheck] = Field(default_factory=list)
    passed: bool = False
    independent_method: str | None = None
    refinement_errors: list[float] = Field(default_factory=list)
    first_integral_latex: str | None = None


class OdeSeries(BaseModel):
    label_latex: str
    unknown: str
    order: int
    values: list[float]


class InitialValue(BaseModel):
    label_latex: str
    value: float


class OdeNumericReport(BaseModel):
    status: Status
    message: str = ""
    interpretation: OdeInterpretation | None = None
    method_requested: NumericMethod | None = None
    method_used: str | None = None
    method_reason: str = ""
    t_start: float | None = None
    t_end: float | None = None
    parameters: dict[str, float] = Field(default_factory=dict)
    initial: list[InitialValue] = Field(default_factory=list)
    rtol: float | None = None
    atol: float | None = None
    steps: int | None = None
    evaluations: int | None = None
    t: list[float] = Field(default_factory=list)
    series: list[OdeSeries] = Field(default_factory=list)
    verification: NumericVerification | None = None
    warnings: list[NumericWarning] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    diagnostics: OdeDiagnostics = Field(default_factory=OdeDiagnostics)
    duration_ms: int = 0
