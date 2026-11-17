"""Models for vector calculus: the request, the result and its diagnostics.

`VectorDiagnostics` holds ids, enums and numbers only (never names, expressions or entries): it is
logged and returned, and a test enforces that no free text reaches it.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..models import ALLOWED_ASSUMPTIONS, Status

MAX_FIELDS = 8
MAX_CONSTANTS = 16
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9]+)?$")

Smoothness = Literal["c1", "c2", "smooth"]
Action = Literal[
    "evaluate",
    "jacobian",
    "convert",
    "identities",
    "apply_identity",
    "substitute",
    "apply",
    "derive",
    "em_derive",
    "integrate",
    "em_compute",
    "distributional",
]
STEP_ACTIONS = ("identities", "apply_identity", "substitute", "apply", "derive", "em_derive")
IntegralKind = Literal[
    "line",
    "line_scalar",
    "flux",
    "volume",
    "gradient_theorem",
    "divergence_theorem",
    "stokes_theorem",
]
ResultKind = Literal["scalar", "vector", "matrix"]
Format = Literal["latex", "text"]

# Why a request was refused or a result withheld (fixed vocabulary; no free text).
Reason = Literal[
    "type_error",
    "undeclared_symbol",
    "bad_declaration",
    "unsupported_expression",
    "needs_smoothness",
    "needs_premises",
    "model_check_failed",
    "singular_in_domain",
    "field_point_in_source",
    "solver_disagreement",
    "missing_geometry",
    "integration_failed",
    "derivative_depth",
    "verification_failed",
    "no_such_rule",
    "no_match",
    "not_found",
    "too_few_points",
    "resource_limit",
    "sandbox_failure",
    "unsupported_singular_field",
    "source_not_real",
    "unsupported_distribution_operation",
]


class FieldDecl(BaseModel):
    """A scalar or vector field.

    Without `components` the field is abstract: an arbitrary function of `depends_on` (default: the
    three coordinates; add `t` for a time-dependent field). With `components` it is given explicitly,
    in the coordinates of the system, always in text syntax (`rho*cos(phi)`, `r**2*sin(theta)`; one
    expression for a scalar, three physical components for a vector). `smoothness` is what the *user* declares; nothing is assumed beyond the existence of the
    derivatives the expression takes.
    """

    name: str
    kind: Literal["scalar", "vector"]
    components: list[str] | None = None
    depends_on: list[str] | None = None
    smoothness: Smoothness | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not NAME_RE.match(value) or len(value) > 24:
            raise ValueError("a field name is a letter or word with an optional subscript")
        return value

    @model_validator(mode="after")
    def _shape(self) -> FieldDecl:
        if self.components is not None:
            expected = 1 if self.kind == "scalar" else 3
            if len(self.components) != expected:
                raise ValueError(f"a {self.kind} field needs {expected} component expression(s)")
            if self.depends_on is not None:
                raise ValueError("an explicit field does not take depends_on")
            if self.smoothness is not None:
                raise ValueError("an explicit field does not take smoothness")
        return self


class Param(BaseModel):
    """A parameter and its range (text syntax: numbers, `pi`, declared constants)."""

    name: str
    lower: str = Field(max_length=120)
    upper: str = Field(max_length=120)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not NAME_RE.match(value) or len(value) > 12 or value == "t":
            raise ValueError("a parameter is a short name other than t (t is time)")
        return value


class Curve(BaseModel):
    """A curve given by the three coordinates of the chosen system as functions of one parameter,
    traversed in the direction of increasing parameter."""

    q: list[str] = Field(min_length=3, max_length=3)
    parameter: Param


class Surface(BaseModel):
    """A surface given by the three coordinates as functions of two parameters over a rectangle. Its
    orientation is that of the normal d_u r x d_v r: the order of `u` and `v` is the orientation."""

    q: list[str] = Field(min_length=3, max_length=3)
    u: Param
    v: Param


class Region(BaseModel):
    """A coordinate box: the lower and upper value of each coordinate of the system, in order. The
    outward boundary is generated from it; nothing else is accepted as a domain."""

    bounds: list[tuple[str, str]] = Field(min_length=3, max_length=3)


class PointCharge(BaseModel):
    """A point charge: its charge and position as text (numbers, `pi`, declared constants)."""

    q: str = Field(max_length=80)
    position: tuple[str, str, str]


class FieldProblem(BaseModel):
    """The electrostatic potential and field at a point, from point charges and/or a charge density
    over a Cartesian box. Everything is in units of k_e = 1/(4 pi epsilon_0)."""

    kind: Literal["field"] = "field"
    charges: list[PointCharge] = Field(default_factory=list, max_length=8)
    density: str | None = Field(default=None, max_length=200)  # rho(x, y, z), text
    box: Region | None = None  # the support of the density, in Cartesian coordinates
    point: tuple[str, str, str]


class MultipoleProblem(BaseModel):
    """Monopole, dipole and quadrupole moments of point charges, and how well the truncated expansion
    reproduces the exact potential along a ray, measured at growing distances."""

    kind: Literal["multipole"] = "multipole"
    charges: list[PointCharge] = Field(min_length=1, max_length=8)
    direction: tuple[str, str, str] = ("0", "0", "1")
    order: Literal[0, 1, 2] = 2


class BoundaryProblem(BaseModel):
    """Laplace's equation (or Poisson's, with `source`) on the rectangle [0, a] x [0, b] with Dirichlet
    data: `bottom` and `top` are functions of x, `left` and `right` functions of y (text; "0" for none).
    Poisson: laplacian(phi) = -source(x, y), then the boundary data must be zero."""

    kind: Literal["boundary_value"] = "boundary_value"
    a: str = Field(max_length=40)
    b: str = Field(max_length=40)
    bottom: str = "0"
    top: str = "0"
    left: str = "0"
    right: str = "0"
    source: str | None = Field(default=None, max_length=200)
    terms: int = Field(default=40, ge=4, le=200)
    points: list[tuple[str, str]] = Field(default_factory=list, max_length=6)


class FaceCondition(BaseModel):
    """phi on one face of a box, as a function of the other two coordinates (text)."""

    face: Literal["x=lo", "x=hi", "y=lo", "y=hi", "z=lo", "z=hi"]
    value: str = Field(max_length=200)


class CheckProblem(BaseModel):
    """Does a candidate potential satisfy laplacian(phi) = -source and the stated boundary values?"""

    kind: Literal["check_solution"] = "check_solution"
    potential: str = Field(max_length=300)
    source: str = Field(default="0", max_length=200)
    box: Region
    conditions: list[FaceCondition] = Field(default_factory=list, max_length=6)


EMProblem = FieldProblem | MultipoleProblem | BoundaryProblem | CheckProblem


class VectorRule(BaseModel):
    """A vector identity handed over by the knowledge layer: `lhs = rhs` in text syntax over typed
    pattern variables. `requires` is the smoothness the identity needs (a handbook fact, not an
    assumption): `c2` means equal mixed partial derivatives."""

    id: str
    lhs: str = Field(max_length=300)
    rhs: str = Field(max_length=300)
    variables: dict[str, Literal["scalar", "vector"]]
    requires: list[Literal["c2"]] = Field(default_factory=list)


class Premise(BaseModel):
    """A field equation the USER supplies (`curl(E) = -dt(B)`). Minerva never checks that it is true:
    a result that uses one is `conditional` and names it."""

    equation: str = Field(max_length=500)
    label: str | None = Field(default=None, max_length=80)


class VectorInput(BaseModel):
    """`expression` over the declared `fields` and `constants`, in one coordinate system."""

    expression: str = Field("", max_length=2000)
    format: Format = "latex"
    coordinates: Literal["cartesian", "cylindrical", "spherical"] = "cartesian"
    fields: list[FieldDecl] = Field(default_factory=list, max_length=MAX_FIELDS)
    constants: list[str] = Field(default_factory=list, max_length=MAX_CONSTANTS)
    assumptions: dict[str, list[str]] = Field(default_factory=dict)
    action: Action = "evaluate"
    to: Literal["cartesian", "cylindrical", "spherical"] | None = None
    # Controlled steps (see steps.py). `rules` are the candidate identities from the handbook.
    rules: list[VectorRule] = Field(default_factory=list, max_length=60)
    rule_id: str | None = None
    path: list[int] | None = Field(default=None, max_length=24)
    reverse: bool = False
    premises: list[Premise] = Field(default_factory=list, max_length=8)
    premise: int | None = Field(default=None, ge=0, le=7)
    operator: Literal["grad", "div", "curl", "lap", "dt"] | None = None
    target: str | None = Field(default=None, max_length=1000)
    template: str | None = Field(default=None, max_length=40)
    em: EMProblem | None = Field(default=None, discriminator="kind")
    # Integrals: the domain is always given, never invented (see integrals.py).
    integral: IntegralKind | None = None
    curve: Curve | None = None
    surface: Surface | None = None
    region: Region | None = None
    trace_id: str | None = None

    @field_validator("assumptions")
    @classmethod
    def _assumptions(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        for name, items in value.items():
            if not NAME_RE.match(name):
                raise ValueError(f"invalid constant name {name!r}")
            unknown = set(items) - ALLOWED_ASSUMPTIONS
            if unknown:
                raise ValueError(f"unknown assumption(s): {sorted(unknown)}")
        return value

    @field_validator("constants")
    @classmethod
    def _constants(cls, value: list[str]) -> list[str]:
        for name in value:
            if not NAME_RE.match(name):
                raise ValueError(f"invalid constant name {name!r}")
        return value


class Blocked(BaseModel):
    """Something the result needs and the declarations do not give."""

    needs: Literal["c2", "smooth"]
    fields: list[str]
    note: str


class VectorCheck(BaseModel):
    name: str
    passed: bool | None
    note: str = ""


class VectorVerification(BaseModel):
    """Evidence for a result. `numeric_passed` is the veto; `symbolic_zero` is reported separately and
    is None when no independent symbolic route was run. Numerical agreement is evidence, not proof."""

    method: str = "cartesian_finite_difference"
    numeric_passed: bool = False
    points_checked: int = 0
    points_agreed: int = 0
    points_skipped: int = 0
    max_residual: float | None = None
    symbolic_zero: bool | None = None
    checks: list[VectorCheck] = Field(default_factory=list)
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.numeric_passed and self.symbolic_zero is not False


class VectorStage(BaseModel):
    name: Literal["parse", "lower", "tidy", "verify", "render", "match", "search", "singular"]
    duration_ms: int = 0


class VectorDiagnostics(BaseModel):
    """Ids, enums and numbers only."""

    action: Action = "evaluate"
    coordinates: Literal["cartesian", "cylindrical", "spherical"] = "cartesian"
    result_kind: ResultKind | None = None
    operator_depth: int = 0
    abstract_fields: int = 0
    explicit_fields: int = 0
    stages: list[VectorStage] = Field(default_factory=list)
    reason: Reason | None = None
    points: int = 0
    trace_id: str | None = None


StepEvidence = Literal["identity", "conditional"]


class VectorStep(BaseModel):
    """One controlled move. `evidence` is "identity" when it holds with no premise (under the declared
    smoothness) and "conditional" when it holds only given the premises in `given`."""

    action: Action
    title: str
    before_latex: str
    after_latex: str
    rule_id: str | None = None
    reverse: bool = False
    path: list[int] = Field(default_factory=list)
    part_latex: str | None = None
    evidence: StepEvidence = "identity"
    given: list[str] = Field(default_factory=list)
    needs: list[str] = Field(default_factory=list)
    verification: VectorVerification | None = None
    symbolic_zero: bool | None = None
    chain: int | None = None  # which route of an electromagnetic derivation (0-based)


class ModelCheck(BaseModel):
    """A derived statement evaluated on exact solutions of the premises it was derived from
    (plane waves in vacuum, fields built from random potentials with the sources they imply). A
    pass is "consistent with N exact solutions", never a verification; a failure withholds the result."""

    status: Literal["passed", "failed", "not_run"]
    kind: str = ""
    models: int = 0
    points: int = 0
    max_residual: float | None = None
    note: str = ""


class IntegralValue(BaseModel):
    """One integral: its exact value when SymPy found one, and always an independent quadrature."""

    label: str
    latex: str | None = None  # exact value; None when only a numerical value exists
    text: str | None = None
    numeric: str | None = None  # decimal string; None while constants are symbols
    exact: bool = False
    quadrature_error: float | None = None
    agreed: bool | None = (
        None  # exact value against the independent quadrature (None: no exact value)
    )
    points: int = 0


class Hypothesis(BaseModel):
    """A hypothesis of a theorem and what is known about it: `established` (by construction),
    `checked` (a finite scan found no violation: not a proof) or `not_established`."""

    text: str
    status: Literal["established", "checked", "not_established"]
    note: str = ""


class IntegralResult(BaseModel):
    kind: IntegralKind
    values: list[IntegralValue] = Field(default_factory=list)  # one integral, or the two sides
    theorem: str | None = None
    holds: bool | None = None  # theorems: the two sides agree
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    orientation: str = ""
    method: Literal["symbolic", "numerical"] = "symbolic"


class EMItem(BaseModel):
    label: str
    latex: str | None = None
    numeric: str | None = None
    note: str = ""


class EMComputation(BaseModel):
    """The result of a numerical or analytical electromagnetic computation, with how it was checked."""

    kind: Literal["field", "multipole", "boundary_value", "check_solution"]
    items: list[EMItem] = Field(default_factory=list)
    method: str = ""
    checks: list[str] = Field(default_factory=list)
    approximate: bool = False
    units: str = ""
    series_terms: int | None = None
    agreement: float | None = None  # largest measured disagreement between two methods
    holds: bool | None = None  # check_solution: every requirement is met


class Applicable(BaseModel):
    """An identity that matches a part of the expression (listed, not applied or verified)."""

    rule_id: str
    reverse: bool
    path: list[int]
    part_latex: str
    result_latex: str
    needs: list[str] = Field(default_factory=list)
    blocked: bool = False
    note: str = ""


SingularStatus = Literal["established", "checked", "not_established"]


class SingularStep(BaseModel):
    """One step of the shrinking-sphere derivation and what is known about it: `established` (an exact
    symbolic computation, or true by construction), `checked` (numerical agreement: not a proof) or
    `not_established`."""

    title: str
    latex: str
    status: SingularStatus
    note: str = ""


class WeakPairing(BaseModel):
    """One independent numerical pairing: the distribution applied to a rapidly decreasing test function,
    computed by quadrature over R^3 (or the flux through a sphere) from the *input* field, against the
    value the derived identity predicts. A measurement on sampled constants, never a bound."""

    label: str
    test_function_latex: str = ""
    expected: float
    measured: float
    error: float
    passed: bool


class Regularisation(BaseModel):
    """`\\int phi div(F_eps)` for the smoothed field `F_eps = C (r-a)/(|r-a|^2 + eps^2)^{3/2}` at one
    width: supporting evidence that the limit is the derived coefficient, reported and never deciding."""

    eps: float
    value: float
    expected: float
    error: float


class SingularClaim(BaseModel):
    """A statement the user made about the distribution (`target`), compared exactly with the derived one."""

    latex: str
    holds: bool
    reason: Literal[
        "agrees",
        "no_delta_term",
        "different_strength",
        "different_source",
        "not_readable",
    ]
    note: str = ""


class SingularResult(BaseModel):
    """The distributional divergence (or Laplacian) of a point-source Coulomb kernel, with its evidence.

    The classical result (`classical_latex`, valid away from the source) and the distributional one
    (`distributional_latex`) are separate on purpose: the classical divergence of `r/r^3` is 0 at every
    point where it is defined, and as a distribution it is `4 pi delta^(3)(r)`. `proof_status` is
    `derived` only when every hypothesis of the shrinking-sphere argument is an exact symbolic fact for
    this input and every independent pairing agreed; nothing here is called proved, and the numerical
    pairings and the regularisation table are supporting evidence.
    """

    family: Literal["divergence", "laplacian"]
    classical_latex: str
    classical_domain_latex: str
    distributional_latex: str
    strength_latex: str
    source_latex: str
    support: str = "the single point where the kernel is singular"
    derivation: list[SingularStep] = Field(default_factory=list)
    flux_latex: str = ""
    weak_checks: list[WeakPairing] = Field(default_factory=list)
    flux_checks: list[WeakPairing] = Field(default_factory=list)
    regularisation: list[Regularisation] = Field(default_factory=list)
    proof_status: Literal["derived", "not_established"] = "not_established"
    scope: str = ""
    unresolved: list[str] = Field(default_factory=list)
    claim: SingularClaim | None = None


class VectorResult(BaseModel):
    status: Status
    message: str = ""
    action: Action = "evaluate"
    coordinates: str = "cartesian"
    to: str | None = None
    result_kind: ResultKind | None = None
    # What was read (echo of the parsed input) and the result.
    input_latex: str = ""
    result_latex: str | None = None
    result_text: str | None = None
    components_latex: list[str] = Field(default_factory=list)
    components_text: list[str] = Field(default_factory=list)
    basis_latex: list[str] = Field(default_factory=list)
    scale_factors_latex: list[str] = Field(default_factory=list)
    determinant_latex: str | None = None
    verification: VectorVerification | None = None
    conventions: list[str] = Field(default_factory=list)
    assumptions_used: list[str] = Field(default_factory=list)
    restrictions_latex: list[str] = Field(default_factory=list)
    restrictions_complete: bool = True
    blocked: list[Blocked] = Field(default_factory=list)
    undeclared: list[str] = Field(default_factory=list)
    steps: list[VectorStep] = Field(default_factory=list)
    applicable: list[Applicable] = Field(default_factory=list)
    evidence: StepEvidence | None = None
    given: list[str] = Field(default_factory=list)
    search_complete: bool | None = None
    equations_latex: list[str] = Field(default_factory=list)
    needs_premises: list[str] = Field(default_factory=list)
    model_check: ModelCheck | None = None
    derivation: str | None = None
    integral: IntegralResult | None = None
    em: EMComputation | None = None
    singular: SingularResult | None = None
    singular_hint: Literal["point_source"] | None = None
    diagnostics: VectorDiagnostics = Field(default_factory=VectorDiagnostics)
    duration_ms: int = 0
