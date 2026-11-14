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
    name: Literal["parse", "lower", "tidy", "verify", "render", "match", "search"]
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
    diagnostics: VectorDiagnostics = Field(default_factory=VectorDiagnostics)
    duration_ms: int = 0
