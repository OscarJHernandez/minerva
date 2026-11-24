"""Tools the agent may call. Each is a pure, sandboxed computation or a read-only handbook lookup.

A tool returns two things: a `Card` (rendered by Minerva from the tool's own output, never by the
model) and a compact `for_model` text (what the model gets to read). Tools have no side effects, so a
confused or manipulated model can at worst produce a poor explanation.
"""

from __future__ import annotations

import contextlib
import json
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from minerva_engine import (
    CompareRequest,
    DiffVariable,
    DistributionInput,
    EndpointCondition,
    Engine,
    ExpressionInput,
    GeometryInput,
    IntegrateResult,
    MatrixCalculusInput,
    MatrixDecl,
    MotionRequest,
    OdeAnalysis,
    OdeAnalyzeInput,
    OdeInput,
    OdeNumericInput,
    OdeNumericReport,
    OdeSolution,
    OdeSolveInput,
    Pins,
    QuantitySpec,
    SeriesEvidence,
    SeriesResult,
    SeriesTermEvidence,
    TensorInput,
    TensorSpec,
    UnitsCheckReport,
    VariationInput,
    VectorInput,
    VectorResult,
)
from minerva_engine.claims import mentions_name
from minerva_engine.geometry import QUANTITIES, FunctionDecl, TensorField
from minerva_engine.integration_strategy.models import StrategyName
from minerva_engine.notation.models import InputDiagnostic, Normalization
from minerva_engine.vector import Curve, FieldDecl, Premise, Region, Surface
from minerva_knowledge import KnowledgeBase, SearchQuery
from minerva_knowledge.identities import (
    TransformResponse,
    constant_candidates,
    find_identities,
    premise_options,
    resolve_premise_ids,
    simplify_expression,
    summarize,
    vector_identity_rules,
)
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator
from pydantic.json_schema import SkipJsonSchema

from ..diagnostics import log_distribution
from ..geometry_log import log_geometry
from ..logging import error_fields
from ..ode_log import log_ode
from ..search_log import log_search_events
from ..variation_log import log_variation
from ..vector_log import log_vector
from .choice import numbers_in
from .constants import KnownConstant, Refusal, expand_constants
from .events import Card, Evidence
from .evidence import evidence_for
from .mathinput import (
    FORMAT_DESCRIPTION,
    MATH_FIELDS,
    SyntaxChoice,
    diagnostic_line,
    input_lines,
    input_record,
    resolve_syntax,
    strip_assignment,
    transport_line,
)
from .prompts import GROUP_SUMMARY, GROUP_TOOLS
from .segments import normalize
from .toolsets import GROUPS, requested_groups
from .tracing import category_for
from .unit_statements import assumed_quantities

MAX_INPUT = 2000


@dataclass
class ToolContext:
    engine: Engine
    kb: KnowledgeBase
    # Names this conversation declared as matrices or operators (set by the loop before each call).
    # Tools that treat symbols as commuting numbers refuse expressions that mention one.
    matrices: frozenset[str] = frozenset()
    # Where a declared name came from, for the refusal: name -> "your message: «sentence»" (set by the
    # loop). Never logged; shown only to the user who wrote the sentence.
    matrix_sources: dict[str, str] = field(default_factory=dict)
    # Of `matrices`, the names declared as vectors or vector fields (not matrices). A vector's components are
    # scalars (`r_i`, a coordinate `r`), so the tools that only a declared MATRIX makes wrong keep answering
    # for them (`MATRIX_ONLY_TOOLS`).
    vector_names: frozenset[str] = frozenset()
    # What the USER answered on a card (`delta(...)` is a Dirac delta). Only the direct-post route fills
    # it; the agent loop never does, so the model cannot make such an answer.
    notation: dict[str, str] = field(default_factory=dict)
    # The numbers and the action of a numerical `variation` run (parameter values, trial paths, the
    # starting guess): only the direct-post route fills it, from a card's form. The model's schema
    # has no such fields, so it can neither choose them nor run a numerical action.
    numeric: dict[str, Any] = field(default_factory=dict)
    # What the USER answered or chose for statistical mechanics (phase-space cell, indistinguishable
    # copies, numbers, sweep). Only the direct-post route fills it; the model cannot make these choices.
    statmech_answers: dict[str, Any] = field(default_factory=dict)
    # The USER's messages of this conversation, oldest first (set by the chat route; None: not a chat
    # turn). The loop reads them to decide whether the user handed the choice of numbers to the assistant.
    user_messages: Sequence[str] | None = None
    # Set by the loop for the model's own calls only, never by the model and never by the card form
    # (which posts the user's numbers): whether the user handed the numbers over, and every number the
    # user wrote. `user_numbers is None` means the numbers came from the user.
    numbers_delegated: bool = False
    user_numbers: frozenset[float] | None = None
    # Set by the loop for one call (and cleared after it): what `mathinput.prepare_arguments` removed from
    # the model's arguments (markdown escapes, a `$` wrapper, `L =`...). `execute` shows it with the result.
    input_notes: tuple[Normalization, ...] = ()
    # The constants of motion earlier results of this conversation found and verified (set by the chat route
    # from the stored discovery cards, never by the model): `time_derivative` inserts their expressions for
    # the names `C_{1}`... that its quantity uses (`agent/constants.py`).
    constants: tuple[KnownConstant, ...] = ()


@dataclass
class ToolOutput:
    card: Card
    for_model: str
    # Runners put every formula of the result here; `execute` then keeps in it only the formulas a
    # verified result produced, and moves the rest to `quoted_latex` (formula, reason it is not verified).
    known_latex: list[str] = field(default_factory=list)
    status: str = "ok"  # the tool's own status, for the debug trace
    quoted_latex: list[tuple[str, str]] = field(default_factory=list)
    # Filled in by `execute` from the result itself (see evidence.py), never by the model.
    evidence: Evidence | None = None
    arguments_valid: bool = True  # False: the call was refused before any computation
    error_category: str | None = None  # tracing.ErrorCategory value, when the result is a failure
    # The ids of the methods an analysis found applicable, in its order (`ode_analyze`); the loop asks
    # the user which to run when there are several and their message named none.
    choices: tuple[str, ...] = ()
    # For the log and the trace only (see tracing.ToolOutcome): a fixed code for why the call failed
    # (a validation error type), the schema paths that failed validation with list indices and
    # dictionary keys replaced by `*`, and where an unexpected exception was raised. Never a value, a
    # symbol name or an exception message.
    reason_code: str | None = None
    invalid_fields: tuple[str, ...] = ()
    error_site: dict[str, Any] = field(default_factory=dict)
    # What each formula is (`equation_of_motion`, `acceleration`, `momentum`, `conserved`), and the names of
    # the system's coordinates: used only to compare the model's later equations with this result
    # (`claim_evidence.py`). Never read from the model.
    formula_roles: dict[str, str] = field(default_factory=dict)
    variables: tuple[str, ...] = ()
    # A verified Taylor series as a typed record (`SeriesEvidence.model_dump()`), built from the engine's own result:
    # the claim check links a model equation `f = P + O(...)` to it only if every part agrees, and reads the centre and
    # the places where the error was measured as parameters of this call, not definitions.
    series: dict[str, Any] | None = None


# --- argument models (also the JSON schema the model sees) ------------------------------------
class SimplifyArgs(BaseModel):
    latex: str = Field(
        max_length=MAX_INPUT, description="The expression in LaTeX, e.g. \\frac{x^2-1}{x-1}"
    )
    assumptions: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Properties of symbols: real, positive, negative, nonnegative, nonpositive, "
        'nonzero, integer, rational. Example: {"x": ["positive"]}',
    )


class SimplifyToolArgs(SimplifyArgs):
    format: SkipJsonSchema[SyntaxChoice] = "latex"
    goal: Literal["simplify", "expand", "factor", "rewrite"] = Field(
        default="simplify",
        description="What the result should be: simplify (fewest operations), expand, factor, or "
        "rewrite (in terms of the functions listed in `target`).",
    )
    target: list[str] = Field(
        default_factory=list,
        max_length=8,
        description='Only for goal "rewrite": function names to express the result in, e.g. ["sin", "cos"].',
    )


class CheckIdentityArgs(SimplifyArgs):
    format: SkipJsonSchema[SyntaxChoice] = "latex"
    latex: str = Field(
        max_length=MAX_INPUT,
        description="One equation `left = right` in LaTeX, as the user wrote it, e.g. \\sqrt{x^2} = x",
    )


class DifferentiateArgs(SimplifyArgs):
    format: SkipJsonSchema[SyntaxChoice] = "latex"
    latex: str = Field(
        max_length=MAX_INPUT, description="The expression to differentiate, in LaTeX, e.g. x^x"
    )
    variables: list[DiffVariable] = Field(
        min_length=1,
        max_length=4,
        description="Variables to differentiate by, in order, each with an order (1 to 10; total "
        'at most 10). Example: [{"name": "x", "order": 2}] is the second derivative in x; '
        '[{"name": "x"}, {"name": "y"}] is the mixed partial in x then y.',
    )


class SeriesArgs(SimplifyArgs):
    format: SkipJsonSchema[SyntaxChoice] = "latex"
    latex: str = Field(
        max_length=MAX_INPUT,
        description="The expression to expand, in LaTeX, e.g. \\frac{1}{\\sqrt{1-x^2}}",
    )
    variable: str = Field(min_length=1, max_length=32, description="The variable to expand in")
    point: str = Field(
        default="0",
        max_length=64,
        description="Finite real point to expand about, as a number (text syntax: 0, 1, 1/2, pi)",
    )
    order: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Highest power of (variable - point) kept, 1 to 12. The remainder is O of the next power.",
    )


class MatrixCalculusArgs(SimplifyArgs):
    latex: str = Field(
        max_length=MAX_INPUT,
        description="Expression in LaTeX (text syntax for the gradient actions) in the declared matrices. Write exponentials as \\exp(tA), never as e^{tA} "
        "(a bare e is an ordinary symbol). Products keep their order. Matrices are constant unless "
        "declared with depends_on; a matrix that depends on the variable is written A(t), and tA "
        "always means t times A. For action evolve this is the generator A(t) of dU/dt = A(t)U.",
    )
    matrices: list[MatrixDecl] = Field(
        min_length=1,
        max_length=6,
        description="Every matrix in the expression: a size (rows, cols) or explicit entries. "
        "Ask the user for sizes or entries; never invent them. Set depends_on to the variable only "
        "when the USER says the matrix depends on it.",
    )
    action: Literal[
        "evaluate",
        "simplify",
        "differentiate",
        "integrate",
        "evolve",
        "gradient",
        "jacobian",
        "hessian",
    ] = Field(
        default="simplify",
        description="gradient / jacobian / hessian: derivative of a scalar (or vector) function with "
        "respect to a MATRIX or VECTOR `variable`, over the reals. For these three actions write the "
        "expression in the PLAIN-TEXT syntax in `latex`: tr(A*X*B), transpose(x), inv(S), det(X), "
        "logdet(S) (= log|det|), `*` for the matrix product, e.g. "
        '"-1/2*logdet(S) - 1/2*transpose(x - mu)*inv(S)*(x - mu)". Declare `symmetric` or '
        "`positive_definite` on a covariance ONLY if the USER says so (the result then is the gradient "
        "within the symmetric matrices, with the entrywise form labelled separately); inv, logdet and "
        "det need the matrix listed in `invertible` (or positive definite), which only the USER may "
        "state. evaluate: explicit matrices only. simplify: merge what is provably mergeable. "
        "differentiate/integrate need `variable`; integrate may take bounds `lower` and `upper`; "
        "differentiate also handles exponents that do not commute with their derivative (Frechet "
        "derivative) and time-dependent matrices. integrate recognises Duhamel integrals "
        "(exp((t-s)A) B exp(sA) over s). evolve solves dU/dt = A(t)U, U(t0)=I: `variable` is time, "
        "`lower` the initial time (default 0), `upper` the end of the span.",
    )
    variable: str | None = Field(
        default=None, max_length=32, description="Scalar variable of differentiation or integration"
    )
    lower: str | None = Field(
        default=None,
        max_length=200,
        description="Lower bound (a scalar expression); for evolve the initial time",
    )
    upper: str | None = Field(
        default=None,
        max_length=200,
        description="Upper bound (a scalar expression); for evolve the end of the numerical span: "
        "a number only the USER may choose, so ask and never pick one",
    )
    commuting: list[tuple[str, str]] = Field(
        default_factory=list,
        max_length=16,
        description="Pairs of matrices the USER says commute (for time-dependent matrices: at all "
        "times). Never add a pair on your own.",
    )
    self_commuting: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Time-dependent matrices A the USER says satisfy A(t1)A(t2) = A(t2)A(t1) at "
        "all times. Never add one on your own.",
    )
    invertible: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Matrices the USER says are invertible. Never add one on your own.",
    )


class GeometryArgs(BaseModel):
    """Arguments of `geometry`. The whole geometry is sent every time: nothing is remembered between calls."""

    coordinates: list[str] = Field(
        min_length=1,
        max_length=4,
        description="Coordinate names in order, plain names as the LaTeX parser reads them: "
        '["t", "r", "theta", "phi"]. They are real symbols.',
    )
    metric: list[list[str]] = Field(
        min_length=1,
        max_length=4,
        description="The metric components g_{mu nu} as an n by n grid of strings in one syntax (see `format`), one row per "
        'coordinate, e.g. [["-(1-2M/r)", "0"], ["0", "(1-2M/r)^{-1}"]]. Use \\sin^2\\theta, '
        "not \\sin 2\\theta. Ask the user for the metric; never invent one.",
    )
    format: SyntaxChoice = Field(
        default="auto",
        description="Syntax of the `metric` entries and `tensor` components: LaTeX (\\sin^2\\theta, \\frac{2M}{r}) "
        "or text (sin(theta)^2, 2*M/r), the same for every entry. " + FORMAT_DESCRIPTION,
    )
    assumptions: dict[str, list[str]] = Field(
        default_factory=dict,
        description='Properties of symbols (e.g. {"M": ["positive"]}). Coordinates are real already. '
        "Do not add assumptions the user did not state.",
    )
    functions: list[FunctionDecl] = Field(
        default_factory=list,
        max_length=6,
        description='Unspecified real functions of coordinates used in the metric, e.g. [{"name": "f", '
        '"args": ["r"]}] for f(r). A function must be declared before it is written f(r).',
    )
    quantities: list[Literal[*QUANTITIES]] = Field(  # type: ignore[valid-type]
        min_length=1,
        max_length=len(QUANTITIES),
        description="What to compute: inverse_metric, determinant, christoffel, riemann, ricci, ricci_scalar, "
        "einstein, kretschmann, ricci_squared, geodesics (symbolic geodesic equations), "
        "covariant_derivative, raise_index, lower_index, and for electromagnetic fields on a metric you give: "
        "exterior_derivative (F = dA from a 1-form, or dF of a 2-form), antisymmetric_divergence "
        "(J^nu = nabla_mu F^(mu nu)), field_invariant (F_(mu nu) F^(mu nu)), hodge_dual (four dimensions). "
        "Everything from covariant_derivative on needs `tensor` (covariant components for the "
        "electromagnetic ones). The metric, signature and units are the user's; never assume them.",
    )
    tensor: TensorField | None = Field(
        default=None,
        description="A tensor field given by variance (up/down per index) and its n**rank components, "
        "row-major, as LaTeX strings. Only for the tensor quantities.",
    )
    tensor_index: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Which index (0-based) raise_index or lower_index acts on",
    )


class VectorCalculusArgs(BaseModel):
    """Arguments of `vector_calculus`. The declarations are sent every time: nothing is remembered."""

    expression: str = Field(
        default="",
        max_length=MAX_INPUT,
        description="A scalar or vector expression in LaTeX, e.g. \\nabla\\times(f\\mathbf{A}), "
        "\\nabla\\cdot(\\mathbf{A}\\times\\mathbf{B}), \\nabla^2\\mathbf{A}, "
        "(\\mathbf{A}\\cdot\\nabla)\\mathbf{B}, \\frac{\\partial\\mathbf{E}}{\\partial t}, "
        "\\mathbf{r}/r^3 (position vector; r is a coordinate only in spherical coordinates, in Cartesian "
        "write the distance as |\\mathbf{r}|). \\nabla acts on the ONE factor after it: write \\nabla(fg), "
        "never \\nabla f g. \\cdot between two vectors is the dot product, \\times between two "
        "vectors the cross product; the product of two vectors written side by side is refused. "
        "For integrate, the expression is the integrand written with the NAMES of declared fields (F, or "
        "\\nabla\\times F), never the tuple of components: those go in `fields`. "
        "Not needed for action=jacobian.",
    )
    coordinates: Literal["cartesian", "cylindrical", "spherical"] = Field(
        default="cartesian",
        description="cartesian (x, y, z), cylindrical (rho, phi, z) or spherical (r, theta, phi, theta "
        "the polar angle). Vector components are physical components along the unit vectors. "
        "Use the coordinate the user's problem is written in; do not change it.",
    )
    fields: list[FieldDecl] = Field(
        default_factory=list,
        max_length=8,
        description='Every scalar or vector field in the expression, e.g. [{"name": "A", "kind": '
        '"vector"}, {"name": "f", "kind": "scalar"}]. Without `components` the field is an arbitrary '
        'function of the coordinates (add depends_on: [..., "t"] only if the user says it depends on '
        "time). `components` gives it explicitly in TEXT syntax (rho*cos(phi), r**2*sin(theta); 1 "
        "expression for a scalar, 3 for a vector). `smoothness` (c1, c2, smooth) is only what the USER "
        "states: without it a result that needs mixed partial derivatives to commute (curl of a "
        "gradient, divergence of a curl) is withheld; ask the user, never declare it yourself.",
    )
    constants: list[str] = Field(
        default_factory=list,
        max_length=16,
        description="Names of constants (k, mu_0, c) that do not depend on position or time. Every symbol "
        "that is not a coordinate, t or a declared field must be listed here.",
    )
    assumptions: dict[str, list[str]] = Field(
        default_factory=dict,
        description='Properties of constants (e.g. {"k": ["positive"]}). Do not add what the user did '
        "not state.",
    )
    action: Literal[
        "evaluate",
        "jacobian",
        "convert",
        "identities",
        "apply_identity",
        "substitute",
        "apply",
        "derive",
        "integrate",
        "distributional",
    ] = Field(
        default="evaluate",
        description="evaluate: compute the expression in the components of the system. jacobian: the "
        "Jacobian, determinant and scale factors of the map to Cartesian coordinates. convert: re-express "
        "an expression of EXPLICIT fields in the system `to`. identities: list the handbook identities "
        "that match a part of the expression (applies none). apply_identity: apply `rule_id` at `path` "
        "(or right to left with reverse=true), checked by comparing the whole expressions before and "
        "after. substitute: replace one side of a premise the USER gave by the other. apply: apply "
        "`operator` to both sides of an equation (`expression` is `lhs = rhs`). derive: search a chain "
        "of identity (and premise) steps to `target`, or to a smaller form without one. integrate: a line, "
        "flux, volume integral or one of the three theorems, over a domain the USER gives (see `integral`). "
        "distributional: the DISTRIBUTIONAL divergence of a point-source Coulomb field C (r-a)/|r-a|^3, or the "
        "Laplacian of C/|r-a|. `expression` MUST contain the operator, e.g. "
        "\\nabla\\cdot\\left(\\frac{\\mathbf{r}}{r^{3}}\\right) in spherical coordinates (Cartesian: write "
        "|\\mathbf{r}|), never the field alone. The engine returns the classical value 0 away from the source AND "
        "the delta term with its coefficient: use it for a delta function, a point charge, or a zero divergence "
        "with a nonzero flux; never write the coefficient yourself; put the user's own statement in `target`.",
    )
    to: Literal["cartesian", "cylindrical", "spherical"] | None = Field(
        default=None, description="Target system for convert."
    )
    rule_id: str | None = Field(
        default=None,
        description="apply_identity: the handbook id of the identity, e.g. vector.curl_curl (from a "
        "previous `identities` call).",
    )
    path: list[int] | None = Field(
        default=None,
        max_length=24,
        description="apply_identity or substitute: the part to act on, as the path the `identities` "
        "call listed ([] is the whole expression). Never guess it: list first when several places match.",
    )
    integral: (
        Literal[
            "line",
            "line_scalar",
            "flux",
            "volume",
            "gradient_theorem",
            "divergence_theorem",
            "stokes_theorem",
        ]
        | None
    ) = Field(
        default=None,
        description="integrate: to verify or compare both sides of Stokes' theorem use stokes_theorem, never flux. line (the work of a vector field along `curve`), line_scalar, flux "
        "(through `surface`), volume (of a scalar over `region`), gradient_theorem (scalar f along "
        "`curve`), divergence_theorem (vector F over `region`: the volume integral of div F and the flux "
        "through the box's outward faces, computed separately), stokes_theorem (vector F over `surface`: "
        "the flux of curl F and the circulation around the boundary). `expression` is the integrand; "
        "its fields need explicit `components`. The domain is NEVER invented: ask the user for the curve, "
        "surface or region and their orientation.",
    )
    curve: Curve | None = Field(
        default=None,
        description="A curve: the three coordinates of `coordinates` as text functions of one parameter "
        "with a range, traversed in the direction of increasing parameter.",
    )
    surface: Surface | None = Field(
        default=None,
        description="A surface: the three coordinates as text functions of two parameters over a "
        "rectangle. The orientation is the normal d_u r x d_v r: the ORDER of u and v is the orientation, "
        "so state it and never choose it for the user. The surface is only `q`, `u` and `v`; the FIELD is "
        "declared in `fields` in the system's own coordinates (x, y, z), never in u and v. Example shape "
        "(not a problem to copy): F = (y, z, x) over r(u, v) = (u, v, u + v), 0 <= u, v <= 2 is "
        "integral stokes_theorem, expression F, fields [{name: F, kind: vector, components: [y, z, x]}], "
        "surface {q: [u, v, u + v], u: {name: u, lower: 0, upper: 2}, v: {name: v, lower: 0, upper: 2}}.",
    )
    region: Region | None = Field(
        default=None,
        description="A coordinate box: (lower, upper) for each coordinate of the system, in order. "
        "Nothing else is accepted as a volume.",
    )
    reverse: bool = Field(default=False, description="Read the identity or premise right to left.")
    premises: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Field equations the USER stated, each `lhs = rhs` in the same syntax as the "
        "expression, e.g. \\nabla\\times\\mathbf{E} = -\\frac{\\partial\\mathbf{B}}{\\partial t}. "
        "Never invent or complete one: a result that uses a premise holds only given it, and Minerva does "
        "not check that the premise is true.",
    )
    premise_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Handbook field equations (em.maxwell.faraday, em.maxwell.ampere_vacuum, ...) the USER "
        "chose to adopt as premises, listed by `electromagnetism` with action setup. Adopt one only when "
        "the user said to use that law; the equation then holds in the result only as a premise.",
    )
    premise: int | None = Field(
        default=None, ge=0, le=7, description="substitute: which premise (0-based) to use."
    )
    operator: Literal["grad", "div", "curl", "lap", "dt"] | None = Field(
        default=None, description="apply: the operator to apply to both sides."
    )
    target: str | None = Field(
        default=None,
        max_length=MAX_INPUT,
        description="derive: the expression to reach (same syntax as `expression`). distributional: the "
        "statement the USER made about the result, written as they wrote it (e.g. 4\\pi\\delta^{(3)}(\\mathbf{r}) "
        "or 0); the engine compares it exactly and says whether it holds.",
    )


class ElectromagnetismArgs(BaseModel):
    """Arguments of `electromagnetism`."""

    action: Literal["setup", "derive", "field", "multipole", "boundary_value", "check_solution"] = (
        Field(
            description="setup: list the derivations Minerva can carry out, the field equations (premises) each "
            "needs, and the field equations of the handbook the user may adopt. derive: carry one out. "
            "field: potential and electric field of point charges and/or a charge density at a point. "
            "multipole: monopole, dipole and quadrupole moments of point charges with a measured far-field "
            "check. boundary_value: Laplace or Poisson on a rectangle with Dirichlet data (a numerical "
            "approximation, shown only if two methods agree). check_solution: does a candidate potential "
            "satisfy Poisson's equation and the stated face values. All positions, charges, sizes and "
            "values come from the USER; never choose them. Units: k_e = 1/(4 pi epsilon_0). Cartesian only."
        )
    )
    derivation: str | None = Field(
        default=None,
        max_length=40,
        description="derive: wave_equation_e, wave_equation_b, continuity, poynting, gauge_invariance, "
        "poisson or laplace.",
    )
    premise_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="derive: the handbook field equations the USER chose to adopt as premises (from setup). "
        "Never adopt one the user did not choose; the result holds only given them.",
    )
    premises: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="derive: further field equations the USER stated, `lhs = rhs` in text syntax "
        "(curl(E) = -dt(B)).",
    )
    smoothness: Literal["c2", "smooth"] | None = Field(
        default=None,
        description="derive: the smoothness the USER stated for the fields (twice continuously "
        "differentiable, or smooth). Without it identities that need equal mixed partial derivatives are "
        "blocked; never add it yourself.",
    )
    charges: list[tuple[str, str, str, str]] = Field(
        default_factory=list,
        max_length=12,
        description="field/multipole: point charges as (q, x, y, z) text, e.g. ('q', '0', '0', 'd').",
    )
    point: tuple[str, str, str] | None = Field(
        default=None, description="field: where to evaluate (x, y, z), given by the user."
    )
    density: str | None = Field(
        default=None,
        max_length=200,
        description="field: charge density rho(x, y, z) as text syntax over `box`.",
    )
    box: list[tuple[str, str]] | None = Field(
        default=None,
        max_length=3,
        description="field (density) / check_solution: [(x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)].",
    )
    direction: tuple[str, str, str] = Field(
        default=("0", "0", "1"), description="multipole: ray along which the far field is compared."
    )
    order: Literal[0, 1, 2] = Field(default=2, description="multipole: highest multipole kept.")
    side_a: str | None = Field(
        default=None, max_length=60, description="boundary_value: width (x)."
    )
    side_b: str | None = Field(
        default=None, max_length=60, description="boundary_value: height (y)."
    )
    bottom: str = Field(default="0", max_length=200, description="boundary_value: phi(x, 0), in x.")
    top: str = Field(default="0", max_length=200, description="boundary_value: phi(x, b), in x.")
    left: str = Field(default="0", max_length=200, description="boundary_value: phi(0, y), in y.")
    right: str = Field(default="0", max_length=200, description="boundary_value: phi(a, y), in y.")
    source: str | None = Field(
        default=None,
        max_length=200,
        description="boundary_value: Poisson source rho/eps0 in x, y (zero edges only); "
        "check_solution: the source in laplacian(phi) = -source.",
    )
    sample_points: list[tuple[str, str]] = Field(
        default_factory=list, max_length=8, description="boundary_value: interior points to report."
    )
    potential: str | None = Field(
        default=None, max_length=300, description="check_solution: candidate phi(x, y, z)."
    )
    faces: list[tuple[str, str]] = Field(
        default_factory=list,
        max_length=6,
        description="check_solution: (face, value) with face in x=lo, x=hi, y=lo, y=hi, z=lo, z=hi.",
    )
    constants: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="field/multipole/...: declared constant names.",
    )


class OdeAnalyzeArgs(OdeAnalyzeInput):
    """Arguments of `ode_analyze`: the equation(s) exactly as the user wrote them. `detail=qualitative`
    adds equilibria, linearisation and local stability (and a direction field with a window)."""


class OdeSymbolicArgs(OdeSolveInput):
    """Arguments of `ode_solve_symbolic`: `closed_form`, `series` or `perturbation`."""


class OdeNumericArgs(OdeNumericInput):
    """Arguments of `ode_solve_numerical`. Every number comes from the user."""


class VariationArgs(BaseModel):
    """Arguments of `variation`. The whole functional is sent every time: nothing is remembered."""

    action: Literal["analyze", "first", "second"] = Field(
        default="first",
        description="analyze: structure (orders, regularity, boundary slots, next steps); first: the "
        "first variation with every integration by parts, the Euler-Lagrange equations of any order, the "
        "complete boundary form and the natural boundary conditions; second: the second variation as a "
        "quadratic form. Numerical actions (evaluating the action, verifying by finite differences, "
        "finding stationary paths) are run by the USER from the card: you cannot run them.",
    )
    integrand: str = Field(
        max_length=MAX_INPUT,
        description="The integrand L of the action S = integral of L dt, as LaTeX, using the unknown "
        "functions and their derivatives: x, \\dot{x}, \\ddot{x}, \\dddot{x}, x^{(4)}; with another independent "
        "variable y the derivative of x is \\frac{dx}{dy} (or x'), \\frac{d^2x}{dy^2}; write no other "
        "d-notation. Example: "
        "\\frac{m}{2}\\dot{x}^2+\\frac{\\alpha}{2}\\ddot{x}^2-V(x). Ask the user for it; never invent one.",
    )
    variable: str = Field(default="t", max_length=32, description="The independent variable")
    functions: list[str] = Field(
        min_length=1,
        max_length=4,
        description='The unknown functions of the variable, e.g. ["x"] or ["x", "y"].',
    )
    given_functions: list[FunctionDecl] = Field(
        default_factory=list,
        max_length=8,
        description='Given (not unknown) functions used in the integrand, e.g. [{"name": "V", "args": '
        '["x"]}] for V(x). A function must be declared before it is written V(x).',
    )
    lower: str | None = Field(
        default=None,
        max_length=200,
        description="The lower bound, only if the user gave it (else a)",
    )
    upper: str | None = Field(
        default=None,
        max_length=200,
        description="The upper bound, only if the user gave it (else b)",
    )
    lower_term: str | None = Field(
        default=None,
        max_length=MAX_INPUT,
        description="A term of the action evaluated at the lower end, in the unknowns and their "
        "derivatives there, e.g. \\frac{k}{2}x^2. Only if the user stated it.",
    )
    upper_term: str | None = Field(
        default=None, max_length=MAX_INPUT, description="Same, at the upper end"
    )
    conditions: list[EndpointCondition] = Field(
        default_factory=list,
        max_length=64,
        description="Endpoint conditions the USER stated: kind fixed (with its value) or free, for a "
        "function, a derivative order and an end (lower or upper). Never default, infer or complete "
        "them: a derivative that is not listed is left undecided and its boundary term is kept.",
    )
    assumptions: dict[str, list[str]] = Field(
        default_factory=dict,
        description='Properties of symbols (e.g. {"alpha": ["positive"]}). Do not add any the user did not state.',
    )


def variation_input(args: VariationArgs, ctx: ToolContext) -> VariationInput:
    payload: dict[str, Any] = {**args.model_dump(), "format": "latex"}
    numeric = dict(ctx.numeric)
    if "action" in numeric:
        payload["action"] = numeric.pop("action")
    payload.update(numeric)
    return VariationInput(**payload)


class DistributionArgs(SimplifyArgs):
    latex: str = Field(
        max_length=MAX_INPUT,
        description="LaTeX expression with Dirac delta or Heaviside functions, as the user wrote it: "
        "\\delta(x-a), \\theta(x), or the explicit \\operatorname{DiracDelta}(x-a) and "
        "\\operatorname{Heaviside}(x). A user who writes Heaviside(...) or DiracDelta(...) by name has "
        "said what it is: send \\operatorname{Heaviside}(...) or \\operatorname{DiracDelta}(...), never "
        "\\theta or \\delta. \\delta(...) and \\theta(...) are read as distributions only after "
        "the USER confirms it on the card; do not rewrite them yourself. For integrate give only the "
        "integrand (no \\int, no dx).",
    )
    action: Literal["simplify", "integrate", "differentiate"] = Field(
        default="simplify",
        description="simplify: normal form sum c_i delta(x - x_i). integrate: needs `lower` and `upper` "
        "(numbers, oo, -oo). differentiate: the distributional derivative of steps and smooth terms.",
    )
    variable: str = Field(
        default="x",
        max_length=32,
        description="The real variable. Declare it real in `assumptions`; distributions need a real line.",
    )
    lower: str | None = Field(default=None, max_length=64, description="Lower limit (integrate)")
    upper: str | None = Field(default=None, max_length=64, description="Upper limit (integrate)")
    endpoint: Literal["half", "include", "exclude"] | None = Field(
        default=None,
        description="How a delta exactly on an integration limit counts. Set it ONLY if the USER chose "
        "one; never choose it yourself. Without it such an integral is reported as ambiguous.",
    )


class IntegrateArgs(SimplifyArgs):
    format: SkipJsonSchema[SyntaxChoice] = "latex"
    latex: str = Field(
        max_length=MAX_INPUT, description="The integrand, in LaTeX, e.g. x\\cos x (no \\int or dx)"
    )
    variable: str = Field(
        min_length=1, max_length=32, description="The variable of integration, e.g. x"
    )
    lower: str | None = Field(
        default=None,
        max_length=200,
        description="Lower bound of a definite integral in LaTeX: a real number such as 0, \\pi or "
        "\\frac{\\pi}{2} (never pi/2), or -\\infty for minus infinity. Give both bounds or neither.",
    )
    upper: str | None = Field(
        default=None,
        max_length=200,
        description="Upper bound of a definite integral in LaTeX: a real number such as \\frac{\\pi}{2}, or "
        "\\infty for infinity.",
    )
    strategy: StrategyName | None = Field(
        default=None,
        description="Optional. Leave it out first. After a failed or unexplained result, the "
        "`strategy` block of the card lists the techniques that apply (suggestions): call again "
        "with one of them (substitution, parts, trigonometric, partial_fractions, parity, "
        "reflection, frullani, schwinger, parameter_differentiation, numerical), or `auto` to get "
        "a result explained by the technique that finds it. Never invent a technique.",
    )


class IdentitiesArgs(SimplifyArgs):
    format: SkipJsonSchema[SyntaxChoice] = "latex"
    query: str = Field(default="", max_length=200, description="Optional extra search words")


class TensorArgs(BaseModel):
    expression: str = Field(
        max_length=MAX_INPUT,
        description="In text syntax (not LaTeX), e.g. S(a,b)*A(-a,-b); a leading - lowers an index",
    )
    tensors: dict[str, TensorSpec] = Field(description="Rank and symmetry of every tensor used")
    dimension: int = Field(default=4, ge=1, le=64)


class SolveArgs(BaseModel):
    equation: str = Field(
        max_length=MAX_INPUT,
        description="A polynomial equation in LaTeX with one = sign, e.g. x^3+x^2+x+1=0",
    )
    variable: str | None = Field(
        default=None, max_length=32, description="The unknown, if the equation has several symbols"
    )
    assumptions: dict[str, list[str]] = Field(default_factory=dict)


class LagrangianArgs(BaseModel):
    lagrangian: str = Field(
        max_length=MAX_INPUT,
        description=r"L(q, qdot, t) in LaTeX, using \dot{q} for velocities, e.g. \frac12 m\dot{x}^2 - \frac12 k x^2",
    )
    coordinates: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Names of the generalised coordinates, e.g. ['r', 'theta']",
    )
    time: str = Field(
        default="t",
        max_length=16,
        description="Name of the time symbol; not a coordinate's name (with a coordinate `t`, use `s`).",
    )
    assumptions: dict[str, list[str]] = Field(default_factory=dict)


class AnalyzeLagrangianArgs(LagrangianArgs):
    format: SyntaxChoice = Field(
        default="auto",
        description="Syntax of `lagrangian`: LaTeX (\\dot{\\theta}) or text (theta_dot, sin(theta)*phi_dot^2). "
        + FORMAT_DESCRIPTION,
    )


class TimeDerivativeArgs(SimplifyArgs):
    format: SkipJsonSchema[SyntaxChoice] = "latex"
    latex: str = Field(
        max_length=MAX_INPUT,
        description=r"The quantity in LaTeX, using q for coordinates and \dot{q} for velocities, e.g. "
        r"\mu r^2 \dot{\theta}. Not an equation: send only the expression.",
    )
    coordinates: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Names of the coordinates that change along the motion, e.g. ['r', 'theta']. "
        "Every velocity in the expression must belong to one of them.",
    )
    lagrangian: str | None = Field(
        default=None,
        max_length=MAX_INPUT,
        description=r"Optional L(q, qdot, t) of the system in LaTeX, as an expression (not 'L = ...'). Given, the "
        "derivative is also evaluated on that system's equations of motion and says whether it vanishes "
        "there (is constant along its motions).",
    )
    time: str = Field(
        default="t",
        max_length=16,
        description="Name of the time symbol; not a coordinate's name (with a coordinate `t`, use `s`).",
    )


class ExpandMotionArgs(LagrangianArgs):
    equilibrium: dict[str, str] = Field(
        default_factory=dict,
        description="Equilibrium value of each coordinate as text (numbers, pi, or the Lagrangian's "
        'parameters), e.g. {"theta": "pi"} or {"x": "m*g/k"}. Coordinates left out are 0. Only an '
        "equilibrium is accepted; Minerva checks that it is one.",
    )
    order: int = Field(
        default=3,
        ge=1,
        le=12,
        description="Highest degree in the deviations kept in the accelerations, 1 to 12. "
        "1 gives the linearised equations.",
    )


class PerturbationArgs(LagrangianArgs):
    small_parameter: str | None = Field(
        default=None,
        max_length=32,
        description="Name of a parameter of L that is small, e.g. the coefficient of a quartic term "
        "(the system must be a linear oscillator when it is 0). Leave empty to use the amplitude of the "
        "initial displacement as the small parameter, which suits the pendulum and any smooth well.",
    )
    equilibrium: str = Field(
        default="0",
        max_length=64,
        description="Equilibrium value of the single coordinate (number, pi, or the parameters of L).",
    )
    order: int = Field(
        default=2,
        ge=1,
        le=6,
        description="Highest power of the small parameter kept, 1 to 6.",
    )
    compare: CompareRequest | None = Field(
        default=None,
        description="Optional numerical comparison. Only with numbers the USER gave you: values of every "
        "parameter of L, the small parameter or amplitude, and an end time. Never choose them yourself.",
    )


class InvariantArgs(BaseModel):
    lagrangian: str = Field(
        max_length=MAX_INPUT,
        description=r"L(q, qdot) in LaTeX, using \dot{q} for velocities. The same Lagrangian as for analyze_lagrangian.",
    )
    format: SyntaxChoice = Field(
        default="auto",
        description="Syntax of `lagrangian`: LaTeX or text (theta_dot). " + FORMAT_DESCRIPTION,
    )
    coordinates: list[str] = Field(
        min_length=1,
        max_length=3,
        description="Names of the generalised coordinates (at most 3), e.g. ['r', 'theta']",
    )
    angles: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="The coordinates that are angles (periodic), e.g. ['theta']. Only those are searched "
        "with cos and sin; every other coordinate with powers. Say so to the user if you chose them.",
    )
    degree: int = Field(
        default=2, ge=1, le=2, description="Highest degree in the velocities that is searched"
    )
    power_bound: int = Field(
        default=3, ge=0, le=4, description="Powers q^a with |a| at most this are searched"
    )
    harmonics: int = Field(
        default=1, ge=1, le=2, description="Highest harmonic of an angle that is searched"
    )
    time: str = Field(
        default="t",
        max_length=16,
        description="Name of the time symbol; not a coordinate's name (with a coordinate `t`, use `s`).",
    )
    assumptions: dict[str, list[str]] = Field(default_factory=dict)


class DiscoveryArgs(BaseModel):
    lagrangian: str | None = Field(
        default=None,
        max_length=MAX_INPUT,
        description=r"L(q, qdot), using \dot{q} for velocities in LaTeX and q_dot in text syntax (see `format`). "
        "Give exactly one of `lagrangian` and `hamiltonian`.",
    )
    hamiltonian: str | None = Field(
        default=None,
        max_length=MAX_INPUT,
        description="H(q, p), with the canonical momenta as symbols such as p_x or p_\\theta (p_theta in text "
        "syntax), in the syntax `format` names.",
    )
    format: SyntaxChoice = Field(
        default="auto",
        description="Syntax of `lagrangian` or `hamiltonian`: LaTeX or text (theta_dot). "
        + FORMAT_DESCRIPTION,
    )
    coordinates: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Names of the generalised coordinates (the search supports at most 3; more is refused "
        "with the reason and nothing is searched), e.g. ['r', 'theta']",
    )
    momenta: list[str] | None = Field(
        default=None,
        max_length=8,
        description="Hamiltonian only: the momentum symbols in the order of the coordinates. Default p_<coordinate>.",
    )
    angles: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Optional: coordinates known to be angles. Not needed: angles are detected (a coordinate "
        "that appears only through cos and sin, or that does not appear in the equations at all).",
    )
    budget: Literal["quick", "standard", "thorough"] = Field(
        default="standard",
        description="How far the search may go. Use the default unless the user asks for more or less.",
    )
    degree: int | None = Field(
        default=None,
        ge=1,
        le=4,
        description="Advanced, only on request: pin one family (momentum degree)",
    )
    power_bound: int | None = Field(
        default=None, ge=0, le=4, description="Advanced, only on request: pin one family (powers)"
    )
    harmonics: int | None = Field(
        default=None,
        ge=1,
        le=2,
        description="Advanced, only on request: pin one family (harmonics)",
    )
    time_degree: int | None = Field(
        default=None,
        ge=0,
        le=2,
        description="Advanced, only on request: pin one family (powers of t)",
    )
    time: str = Field(
        default="t",
        max_length=16,
        description="Name of the time symbol; not a coordinate's name (with a coordinate `t`, use `s`).",
    )
    assumptions: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_function(self) -> DiscoveryArgs:
        if (self.lagrangian is None) == (self.hamiltonian is None):
            raise ValueError("give exactly one of lagrangian and hamiltonian")
        return self


class MotionArgs(MotionRequest):
    """Arguments of integrate_motion: the engine's own request model (also the JSON schema)."""


class DimensionalArgs(BaseModel):
    quantities: list[QuantitySpec] = Field(
        min_length=1,
        max_length=16,
        description="Every quantity of the problem. Each has a name (letters, digits, underscore: "
        "theta, rho, x_0), exactly one of `unit` (SI base or derived units: kg, m, s, A, K, mol, cd, N, J, "
        "W, Pa, Hz, C, V, F, ohm, S, Wb, T, H, rad; e.g. 'kg m/s^2') or `kind` (a named quantity such as "
        "'velocity', 'density', 'dynamic viscosity', 'surface tension'), and optionally `meaning`.",
    )
    unknown: str | None = Field(
        default=None,
        max_length=32,
        description="Name of the quantity to find, if there is one. Enables the missing-quantity check.",
    )
    repeating: list[str] = Field(
        default_factory=list,
        max_length=16,
        description="Optional: quantities to use as repeating variables (e.g. rho, v, D for pipe flow).",
    )


class UnitsCheckArgs(BaseModel):
    equation: str = Field(
        max_length=MAX_INPUT,
        description="An equation in LaTeX, e.g. F = m a. Symbol names must match `quantities` "
        "(Greek letters by name: \\theta is `theta`). E is reserved: rename it (W, E_n).",
    )
    quantities: list[QuantitySpec] = Field(
        min_length=1,
        max_length=16,
        description="Unit or kind of every symbol in the equation, except the ones in `unknown`. "
        "Never give a quantity the user is asking about an entry here. A unit or kind you were not "
        "told is an assumption: the card says so. `x_dot` and `x_ddot` need no entry when `x` has one.",
    )
    unknown: list[str] = Field(
        default_factory=list,
        max_length=4,
        description='Names whose unit or dimension is wanted ("what are the units of C_3?"): put the '
        "equation that defines it in `equation` and its name here, with no entry in `quantities`. The "
        "dimension is found from the equation and the other symbols' units; if it is not fixed, the "
        "result says which stay free.",
    )


class SearchArgs(BaseModel):
    query: str = Field(max_length=200, description="Words describing the formula or method")
    chapter: str | None = Field(default=None, max_length=64)
    kind: str | None = Field(default=None, max_length=32)
    limit: int = Field(default=6, ge=1, le=10)


class EntryArgs(BaseModel):
    id: str = Field(max_length=80, description="Handbook entry id, e.g. trig.pythagorean")


class LoadToolsArgs(BaseModel):
    groups: list[str] = Field(
        min_length=1,
        max_length=11,
        description="Names of the tool groups to load, from the list in the system prompt "
        '(for example ["matrix"]). Their tools are available from the next step.',
    )


def _as_read(latex: str | None) -> str:
    """`input_as_read=$...$; ` once the input was read, nothing when it could not be (never `$None$`)."""
    return f"input_as_read=${latex}$; " if latex else ""


def _expr(args: SimplifyArgs) -> ExpressionInput:
    """The formula as the tool reads it: LaTeX unless the tool takes a `format` and it says text."""
    syntax = getattr(args, "format", "latex")
    return ExpressionInput(
        input=args.latex,
        format="text" if syntax == "text" else "latex",
        assumptions=args.assumptions,
    )


ODE_SOLVERS = frozenset({"ode_solve_symbolic", "ode_solve_numerical"})
NEEDS_CHOICE = (
    "The user has not said which method to use and several apply. Do not run one: list the applicable "
    "methods from the analysis, in its order, with what each needs and its limits, and ask which one to "
    "run. Stop calling tools."
)


def _error(
    tool: str, message: str, category: str = "tool_failure", arguments_valid: bool = True
) -> ToolOutput:
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool=tool,
        type="error",
        title="Could not run",
        data={"message": message},
    )
    return ToolOutput(
        card=card,
        for_model=f"error: {message}",
        status="error",
        evidence=Evidence(check="none", accepted=False, reason="error"),
        arguments_valid=arguments_valid,
        error_category=category,
    )


# --- tool implementations ----------------------------------------------------------------------
MAX_STEPS_FOR_MODEL = 6
LONG_RESULT = 800  # characters of LaTeX; a longer total derivative stays on the card
# What a result id of `time_derivative` names. The conclusion on the equations of motion has no id (a plain 0
# would drop the system it was decided on), so the model is told, never left to guess.
TOTAL_DERIVATIVE_HANDLE_NOTE = (
    "reuse: the result id below names the total derivative as computed, before any equations of motion are "
    "used; the conclusion on the equations of motion is a separate statement about the Lagrangian given and "
    "has no result id, so it cannot be reused by @rN or @eN (state it in words); "
)


NOT_APPLIED = "a handbook identity that was not applied: its conditions are not established"
NOT_VERIFIED = "a candidate that was not verified"


def _chain_lines(result: TransformResponse) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """What the model may quote about a chain: each step's identity, forms and evidence.

    Returns the lines, the formulas the verified chain produced, and the formulas that are only quoted
    (statements of identities that were not applied).
    """
    lines: list[str] = []
    known: list[str] = []
    quoted: list[tuple[str, str]] = []
    for i, step in enumerate(result.steps[:MAX_STEPS_FOR_MODEL], start=1):
        v = step.verification
        evidence = (
            f"checked numerically at {v.numeric_points_agreed}/{v.numeric_points_checked} "
            f"points; symbolic_zero={v.symbolic_zero}{_probe_cut_note(v)}"
        )
        entry = result.entries.get(step.name)
        if step.kind == "rule" and entry is not None:
            how = f"{step.name} ({entry.title}, read {step.direction}); statement=${entry.statement_latex}$"
            known.append(entry.statement_latex)
        else:
            how = f"{step.name} (a SymPy move, not a handbook identity)"
        lines.append(
            f"- step {i}: {how}; ${step.before_latex}$ -> ${step.after_latex}$; {evidence}"
        )
        known += [step.before_latex, step.after_latex]
    for b in result.blocked:
        entry = result.entries.get(b.rule_id)
        needs = ", ".join(f"{c.variable} {c.assumption}" for c in b.unmet_conditions)
        lines.append(
            f"- NOT APPLIED, conditions not provable from the declared assumptions: {b.rule_id} "
            f"matches ${b.matched_latex}$ but needs: {needs}. Ask whether they hold; this is not a result"
            + (f"; statement=${entry.statement_latex}$" if entry else "")
        )
        if entry:
            quoted.append((entry.statement_latex, NOT_APPLIED))
    for alt in result.alternatives:
        lines.append(f"- other form: ${alt.result_latex}$ ({len(alt.steps)} steps)")
        known.append(alt.result_latex)
    return lines, known, quoted


def run_simplify(args: SimplifyToolArgs, ctx: ToolContext) -> ToolOutput:
    result = simplify_expression(ctx.kb, ctx.engine, _expr(args), args.goal, args.target)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="simplify",
        type="simplify",
        title="Simplify",
        data={
            **result.model_dump(mode="json"),
            "original_request": args.model_dump(mode="json"),
        },
    )
    v = result.verification
    head = (
        f"status={result.status.value}; goal={args.goal}; {_as_read(result.input_latex)}"
        + (
            f"result=${result.result_latex}$; "
            if result.status.value == "ok"
            else "result=none (the input is unchanged: no rewrite was applied, so there is no "
            "simplified form to report; this is the answer, so do not try the same input with another "
            "tool to get one); "
        )
        + (
            f"end to end: numeric {v.numeric_points_agreed}/{v.numeric_points_checked}, "
            f"symbolic_zero={v.symbolic_zero} "
            f"({'symbolic and numeric evidence' if v.symbolic_zero else 'numeric evidence only, not a proof'})"
            f"{_probe_cut_note(v)}; "
            + (
                "valid only where " + ", ".join(f"${r}$" for r in result.restrictions_latex) + "; "
                if result.restrictions_latex
                else ""
            )
            if v
            else ""
        )
        + result.message
    )
    lines, known_steps, quoted = _chain_lines(result)
    known = [x for x in (result.input_latex, result.result_latex) if x] + known_steps
    if result.status.value == "ok":
        # Where the result holds is the engine's own statement, like the restrictions of the other tools:
        # a reply that repeats it is not an invention.
        known += result.restrictions_latex
    return ToolOutput(
        card, "\n".join([head, *lines]), known, result.status.value, quoted_latex=quoted
    )


def run_check_identity(args: CheckIdentityArgs, ctx: ToolContext) -> ToolOutput:
    """Is the user's one equation an identity? The engine's verdict, with the point where it fails."""
    result = ctx.engine.check_identity(args.latex, args.assumptions)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="check_identity",
        type="identity",
        title="Is it an identity?",
        data={
            **result.model_dump(mode="json"),
            "original_request": args.model_dump(mode="json"),
        },
    )
    parts = [f"status={result.status.value}"]
    if result.verdict:
        parts.append(f"verdict={result.verdict}")
    if result.lhs_latex and result.rhs_latex:
        parts.append(f"equation as read: ${result.lhs_latex} = {result.rhs_latex}$")
    known: list[str] = []
    v = result.verification
    if result.verdict in ("proved_equal", "numerically_equal"):
        known.append(f"{result.lhs_latex} = {result.rhs_latex}")
        if v is not None:
            parts.append(
                f"numeric {v.numeric_points_agreed}/{v.numeric_points_checked}, "
                f"symbolic_zero={v.symbolic_zero} "
                f"({'symbolic and numeric evidence' if v.symbolic_zero else 'numeric evidence only, not a proof'})"
            )
    if v is not None and v.probes_cut_short:
        parts.append(_probe_cut_note(v).removeprefix("; "))
    if result.condition:
        parts.append(f"condition: {result.condition}")
    cx = result.counterexample
    if cx is not None:
        point = ", ".join(f"${name} = {value}$" for name, value in cx.substitution.items())
        parts.append(
            f"counterexample at {point}: left side ${cx.lhs_value}$, right side ${cx.rhs_value}$ "
            f"(exact point, confirmed at 60 and 120 digits; found by a {cx.probe} probe)"
        )
        known += [f"{name} = {value}" for name, value in cx.substitution.items()]
        known += [cx.lhs_value, cx.rhs_value]
    text = "; ".join(parts) + "; " + result.message
    return ToolOutput(card, text, known, result.status.value)


def run_differentiate(args: DifferentiateArgs, ctx: ToolContext) -> ToolOutput:
    result = ctx.engine.differentiate(_expr(args), args.variables)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="differentiate",
        type="differentiate",
        title="Differentiate",
        data={
            **result.model_dump(mode="json"),
            "original_request": args.model_dump(mode="json"),
        },
    )
    v = result.verification
    text = (
        f"status={result.status.value}; {_as_read(result.input_latex)}"
        + (f"derivative=${result.result_latex}$; " if result.result_latex else "")
        + (
            f"numerically checked: {v.steps_checked} step(s), numeric {v.points_agreed}/{v.points_checked} "
            f"({'+'.join(v.directions)} direction); "
            if v
            else ""
        )
        + result.message
        + "".join(f" note: {n}" for n in result.notes)
    )
    known = [
        x
        for x in (
            result.input_latex,
            result.notation_latex,
            result.result_latex,
            *(s.latex for s in result.steps),
        )
        if x
    ]
    return ToolOutput(card, text, known, result.status.value)


def _constants_refusal(refusal: Refusal) -> ToolOutput:
    out = _error("time_derivative", refusal.message, "unsupported")
    diagnostic = InputDiagnostic(
        code=refusal.code, field="latex", symbols=list(refusal.symbols), recoverable=False
    )
    out.card.data["diagnostic"] = diagnostic.model_dump(mode="json")
    out.reason_code = refusal.code.value
    return out


def reference_refusal(tool: str, message: str, diagnostic: InputDiagnostic) -> ToolOutput:
    """The refusal of an `@eN` of the wrong kind for a field: typed, never run, never retried unchanged.

    The model sees the message and the `diagnostic:` line; the card carries the diagnostic; the log gets the code.
    """
    out = _error(tool, message, "invalid_arguments", arguments_valid=False)
    out.card.data["diagnostic"] = diagnostic.model_dump(mode="json")
    line = diagnostic_line("parse_error", diagnostic)
    if line is not None:
        out.for_model = f"{out.for_model}\n{line}"
    out.reason_code = diagnostic.code.value
    return out


def run_time_derivative(args: TimeDerivativeArgs, ctx: ToolContext) -> ToolOutput:
    expansion = expand_constants(args.latex, ctx.constants)
    if isinstance(expansion, Refusal):
        return _constants_refusal(expansion)
    # The system of the constants is the one whose equations of motion the zero is decided on, unless the
    # call names its own Lagrangian.
    lagrangian = args.lagrangian or (expansion.function if expansion else None)
    result = ctx.engine.time_derivative(
        _expr(args),
        args.coordinates,
        args.time,
        lagrangian,
        expansion.definitions if expansion else None,
    )
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="time_derivative",
        type="differentiate",
        title="Total time derivative",
        data=result.model_dump(mode="json"),
    )
    v = result.verification
    text = (
        f"status={result.status.value}; {_as_read(result.input_latex)}"
        + (
            "constants replaced by their verified expressions from the earlier result: "
            + "; ".join(f"${d}$" for d in result.definitions_used)
            + "; "
            if result.definitions_used
            else ""
        )
        + (
            f"total time derivative=${result.result_latex}$; "
            if result.result_latex and len(result.result_latex) <= LONG_RESULT
            else (
                "total time derivative: a long expression, on the card and not repeated here "
                "(reuse it exactly by its result id, given below, never by retyping it); "
                if result.result_latex
                else ""
            )
        )
        + (
            TOTAL_DERIVATIVE_HANDLE_NOTE
            if result.result_latex and result.status.value == "ok"
            else ""
        )
        + (
            f"numerically checked along random motions: {v.points_agreed}/{v.points_checked} points; "
            if v
            else ""
        )
        + (
            "zero on the equations of motion of the Lagrangian given; "
            if result.zero_on_motion is True
            else (
                f"NOT zero on the equations of motion: ${result.on_motion_latex}$; "
                if result.zero_on_motion is False
                else ""
            )
        )
        + result.message
        + "".join(f" note: {n}" for n in result.notes)
    )
    known = [x for x in (result.input_latex, result.notation_latex, result.result_latex) if x]
    quoted: list[tuple[str, str]] = []
    if result.zero_on_motion is True and result.on_motion_latex:
        known.append(result.on_motion_latex)
    elif result.zero_on_motion is False and result.on_motion_latex:
        quoted.append(
            (
                result.on_motion_latex,
                "an exact substitution of the verified accelerations, not separately sampled",
            )
        )
    return ToolOutput(card, text, known, result.status.value, quoted_latex=quoted)


def run_series(args: SeriesArgs, ctx: ToolContext) -> ToolOutput:
    result = ctx.engine.series(_expr(args), args.variable, args.point, args.order)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="series",
        type="series",
        title="Series",
        data={
            **result.model_dump(mode="json"),
            "original_request": args.model_dump(mode="json"),
        },
    )
    v = result.verification
    text = (
        f"status={result.status.value}; {_as_read(result.input_latex)}"
        + (
            f"about {result.variable}=${result.point_latex}$, order {result.order}; "
            if result.point_latex
            else ""
        )
        + (
            f"series=${result.result_latex} + {result.remainder_latex}$; "
            if result.result_latex and result.remainder_latex
            else (f"series=${result.result_latex}$; " if result.result_latex else "")
        )
        + (
            f"numerically checked: coefficients (Cauchy integral) {'agree' if v.coefficients_passed else 'DISAGREE'}, "
            f"remainder order {'confirmed' if v.remainder_passed else 'NOT confirmed'}, "
            f"{v.parameter_sets_checked} parameter set(s); "
            if v
            else ""
        )
        + result.message
        + "".join(f" note: {n}" for n in result.notes)
        + (
            " Measured truncation error (not a bound): "
            + ", ".join(
                f"{m.absolute_error} at distance {m.distance}" for m in result.measured_errors
            )
            if result.measured_errors
            else ""
        )
    )
    known = [x for x in (result.input_latex, result.result_latex) if x]
    known += [t.coefficient_latex for t in result.terms]
    if result.result_latex and result.remainder_latex:
        known.append(
            f"{result.result_latex} + {result.remainder_latex}"
        )  # printed in the text above
    return ToolOutput(card, text, known, result.status.value, series=_series_record(result, args))


def _series_record(result: SeriesResult, args: SeriesArgs) -> dict[str, Any] | None:
    """The verified series as a typed record, or None unless the engine's checks passed."""
    if (
        result.status.value != "ok"
        or result.verification is None
        or not result.verification.passed
        or not result.input_latex
        or not result.variable
        or result.point_latex is None
        or not result.result_latex
    ):
        return None
    return SeriesEvidence(
        expression_latex=result.input_latex,
        variable=result.variable,
        centre_latex=result.point_latex,
        order=result.order,
        terms=[
            SeriesTermEvidence(power=t.power, coefficient_latex=t.coefficient_latex)
            for t in result.terms
        ],
        distances=[m.distance for m in result.measured_errors],
        assumptions={k: list(v) for k, v in args.assumptions.items()},
    ).model_dump()


def run_integrate(args: IntegrateArgs, ctx: ToolContext) -> ToolOutput:
    result = ctx.engine.integrate(_expr(args), args.variable, args.lower, args.upper, args.strategy)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="integrate",
        type="integrate",
        title="Integrate",
        data={
            **result.model_dump(mode="json"),
            "original_request": args.model_dump(mode="json"),
        },
    )
    v = result.verification
    text = (
        f"status={result.status.value}; {_as_read(result.input_latex)}"
        + (
            f"{'value' if result.definite else 'antiderivative'}=${result.result_latex}$; "
            if result.result_latex
            else ""
        )
        + (f"approx={result.numeric}; " if result.numeric else "")
        + (
            f"numerically checked: numeric derivative {v.points_agreed}/{v.points_checked} "
            f"({'+'.join(v.directions)} direction, {v.branches_checked} case(s)); "
            if v and v.passed
            else ""
        )
        + (
            f"numerically checked: quadrature {q.points_agreed}/{q.points_checked} points; "
            if (q := result.quadrature) and q.passed
            else ""
        )
        + "".join(f"condition: ${c}$; " for c in result.conditions_latex)
        + result.message
        + "".join(f" note: {n}" for n in result.notes)
        + _strategy_text(result)
    )
    known = [
        x for x in (result.input_latex, result.notation_latex, result.result_latex) if x
    ] + result.conditions_latex
    if (
        result.status.value == "ok"
        and result.definite
        and result.result_kind == "closed_form"
        and result.notation_latex
        and result.result_latex
    ):
        # The equation the tool verified (quadrature of this integral against this value), written out. The
        # integral alone is the user's input and stays quoted; the value alone is not a statement about it.
        known.append(f"{result.notation_latex} = {result.result_latex}")
    strategy = result.strategy
    if strategy is not None and result.status.value == "ok" and strategy.technique != "direct":
        known += [step.latex for step in strategy.steps]  # intermediate forms the engine printed
    return ToolOutput(card, text, known, result.status.value)


def _strategy_text(result: IntegrateResult) -> str:
    """What the model may say about the technique: only what the engine reported."""
    s = result.strategy
    if s is None:
        return ""
    lines: list[str] = []
    if result.result_kind == "numeric" and result.numeric:
        lines.append(
            f" NUMERICAL ESTIMATE ONLY: {result.numeric} (estimated error {result.numeric_error}). "
            "It is not a formula; say so, and never present it as a closed form."
        )
    if s.technique and s.technique != "direct":
        lines.append(
            f" technique={s.technique} (why: {s.why}); verification={s.verification}"
            " (the result was verified independently of the technique)."
        )
        for step in s.steps:
            lines.append(f" step [{step.what}]: ${step.latex}$")
    for c in s.conditions_latex:
        lines.append(f" established: {c}")
    if s.suggestions and result.status.value != "ok":
        names = ", ".join(x.technique for x in s.suggestions)
        lines.append(
            f" suggestions: {names}. You may call integrate again with `strategy` set to one of them."
        )
    for a in s.attempts:
        if a.outcome in ("blocked", "failed_verification"):
            lines.append(
                f" tried {a.technique}: {a.outcome}{' (' + a.detail + ')' if a.detail else ''}; "
                "nothing from it is shown."
            )
    if s.verification == "numerically_checked":
        lines.append(" Say 'checked numerically' for a definite value, never 'proved'.")
    return "".join(lines)


def geometry_request(args: GeometryArgs) -> GeometryInput:
    """The engine request; `format: auto` picks one strict parser for the whole metric as written (never rewritten).

    Any backslash anywhere makes it LaTeX; text markers (`*`, `sin(`) with no backslash make it text, so a metric
    sent as `2*M/r` and `sin(theta)^2` is read as text instead of failing three times as LaTeX.
    """
    cells = [str(c) for row in args.metric for c in row]
    if args.tensor is not None:
        cells += [str(c) for c in args.tensor.components]
    syntax = resolve_syntax(" ".join(cells), args.format)
    return GeometryInput(**args.model_dump(exclude={"format"}), format=syntax)


def run_geometry(args: GeometryArgs, ctx: ToolContext) -> ToolOutput:
    request = geometry_request(args)
    result = ctx.engine.geometry(request)
    log_geometry(result)  # stage names and counters only
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="geometry",
        type="geometry",
        title="Differential geometry",
        data={
            "result": result.model_dump(mode="json"),
            # the whole problem, so that the card's buttons can ask for more about the same geometry
            "request": request.model_dump(mode="json"),
        },
    )
    head = f"status={result.status.value}; dimension={result.dimension}; "
    known: list[str] = []
    if result.metric_latex:
        head += f"metric=${result.metric_latex}$; "
        known.append(result.metric_latex)
    if result.signature:
        head += f"signature={result.signature}; "
    elif result.status.value == "ok":
        head += "signature=NOT DETERMINED (do not call the metric Lorentzian or Riemannian); "
    lines = [head + result.message]
    if result.status.value == "ok" or result.quantities:
        lines.extend(f"- convention: {c}" for c in result.conventions)
        lines.extend(f"- assumption: {a}" for a in result.assumptions_used)
        if result.restrictions_latex:
            lines.append(
                "- the coordinate expressions are valid only where "
                + ", ".join(f"${r}$" for r in result.restrictions_latex)
                + (
                    ""
                    if result.restrictions_complete
                    else " (and possibly elsewhere: root, logarithm and branch domains were not analysed)"
                )
                + ". This says where the coordinates break down, not whether the geometry is singular."
            )
            known.extend(result.restrictions_latex)
        elif not result.restrictions_complete:
            lines.append("- domain restrictions could not be fully determined")
        if result.signature is None and result.signature_note:
            lines.append(f"- signature: {result.signature_note}")
    for q in result.quantities:
        evidence = ""
        v = q.verification
        if v is not None and q.status.value == "ok":
            numeric = [c for c in v.checks if c.kind == "numeric"]
            points = min((c.points_agreed for c in numeric), default=0)
            evidence = f"; numerically checked ({len(numeric)} independent checks, at least {points} points each)"
            if v.symbolic_zero is True:
                evidence += "; also proved by a symbolic zero test"
            else:
                evidence += "; NO symbolic proof (say 'numerically checked', never 'proved')"
        lines.append(
            f"- {q.kind}: status={q.status.value}{evidence}"
            + (f"; {q.message}" if q.message else "")
        )
        if q.status.value != "ok":
            continue
        if q.scalar_latex is not None:
            lines.append(f"  value=${q.scalar_latex}$")
            known.append(q.scalar_latex)
        elif q.all_zero and q.kind != "geodesics":
            lines.append("  every component is zero")
        for c in q.components[:40]:
            statement = c.latex if q.kind == "geodesics" else f"{c.label_latex} = {c.latex}"
            lines.append(f"  ${statement}$")
            known.extend([statement, c.latex.split(" = ")[-1]])
        if len(q.components) > 40:
            lines.append(f"  ... and {len(q.components) - 40} more components (see the card)")
        if q.symmetry_note and q.kind != "geodesics":
            lines.append(f"  ({q.symmetry_note})")
        lines.extend(f"  note: {n}" for n in q.notes)
    slow = [d for d in result.diagnostics if d.status == "failed"]
    if slow:
        lines.append("- failed stage(s): " + ", ".join(f"{d.stage} ({d.failure})" for d in slow))
    lines.append(
        "to compute more about THIS geometry, call geometry again with the same coordinates, metric, "
        "assumptions and functions."
    )
    return ToolOutput(card, "\n".join(lines), known, result.status.value)


def run_vector_calculus(args: VectorCalculusArgs, ctx: ToolContext) -> ToolOutput:
    fields = args.model_dump(exclude={"premises", "premise_ids"})
    premises = [Premise(equation=p) for p in args.premises]
    try:
        premises += [
            Premise(equation=o.equation, label=o.title)
            for o in resolve_premise_ids(ctx.kb, args.premise_ids)
        ]
    except KeyError as exc:
        return _error(
            "vector_calculus",
            f"{exc.args[0]} is not an adoptable handbook field equation (see electromagnetism setup)",
            "invalid_arguments",
            arguments_valid=False,
        )
    request = VectorInput(**fields, premises=premises, format="latex")
    if args.action in ("identities", "apply_identity", "derive"):
        # The handbook proposes; the engine matches the identities and checks every rewrite.
        request = request.model_copy(update={"rules": vector_identity_rules(ctx.kb)})
    return _vector_output(ctx.engine.vector_calculus(request), request)


def _singular_lines(result: VectorResult, lines: list[str], known: list[str]) -> None:
    """The distributional result: classical and distributional statements apart, with their evidence."""
    s = result.singular
    if s is None:
        return
    lines.append(
        f"- CLASSICAL result (pointwise, away from the source): ${result.input_latex} = {s.classical_latex}$ "
        f"valid where ${s.classical_domain_latex}$"
    )
    lines.append(
        f"- DISTRIBUTIONAL result (as a distribution on R^3, equality as distributions, never pointwise): "
        f"${s.distributional_latex}$; strength ${s.strength_latex}$, source ${s.source_latex}$"
    )
    lines.append(f"- proof status: {s.proof_status} (this is NOT called proved); scope: {s.scope}")
    for step in s.derivation:
        lines.append(f"  - {step.status}: {step.title}: ${step.latex}$ ({step.note})")
        if step.status == "established":
            known.append(step.latex)
    agreed = sum(w.passed for w in s.weak_checks)
    lines.append(
        f"- independent pairings with test functions: {agreed}/{len(s.weak_checks)} agree (values at the "
        f"source differ, including 0); flux through {len(s.flux_checks)} spheres agrees. These are supporting "
        "evidence, not a proof: say so"
    )
    if s.regularisation:
        shown = ", ".join(f"eps={r.eps:g}: error {r.error:.3g}" for r in s.regularisation)
        lines.append(
            f"- regularised field r/(r^2+eps^2)^(3/2): pairing error shrinks as eps does ({shown}); "
            "supporting evidence only"
        )
    if s.claim is not None:
        verdict = "HOLDS" if s.claim.holds else f"does NOT hold ({s.claim.reason})"
        lines.append(
            f"- the user's statement ${s.claim.latex}$ {verdict}: {s.claim.note}. State this plainly"
        )
    known.extend(
        [s.distributional_latex, s.strength_latex, f"{result.input_latex} = {s.classical_latex}"]
    )


def _vector_output(result: VectorResult, request: VectorInput) -> ToolOutput:
    log_vector(result)  # status, reason, counters: never a field name or an expression
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="vector_calculus",
        type="vector",
        title="Vector calculus",
        data={
            "result": result.model_dump(mode="json"),
            # the whole problem (without the handbook rules), so a follow-up can resend it
            "request": request.model_dump(mode="json", exclude={"rules"}),
        },
    )
    ok = result.status.value == "ok"
    head = f"status={result.status.value}; coordinates={result.coordinates}; "
    known: list[str] = []
    lines: list[str] = []
    if ok:
        if result.result_kind in ("scalar", "vector") and result.result_latex:
            head += f"result=${result.result_latex}$; "
            known.append(result.result_latex)
            for label, comp in zip(result.basis_latex, result.components_latex, strict=False):
                lines.append(f"- component along ${label}$: ${comp}$")
                known.append(comp)
        elif result.result_kind == "matrix":
            head += f"jacobian=${result.result_latex}$; determinant=${result.determinant_latex}$; "
            known.extend(x for x in (result.result_latex, result.determinant_latex) if x)
            known.extend(result.scale_factors_latex)
        for i, step in enumerate(result.steps, start=1):
            kind = (
                "CONDITIONAL on " + "; ".join(step.given)
                if step.evidence == "conditional"
                else "identity"
            )
            lines.append(
                f"- step {i} ({kind}): {step.title}: ${step.before_latex}$ -> ${step.after_latex}$"
            )
            if step.evidence == "identity" and step.needs:
                lines.append(
                    f"  (holds for fields that are {', '.join(step.needs)}; the USER declared that)"
                )
        if result.evidence == "conditional":
            lines.append(
                "- CONDITIONAL: this holds only given "
                + "; ".join(f"${g}$" for g in result.given)
                + ". The user supplied it and Minerva does not check it: never call the result an "
                "identity, a proof, or a consequence of Maxwell's equations unless that is what the "
                "premises were, and say which premises it used."
            )
        elif result.evidence == "identity":
            lines.append(
                "- every identity step was checked by comparing the whole expressions before and after "
                "(exact component difference zero, and an independent Cartesian evaluation agreeing)"
            )
        if result.integral is not None:
            integral = result.integral
            for iv in integral.values:
                what = "exact" if iv.exact else "NUMERICAL APPROXIMATION (no closed form found)"
                shown = iv.latex if iv.latex is not None else (iv.numeric or "?")
                lines.append(f"- {what}: ${iv.label} = {shown}$")
                if iv.latex is not None:
                    known.append(iv.latex)
                if iv.numeric is not None:
                    lines.append(f"  numerical value {iv.numeric}")
            lines.append(f"- orientation: {integral.orientation}")
            if integral.theorem is not None:
                lines.append(
                    f"- {integral.theorem}: the two sides were computed separately and "
                    f"{'agree' if integral.holds else 'DISAGREE'}. Hypotheses:"
                )
                for h in integral.hypotheses:
                    lines.append(f"  - {h.text}: {h.status} ({h.note})")
                lines.append(
                    "  'checked' is a finite scan, never a proof of smoothness: do not say the "
                    "hypotheses were proved"
                )
            if integral.method == "numerical":
                lines.append(
                    "- this is a numerical approximation, not an exact value: say so, and give the "
                    "digits only as approximate"
                )
        for eq in result.equations_latex:
            lines.append(f"- derived equation: ${eq}$")
            known.append(eq)
        if result.model_check is not None:
            mc = result.model_check
            lines.append(
                f"- check on exact solutions of the premises: {mc.status}"
                + (f" ({mc.note})" if mc.note else "")
                + ". This is consistency evidence, not a verification: say so."
            )
        if result.em is not None:
            em = result.em
            for item in em.items:
                shown = item.latex if item.latex is not None else (item.numeric or "?")
                lines.append(
                    f"- {item.label}: ${shown}$" + (f" ({item.note})" if item.note else "")
                )
                if item.latex is not None:
                    known.append(item.latex)
            lines.extend(f"- check: {c}" for c in em.checks)
            lines.append(f"- method: {em.method}" + (f"; units: {em.units}" if em.units else ""))
            if em.approximate:
                lines.append(
                    "- this is a NUMERICAL APPROXIMATION agreeing with an independent method, not an "
                    "exact value or a proof: say so"
                )
            if em.holds is False:
                lines.append("- the candidate does NOT satisfy every requirement: say which")
        if result.singular is not None:
            _singular_lines(result, lines, known)
        if result.singular_hint:
            lines.append(
                "- this is the CLASSICAL value, valid away from the source only. The field has a point "
                "source: as a distribution its divergence is a delta term. Call vector_calculus with "
                "action=distributional to get it (never state its coefficient yourself)"
            )
        for a in result.applicable:
            suffix = f" BLOCKED: {a.note}" if a.blocked else ""
            lines.append(
                f"- match: {a.rule_id}{' (right to left)' if a.reverse else ''} at path {a.path}: "
                f"${a.part_latex}$ -> ${a.result_latex}$ (needs {', '.join(a.needs) or 'nothing'}){suffix}"
            )
        v = result.verification
        if v is not None and not result.steps and result.integral is None:
            lines.append(
                f"- numerically checked at {v.points_agreed} sample points against an independent "
                "evaluation; NO symbolic proof: say 'numerically checked', never 'proved'"
            )
        lines.extend(f"- convention: {c}" for c in result.conventions)
        lines.extend(f"- assumption: {a}" for a in result.assumptions_used)
        if result.restrictions_latex:
            lines.append(
                "- valid only where "
                + ", ".join(f"${r}$" for r in result.restrictions_latex)
                + ("" if result.restrictions_complete else " (and possibly elsewhere)")
                + ". These are where the coordinates break down, not statements about the field."
            )
            known.extend(result.restrictions_latex)
    else:
        for b in result.blocked:
            lines.append(
                f"- blocked: needs {b.needs} smoothness of {', '.join(b.fields)}: {b.note}. Ask the "
                "user whether the field is that smooth; do not declare it yourself."
            )
        if result.needs_premises:
            lines.append(
                "- needs these field equations as premises (the user adopts them; do not assume them): "
                + "; ".join(f"`{n}`" for n in result.needs_premises)
            )
        if result.undeclared:
            lines.append(
                "- undeclared: "
                + ", ".join(result.undeclared)
                + " (declare as a field or constant)"
            )
        for a in result.applicable:
            lines.append(
                f"- candidate: {a.rule_id} at path {a.path}: ${a.part_latex}$ (choose one with `path`)"
            )
    if result.input_latex:
        known.append(result.input_latex)
    lines.append(
        "to compute more about THESE fields call vector_calculus again with the same coordinates, "
        "fields, constants and premises."
    )
    quoted: list[tuple[str, str]] = []
    if result.evidence == "conditional":
        # a result that holds only given the user's premises is shown, never marked as verified
        quoted = [(x, "holds only given the premises you supplied") for x in known]
        known = []
    return ToolOutput(
        card, "\n".join([head + result.message, *lines]), known, result.status.value, quoted
    )


def run_electromagnetism(args: ElectromagnetismArgs, ctx: ToolContext) -> ToolOutput:
    from minerva_engine.vector.em import PLANNED, TEMPLATES, describe

    if args.action == "setup":
        options = premise_options(ctx.kb)
        entries = [ctx.kb.get(o.id) for o in options]
        card = Card(
            id=uuid.uuid4().hex[:8],
            tool="electromagnetism",
            type="entries",
            title="Electromagnetism setup",
            data={
                "query": "field equations you may adopt as premises",
                "entries": [summarize(ctx.kb, e).model_dump(mode="json") for e in entries if e],
            },
        )
        lines = [
            "status=ok; these are the electromagnetic derivations Minerva can carry out. Each needs field "
            "equations as PREMISES: ask the user which laws to adopt; never adopt one for them."
        ]
        for t in TEMPLATES.values():
            d = describe(t)
            needs = ", ".join(f"`{n}`" for n in d["needs"]) or "none (an identity)"
            lines.append(
                f"- {d['id']}: {d['title']}. Needs premises: {needs}. Fields: "
                + ", ".join(f"{f['name']} ({f['kind']})" for f in d["fields"])
                + "; constants: "
                + (", ".join(d["constants"]) or "none")
            )
        lines.append(
            "adoptable field equations (premise_ids): "
            + "; ".join(f"{o.id} = {o.equation}" for o in options)
        )
        lines.append("not available yet: " + "; ".join(PLANNED))
        return ToolOutput(card, "\n".join(lines), [], "ok")
    if args.action != "derive":
        return _em_compute(args, ctx)
    template = TEMPLATES.get(args.derivation or "")
    if template is None:
        return _error(
            "electromagnetism",
            "derive needs `derivation`: " + ", ".join(sorted(TEMPLATES)),
            "invalid_arguments",
            arguments_valid=False,
        )
    try:
        premises = [
            Premise(equation=o.equation, label=o.title)
            for o in resolve_premise_ids(ctx.kb, args.premise_ids)
        ]
    except KeyError as exc:
        return _error(
            "electromagnetism",
            f"{exc.args[0]} is not an adoptable field equation (see setup)",
            "invalid_arguments",
            arguments_valid=False,
        )
    premises += [Premise(equation=p) for p in args.premises]
    fields = [
        FieldDecl(
            name=name,
            kind=kind,
            depends_on=["x", "y", "z", "t"],
            smoothness=args.smoothness if name in template.smooth else None,
        )
        for name, kind in template.fields
    ]
    request = VectorInput(
        expression="",
        format="text",
        coordinates="cartesian",
        fields=fields,
        constants=list(template.constants),
        premises=premises,
        action="em_derive",
        template=template.id,
        rules=vector_identity_rules(ctx.kb),
    )
    return _vector_output(ctx.engine.vector_calculus(request), request)


def _em_compute(args: ElectromagnetismArgs, ctx: ToolContext) -> ToolOutput:
    from minerva_engine.vector.models import (
        BoundaryProblem,
        CheckProblem,
        FaceCondition,
        FieldProblem,
        MultipoleProblem,
        PointCharge,
    )

    def invalid(text: str) -> ToolOutput:
        return _error("electromagnetism", text, "invalid_arguments", arguments_valid=False)

    charges = [PointCharge(q=q, position=(x, y, z)) for q, x, y, z in args.charges]
    region = Region(bounds=args.box) if args.box is not None and len(args.box) == 3 else None
    problem: FieldProblem | MultipoleProblem | BoundaryProblem | CheckProblem
    if args.action == "field":
        if args.point is None:
            return invalid("field needs `point`")
        problem = FieldProblem(
            kind="field", charges=charges, density=args.density, box=region, point=args.point
        )
    elif args.action == "multipole":
        if not charges:
            return invalid("multipole needs `charges`")
        problem = MultipoleProblem(
            kind="multipole", charges=charges, direction=args.direction, order=args.order
        )
    elif args.action == "boundary_value":
        if args.side_a is None or args.side_b is None:
            return invalid("boundary_value needs `side_a` and `side_b`")
        problem = BoundaryProblem(
            kind="boundary_value",
            a=args.side_a,
            b=args.side_b,
            bottom=args.bottom,
            top=args.top,
            left=args.left,
            right=args.right,
            source=args.source,
            points=args.sample_points,
        )
    else:
        if args.potential is None or region is None:
            return invalid("check_solution needs `potential` and `box`")
        try:
            conditions = [FaceCondition(face=f, value=v) for f, v in args.faces]
        except ValidationError:
            return invalid("faces must be (face, value) with face one of x=lo, x=hi, y=lo, ...")
        problem = CheckProblem(
            kind="check_solution",
            potential=args.potential,
            source=args.source or "0",
            box=region,
            conditions=conditions,
        )
    request = VectorInput(
        expression="",
        format="text",
        coordinates="cartesian",
        constants=args.constants,
        action="em_compute",
        em=problem,
    )
    return _vector_output(ctx.engine.vector_calculus(request), request)


def _variation_known(report: Any) -> list[str]:
    """Formulas the engine verified, exactly as printed."""
    known: list[str] = []
    first, second = report.first, report.second
    if first is not None:
        for eq in first.equations:
            if not first.null_lagrangian:
                known += [eq.equation_latex, eq.expression_latex]
        known += [
            first.boundary_form_latex,
            first.first_variation_latex,
            first.delta_integrand_latex,
        ]
        known += [s.integrand_latex for s in first.stages] + [
            s.boundary_latex for s in first.stages
        ]
        for b in first.boundary:
            known.append(b.coefficient_latex)
            if b.natural_condition_latex:
                known.append(b.natural_condition_latex)
        known += list(first.stationarity_latex)
    if second is not None:
        known += [second.quadratic_form_latex, second.legendre_matrix_latex]
    return [k for k in known if k]


def _variation_lines(report: Any) -> list[str]:
    lines: list[str] = []
    a = report.analysis
    if a is not None:
        lines.append(f"- functional: ${a.functional_latex}$")
        lines.extend(f"- reading: {r}" for r in a.reading)
        lines.append(
            f"- highest derivative order {a.highest_order}; Legendre matrix determinant "
            f"${a.hessian_det_latex}$ ({a.regularity}); {a.unspecified} boundary condition(s) undeclared"
        )
    f = report.first
    if f is not None:
        lines.append("- first variation (every integration by parts is on the card):")
        lines.append(f"  ${f.first_variation_latex}$")
        for eq in f.equations:
            lines.append(
                f"  Euler-Lagrange for {eq.function}: ${eq.equation_latex}$ (order {eq.order})"
            )
        if f.null_lagrangian:
            lines.append("  the integrand is a total derivative: the variation is pure boundary")
        for b in f.boundary:
            if b.effect == "vanishes_identically":
                continue
            where = "lower" if b.end == "lower" else "upper"
            what = {
                "removed_by_fixed": "removed by the fixed condition",
                "natural_condition": "natural boundary condition",
                "kept_unspecified": "kept: no condition was declared for it",
            }[b.effect]
            lines.append(
                f"  {where} end, {b.function}^({b.order}): coefficient ${b.coefficient_latex}$ - {what}"
            )
        lines.extend(f"  note: {n}" for n in f.notes)
        lines.append(f"  check: {f.checks.statement}")
    s = report.second
    if s is not None:
        lines.append(
            f"- second variation: ${s.quadratic_form_latex}$ (convention ${s.convention_latex}$)"
        )
        lines.append(f"  check: {s.checks.statement}")
        if s.numeric is not None:
            n = s.numeric
            lines.append(f"  stationary along the tested perturbations: {n.stationary}")
            lines.extend(f"  fact: {x}" for x in n.facts)
            lines.extend(f"  not established: {x}" for x in n.not_established)
    e = report.evaluation
    if e is not None:
        for t in e.trials:
            lines.append(
                f"- action of {t.label}: {t.action} (quadrature disagreement {t.quadrature_error:.1e})"
            )
            lines.extend(
                f"  condition {c.function}^({c.order}) at {c.end}: need {c.required}, path has {c.found}"
                for c in t.conditions
                if not c.met
            )
        if e.ordering:
            lines.append("- order of these paths by action value: " + " < ".join(e.ordering))
        lines.append(f"- {e.statement}")
    v = report.verification
    if v is not None:
        lines.append(f"- finite-difference verification ({v.route}): {v.verdict}. {v.statement}")
    sol = report.solution
    if sol is not None:
        for m in sol.methods:
            lines.append(f"- method {m.method}: converged={m.converged} {m.note}")
        for p in sol.paths:
            lines.append(
                f"- stationary path ({p.method}): action {p.action}; stationarity residual "
                f"{p.stationarity.residual:.1e}"
            )
        lines.append(f"- {sol.statement}")
        lines.extend(f"- note: {n}" for n in sol.notes)
    return lines


def run_variation(args: VariationArgs, ctx: ToolContext) -> ToolOutput:
    request = variation_input(args, ctx)
    report = ctx.engine.variation(request)
    log_variation(report)  # the action, the status and counters only
    shown = {k: v for k, v in args.model_dump().items()}
    features = ["variation", *(["variation:second"] if report.second is not None else [])]
    methods = (
        ctx.kb.search(SearchQuery(features=features, chapter="physics.mechanics", limit=6))
        if report.status.value == "ok"
        else []
    )
    method_data = [summarize(ctx.kb, e).model_dump(mode="json") for e in methods]
    data: dict[str, Any] = {
        "report": report.model_dump(mode="json"),
        "request": shown,
        "methods": method_data,
    }
    if report.analysis is not None:
        data["functional_latex"] = report.analysis.functional_latex
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="variation",
        type="variation",
        title="Variational calculus",
        data=data,
    )
    head = f"action={report.action}; status={report.status.value}; "
    if report.input_latex:
        head += f"input_as_read=${report.input_latex}$; "
    lines = [head + report.message]
    if report.status.value in ("ok", "unchanged"):
        lines.extend(_variation_lines(report))
    if report.analysis is not None:
        steps = ", ".join(
            s.id + (f" (needs {', '.join(s.needs)})" if s.needs else "")
            for s in report.analysis.next_steps
        )
        lines.append(f"- suggested next steps: {steps}")
        lines.append(
            "numerical steps (evaluate the action, verify by finite differences, find stationary paths) "
            "need numbers only the USER can choose: the interval, endpoint conditions, parameter values, "
            "trial paths. Never choose them; ask, or tell the user the card has a form for them."
        )
    known = _variation_known(report)
    if method_data:
        lines.append(
            "handbook methods: "
            + "; ".join(f"{m['id']} ({m['title']}, {m['verification']})" for m in method_data)
        )
        known.extend(m["statement_latex"] for m in method_data)
    return ToolOutput(card, "\n".join(lines), known, report.status.value)


def run_matrix_calculus(args: MatrixCalculusArgs, ctx: ToolContext) -> ToolOutput:
    # gradients, Jacobians and Hessians read the plain-text syntax (transpose, inv, tr, det, logdet)
    plain = args.action in {"gradient", "jacobian", "hessian"}
    request = MatrixCalculusInput(
        expression=ExpressionInput(
            input=args.latex, format="text" if plain else "latex", assumptions=args.assumptions
        ),
        matrices=args.matrices,
        commuting=args.commuting,
        invertible=args.invertible,
        self_commuting=args.self_commuting,
        action=args.action,
        variable=args.variable,
        lower=args.lower,
        upper=args.upper,
    )
    result = ctx.engine.matrix_calculus(request)
    methods = (
        ctx.kb.search(SearchQuery(features=result.features, chapter="math.methods", limit=4))
        if result.features
        else []
    )
    method_data = [summarize(ctx.kb, e).model_dump(mode="json") for e in methods]
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="matrix_calculus",
        type="matrix",
        title="Matrix calculus",
        data={
            "result": result.model_dump(mode="json"),
            "methods": method_data,
            "request": request.model_dump(mode="json", exclude={"expression", "trace_id"}),
        },
    )
    shown = result.status.value == "ok"
    head = (
        f"status={result.status.value}; action={result.action}; {_as_read(result.input_latex)}"
        if result.input_latex
        else f"status={result.status.value}; action={result.action}; "
    )
    if shown:
        head += f"method={result.method}; kind={result.result_kind}; "
        if result.result_kind == "numeric":
            head += (
                "evaluated=NO (a numerical approximation on the given span, not a closed form and "
                "not a proof; say so); "
            )
        else:
            head += (
                "evaluated=yes; "
                if result.evaluated
                else "evaluated=NO (do not call this an evaluated result); "
            )
        if result.result_latex:
            head += f"result=${result.result_latex}$; "
        if result.derivative is not None:
            d = result.derivative
            head += (
                f"derivative={d.kind}; constraint={d.constraint}; convention={d.convention}; "
                "real field (a transpose, not a conjugate transpose); "
            )
            for alt in d.alternatives:
                head += (
                    f"alternative ({alt.convention}, verified {alt.points_agreed}/{alt.points_checked}): "
                    f"${alt.latex}$; "
                )
        if result.equation_latex:
            head += f"equation=${result.equation_latex}$; "
    v = result.verification
    if v and shown:
        head += (
            f"verified: numeric {v.points_agreed}/{v.points_checked}"
            + (
                f", exact series check {'passed' if v.symbolic_zero else 'not run'}"
                if v.symbolic_zero is not None
                else ", no exact check was run"
            )
            + " (checks, not a proof: say 'checked', never 'proved'); "
        )
    lines = [head + result.message]
    if result.structure:
        lines.append(f"- structure: {result.structure}")
    known: list[str] = [
        x for x in (result.input_latex, result.result_latex, result.equation_latex) if x
    ]
    if shown:
        for s in result.steps[:6]:
            parts = [f"- step ({s.method}): {s.title}"]
            if s.before_latex:
                parts.append(f"${s.before_latex}$")
                known.append(s.before_latex)
            if s.after_latex:
                parts.append(f"-> ${s.after_latex}$")
                known.append(s.after_latex)
            if s.note:
                parts.append(f"({s.note})")
            lines.append(" ".join(parts))
        lines.extend(f"- assumption used: {a}" for a in result.assumptions_used)
    lines.extend(
        f"- NOT APPLIED ({b.title}): needs {b.needs}" + (f" for ${b.latex}$" if b.latex else "")
        for b in result.blocked
    )
    quoted_blocked = [(b.latex, NOT_APPLIED) for b in result.blocked if b.latex]
    lines.extend(f"- note: {n}" for n in result.notes)
    if method_data:
        lines.append(
            "handbook methods: "
            + "; ".join(f"{m['id']} ({m['title']}, {m['verification']})" for m in method_data)
        )
        known.extend(m["statement_latex"] for m in method_data if m.get("statement_latex"))
    return ToolOutput(
        card, "\n".join(lines), known, result.status.value, quoted_latex=quoted_blocked
    )


def run_distribution(args: DistributionArgs, ctx: ToolContext) -> ToolOutput:
    request = DistributionInput(
        expression=_expr(args),
        action=args.action,
        variable=args.variable,
        lower=args.lower,
        upper=args.upper,
        endpoint=args.endpoint,
        notation=ctx.notation,
    )
    result = ctx.engine.distribution(request)
    log_distribution(result)
    cited = list(
        dict.fromkeys([s.entry_id for s in result.steps] + [b.entry_id for b in result.blocked])
    )
    entries = [e for entry_id in cited if (e := ctx.kb.get(entry_id)) is not None]
    method_data = [summarize(ctx.kb, e).model_dump(mode="json") for e in entries]
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="distribution",
        type="distribution",
        title="Delta and step functions",
        data={
            "result": result.model_dump(mode="json"),
            "methods": method_data,
            # Everything a card button needs to ask again with the user's answer (the expression too).
            "request": {
                **request.model_dump(mode="json", exclude={"expression", "notation"}),
                "latex": args.latex,
                "assumptions": args.assumptions,
            },
        },
    )
    shown = result.status.value == "ok"
    head = f"status={result.status.value}; action={result.action}; "
    if result.questions:
        names = ", ".join(q.name for q in result.questions)
        head += (
            f"needs_user_confirmation={names} (is it a Dirac delta or step function? ONLY the user "
            "answers, with the buttons on the card; do not answer for them and do not rewrite the "
            "input with the explicit names unless they said so in words); "
        )
    if result.notation:
        head += (
            "notation_confirmed_by_user="
            + ",".join(f"{n}:{r}" for n, r in sorted(result.notation.items()))
            + "; "
        )
    if result.reason:
        head += f"reason={result.reason}; "
    if result.input_latex:
        head += f"{_as_read(result.input_latex)}"
    if shown:
        head += f"result=${result.result_latex}$; "
        head += (
            "equality=value (an ordinary number or expression); "
            if result.equality == "value"
            else "equality=AS DISTRIBUTIONS (not pointwise); "
        )
        if result.endpoint:
            head += f"endpoint_convention={result.endpoint} (chosen by the user); "
    v = result.verification
    if v and shown:
        head += (
            "conditions established exactly: "
            + ("yes" if v.conditions_established else "NO")
            + f"; {v.pairings_agreed}/{v.pairings_checked} test-function pairings agree "
            "(supporting evidence, not a proof); "
        )
    lines = [head + result.message]
    known: list[str] = [x for x in (result.input_latex, result.result_latex) if x]
    if shown:
        for s in result.steps:
            lines.append(
                f"- step ({s.entry_id}): {s.title}: ${s.before_latex}$ -> ${s.after_latex}$"
            )
            known.extend([s.before_latex, s.after_latex])
            for c in s.conditions:
                lines.append(f"  - condition ${c.latex}$ ({c.how})")
                known.append(c.latex)
    lines.extend(f"- NOT APPLIED ({b.entry_id}): {b.detail} [{b.reason}]" for b in result.blocked)
    lines.extend(f"- note: {n}" for n in result.notes)
    if method_data:
        lines.append(
            "handbook entries: "
            + "; ".join(f"{m['id']} ({m['title']}, {m['verification']})" for m in method_data)
        )
        known.extend(m["statement_latex"] for m in method_data if m.get("statement_latex"))
    return ToolOutput(card, "\n".join(lines), known, result.status.value)


def run_identities(args: IdentitiesArgs, ctx: ToolContext) -> ToolOutput:
    result = find_identities(ctx.kb, ctx.engine, _expr(args), text=args.query)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="find_identities",
        type="identities",
        title="Identities",
        data=result.model_dump(mode="json"),
    )
    lines = [f"status={result.status.value}; {_as_read(result.input_latex)}{result.message}"]
    known: list[str] = [result.input_latex] if result.input_latex else []
    for m in result.matches[:8]:
        a = m.application
        line = f"- {m.entry.id} ({m.entry.title}): {a.applicability.value}; statement=${m.entry.statement_latex}$"
        known.append(m.entry.statement_latex)
        if a.rewritten_latex:
            line += f"; rewritten=${a.rewritten_latex}$ (numerically checked)"
            known.append(a.rewritten_latex)
            if a.restrictions_latex:
                line += "; valid only where " + ", ".join(f"${r}$" for r in a.restrictions_latex)
                known.extend(a.restrictions_latex)
        if a.unmet_conditions:
            line += "; unmet: " + ", ".join(
                f"{c.variable} {c.assumption}" for c in a.unmet_conditions
            )
        lines.append(line)
    return ToolOutput(card, "\n".join(lines), known, result.status.value)


def run_tensor(args: TensorArgs, ctx: ToolContext) -> ToolOutput:
    try:
        data = TensorInput(input=args.expression, tensors=args.tensors, dimension=args.dimension)
    except ValidationError as exc:
        return _error("tensor_canonicalize", exc.errors()[0]["msg"])
    result = ctx.engine.tensor_canonicalize(data)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="tensor_canonicalize",
        type="tensor",
        title="Tensor canonical form",
        data=result.model_dump(mode="json"),
    )
    text = f"status={result.status.value}; result={result.result_text}; {result.message}"
    return ToolOutput(
        card, text, [result.result_latex] if result.result_latex else [], result.status.value
    )


def run_solve(args: SolveArgs, ctx: ToolContext) -> ToolOutput:
    report = ctx.engine.solve(args.equation, args.variable, args.assumptions)
    methods = (
        ctx.kb.search(SearchQuery(features=report.features, chapter="math.methods", limit=6))
        if report.features
        else []
    )
    method_data = [summarize(ctx.kb, e).model_dump(mode="json") for e in methods]
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="solve_polynomial",
        type="solve",
        title="Solve",
        data={"report": report.model_dump(mode="json"), "methods": method_data},
    )
    lines = [f"status={report.status.value}; equation=${report.equation_latex}$; {report.message}"]
    known: list[str] = [report.equation_latex] if report.equation_latex else []
    for f in report.factors:
        exact = ", ".join(f"${r}$" for r in f.exact_roots_latex)
        lines.append(
            f"- factor ${f.factor_latex}$ (degree {f.degree}, multiplicity {f.multiplicity}): "
            f"radicals={f.radicals} ({f.radicals_reason})"
            + (f"; exact roots: {exact}" if exact else "")
        )
        known.append(f.factor_latex)
        known.extend(f.exact_roots_latex)
    if report.roots:
        shown = ", ".join(r.numeric + (" (real)" if r.real else "") for r in report.roots[:8])
        lines.append(f"numerical roots: {shown}")
    if report.verification:
        v = report.verification
        lines.append(
            f"verified: substitution={v.roots_substituted_back}, count={v.count_matches_degree}, "
            f"vieta={v.vieta_holds}"
        )
    lines.extend(f"note: {n}" for n in report.notes)
    if method_data:
        lines.append(
            "handbook methods: "
            + "; ".join(f"{m['id']} ({m['title']}, {m['verification']})" for m in method_data)
        )
        known.extend(m["statement_latex"] for m in method_data)
    return ToolOutput(card, "\n".join(lines), known, report.status.value)


def run_lagrangian(args: AnalyzeLagrangianArgs, ctx: ToolContext) -> ToolOutput:
    source = strip_assignment(args.lagrangian, "L")
    report = ctx.engine.analyze_lagrangian(
        source,
        args.coordinates,
        args.time,
        args.assumptions,
        format=resolve_syntax(source, args.format),
    )
    methods = (
        ctx.kb.search(SearchQuery(features=report.features, chapter="physics.mechanics", limit=8))
        if report.features
        else []
    )
    method_data = [summarize(ctx.kb, e).model_dump(mode="json") for e in methods]
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="analyze_lagrangian",
        type="lagrangian",
        title="Lagrangian",
        data={
            "report": report.model_dump(mode="json"),
            "methods": method_data,
            "restrictions_latex": report.restrictions_latex,
            # What was analysed, so the card's numerical-solution form can resend exactly this. A
            # Lagrangian given in text goes back as the LaTeX Minerva printed (the form posts LaTeX).
            "request": {
                "lagrangian": (
                    args.lagrangian
                    if resolve_syntax(source, args.format) == "latex"
                    else report.lagrangian_latex or args.lagrangian
                ),
                "coordinates": args.coordinates,
                "time": args.time,
                "assumptions": args.assumptions,
            },
        },
    )
    lines = [f"status={report.status.value}; {report.message}"]
    known: list[str] = [report.lagrangian_latex] if report.lagrangian_latex else []
    quoted: list[tuple[str, str]] = []
    roles: dict[str, str] = {}
    for e in report.equations:
        lines.append(f"equation of motion ({e.coordinate}): ${e.latex}$")
        known.append(e.latex)
        roles[e.latex] = "equation_of_motion"
    for m in report.momenta:
        lines.append(f"momentum: ${m.latex}$")
        known.append(m.latex)
        roles[m.latex] = "momentum"
    for a in report.accelerations:
        lines.append(f"acceleration: ${a.latex}$")
        known.append(a.latex)
        roles[a.latex] = "acceleration"
    for c in report.conserved:
        flag = "verified on the equations of motion" if c.verified else "NOT verified"
        lines.append(f"conserved ({c.kind}, {flag}): {c.name}: ${c.expression_latex}$; {c.reason}")
        if c.verified:
            known.append(c.expression_latex)
            roles[c.expression_latex] = "conserved"
        else:
            quoted.append((c.expression_latex, NOT_VERIFIED))
    lines.extend(f"tested, not found: {t}" for t in report.tested)
    lines.extend(f"note: {n}" for n in report.notes)
    if report.restrictions_latex:
        lines.append(
            "valid only where "
            + ", ".join(f"${r}$" for r in report.restrictions_latex)
            + " (the accelerations divide by the mass matrix; at these points the formulas say nothing)"
        )
        known.extend(report.restrictions_latex)
    elif not report.restrictions_complete:
        lines.append("domain restrictions could not be fully determined")
    if report.accelerations:
        lines.append(
            "Answer what was asked. Numerical integration (integrate_motion) only if the user asks; it "
            "needs their parameter values, initial conditions and time span, never yours."
        )
    if method_data:
        lines.append(
            "handbook methods: "
            + "; ".join(f"{m['id']} ({m['title']}, {m['verification']})" for m in method_data)
        )
        known.extend(m["statement_latex"] for m in method_data)
    return ToolOutput(
        card,
        "\n".join(lines),
        known,
        report.status.value,
        quoted_latex=quoted,
        formula_roles=roles,
        variables=tuple(report.coordinates),
    )


def run_expand_motion(args: ExpandMotionArgs, ctx: ToolContext) -> ToolOutput:
    report = ctx.engine.expand_motion(
        args.lagrangian,
        args.coordinates,
        args.time,
        args.assumptions,
        args.equilibrium,
        args.order,
    )
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="expand_motion",
        type="expansion",
        title="Expansion about an equilibrium",
        data={
            "report": report.model_dump(mode="json"),
            "request": {
                "lagrangian": args.lagrangian,
                "coordinates": args.coordinates,
                "time": args.time,
                "assumptions": args.assumptions,
            },
        },
    )
    lines = [f"status={report.status.value}; {report.message}"]
    known: list[str] = [report.lagrangian_latex] if report.lagrangian_latex else []
    quoted: list[tuple[str, str]] = []
    if report.equilibrium:
        lines.append(
            "equilibrium: " + ", ".join(f"{e.name} = ${e.latex}$" for e in report.equilibrium)
        )
    for e in report.equations:
        ok = e.verification is not None and e.verification.passed
        lines.append(
            f"expansion ({e.coordinate}, {'verified' if ok else 'NOT verified'}): ${e.full_latex}$"
        )
        if ok:
            known.append(e.full_latex)
            known.extend(t.latex for t in e.terms)
        else:
            quoted.append((e.full_latex, NOT_VERIFIED))
            quoted.extend((t.latex, NOT_VERIFIED) for t in e.terms)
    if report.stiffness_latex:
        lines.append(f"linear part, A (uddot = A u + B v): ${report.stiffness_latex}$")
        known.append(report.stiffness_latex)
    if report.velocity_latex:
        lines.append(f"velocity terms, B: ${report.velocity_latex}$")
        known.append(report.velocity_latex)
    if report.characteristic_latex:
        lines.append(f"characteristic polynomial: ${report.characteristic_latex}$")
        known.append(report.characteristic_latex)
    for m in report.modes:
        claim = {
            "positive": "omega^2 provably positive",
            "negative": "omega^2 provably negative (linearised equations only)",
            "zero": "omega^2 is zero",
            "unknown": "sign not decided by the declared assumptions",
        }[m.sign]
        lines.append(
            f"normal mode (verified={m.verified}; {claim}): ${m.omega_squared_latex}$"
            + (f"; ${m.omega_latex}$" if m.omega_latex else "")
            + (f"; eigenvector ${m.eigenvector_latex}$" if m.eigenvector_latex else "")
        )
        known.append(m.omega_squared_latex)
        known.extend(x for x in (m.omega_latex, m.eigenvector_latex) if x)
    lines.extend(f"note: {n}" for n in report.notes)
    if report.status.value == "ok":
        lines.append(
            "The expansion is local: it says nothing about motion far from the equilibrium. "
            "Do not state stability unless a mode's sign is marked provable."
        )
    return ToolOutput(card, "\n".join(lines), known, report.status.value, quoted_latex=quoted)


def run_perturbation(args: PerturbationArgs, ctx: ToolContext) -> ToolOutput:
    report = ctx.engine.perturbation_solve(
        args.lagrangian,
        args.coordinates,
        args.time,
        args.assumptions,
        args.small_parameter,
        args.equilibrium,
        args.order,
        args.compare,
    )
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="perturbation_solve",
        type="perturbation",
        title="Perturbation solution",
        data={
            "report": report.model_dump(mode="json"),
            # What was solved, so the card's comparison form can resend exactly this with the user's numbers.
            "request": {
                "lagrangian": args.lagrangian,
                "coordinates": args.coordinates,
                "time": args.time,
                "assumptions": args.assumptions,
                "small_parameter": args.small_parameter,
                "equilibrium": args.equilibrium,
                "order": args.order,
            },
        },
    )
    lines = [f"status={report.status.value}; {report.message}"]
    known: list[str] = [report.lagrangian_latex] if report.lagrangian_latex else []
    for label, value in (
        ("unperturbed frequency", report.linear_frequency_latex),
        ("frequency", report.frequency_latex),
        ("period ratio", report.period_ratio_latex),
        ("solution", report.solution_latex),
    ):
        if value:
            lines.append(f"{label}: ${value}$")
            known.append(value)
    for o in report.orders:
        known.append(o.shape_latex)
        if o.frequency_latex:
            known.append(o.frequency_latex)
    if report.verification:
        v = report.verification
        lines.append(
            f"verified: residual falls at the claimed order={v.residual_passed} "
            f"({v.parameter_sets_checked} parameter set(s)), initial conditions={v.initial_conditions_passed}, "
            f"bounded (no secular term)={v.bounded}"
        )
    if report.comparison:
        c = report.comparison
        if c.status.value == "ok":
            lines.append(
                f"comparison with the verified numerical solution (a measurement on this span only): "
                f"max error {c.max_error:.3g} at the given small parameter, {c.max_error_half:.3g} at half; "
                f"observed order {c.observed_order if c.observed_order is None else round(c.observed_order, 2)}"
                f" (predicted {c.predicted_order}); error scaling: {c.scaling}"
            )
        else:
            lines.append(f"comparison not shown ({c.status.value}): {c.message}")
    lines.extend(f"note: {n}" for n in report.notes)
    if report.status.value == "ok" and report.comparison is None:
        lines.append(
            "To see how well it holds, ask the user for numbers (every parameter, the small parameter or "
            "amplitude, an end time) and call again with `compare`; never choose them yourself. The series "
            "is local to small values of the small parameter and says nothing about large ones."
        )
    return ToolOutput(card, "\n".join(lines), known, report.status.value)


def run_discovery(args: DiscoveryArgs, ctx: ToolContext) -> ToolOutput:
    pins = None
    if any(
        x is not None for x in (args.degree, args.power_bound, args.harmonics, args.time_degree)
    ):
        pins = Pins(
            degree=args.degree,
            power_bound=args.power_bound,
            harmonics=args.harmonics,
            time_degree=args.time_degree,
        )
    system = "hamiltonian" if args.hamiltonian is not None else "lagrangian"
    source = args.hamiltonian if args.hamiltonian is not None else str(args.lagrangian)
    source = strip_assignment(source, "H" if system == "hamiltonian" else "L")
    report = ctx.engine.discover_invariants(
        source,
        args.coordinates,
        system,
        args.momenta,
        args.angles,
        args.budget,
        pins,
        args.time,
        args.assumptions,
        format=resolve_syntax(source, args.format),
    )
    log_search_events(report.events)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="find_conserved_quantities",
        type="discovery",
        title="Conserved quantities",
        data={"report": report.model_dump(mode="json")},
    )
    lines = [f"status={report.status.value}; {report.message}"]
    known: list[str] = [report.function_latex] if report.function_latex else []
    if report.angles:
        lines.append("treated as angles (cos and sin): " + ", ".join(report.angles))
    searched = [t for t in report.tiers if t.outcome == "searched"]
    if searched:
        lines.append(
            f"searched {len(searched)} families, from: {searched[0].description}; up to: "
            f"{searched[-1].description}"
        )
    roles: dict[str, str] = {}
    for c in report.constants:
        label = "already known" if c.kind == "known" else "new, verified on the equations of motion"
        lines.append(f"{c.id} ({label}; {c.independence}): ${c.expression_latex}$")
        known.append(c.expression_latex)
        if c.verified:  # the model's later `C_{4} = ...` is compared with it (claim_evidence)
            roles[c.expression_latex] = "conserved"
    if report.independence:
        ind = report.independence
        lines.append(
            f"functionally independent: {ind.rank_lower}"
            + (" (proved)" if ind.rank_proved else f" to {ind.rank_upper} (lower bound proved)")
            + (f"; at most {ind.max_possible} are possible" if ind.max_possible else "")
        )
        for rel in ind.relations:
            lines.append(f"algebraic relation (exact): ${rel.latex}$")
            known.append(rel.latex)
        lines.extend(f"note: {n}" for n in ind.notes)
    if report.unexplored:
        lines.append("not searched: " + "; ".join(report.unexplored[:4]))
    lines.extend(f"note: {n}" for n in report.notes)
    lines.extend(f"limit: {n}" for n in report.limitations)
    if report.status.value == "ok" and not report.new_constants:
        lines.append(
            "No additional constant was found within the explored families. Say exactly that: it does "
            "not mean the system has none."
        )
    if report.new_constants:
        lines.append(
            "These are constants in the searched families only. Naming one (for example as the "
            "Laplace-Runge-Lenz vector) is a recognition you may offer, not something this result proves."
        )
    return ToolOutput(
        card,
        "\n".join(lines),
        known,
        report.status.value,
        formula_roles=roles,
        variables=tuple(report.coordinates),
    )


def run_invariants(args: InvariantArgs, ctx: ToolContext) -> ToolOutput:
    source = strip_assignment(args.lagrangian, "L")
    report = ctx.engine.search_invariants(
        source,
        args.coordinates,
        args.angles,
        args.degree,
        args.power_bound,
        args.harmonics,
        args.time,
        args.assumptions,
        format=resolve_syntax(source, args.format),
    )
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="search_invariants",
        type="invariants",
        title="Search for more constants of motion",
        data={"report": report.model_dump(mode="json")},
    )
    lines = [f"status={report.status.value}; {report.message}"]
    known: list[str] = [report.lagrangian_latex] if report.lagrangian_latex else []
    roles: dict[str, str] = {}
    if report.ansatz:
        lines.append(f"searched: {report.ansatz}")
    if report.known:
        lines.append("already known and factored out: " + "; ".join(report.known))
    for c in report.found:
        lines.append(
            f"new constant of motion (verified on the equations of motion): ${c.expression_latex}$"
        )
        known.append(c.expression_latex)
        roles[c.expression_latex] = "conserved"
    lines.extend(f"note: {n}" for n in report.notes)
    if report.found:
        lines.append(
            "These are constants in the searched family only. Naming them (for example as the "
            "Laplace-Runge-Lenz vector) is a recognition you may offer, not something this result proves."
        )
    return ToolOutput(
        card,
        "\n".join(lines),
        known,
        report.status.value,
        formula_roles=roles,
        variables=tuple(report.coordinates),
    )


def _motion_summary(report: Any) -> list[str]:
    lines: list[str] = []
    values = ", ".join(f"{k}={v:g}" for k, v in report.parameters.items()) or "no parameters"
    lines.append(
        f"values as read: {values}; q(0)={report.initial_positions}; qdot(0)={report.initial_velocities}; "
        f"t from {report.t_start:g} to {report.t_end:g}"
        + (f"; sweeping {report.sweep_name}" if report.sweep_name else "")
    )
    for run in report.runs:
        label = (
            f"{report.sweep_name}={run.parameter_value:g}"
            if run.parameter_value is not None
            else "run"
        )
        v = run.verification
        if run.status.value == "ok":
            ranges = "; ".join(
                f"{name}: min {min(row):.4g}, max {max(row):.4g}, final {row[-1]:.4g}"
                for name, row in zip(report.coordinates, run.q, strict=True)
            )
            lines.append(
                f"- {label}: ok; {ranges}; verified: method difference {v.method_difference:.1e}, "
                f"reversal error {v.reversal_error:.1e}, constants of motion followed {v.drift_checked}"
                + (f", max drift {v.max_drift:.1e}" if v.max_drift is not None else "")
            )
        else:
            lines.append(f"- {label}: {run.status.value}; {run.message}")
    return lines


def _user_wrote(value: float, written: frozenset[float]) -> bool:
    return any(abs(value - w) <= 1e-12 * max(1.0, abs(value)) for w in written)


def values_provenance(texts: list[str], ctx: ToolContext) -> tuple[str, list[str]]:
    """Who chose the numbers of a numerical run: `user` (typed by them, or on the card form),
    `assistant_delegated` (they told the assistant to choose) or `assistant` (the model supplied numbers
    that are in none of the user's messages). The listed values are the ones the user did not write. A
    label, never a refusal: a number the user wrote another way (`10^3` for 1000) is listed too."""
    if ctx.user_numbers is None:
        return "user", []
    if ctx.numbers_delegated:
        return "assistant_delegated", texts
    mine = [t for t in texts if not all(_user_wrote(x, ctx.user_numbers) for x in numbers_in(t))]
    return ("assistant" if mine else "user"), mine


def _provenance_note(who: str, values: list[str]) -> str | None:
    shown = ", ".join(values[:12]) + (" ..." if len(values) > 12 else "")
    if who == "assistant_delegated":
        return (
            "values chosen by the assistant because the user asked it to choose: "
            f"{shown}. Tell the user exactly these values."
        )
    if who == "assistant":
        return (
            "values the user did not write: "
            f"{shown}. The card says so. Tell the user these were not theirs and ask them to confirm "
            "or change them."
        )
    return None


def _ode_value_texts(args: OdeNumericArgs) -> list[str]:
    texts: list[str] = []
    for item in args.conditions:
        if isinstance(item, str):
            texts.append(item)
        else:
            prime = "'" * item.order
            texts.append(f"{item.unknown}{prime}({item.at}) = {item.value}")
    texts.extend(f"{k} = {v:g}" for k, v in args.parameters.items())
    if args.t_end is not None:
        texts.append(f"t_end = {args.t_end:g}")
    return texts


def _motion_value_texts(args: MotionArgs) -> list[str]:
    texts = [f"{k} = {v:g}" for k, v in args.parameters.items()]
    for name, q, v in zip(
        args.coordinates, args.initial_positions, args.initial_velocities, strict=True
    ):
        texts.append(f"{name}(t_start) = {q:g}")
        texts.append(f"d{name}/dt(t_start) = {v:g}")
    if args.t_start != 0:
        texts.append(f"t_start = {args.t_start:g}")
    texts.append(f"t_end = {args.t_end:g}")
    if args.sweep is not None:
        texts.append(f"{args.sweep.name} in " + ", ".join(f"{v:g}" for v in args.sweep.values))
    return texts


def run_motion(args: MotionArgs, ctx: ToolContext) -> ToolOutput:
    report = ctx.engine.integrate_motion(args)
    who, mine = values_provenance(_motion_value_texts(args), ctx)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="integrate_motion",
        type="motion",
        title="Numerical solution",
        data={
            "report": report.model_dump(mode="json"),
            "request": args.model_dump(mode="json"),
            "values_chosen_by": who,
            "assistant_values": mine,
        },
    )
    lines = [f"status={report.status.value}; {report.message}"]
    note = _provenance_note(who, mine)
    if note:
        lines.append(note)
    known: list[str] = []
    if report.lagrangian_latex:
        lines.append(f"Lagrangian: ${report.lagrangian_latex}$")
        known.append(report.lagrangian_latex)
    for a in report.accelerations:
        lines.append(f"acceleration: ${a}$")
        known.append(a)
    if report.runs:
        lines.extend(_motion_summary(report))
    lines.extend(f"note: {n}" for n in report.notes)
    return ToolOutput(card, "\n".join(lines), known, report.status.value)


def _ode_status(result: OdeAnalysis | OdeSolution | OdeNumericReport) -> str:
    return result.status.value


def _ode_card(
    tool: str,
    kind: Literal["ode_analysis", "ode_solution", "ode_numeric"],
    title: str,
    result: OdeAnalysis | OdeSolution | OdeNumericReport,
    request: OdeInput,
) -> Card:
    return Card(
        id=uuid.uuid4().hex[:8],
        tool=tool,
        type=kind,
        title=title,
        data={"report": result.model_dump(mode="json"), "request": request.model_dump(mode="json")},
    )


def _ode_reading(
    result: OdeAnalysis | OdeSolution | OdeNumericReport,
) -> tuple[list[str], list[tuple[str, str]]]:
    """How the equation was read, for the model; the equations are the user's input, so they are
    quoted and never marked as verified."""
    meaning = result.interpretation
    if meaning is None:
        return [], []
    lines = [
        "read as: unknowns "
        + ", ".join(meaning.unknowns)
        + f"; independent variable {meaning.independent}; parameters "
        + (", ".join(meaning.parameters) or "none")
    ]
    lines.extend(f"equation as read: ${eq}$" for eq in meaning.equations_latex)
    lines.extend(f"condition as read: ${c}$" for c in meaning.conditions_latex)
    lines.extend(f"note: {n}" for n in meaning.notes)
    quoted = [(eq, QUOTED_INPUT) for eq in [*meaning.equations_latex, *meaning.conditions_latex]]
    return lines, quoted


def run_ode_analyze(args: OdeAnalyzeArgs, ctx: ToolContext) -> ToolOutput:
    request = OdeAnalyzeInput(**args.model_dump())
    result = ctx.engine.ode_analyze(request)
    log_ode("ode_analyze", _ode_status(result), result.diagnostics, result.duration_ms)
    card = _ode_card("ode_analyze", "ode_analysis", "Differential equation", result, request)
    lines = [f"status={result.status.value}; {result.message}".rstrip("; ")]
    reading, quoted = _ode_reading(result)
    lines.extend(reading)
    known: list[str] = []
    s = result.structure
    if s is not None:
        lines.append(
            f"structure: {s.kind}, order {s.order}, {s.state_dimension} first-order states; "
            f"linear={s.linear}; homogeneous={s.homogeneous}; constant coefficients="
            f"{s.constant_coefficients}; autonomous={s.autonomous}; solved for the highest "
            f"derivative={s.solved_for_highest}"
        )
        lines.append(
            "classes (SymPy's classification; each solver verifies its own result): "
            + (", ".join(result.classes) or "none recognised")
        )
        for solved in result.solved_latex:
            lines.append(f"solved form: ${solved}$")
            known.append(solved)
        for restriction in result.restrictions_latex:
            lines.append(f"valid where ${restriction}$")
            known.append(restriction)
        if result.problem_kind == "general":
            lines.append("conditions: none given")
        else:
            lines.append(f"conditions: {result.problem_kind}")
        if result.conditions_needed:
            lines.append("still needed: " + ", ".join(result.conditions_needed))
    ranked = [m for m in result.methods if m.applicable]
    for m in ranked:
        needs = "; ".join(m.requires)
        lines.append(
            f"{m.rank}. {m.id} (tool {m.tool}{', ' + m.action if m.action else ''}): {m.reason}"
            + (f". Needs: {needs}" if needs else "")
        )
    skipped = [m for m in result.methods if not m.applicable]
    if skipped:
        lines.append("not applicable: " + "; ".join(f"{m.id} ({m.reason})" for m in skipped))
    q = result.qualitative
    if q is not None:
        lines.append(
            f"equilibria and local stability: status={q.status.value}; {q.message}".rstrip("; ")
        )
        for e in q.equilibria:
            lam = ", ".join(f"${x}$" for x in e.eigenvalues_latex) or ", ".join(
                e.eigenvalues_numeric
            )
            lines.append(
                f"- equilibrium ${', '.join(e.point_latex)}$; eigenvalues {lam or 'not given'}; "
                f"{e.classification_text}"
            )
            known.extend(e.point_latex)
        if q.status.value == "ok":
            lines.append(
                f"equilibria found by solving f = 0 exactly (complete list: {q.complete}); each confirmed by "
                "substitution; the Jacobian was checked against numerical differences. "
                + q.local_note
            )
            if q.direction_field is not None:
                lines.append(
                    "a direction field was computed on the user's window (a picture of f, not of a solution)"
                )
        lines.extend(f"note: {n}" for n in q.notes)
    lines.extend(f"note: {n}" for n in result.notes)
    if result.status.value == "ok":
        lines.append(
            "Offer only the applicable methods, in this order, in these words. 'Applicable' means Minerva "
            "will try and check it, not that it will succeed. For a numerical solution the USER must give "
            "every initial value, parameter value and the end of the interval: ask; the card has a form "
            "(only if the user tells you to choose them may you pick them, and say which)."
        )
    return ToolOutput(
        card,
        "\n".join(lines),
        known,
        result.status.value,
        quoted,
        choices=tuple(m.id for m in ranked) if result.status.value == "ok" else (),
    )


def run_ode_symbolic(args: OdeSymbolicArgs, ctx: ToolContext) -> ToolOutput:
    request = OdeSolveInput(**args.model_dump())
    result = ctx.engine.ode_solve_symbolic(request)
    log_ode("ode_solve_symbolic", _ode_status(result), result.diagnostics, result.duration_ms)
    card = _ode_card("ode_solve_symbolic", "ode_solution", "Closed-form solution", result, request)
    lines = [f"status={result.status.value}; {result.message}".rstrip("; ")]
    reading, quoted = _ode_reading(result)
    lines.extend(reading)
    known: list[str] = []
    for item in result.solutions:
        v = item.verification
        proof = (
            "also proved by a symbolic zero test"
            if v.symbolic_zero
            else "NO symbolic proof (say 'numerically checked', never 'proved')"
        )
        family = {
            "general": "general family",
            "particular": "particular solution",
            "partial": "family with fewer independent constants than the order",
        }[item.family]
        where = "" if item.evaluated else "; an integral remains unevaluated (a representation)"
        if item.expansion is not None:
            how = (
                "the truncated result, put into the equation as written, leaves a residual that falls "
                f"at the claimed order ({v.points_agreed} of {v.points_checked} confirmations); "
                "a numerical check, not a proof"
            )
        else:
            how = (
                f"numerically checked at {v.points_agreed} of {v.points_checked} points, derivatives "
                f"taken numerically; {proof}"
            )
        lines.append(
            f"solution ({item.kind}, {family}{', constants ' + ', '.join(item.constants) if item.constants else ''}"
            f"{where}): {how}"
        )
        for equation in item.equations_latex:
            lines.append(f"  ${equation}$")
            known.append(equation)
            known.append(equation.split(" = ", 1)[-1])
        if item.expansion is not None:
            e = item.expansion
            lines.append(
                f"  {e.kind.replace('_', ' ')} to order {e.order}"
                + (f" about ${e.point_latex}$" if e.point_latex else "")
                + f". {e.validity}"
            )
            if e.secular_orders:
                lines.append(
                    "  terms that grow with the variable at order(s) "
                    + ", ".join(map(str, e.secular_orders))
                    + ": valid only on bounded intervals; do not call it valid for long times"
                )
        for c in item.conditions:
            lines.append(
                f"  condition ${c.latex}$: {'satisfied' if c.satisfied else 'NOT satisfied'}"
            )
        if item.restrictions_latex:
            lines.append(
                "  valid only where " + ", ".join(f"${r}$" for r in item.restrictions_latex)
            )
            known.extend(item.restrictions_latex)
    if result.dropped:
        lines.append(f"{result.dropped} candidate(s) failed their checks and are NOT shown")
    lines.extend(f"note: {n}" for n in result.notes[-2:])
    if result.status.value != "ok":
        lines.append(
            "Nothing is claimed about whether a closed form exists. Say what the status means, and "
            "offer the numerical method if the user can give the numbers."
        )
    return ToolOutput(card, "\n".join(lines), known, result.status.value, quoted)


def _series_summary(report: OdeNumericReport) -> list[str]:
    lines = []
    for s in report.series:
        values = s.values
        lines.append(
            f"{s.label_latex}: min {min(values):.6g}, max {max(values):.6g}, "
            f"at the start {values[0]:.6g}, at the end {values[-1]:.6g}"
        )
    return lines


def run_ode_numeric(args: OdeNumericArgs, ctx: ToolContext) -> ToolOutput:
    request = OdeNumericInput(**args.model_dump())
    report = ctx.engine.ode_solve_numerical(request)
    log_ode("ode_solve_numerical", _ode_status(report), report.diagnostics, report.duration_ms)
    card = _ode_card("ode_solve_numerical", "ode_numeric", "Numerical solution", report, request)
    who, mine = values_provenance(_ode_value_texts(args), ctx)
    card.data["values_chosen_by"] = who
    card.data["assistant_values"] = mine
    lines = [f"status={report.status.value}; {report.message}".rstrip("; ")]
    note = _provenance_note(who, mine)
    if note:
        lines.append(note)
    reading, quoted = _ode_reading(report)
    lines.extend(reading)
    known: list[str] = []
    if report.method_used:
        lines.append(
            f"method: {report.method_used} ({report.method_reason}); "
            f"{report.steps or 0} steps, {report.evaluations or 0} evaluations of the equation"
        )
    if report.status.value == "ok":
        lines.append(
            f"interval [{report.t_start:g}, {report.t_end:g}]; rtol {report.rtol:g}, atol {report.atol:g}; "
            "initial values: "
            + ", ".join(f"{i.label_latex}={i.value:g}" for i in report.initial)
            + (
                "; parameters: " + ", ".join(f"{k}={v:g}" for k, v in report.parameters.items())
                if report.parameters
                else ""
            )
            + (
                " (all supplied by the user)"
                if who == "user"
                else " (not all supplied by the user: see above)"
            )
        )
        lines.extend(_series_summary(report))
    if report.verification is not None:
        for c in report.verification.checks:
            detail = (
                f" ({c.value:.2e} against {c.tolerance:.1e})"
                if c.value is not None and c.tolerance
                else ""
            )
            lines.append(f"check {c.name.replace('_', ' ')}: {c.status}{detail}")
        if report.verification.first_integral_latex:
            known.append(report.verification.first_integral_latex)
            lines.append(
                f"conserved quantity checked: ${report.verification.first_integral_latex}$"
            )
    comparison = report.comparison
    if comparison is not None:
        if comparison.status.value == "ok" and comparison.max_difference is not None:
            lines.append(
                f"compared with the closed form given: largest difference {comparison.max_difference:.2e} "
                "(relative to the size of the closed form); two separately verified results, a "
                "measurement and not a bound on either's error"
            )
        else:
            lines.append(f"comparison with the closed form not made: {comparison.message}")
    if report.warnings:
        lines.append("warnings: " + ", ".join(report.warnings))
    if report.status.value == "ok":
        lines.append(
            "This is a numerical approximation on the interval given: not exact, not valid beyond it. "
            "The checks are measurements, not an error bound."
        )
    else:
        lines.append("No trajectory is shown. Do not describe or estimate one.")
    return ToolOutput(card, "\n".join(lines), known, report.status.value, quoted)


def run_dimensional(args: DimensionalArgs, ctx: ToolContext) -> ToolOutput:
    report = ctx.engine.dimensional_analysis(
        args.quantities,
        args.unknown,
        args.repeating,
        constant_candidates(ctx.kb) if args.unknown else [],
    )
    methods = (
        ctx.kb.search(SearchQuery(features=report.features, chapter="physics.methods", limit=4))
        if report.features
        else []
    )
    method_data = [summarize(ctx.kb, e).model_dump(mode="json") for e in methods]
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="dimensional_analysis",
        type="dimensional",
        title="Dimensional analysis",
        data={"report": report.model_dump(mode="json"), "methods": method_data},
    )
    lines = [f"status={report.status.value}; {report.message}"]
    known: list[str] = []
    if report.quantities:
        lines.append(
            "units as read: "
            + "; ".join(f"{q.name} = [{q.dimension_text}]" for q in report.quantities)
        )
    for g in report.groups:
        lines.append(f"group ${g.label}$ = ${g.latex}$ (dimensionless, verified)")
        known.append(g.latex)
    if report.closure:
        lines.append(f"closure={report.closure}")
    if report.relation_latex:
        lines.append(
            f"relation: ${report.relation_latex}$ (C and f are not determined by dimensions)"
        )
        known.append(report.relation_latex)
    if report.cannot_enter:
        lines.append("cannot enter by dimensions alone: " + ", ".join(report.cannot_enter))
    if report.missing:
        m = report.missing
        lines.append(
            f"MISSING: a quantity with dimension [{m.missing_dimension_text}] "
            f"(unbalanced base units: {', '.join(m.unbalanced)}); {m.explanation}"
        )
        for sug in m.suggestions:
            lines.append(
                f"- adding {' and '.join(sug.ids)} closes it ({sug.closure}): ${sug.relation_latex}$"
            )
            if sug.relation_latex:
                known.append(sug.relation_latex)
        if not m.suggestions:
            lines.append("no single constant or pair of handbook constants closes it")
    lines.extend(report.tested)
    lines.extend(f"note: {n}" for n in report.notes)
    if method_data:
        lines.append(
            "handbook methods: "
            + "; ".join(f"{x['id']} ({x['title']}, {x['verification']})" for x in method_data)
        )
        known.extend(x["statement_latex"] for x in method_data)
    return ToolOutput(card, "\n".join(lines), known, report.status.value)


def _assumed_units(report: UnitsCheckReport, ctx: ToolContext) -> list[str]:
    """The quantities of the call whose unit the user did not state (`unit_statements`): a label on the card
    and in the tool text, never a refusal and never an input to the computation. `ctx.user_messages` is None
    outside a chat turn, where the numbers are the user's own."""
    if ctx.user_messages is None:
        return []
    pairs = [(q.name, q.dimension_text) for q in report.quantities if q.origin == "given"]
    return assumed_quantities(pairs, list(ctx.user_messages))


def run_units_check(args: UnitsCheckArgs, ctx: ToolContext) -> ToolOutput:
    report = ctx.engine.check_units(args.equation, args.quantities, args.unknown)
    assumed = _assumed_units(report, ctx)
    methods = ctx.kb.search(
        SearchQuery(features=["dimensional:units-check"], chapter="physics.methods", limit=3)
    )
    method_data = [summarize(ctx.kb, e).model_dump(mode="json") for e in methods]
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="check_units",
        type="units",
        title="Unit check",
        data={
            "report": report.model_dump(mode="json"),
            "methods": method_data,
            "assumed_units": assumed,
        },
    )
    lines = [f"status={report.status.value}; {report.message}"]
    known: list[str] = []
    if report.equation_latex:
        lines.append(f"equation_as_read=${report.equation_latex}$")
        known.append(report.equation_latex)
    for i, side in enumerate(report.sides, start=1):
        lines.append(f"part {i}: ${side.latex}$ has dimension [{side.dimension_text}]")
    if report.consistent is not None:
        lines.append(f"consistent={report.consistent}")
    for d in report.derived:
        lines.append(f"dimension of {d.name}, given these inputs: [{d.dimension_text}]")
    if report.undetermined:
        lines.append(
            f"not determined: {', '.join(report.undetermined)} stay free (the equation fixes only "
            "combinations of them); say so and ask for a unit, never pick one"
        )
    for r in report.quantities:
        if r.origin == "time_derivative":
            lines.append(f"read {r.name} as {r.given}: [{r.dimension_text}]")
    if assumed:
        lines.append(
            "units not stated by the user (assumed by the assistant): "
            f"{', '.join(assumed)}. The card says so. Tell the user these were assumed, that the "
            "result holds given them, and ask them to confirm or correct them."
        )
    if report.missing_factor_text:
        lines.append(f"missing factor dimension: [{report.missing_factor_text}]")
    if report.problem:
        lines.append(
            f"problem: {report.problem}"
            + "".join(f"; ${t.latex}$ is [{t.dimension_text}]" for t in report.problem_terms)
        )
    lines.extend(f"note: {n}" for n in report.notes)
    return ToolOutput(card, "\n".join(lines), known, report.status.value)


def _entry_card_data(ctx: ToolContext, entry_ids: list[str]) -> list[dict[str, Any]]:
    out = []
    for entry_id in entry_ids:
        e = ctx.kb.get(entry_id)
        if e is not None:
            out.append(summarize(ctx.kb, e).model_dump(mode="json"))
    return out


def run_search(args: SearchArgs, ctx: ToolContext) -> ToolOutput:
    found = ctx.kb.search(
        SearchQuery(text=args.query, chapter=args.chapter, kind=args.kind, limit=args.limit)
    )
    ids = [e.id for e in found]
    entries = _entry_card_data(ctx, ids)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="search_handbook",
        type="entries",
        title=f"Handbook: {args.query}",
        data={"query": args.query, "entries": entries},
    )
    if not entries:
        return ToolOutput(
            card,
            "no handbook entry matched this query. The handbook may simply not cover it: say so. "
            "Try different words at most once; never describe an entry that was not returned.",
            [],
            "no_results",
        )
    lines = [
        f"- {e['id']} ({e['title']}; verification={e['verification']}): ${e['statement_latex']}$"
        + (f" Valid: {e['validity']}" if e["validity"] else "")
        for e in entries
    ]
    return ToolOutput(card, "\n".join(lines), [e["statement_latex"] for e in entries])


def run_get_entry(args: EntryArgs, ctx: ToolContext) -> ToolOutput:
    entry = ctx.kb.get(args.id)
    if entry is None:
        return _error("get_entry", f"no handbook entry with id {args.id!r}; use search_handbook")
    data = summarize(ctx.kb, entry).model_dump(mode="json")
    data["symbols"] = [s.model_dump() for s in entry.symbols]
    data["notes"] = entry.notes
    card = Card(
        id=uuid.uuid4().hex[:8], tool="get_entry", type="entry", title=entry.title, data=data
    )
    text = (
        f"{entry.id}: {entry.title}; verification={entry.verification}; statement=${data['statement_latex']}$; "
        f"valid: {entry.validity or 'not restricted'}; notes: {entry.notes or '-'}"
    )
    return ToolOutput(card, text, [data["statement_latex"]])


def run_load_tools(args: LoadToolsArgs, ctx: ToolContext) -> ToolOutput:
    """Ask for the tools of one or more groups. The loop adds them to the next request."""
    found = requested_groups({"groups": args.groups})
    unknown = [g for g in args.groups if str(g).strip().lower() not in found]
    if not found:
        return _error(
            "load_tools",
            f"unknown tool group(s) {', '.join(map(repr, args.groups))}; groups: {', '.join(GROUPS)}",
            "invalid_arguments",
            arguments_valid=False,
        )
    tools = [t for g in found for t in GROUP_TOOLS[g]]
    message = (
        f"Loaded the {' and '.join(found)} group(s); the tools are now available. Call them by "
        f"exactly these names, nothing before or after: {', '.join(tools)}."
    )
    if unknown:
        message += f" Unknown groups ignored: {', '.join(map(str, unknown))}."
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="load_tools",
        type="note",
        title="Tools loaded",
        data={"message": message, "groups": found},
    )
    return ToolOutput(card, message, [])


# --- controlled steps: substitute, rewrite one part, isolate a variable -----------------------------
class SubstituteArgs(SimplifyArgs):
    relations: list[str] = Field(
        min_length=1,
        max_length=4,
        description="Relations to use, each `left = right` in LaTeX, e.g. y^2 = 1 - x^2. They come "
        "from the user: never invent one. The left side is replaced by the right side.",
    )
    order: Literal["given", "simultaneous"] | None = Field(
        default=None,
        description="Only if the user said how several relations combine: `given` applies them one after "
        "another in the order listed, `simultaneous` all at once. Leave empty otherwise: the tool refuses "
        "when the order would change the result, and then you must ask the user.",
    )


class RewritePartArgs(SimplifyArgs):
    action: Literal["simplify", "expand", "factor", "cancel", "collect", "rule"] = Field(
        description="The move to apply to the chosen part. `rule` applies the handbook identity `rule_id`."
    )
    rule_id: str | None = Field(
        default=None, max_length=128, description="Handbook entry id when action is `rule`."
    )
    reverse: bool = Field(default=False, description="Read the identity right to left.")
    part: str | None = Field(
        default=None,
        max_length=400,
        description="The part of the expression to change, in LaTeX. Empty: the whole expression.",
    )
    occurrence: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Which occurrence, when the part appears more than once. Ask the user; never guess.",
    )
    by: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="For `collect`: the variables or parts to collect by.",
    )


class IsolateArgs(BaseModel):
    equation: str = Field(
        max_length=MAX_INPUT,
        description="The equation `left = right` in LaTeX, e.g. s = u t + \\frac{1}{2} a t^2",
    )
    variable: str = Field(max_length=32, description="The unknown to solve for, e.g. t")
    assumptions: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Properties of symbols: real, positive, negative, nonnegative, nonpositive, nonzero, "
        "integer, rational.",
    )


def _step_card(tool: str, title: str, operation: str, result: Any) -> Card:
    return Card(
        id=uuid.uuid4().hex[:8],
        tool=tool,
        type="step",
        title=title,
        data={"operation": operation, "result": result.model_dump(mode="json")},
    )


def _step_known(result: Any) -> list[str]:
    """Every formula in a step result that the model may quote."""
    known = [x for x in (result.input_latex, result.result_latex) if x]
    for step in result.steps:
        known += [step.before_latex, step.after_latex, *step.conditions_latex]
        if step.display_latex:
            known.append(step.display_latex)
        if step.matched_latex:
            known.append(step.matched_latex)
    for premise in result.given:
        known.append(premise.latex)
        known += [side for side in premise.latex.split(" = ") if side]
    known += [*result.conditions_latex, *result.restrictions_latex]
    return known


def _probe_cut_note(v: Any) -> str:
    """Said when the targeted probes were stopped by the clock before their point cap (V-017): the check is
    incomplete, which is not a disagreement, and it did not pass."""
    if not getattr(v, "probes_cut_short", False):
        return ""
    return (
        "; check incomplete: the targeted probes were cut short by the time limit (not a disagreement), "
        "so this is not verified"
    )


def _evidence_clause(result: Any) -> str:
    v = result.verification
    if v is None:
        return ""
    return (
        f"checked numerically at {v.numeric_points_agreed}/{v.numeric_points_checked} points; "
        f"symbolic_zero={v.symbolic_zero}{_probe_cut_note(v)}"
    )


def run_substitute(args: SubstituteArgs, ctx: ToolContext) -> ToolOutput:
    result = ctx.engine.substitute(_expr(args), args.relations, order=args.order)
    card = _step_card("substitute", "Substitute", "substitute", result)
    head = f"status={result.status.value}; input_as_read=${result.input_latex}$"
    lines = []
    if result.status.value == "ok":
        head += f"; result=${result.result_latex}$; evidence={result.evidence}"
        if result.evidence == "conditional":
            head += (
                "; the result HOLDS GIVEN the user's relations "
                + ", ".join(f"${p.latex}$" for p in result.given)
                + ". Minerva did not check that they are true: say 'given', never 'proved'"
            )
        head += f"; {_evidence_clause(result)}"
        for i, step in enumerate(result.steps, start=1):
            shown = (
                f" (written ${step.display_latex}$ before combining)" if step.display_latex else ""
            )
            lines.append(f"- step {i}: ${step.before_latex}$ -> ${step.after_latex}${shown}")
        if result.restrictions_latex:
            head += "; valid only where " + ", ".join(f"${r}$" for r in result.restrictions_latex)
        if result.order_dependent:
            head += "; ANOTHER ORDER of the relations would give a different expression"
    else:
        head += f"; result=none ({result.reason.value if result.reason else result.message})"
        if result.reason is not None and result.reason.value in {
            "order_dependent",
            "cyclic",
            "deadline",
        }:
            head += (
                ". Ask the user whether to apply the relations in the order listed or all at once"
            )
    return ToolOutput(
        card, "\n".join([head, result.message, *lines]), _step_known(result), result.status.value
    )


def run_rewrite_part(args: RewritePartArgs, ctx: ToolContext) -> ToolOutput:
    rule = None
    if args.action == "rule":
        entry = ctx.kb.get(args.rule_id or "")
        if entry is None or entry.rule is None:
            return _error(
                "rewrite_part", "that handbook entry is not an identity that can be applied"
            )
        rule = entry.rule
    result = ctx.engine.rewrite_at(
        _expr(args),
        args.action,
        part=args.part,
        occurrence=args.occurrence,
        rule=rule,
        reverse=args.reverse,
        by=args.by,
    )
    card = _step_card("rewrite_part", "Change one part", "rewrite", result)
    head = f"status={result.status.value}; {_as_read(result.input_latex)}action={args.action}"
    lines: list[str] = []
    if result.status.value == "ok":
        head += f"; result=${result.result_latex}$; {_evidence_clause(result)}"
        for step in result.steps:
            head += f"; applied {step.name} to the part ${step.matched_latex}$"
        if result.restrictions_latex:
            head += "; valid only where " + ", ".join(f"${r}$" for r in result.restrictions_latex)
    elif result.status.value == "conditions_not_satisfied":
        needs = ", ".join(f"{c.variable} {c.assumption}" for c in result.unmet_conditions)
        head += f"; NOT APPLIED: the identity matches this part but needs {needs}. Ask whether they hold"
    else:
        head += "; result=none"
        if result.reason is not None and result.reason.value == "ambiguous_selection":
            for sub in result.subterms[:8]:
                lines.append(
                    f"- occurrence {sub.occurrence} of {sub.occurrences}: ${sub.latex}$ in ${sub.highlight_latex}$"
                )
            head += ". The part occurs more than once: ask the user which occurrence, then call again with `occurrence`"
    known = (
        _step_known(result)
        + [s.latex for s in result.subterms]
        + [s.highlight_latex for s in result.subterms]
    )
    return ToolOutput(card, "\n".join([head, result.message, *lines]), known, result.status.value)


def run_isolate(args: IsolateArgs, ctx: ToolContext) -> ToolOutput:
    result = ctx.engine.isolate(
        ExpressionInput(input=args.equation, format="latex", assumptions=args.assumptions),
        args.variable,
    )
    card = _step_card("isolate_variable", "Solve for a variable", "isolate", result)
    head = f"status={result.status.value}; equation_as_read=${result.equation_latex}$; unknown={args.variable}"
    known: list[str] = [x for x in [result.equation_latex] if x]
    lines: list[str] = []
    if result.status.value == "ok":
        if not result.solutions:
            head += "; NO SOLUTION (every candidate was excluded by a condition)"
        for i, sol in enumerate(result.solutions[:12], start=1):
            value = sol.latex.split(" = ", 1)[-1]
            needs = (
                f"; needs {', '.join(f'${c}$' for c in sol.conditions_latex)}"
                if sol.conditions_latex
                else ""
            )
            family = f"; family: {sol.branch}" if sol.branch else ""
            lines.append(f"- solution {i}: ${sol.latex}${needs}{family}")
            known += [sol.latex, value, *sol.conditions_latex]
        if result.conditions_latex:
            head += "; ALL solutions need " + ", ".join(f"${c}$" for c in result.conditions_latex)
        if result.excluded_latex:
            head += "; the equation itself excludes " + ", ".join(
                f"${c}$" for c in result.excluded_latex
            )
        if result.exceptional:
            head += (
                "; NOT COVERED when "
                + ", ".join(f"${c}$" for c in result.exceptional)
                + " (the equation degenerates there: say so)"
            )
        if result.assumption_note:
            head += f"; {result.assumption_note}"
        head += f"; every solution was substituted back; complete={result.complete}"
        known += [*result.conditions_latex, *result.excluded_latex, *result.exceptional]
    else:
        head += "; result=none (do not offer a partial answer)"
    return ToolOutput(card, "\n".join([head, result.message, *lines]), known, result.status.value)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[Any, ToolContext], ToolOutput]

    def schema(self) -> dict[str, Any]:
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "search_handbook",
            "Search the handbook of verified formulas, identities, integrals, series, physics equations and constants.",
            SearchArgs,
            run_search,
        ),
        Tool(
            "get_entry",
            "Read one handbook entry by id (statement, validity conditions, how it was verified).",
            EntryArgs,
            run_get_entry,
        ),
        Tool(
            "simplify",
            "Simplify, expand, factor or rewrite an expression. Searches chains of handbook identities and "
            "SymPy moves toward the goal and verifies every step; the result lists the identities used "
            "(quote them by id), what was checked numerically versus symbolically, and identities that "
            "match but whose conditions are not provable from the declared assumptions (nothing is applied "
            "then). Reports 'unchanged' if nothing better was found, which does not show that none exists.",
            SimplifyToolArgs,
            run_simplify,
        ),
        Tool(
            "check_identity",
            'Is ONE equation the user stated an identity? For "is it true that A = B?". Returns proved equal, '
            "numerically equal (evidence), true only under a stated assumption, or not an identity with the "
            "point where the sides differ. Refuses definitions, equations of motion, matrix, vector and sum "
            "statements.",
            CheckIdentityArgs,
            run_check_identity,
        ),
        Tool(
            "substitute",
            "Replace the left side of the user's relation(s) `left = right` by the right side in an "
            "expression, step by step (for example x^2 + y^2 with y^2 = 1 - x^2 gives 1). The relations "
            "are the user's premises: Minerva does not check them, so a result that depends on them holds "
            "'given' them; say so, never 'proved'. If several relations give different results in "
            "different orders, or rewrite each other, the tool refuses until the user states the order.",
            SubstituteArgs,
            run_substitute,
        ),
        Tool(
            "rewrite_part",
            "Apply ONE move (a handbook identity by id, expand, factor, cancel, simplify, or collect terms "
            "by chosen variables) to one part of an expression and leave the rest alone; the whole "
            "expression is verified. Name the part in LaTeX, and the occurrence if it appears more than "
            "once (ask the user, never guess). An identity whose conditions are not provable from the "
            "assumptions is reported and not applied.",
            RewritePartArgs,
            run_rewrite_part,
        ),
        Tool(
            "isolate_variable",
            "Solve an equation for one variable: lists EVERY solution and the conditions each needs "
            "(for example a != 0), values the equation excludes, and parameter values where it degenerates. "
            "Covers polynomial and rational equations, and a variable that occurs once inside powers, "
            "roots, exp and log (a family with an integer k_1 where the logarithm is multi-valued). "
            "Each solution is substituted back and a separate check shows none was dropped. Periodic "
            "inverses (sin, cos), higher-degree symbolic polynomials and several occurrences of the "
            "variable in other forms are 'unsupported': report that, never a partial answer.",
            IsolateArgs,
            run_isolate,
        ),
        Tool(
            "differentiate",
            "Differentiate an expression symbolically with respect to one or more variables, to any "
            "order up to 10 (mixed partials too). Every step is checked against an independent "
            "numerical derivative. Reports 'unsupported' where no complex derivative exists (for "
            "example |x| for a complex x: declare x real).",
            DifferentiateArgs,
            run_differentiate,
        ),
        Tool(
            "series",
            "Taylor expansion of an expression in one variable about a finite real point, to order 1-12. "
            "The coefficients are re-derived by an independent Cauchy-integral estimate and the "
            "remainder order is measured; a series is shown only if both agree. Reports 'unsupported' "
            "for poles, branch points, logarithms and functions like |x| (no Laurent, Puiseux or "
            "asymptotic series). For an approximation to a physical system, expand its equations of motion, not the answer.",
            SeriesArgs,
            run_series,
        ),
        Tool(
            "integrate",
            "Integrate an expression in one variable. Without bounds: an antiderivative, shown only "
            "if an independent numerical derivative of it matches the integrand. With both bounds "
            "(numbers, or oo / -oo): a definite integral, shown only if independent numerical "
            "quadrature agrees at sampled parameter values. Reports 'conditions_not_satisfied' "
            "when convergence needs assumptions that were not declared (ask the user), and "
            "'unsupported' when no closed form is found (that does not prove none exists).",
            IntegrateArgs,
            run_integrate,
        ),
        Tool(
            "matrix_calculus",
            "Calculus with noncommuting matrices and their exponentials: evaluate (explicit entries), "
            "simplify, differentiate, integrate or evolve. Differentiate \\exp(X(t)) for noncommuting "
            "X and X' (the Frechet derivative integral, never assumed to equal X' exp(X)), including "
            "time-dependent matrices A(t); integrate products such as \\exp(wA)\\exp(-wB) and Duhamel "
            "integrals \\exp((t-s)A) B \\exp(sA); evolve dU/dt = A(t)U with U(0)=I (closed form "
            "exp of the integral only when the generator commutes with itself at all times; explicit "
            "entries on a span the user gives are solved numerically). Declare every matrix (a size, "
            "or explicit entries). The method is chosen from the structure and the result is shown "
            "only if independent checks pass. A Sylvester equation, a block-matrix or integral "
            "representation is NOT an evaluated result and a numerical solution is an approximation "
            "on its span, not a closed form: say so. Matrices are never assumed to commute or to be "
            "invertible; ask the user before declaring either.",
            MatrixCalculusArgs,
            run_matrix_calculus,
        ),
        Tool(
            "geometry",
            "Differential geometry in explicit coordinates: give the coordinates and the metric once and "
            "ask for the inverse metric, determinant, Christoffel symbols, Riemann, Ricci and Einstein "
            "tensors, the Ricci scalar, Kretschmann and Ricci-squared invariants, symbolic geodesic "
            "equations, and covariant derivatives or index raising/lowering of a tensor field. Conventions "
            "are fixed and returned. Every quantity is shown only if independent numerical checks pass "
            "(metric compatibility, curvature symmetries, Bianchi identities, Euler-Lagrange); a symbolic "
            "proof is reported separately. The signature is never assumed, a degenerate or asymmetric "
            "metric is refused, and where the coordinates break down is reported. The geometry is not "
            "remembered: resend it with every call.",
            GeometryArgs,
            run_geometry,
        ),
        Tool(
            "vector_calculus",
            "Vector calculus in Cartesian, cylindrical or spherical coordinates: gradient, divergence, curl, "
            "Laplacian (scalar and vector), dot, cross and triple products, directional derivatives, partial "
            "derivatives with respect to a coordinate or time, the Jacobian of a coordinate system, and "
            "conversion of explicit fields between systems, and line, flux and volume integrals with the "
            "gradient, divergence and Stokes theorems over a domain the USER gives (action integrate). "
            "Declare each field (scalar or vector, arbitrary "
            "or explicit) and each constant. A result is shown only if an independent Cartesian "
            "finite-difference evaluation agrees (numerically checked, not a proof). Nothing is assumed "
            "smooth: a result that needs mixed partial derivatives to commute is withheld until the USER "
            "states the smoothness. The origin and axes are reported as restrictions, never analysed. "
            "The declarations are not remembered: resend them with every call.",
            VectorCalculusArgs,
            run_vector_calculus,
        ),
        Tool(
            "electromagnetism",
            "Derivations in electromagnetism from field equations the USER adopts as premises (Maxwell's "
            "equations in SI, potentials, statics): the wave equations for E and B in vacuum, charge "
            "conservation, Poynting's theorem, gauge invariance of the fields, Poisson's and Laplace's "
            "equations. `setup` lists them with the premises each needs. Each derivation is a set of routes "
            "found by general vector-calculus steps and checked step by step; the result is evaluated on "
            "exact solutions of its premises. A result that used a premise is CONDITIONAL on it and is never "
            "called an identity, a proof or unconditional; only gauge invariance needs no premise. "
            "Minerva never checks that a premise is true. Cartesian coordinates, fields depend on x, y, z, t. "
            "Also electrostatics with numbers the USER gives: `field` (potential and E of point charges or a "
            "density, two methods must agree), `multipole` (moments and a measured far-field error), "
            "`boundary_value` (Laplace/Poisson on a rectangle by a series, a numerical approximation shown "
            "only if a finite-difference solve agrees) and `check_solution` (does a candidate potential "
            "satisfy the equation and the face values). Not available: Fourier/Laplace transforms, "
            "retarded Green's functions, FEM or FDTD, general PDE solving.",
            ElectromagnetismArgs,
            run_electromagnetism,
        ),
        Tool(
            "variation",
            "Variational calculus for a functional S = integral of L(t, x, x', x'', ...) dt of one or "
            "several functions, with derivatives of any order up to four and fixed bounds. action=first "
            "derives the first variation with every integration by parts listed, the Euler-Lagrange "
            "equations (sum of (-1)^k d^k/dt^k of dL/dx^(k)), the complete boundary form, which "
            "boundary terms the declared conditions remove, and the natural boundary conditions; "
            "action=second gives the second variation; action=analyze reports the structure and "
            "suggests next steps. Fixing a value does not fix its derivative: undeclared conditions "
            "are never assumed and their boundary terms are kept. Every result is shown only if finite "
            "differences of the action itself agree (numerical evidence, not a proof). Numerical steps "
            "(action values, finite-difference checks, stationary paths) need numbers that only the USER "
            "can choose and are run from the card's form, not by you; never choose an interval, an "
            "endpoint condition, a parameter value or a trial path. The functional is not remembered: "
            "resend it with every call.",
            VariationArgs,
            run_variation,
        ),
        Tool(
            "distribution",
            "Calculus with Dirac delta and Heaviside (step) functions: simplify a delta of a polynomial "
            "(scaling, translation, several simple roots, a continuous factor taking its value), integrate "
            "against deltas over given limits, or differentiate steps (d/dx of a step is a delta). Equality "
            "is as distributions, never pointwise. Each step cites a handbook identity with the conditions "
            "that were established; the result is shown only if independent test-function pairings agree. "
            "Declare the variable and every symbol in a delta or step real. Refuses (with a reason) repeated "
            "roots, unknown signs or zero scales, a delta on an integration limit without a convention the "
            "USER chose, products of distributions and derivatives of deltas. If the input uses \\delta(...) "
            "or \\theta(...) the card asks the user whether it is a Dirac delta or a step function and "
            "nothing is computed until they answer.",
            DistributionArgs,
            run_distribution,
        ),
        Tool(
            "solve_polynomial",
            "Solve a polynomial equation: factors it, says whether a formula in radicals exists "
            "(Galois criterion, Abel-Ruffini), gives exact roots where possible and verified numerical "
            "roots, and suggests handbook methods. Not for non-polynomial equations.",
            SolveArgs,
            run_solve,
        ),
        Tool(
            "analyze_lagrangian",
            "Given a Lagrangian and its generalised coordinates, derive the Euler-Lagrange equations, "
            "conjugate momenta and accelerations, and find verified constants of motion (cyclic "
            "coordinates, energy, rotations). States which symmetries were not searched for.",
            AnalyzeLagrangianArgs,
            run_lagrangian,
        ),
        Tool(
            "time_derivative",
            "The TOTAL time derivative of a quantity along a motion: coordinates, velocities and time all "
            "change (chain rule; accelerations appear). Use it for d/dt of a momentum, an energy, an "
            "angular momentum or any candidate constant, never `differentiate` (that is a partial "
            "derivative and holds velocities fixed). With a Lagrangian it also evaluates the result on "
            "its equations of motion and says whether it is zero there.",
            TimeDerivativeArgs,
            run_time_derivative,
        ),
        Tool(
            "expand_motion",
            "Expand the equations of motion of a Lagrangian about an equilibrium the user gives "
            "(default: every coordinate 0): the accelerations as a series in the deviations, up to a chosen "
            "degree (1 = linearised), the linear matrix, and for up to two coordinates without damping the "
            "normal-mode frequencies. Every expansion is verified independently of SymPy; a point that is "
            "not an equilibrium is refused. Use this to approximate, not to solve; the result is local.",
            ExpandMotionArgs,
            run_expand_motion,
        ),
        Tool(
            "perturbation_solve",
            "Perturbation solution (Poincare-Lindstedt) of a one-degree-of-freedom oscillator whose "
            "acceleration depends on the position only: the frequency shift, the period series and the "
            "periodic solution as a series in a small parameter (a named parameter of L, or the amplitude). "
            "Verified by putting the truncated solution into the unexpanded equation of motion and checking "
            "that the residual falls at the claimed order. Optionally compares with the verified numerical "
            "solution for numbers the user gives. Not for damping, driving or several coordinates.",
            PerturbationArgs,
            run_perturbation,
        ),
        Tool(
            "find_conserved_quantities",
            "Find the conserved quantities of a Lagrangian or Hamiltonian system without being told where "
            "to look: searches a deterministic ladder of larger and larger families (polynomials in the "
            "velocities or momenta times powers of the coordinates, cos and sin of angles, powers of t), "
            "solves the conservation condition exactly in each, verifies every constant on the equations "
            "of motion, removes products of what is known, and reports which are functionally independent "
            "(with exact relations such as |A|^2 in terms of the energy and angular momentum). Use this "
            "for 'find the conserved quantities', 'is there a hidden symmetry', 'more constants of "
            "motion'. Reports exactly which families were searched and which were not; finding nothing "
            "means nothing within them, never that the list is complete. Not for square roots (Cartesian Kepler); more than 3 coordinates is refused (nothing is searched).",
            DiscoveryArgs,
            run_discovery,
        ),
        Tool(
            "search_invariants",
            "Look for constants of motion of a Lagrangian beyond cyclic coordinates, energy and rotations "
            "(hidden symmetries such as the Laplace-Runge-Lenz vector): solves exactly for combinations "
            "of functions of the coordinates times velocity monomials, and shows only those verified on "
            "the equations of motion. Needs the coordinates and which of them are angles. Reports the "
            "exact family searched; finding none proves nothing outside it. Not for time-dependent "
            "Lagrangians or ones with square roots.",
            InvariantArgs,
            run_invariants,
        ),
        Tool(
            "integrate_motion",
            "Solve the equations of motion of a Lagrangian numerically for given parameter values, initial "
            "conditions and time span, optionally sweeping one parameter over 2-6 values. Every run is "
            "verified (second method, time reversal, constants of motion) and shown only if it passes. "
            "The numbers come from the USER: ask. Choose them yourself only if the user explicitly told "
            "you to choose; then say which values you used.",
            MotionArgs,
            run_motion,
        ),
        Tool(
            "ode_analyze",
            "Read an ordinary differential equation or system and describe it: the unknowns, the "
            "independent variable, order, linearity, constant coefficients, autonomy, whether it can be "
            "solved for its highest derivative, the equation class (separable, linear, exact, Bernoulli, "
            "homogeneous, Euler-Cauchy, ...), whether conditions make it an initial-value or "
            "boundary-value problem, and a RANKED list of approaches that can be tried, each with what "
            "it needs and its limits. Call this first for any differential equation and use ITS method "
            "names and order; do not classify the equation or choose a method yourself. It solves "
            "nothing. detail=qualitative also gives, for an autonomous system of at most three states, "
            "its equilibria (each confirmed), the Jacobian, eigenvalues and a classification that is "
            "LOCAL and from the linearisation only (never say 'globally stable'; a zero real part or "
            "unknown parameters leave it undecided), and with a `window` a direction field. The "
            "parameter values and the window come from the USER: ask, never choose. Ask the user "
            "which variable is independent if the equation does not say.",
            OdeAnalyzeArgs,
            run_ode_analyze,
        ),
        Tool(
            "ode_solve_symbolic",
            "Solve an ordinary differential equation or system analytically. method=closed_form (default): "
            "a formula from SymPy, accepted only if the equation as the user wrote it holds when it is put "
            "in (derivatives taken numerically); conditions are applied afterwards and checked separately. "
            "method=series: a Taylor polynomial about the point of the initial values, which the user must "
            "give in full; local, with no radius of convergence claimed. method=perturbation: an expansion "
            "in a small parameter the USER names (never choose it); with an `equilibrium` it is the "
            "Poincare-Lindstedt method for periodic motion about it (y'' = a(y) only), without one a "
            "regular expansion that needs every initial value and may show terms that grow with time. "
            "Every result says what it is and where it can be trusted. A failure means nothing was "
            "found, never that no solution exists.",
            OdeSymbolicArgs,
            run_ode_symbolic,
        ),
        Tool(
            "ode_solve_numerical",
            "Solve an ordinary differential equation or system numerically. kind=ivp (default): an "
            "initial-value problem; higher orders become first-order systems and the integrator "
            "(explicit or stiff) is chosen from measured behaviour. kind=bvp: conditions at two points, "
            "the interval is between them. A result is shown only if independent checks pass (a different "
            "integrator or shooting, refinement, the equation's residual, boundary values, a verified "
            "conserved quantity). It is an approximation on the interval given; a boundary-value "
            "solution is one of possibly several. Every number (conditions, parameter values, the end "
            "of the interval for ivp) comes from the USER: ask. Choose them yourself only if the user "
            "explicitly told you to choose; then say which values you used.",
            OdeNumericArgs,
            run_ode_numeric,
        ),
        Tool(
            "dimensional_analysis",
            "Dimensional analysis from the quantities of a problem (each with a unit or kind) and optionally "
            "the unknown: finds the independent dimensionless groups (Buckingham Pi), says whether the "
            "unknown is fixed by dimensions alone, and if its units cannot be built from the knowns, "
            "which dimension is missing and which handbook constants would supply it. Ask the user for "
            "any unit you are unsure of; never guess groups yourself.",
            DimensionalArgs,
            run_dimensional,
        ),
        Tool(
            "check_units",
            "Check an equation for dimensional consistency given the unit of each symbol: reports whether "
            "all terms and both sides agree, and the dimension of the factor missing when they do not.",
            UnitsCheckArgs,
            run_units_check,
        ),
        Tool(
            "find_identities",
            "Find handbook identities that match parts of an expression, check their validity conditions against the declared assumptions, and apply them with verification.",
            IdentitiesArgs,
            run_identities,
        ),
        Tool(
            "tensor_canonicalize",
            "Canonical form of a tensor expression using declared index symmetries (symmetric, antisymmetric, riemann). Applies monoterm symmetries only.",
            TensorArgs,
            run_tensor,
        ),
    ]
}


TOOLS["load_tools"] = Tool(
    "load_tools",
    "Load the tools of a group that is not in your tool list yet, when the user's problem needs them. "
    "Groups: " + "; ".join(f"{n} ({sm})" for n, sm in GROUP_SUMMARY.items()) + ". "
    "The tools are available from your next step.",
    LoadToolsArgs,
    run_load_tools,
)


def schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in TOOLS.values()]


# Examples of the usual wrong shapes. An example is shown only if it validates against the very field
# that failed (`_shape_hint`): the same field name has different shapes in different tools (`quantities`
# is a list of objects for the unit tools and a list of names for `geometry`), and a hint that teaches a
# key the schema does not have makes the model repeat the same wrong call.
_SHAPE_EXAMPLES: dict[str, Any] = {
    "assumptions": {"x": ["positive"]},
    "quantities": [{"name": "F", "unit": "N"}],
    "matrices": [{"name": "A", "rows": 2}],
    "variables": ["x", "y"],
    "conditions": ["y(0) = 1", {"unknown": "y", "order": 0, "at": "0", "value": "1"}],
    # `statistical_mechanics`: a system is an object with a `kind`, then that kind's own keys
    "system": {
        "kind": "levels",
        "levels": [{"energy": "0"}, {"energy": "epsilon", "degeneracy": "2"}],
    },
}
_SHAPE_MAX_KEYS = 8


def _shape_of(schema: dict[str, Any], defs: dict[str, Any], depth: int = 0) -> str:
    """The shape a JSON schema asks for, in words, with the schema's own key names."""
    if "$ref" in schema:
        schema = defs.get(str(schema["$ref"]).rsplit("/", 1)[-1], {})
    if "anyOf" in schema:
        parts = [_shape_of(s, defs, depth) for s in schema["anyOf"] if s.get("type") != "null"]
        return " or ".join(dict.fromkeys(parts))
    if "enum" in schema:
        return "one of " + ", ".join(f'"{v}"' for v in schema["enum"])
    if "oneOf" in schema and isinstance(schema.get("discriminator"), dict):
        return _tagged_shape(schema, defs, depth)
    kind = schema.get("type")
    if kind == "array":
        item = _shape_of(schema.get("items", {}), defs, depth)
        if item.startswith("an object with"):
            item = "objects, each with" + item[len("an object with") :]
        return "a list of " + item
    if kind == "object":
        props = schema.get("properties")
        if props and depth < 2:
            required = [k for k in props if k in set(schema.get("required", []))]
            others = [k for k in props if k not in required]
            keys = []
            for key in (required + others)[:_SHAPE_MAX_KEYS]:
                tag = ", required" if key in required else ""
                keys.append(f"{key} ({_shape_of(props[key], defs, depth + 1)}{tag})")
            more = len(props) - len(keys)
            tail = f", and {more} more" if more > 0 else ""
            return "an object with the keys " + ", ".join(keys) + tail
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            return "an object mapping names to " + _shape_of(extra, defs, depth)
        return "an object"
    words = {
        "string": "text",
        "integer": "an integer",
        "number": "a number",
        "boolean": "true or false",
    }
    return words.get(str(kind), "a value")


def _tagged_shape(schema: dict[str, Any], defs: dict[str, Any], depth: int) -> str:
    """A union chosen by a tag (`kind`): each tag with the keys that member needs, so a model that sent the
    wrong tag, or keys of another member, can see what each one is made of."""
    disc = schema["discriminator"]
    tag = str(disc.get("propertyName", "kind"))
    members = []
    for name, ref in (disc.get("mapping") or {}).items():
        member = defs.get(str(ref).rsplit("/", 1)[-1], {})
        props = member.get("properties") or {}
        required = [k for k in props if k in set(member.get("required", [])) and k != tag]
        optional = [k for k in props if k not in required and k != tag]
        keys = ", ".join(
            [f"{k} ({_shape_of(props[k], defs, depth + 1)}, required)" for k in required]
            + [f"{k} ({_shape_of(props[k], defs, depth + 1)})" for k in optional[:3]]
        )
        members.append(f'"{name}": {keys or "no other keys"}')
    return f'an object whose "{tag}" is one of these, each with its own keys: ' + "; ".join(members)


def _model_of(candidates: list[Any], value: dict[str, Any]) -> type[BaseModel] | None:
    """The model a dict is meant to be: the only candidate, or the one whose `kind` default or const matches."""
    models = [c for c in candidates if isinstance(c, type) and issubclass(c, BaseModel)]
    if len(models) == 1:
        return models[0]
    for tp in models:
        info = tp.model_fields.get("kind")
        if info is not None and value.get("kind") == info.default:
            return tp
    return None


def _unknown_keys(
    candidates: list[Any], value: Any, path: str, found: list[tuple[str, list[str], list[str]]]
) -> None:
    """Keys the model sent that the schema does not have, with the keys it does have (names from the
    schema; the unknown ones are the model's own and are shown back to it only)."""
    if isinstance(value, list):
        item_types = [
            a for c in candidates if get_origin(c) in (list, tuple) for a in get_args(c)[:1]
        ]
        for item in value[:3]:
            _unknown_keys(
                [t for it in item_types for t in _candidates(it)], item, path + "[]", found
            )
        return
    if not isinstance(value, dict):
        return
    tp = _model_of(candidates, value)
    if tp is None:
        return
    valid = [k for k in tp.model_fields if k != "kind"]
    extra = [str(k) for k in value if k not in tp.model_fields and k != "kind"]
    if extra:
        found.append((path or "(top level)", extra[:6], valid[:10]))
    for key, sub in value.items():
        info = tp.model_fields.get(str(key))
        if info is not None:
            _unknown_keys(
                _candidates(info.annotation), sub, f"{path}.{key}" if path else key, found
            )


def _unknown_key_note(model: type[BaseModel], arguments: dict[str, Any]) -> str:
    found: list[tuple[str, list[str], list[str]]] = []
    _unknown_keys([model], arguments, "", found)
    return "".join(
        f" Unknown key{'s' if len(extra) > 1 else ''} in {where}: {', '.join(f'`{k}`' for k in extra)}"
        f" (ignored; the keys there are {', '.join(f'`{k}`' for k in valid)})."
        for where, extra, valid in found[:2]
    )


def _focused_hint(model: type[BaseModel], field: str, value: Any) -> str | None:
    """For a tagged union whose tag the model got right: only that member's keys, not every member's."""
    if not isinstance(value, dict):
        return None
    schema = model.model_json_schema()
    prop = schema.get("properties", {}).get(field) or {}
    disc = prop.get("discriminator")
    if "oneOf" not in prop or not isinstance(disc, dict):
        return None
    tag = str(disc.get("propertyName", "kind"))
    ref = (disc.get("mapping") or {}).get(str(value.get(tag)))
    if ref is None:
        return None
    member = schema.get("$defs", {}).get(str(ref).rsplit("/", 1)[-1], {})
    props = {k: v for k, v in (member.get("properties") or {}).items() if k != tag}
    keys = ", ".join(
        f"{k} ({_shape_of(v, schema.get('$defs', {}), 1)}{', required' if k in member.get('required', []) else ''})"
        for k, v in props.items()
    )
    return f'an object with "{tag}": "{value[tag]}" and the keys {keys}'


def _shape_hint(model: type[BaseModel], field: str) -> str | None:
    """What `field` of `model` must look like, from the schema itself, plus an example that validates."""
    info = model.model_fields.get(field)
    if info is None:
        return None
    schema = model.model_json_schema()
    prop = schema.get("properties", {}).get(field)
    if prop is None:
        return None
    structured = '"$ref"' in json.dumps(prop) or '"properties"' in json.dumps(prop)
    example = _SHAPE_EXAMPLES.get(field)
    if example is not None and info.annotation is not None:
        try:
            TypeAdapter(info.annotation).validate_python(example)
        except ValidationError:
            example = None
    if example is None and not structured:
        return None
    described = _shape_of(prop, schema.get("$defs", {}))
    if example is None and described in ("a value", "an object"):
        return None  # nothing the model could act on
    return f"{described}, e.g. {json.dumps(example)}" if example is not None else described


def _coerce_numbers(model: type[BaseModel], arguments: dict[str, Any]) -> dict[str, Any]:
    """A number where the schema wants text (`point: 0`) is the same thing written differently."""
    fields = model.model_fields
    out = dict(arguments)
    for key, value in arguments.items():
        field_info = fields.get(key)
        if (
            field_info is not None
            and isinstance(value, int | float)
            and not isinstance(value, bool)
            and field_info.annotation in (str, str | None)
        ):
            out[key] = str(value)
    return out


_REASON_CODE = re.compile(r"[a-z][a-z_]{0,39}")
_UNION_NOISE = ("function-after", "function-before", "function-wrap", "json-or-python")


def _candidates(annotation: Any) -> list[Any]:
    """The concrete types an annotation allows (the members of a union, without `None`)."""
    if get_origin(annotation) in (Union, UnionType):
        return [a for a in get_args(annotation) if a is not type(None)]
    return [annotation]


def _step(candidates: list[Any], segment: Any) -> tuple[str | None, list[Any]] | None:
    """Follow one segment of a validation error's location through the argument model.

    Returns `(text to log or None, the types at the next level)`, or `None` when the segment cannot be
    placed in the schema. Only names of the model's own fields are ever logged: an index or a
    dictionary key (which can be a symbol the user chose) becomes `*`."""
    for tp in candidates:
        origin = get_origin(tp)
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            if isinstance(segment, str) and segment in tp.model_fields:
                return segment, _candidates(tp.model_fields[segment].annotation)
        elif origin in (list, set, frozenset, tuple):
            if isinstance(segment, int):
                inner = get_args(tp)
                return "*", _candidates(inner[0]) if inner else []
        elif origin is dict:
            inner = get_args(tp)
            return "*", _candidates(inner[1]) if len(inner) > 1 else []
    # a union member's own name (`OdeCondition`, `str`) or a validator wrapper is not a path segment
    names = {getattr(tp, "__name__", None) for tp in candidates}
    if isinstance(segment, str) and (segment in names or segment.startswith(_UNION_NOISE)):
        return None, candidates
    return None


def _safe_path(model: type[BaseModel], loc: tuple[Any, ...]) -> str:
    """`loc` as a schema path that cannot hold user data, cut with `*` where it cannot be placed."""
    parts: list[str] = []
    candidates: list[Any] = [model]
    for segment in loc:
        step = _step(candidates, segment)
        if step is None:
            parts.append("*")
            break
        label, candidates = step
        if label is not None:
            parts.append(label)
    return ".".join(parts) or "*"


def _validation_fields(
    exc: ValidationError, model: type[BaseModel]
) -> tuple[str | None, tuple[str, ...]]:
    """A fixed reason code and the schema paths of the failed fields, for the log."""
    code: str | None = None
    paths: list[str] = []
    for err in exc.errors():
        path = _safe_path(model, tuple(err["loc"]))
        if path not in paths:
            paths.append(path)
        kind = str(err.get("type", ""))
        if code is None and _REASON_CODE.fullmatch(kind):
            code = kind
    return code, tuple(paths[:6])


# Fixed advice appended when a field of a tool fails validation, for mistakes the benchmark showed repeating.
# Text only: nothing is rewritten, and no value of the call is quoted.
_FIELD_HINTS: dict[tuple[str, str], str] = {
    ("StatMechArgs", "quantities"): (
        " Heat capacity, pressure and other derivatives are not quantities: put them in `partials`, for "
        'example heat capacity is {"of": "U", "variable": "T", "held_fixed": [], "label": "C"}.'
    ),
}


def _validation_message(
    exc: ValidationError, model: type[BaseModel], arguments: dict[str, Any] | None = None
) -> str:
    """Every distinct problem (up to three) with the fix for the usual wrong shapes.

    A list of four conditions that all lack the same key is one problem, not four, and the error of
    the branch of a union (`OdeCondition | str`) that the value never tried to be is noise."""
    problems: dict[tuple[str, str], list[Any]] = {}
    for err in exc.errors():
        loc = [p for p in err["loc"] if p not in ("OdeCondition", "str")]
        given = err.get("input")
        union_noise = (err["loc"][-1:] == ("str",) and isinstance(given, dict)) or (
            "OdeCondition" in err["loc"] and not isinstance(given, dict)
        )
        if union_noise:
            continue
        where = ".".join("*" if isinstance(p, int) else str(p) for p in loc)
        entry = problems.setdefault((where, err["msg"]), [err, 0])
        entry[1] += 1
    shown = list(problems.items())[:3]
    parts: list[str] = []
    hints: list[str] = []
    for (where, msg), (err, count) in shown:
        first = ".".join(str(p) for p in err["loc"] if p not in ("OdeCondition", "str"))
        label = first if count == 1 else f"{where} ({count} items, e.g. {first})"
        parts.append(f"{label}: {msg}")
        top = str(err["loc"][0]) if err["loc"] else None
        hint = (
            _focused_hint(model, top, (arguments or {}).get(top)) if top and arguments else None
        ) or (_shape_hint(model, top) if top else None)
        if hint and hint not in hints:
            hints.append(hint)
    more = len(problems) - len(shown)
    text = "; ".join(parts) + (f"; and {more} more" if more > 0 else "")
    unknown = _unknown_key_note(model, arguments) if arguments else ""
    advice = "".join(
        dict.fromkeys(
            _FIELD_HINTS.get((model.__name__, str(err["loc"][0])), "")
            for _, (err, _) in shown
            if err["loc"]
        )
    )
    return (
        f"invalid arguments ({text})"
        + "".join(f" It must be {hint}." for hint in hints)
        + unknown
        + advice
    )


_BARE_E_POWER = re.compile(r"(?<![A-Za-z\\])e\s*\^")


# After a calculation that ran out of time or memory. The same call would run out again, and the
# model used to spend the rest of the turn on calls that answered a different question.
RESOURCE_LIMIT_HINT = (
    "hint: this calculation is too large for the limit; the same call will fail again, so do not "
    "repeat it. Only make another call if it is a smaller version of what the user asked for. "
    "Otherwise stop calling tools and tell the user that it could not be computed within the limit."
)
RESOURCE_LIMIT_HINTS = {
    "expand_motion": (
        "hint: this expansion is too large for the limit; the same call will fail again, so do not "
        "repeat it. If `order` was above 1 and the user asked for small oscillations or a "
        "linearisation, `order: 1` is the smaller call. Otherwise stop calling tools and tell the "
        "user that the expansion could not be computed within the limit. Do not replace it with "
        "an expansion in one variable unless that is what the user asked for."
    ),
}


def _add_hints(name: str, arguments: dict[str, Any], output: ToolOutput) -> None:
    """Fixed advice for the model on failures it keeps repeating. It only ever adds text for the
    model; the card and the input are unchanged, and nothing is reinterpreted."""
    text = " ".join(text for _, text in _argument_texts(arguments))
    hints: list[str] = []
    if output.status in ("unsupported", "unchanged", "parse_error") and _BARE_E_POWER.search(text):
        hints.append(
            "hint: `e` was read as an ordinary symbol, not Euler's number; write exponentials as "
            "\\exp(...) (a bare `e^x` is the symbol e to the power x)."
        )
    if output.status == "parse_error" and "reserved" in output.for_model:
        hints.append(
            "hint: E, I, pi and oo are reserved names; rename a symbol called E or I (for example "
            "En or Ik) and try again."
        )
    if name == "differentiate" and output.status == "parse_error" and "\\int" in text:
        hints.append(
            "hint: differentiating an integral with respect to a parameter is not supported. If the "
            "integrand is explicit, evaluate it with `integrate` first; an integral over an unknown "
            "function is a functional, and `variation` gives its Euler-Lagrange equation. Say what is "
            "not supported; do not carry out the differentiation yourself."
        )
    if name == "tensor_canonicalize" and output.status == "parse_error":
        hints.append(
            "hint: write each index as a call, e.g. S(a,b)*A(-a,-b), with - for a lower index; "
            "LaTeX subscripts such as S_{ab} are not read."
        )
    if (
        name == "integrate"
        and output.status in ("unsupported", "conditions_not_satisfied", "not_verified")
        and arguments.get("lower") is not None
        and arguments.get("upper") is not None
    ):
        hints.append(
            "hint: if this integrand depends on a parameter (besides the integration variable) and "
            "is a product of rational functions, powers and exponentials, `creative_telescoping` "
            "may derive a verified differential equation for the integral."
        )
    if output.status == "resource_limit":
        hints.append(RESOURCE_LIMIT_HINTS.get(name, RESOURCE_LIMIT_HINT))
    if hints:
        output.for_model += "\n" + "\n".join(hints)


# Tools whose engine treats every symbol as a commuting number: on a declared matrix they would give a
# plausible but wrong answer, which the loop would then mark as verified.
SCALAR_TOOLS = frozenset(
    {
        "simplify",
        "differentiate",
        "time_derivative",
        "series",
        "integrate",
        "isolate_variable",
        "solve_polynomial",
        "find_identities",
        # decides whether one stated equation holds by sampling commuting numbers: on `(A+B)^2 = A^2 + 2AB + B^2`
        # with A and B matrices it "proved" a false identity and the loop marked it verified (A-009)
        "check_identity",
        "distribution",
        "ode_analyze",
        "ode_solve_symbolic",
        "ode_solve_numerical",
        "variation",
        "creative_telescoping",
        "approximation",
    }
)
_EXPRESSION_FIELDS = MATH_FIELDS


# Every other tool, and why a declared matrix cannot make it show a wrong result as verified. A new tool must be in exactly one
# of `SCALAR_TOOLS`, `NONCOMMUTATIVE_AWARE_TOOLS`, `MATRIX_ONLY_TOOLS` or here, with a reason
# (`test_tool_matrix_classes.py`): the holes of A-009 were tools nobody had asked the question of.
MATRIX_SAFE: dict[str, str] = {
    "matrix_calculus": "the matrix tool itself: matrices are noncommutative symbols there",
    "vector_calculus": "typed vector fields; refuses what it cannot type",
    "check_units": "compares dimensions only; it makes no claim that two sides are equal",
    "dimensional_analysis": "dimensions of the declared quantities only",
    "expectation": "random variables the caller declares; a matrix is not one and undeclared symbols are refused",
    "statistical_model": "families and parameters typed by their own schema",
    "statistical_mechanics": "systems typed by their own schema; refuses what it cannot build",
    "geometry": "explicit coordinates and metric components; the quantities are numbers or functions of them",
    "electromagnetism": "its own typed charges, fields and premises",
    "tensor_canonicalize": "abstract-index tensors by index structure only, never by sampled values",
    "search_handbook": "retrieval; no mathematics is evaluated",
    "get_entry": "retrieval; no mathematics is evaluated",
    "load_tools": "widens the toolset; evaluates nothing",
}


# The fields that hold the system a quantity lives in, not the quantity itself. A Lagrangian in a vector's
# coordinates and components (`r`, `theta`) is the ordinary case, and the Lagrangian tools already answer it
# (`MATRIX_ONLY_TOOLS`); `time_derivative` refused the same Lagrangian for an expression without `r` in it.
_SYSTEM_FIELDS = frozenset({"lagrangian", "hamiltonian"})


def _argument_texts(arguments: dict[str, Any]) -> list[tuple[str, str]]:
    """(field, text) for the strings in a call that hold mathematics: expression fields, and for a
    differential equation its equations and written conditions."""
    texts = [(f, arguments[f]) for f in _EXPRESSION_FIELDS if isinstance(arguments.get(f), str)]
    for field_name in ("equations", "conditions"):
        value = arguments.get(field_name)
        if isinstance(value, list):
            texts.extend((field_name, item) for item in value if isinstance(item, str))
    return texts


# Tools that can work with noncommuting symbols, but only if the call declares them `noncommutative` in its
# assumptions (substitute, rewrite_part: polynomial expressions, exact). A matrix a conversation declared in words
# or through `matrix_calculus` is not that declaration: without it these tools read A and B as commuting numbers,
# expanded `(A+B)^2` to `A^2 + 2AB + B^2` and called it checked at 20/20 points (A-009).
NONCOMMUTATIVE_AWARE_TOOLS = frozenset({"substitute", "rewrite_part"})

# Tools that treat symbols as commuting numbers but are also used with a vector's coordinates and components
# (`r`, `r_i`): only a declared matrix makes them wrong. `sums` expanded `sum (A_i + B_i)^2` to
# `2 sum A_i B_i + sum A_i^2 + sum B_i^2` and called it agreed on explicit instances; the Lagrangian tools
# solved `A x'' + B x = 0` for `x'' = -B x / A` with a matrix `A` (A-009).
MATRIX_ONLY_TOOLS = frozenset(
    {
        "sums",
        "analyze_lagrangian",
        "expand_motion",
        "perturbation_solve",
        "find_conserved_quantities",
        "search_invariants",
        "integrate_motion",
    }
)


def _declared_noncommutative(arguments: dict[str, Any]) -> set[str]:
    declared = arguments.get("assumptions")
    if not isinstance(declared, dict):
        return set()
    return {
        str(n)
        for n, items in declared.items()
        if isinstance(items, list) and "noncommutative" in items
    }


def _matrix_refusal(name: str, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutput | None:
    aware = name in NONCOMMUTATIVE_AWARE_TOOLS
    matrix_only = name in MATRIX_ONLY_TOOLS
    if not (name in SCALAR_TOOLS or aware or matrix_only) or not ctx.matrices:
        return None
    texts = _argument_texts(arguments)
    declared_here = _declared_noncommutative(arguments) if aware else set()
    # Only a declared matrix makes a Lagrangian or Hamiltonian wrong; a declared vector blocks the
    # expression fields of a scalar tool and nothing else.
    matrices_only = ctx.matrices - ctx.vector_names
    hit = sorted(
        n
        for n in ctx.matrices
        if n not in declared_here
        and any(
            mentions_name(text, {n})
            for field_name, text in texts
            if n in matrices_only or not (matrix_only or field_name in _SYSTEM_FIELDS)
        )
    )
    if not hit:
        return None
    sources = "; ".join(
        f"{n} comes from {ctx.matrix_sources[n]}" for n in hit if n in ctx.matrix_sources
    )
    return _error(
        name,
        f"{', '.join(hit)} was declared as a matrix, operator or vector field in this conversation"
        + (f" ({sources})" if sources else "")
        + f"; `{name}` treats symbols as commuting numbers and would give a wrong answer for it"
        + (
            " unless the call declares the name `noncommutative` in its assumptions. "
            if aware
            else ". "
        )
        + "Use `matrix_calculus` for a matrix (it needs its size or entries), `vector_calculus` for a "
        "vector field, or say that this is not supported. If it is only a scalar, tell the user the "
        'name was read as a matrix or vector and that saying "<name> is a scalar" clears it (so does '
        "defining it as a norm, `r = |\\mathbf{r}|`); do not compute it yourself.",
        "unsupported",
    )


def execute(name: str, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutput:
    """Validate and run one tool call. Never raises: problems become an error result the model can read."""
    tool = TOOLS.get(name)
    if tool is None:
        return _error(
            name,
            f"unknown tool {name!r}. Call a tool by exactly its name, with no group or namespace "
            f"in front of it. Tools: {', '.join(TOOLS)}.",
            "unknown_tool",
            arguments_valid=False,
        )
    try:
        args = tool.args_model(**_coerce_numbers(tool.args_model, arguments))
    except ValidationError as exc:
        invalid = _error(
            name,
            _validation_message(exc, tool.args_model, arguments),
            "invalid_arguments",
            arguments_valid=False,
        )
        invalid.reason_code, invalid.invalid_fields = _validation_fields(exc, tool.args_model)
        return invalid
    except TypeError:
        return _error(name, "invalid arguments", "invalid_arguments", arguments_valid=False)
    refused = _matrix_refusal(name, arguments, ctx)
    if refused is not None:
        return refused
    try:
        output = tool.run(args, ctx)
    except ValidationError as exc:
        return _error(
            name,
            f"invalid arguments: {exc.errors()[0]['msg']}",
            "invalid_arguments",
            arguments_valid=False,
        )
    except (
        Exception
    ) as exc:  # a bug in one tool must not end the turn; the text is fixed, never the cause
        failed = _error(name, "the tool failed unexpectedly")
        failed.error_site = error_fields(exc)  # class and code location only, never the message
        return failed
    # What the engine established, read from its own result: the model has no say in it.
    with contextlib.suppress(Exception):
        output.evidence = evidence_for(name, output.status, output.card.data)
    category = category_for(output.status)
    output.error_category = category.value if category else None
    _classify_known(name, output)
    _add_hints(name, arguments, output)
    reading, diagnostic = input_record(output.card.data)
    lines, code = input_lines(output.status, reading, diagnostic)
    if lines:
        output.for_model = "\n".join([output.for_model, *lines])
    if code and output.reason_code is None:
        output.reason_code = code
    if ctx.input_notes:
        line = transport_line(ctx.input_notes)
        if line is not None:
            output.for_model = f"{output.for_model}\n{line}"
        output.card.data["input_cleaned"] = [n.model_dump(mode="json") for n in ctx.input_notes]
    return output


# Fields of a result that hold the user's own input as Minerva read it (not something it computed).
# `equation_latex` is an input for solve and check_units, but a computed result for matrix_calculus.
_INPUT_FIELDS = (
    "input_latex",
    "lagrangian_latex",
    "notation_latex",
    "metric_latex",
    "functional_latex",
)
_EQUATION_IS_INPUT = frozenset({"solve_polynomial", "check_units"})
QUOTED_INPUT = "your input, as read by Minerva"
QUOTED_STATED = "a handbook entry that is only stated, not machine-checked"
QUOTED_UNVERIFIED = "part of a result that was not verified"


def _walk(value: Any, depth: int = 0) -> Any:
    """Every dict inside a card's data (depth-limited)."""
    if depth > 6:
        return
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item, depth + 1)


def _classify_known(tool: str, output: ToolOutput) -> None:
    """Keep in `known_latex` only the formulas that a verified result produced.

    A formula the user typed (echoed as "input as read"), a handbook statement that is only `stated`, or
    anything from a result that was not accepted is *quoted*: the model may repeat it, but it is not shown
    as verified. Only the result's own fields decide; nothing is parsed from the model's words.
    """
    data = output.card.data
    inputs: set[str] = set()
    stated: set[str] = set()
    input_fields = (
        (*_INPUT_FIELDS, "equation_latex") if tool in _EQUATION_IS_INPUT else _INPUT_FIELDS
    )
    for node in _walk(data):
        for key in input_fields:
            value = node.get(key)
            if isinstance(value, str) and value:
                inputs.add(normalize(value))
        statement = node.get("statement_latex")
        if isinstance(statement, str) and node.get("verification") == "stated":
            stated.add(normalize(statement))
    accepted = output.evidence.accepted if output.evidence else False
    verified: list[str] = []
    quoted: dict[str, str] = {normalize(x): why for x, why in output.quoted_latex if x}
    for formula in output.known_latex:
        key = normalize(formula)
        if not key:
            continue
        if key in inputs:
            quoted.setdefault(key, QUOTED_INPUT)
        elif key in stated:
            quoted.setdefault(key, QUOTED_STATED)
        elif accepted and key not in quoted:
            verified.append(formula)
        else:
            quoted.setdefault(key, QUOTED_UNVERIFIED)
    output.known_latex = verified
    output.quoted_latex = list(quoted.items())  # (normalised formula, reason)


def check_arguments(name: str, arguments: dict[str, Any]) -> bool:
    """Would `execute` accept these arguments? Validation only: nothing is computed."""
    tool = TOOLS.get(name)
    if tool is None:
        return False
    try:
        tool.args_model(**_coerce_numbers(tool.args_model, arguments))
    except (ValidationError, TypeError):
        return False
    return True


def known_formulas(outputs: list[ToolOutput]) -> set[str]:
    """Normalised formulas that verified results (or machine-checked handbook entries) contain."""
    return {normalize(x) for o in outputs for x in o.known_latex if x}


def quoted_formulas(outputs: list[ToolOutput]) -> dict[str, str]:
    """Normalised formulas that appeared in tool output without being a verified result, with why."""
    quoted: dict[str, str] = {}
    for o in outputs:
        for key, why in o.quoted_latex:
            quoted.setdefault(key, why)
    return quoted


def describe_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False)[:400]


# The statistics tools live in their own module; they import the helpers above, so they are registered here.
from .stat_tools import (  # noqa: E402
    ExpectationArgs,
    StatModelArgs,
    SumsArgs,
    run_expectation,
    run_statistical_model,
    run_sums,
)

TOOLS["sums"] = Tool(
    "sums",
    "Finite sums and products with symbolic bounds: expand by linearity, closed forms of sums without "
    "data, factor a double sum, log of a product (only for provably positive factors), differentiate "
    "through a sum. Indexed data x_i inside \\sum are separate unknowns. A result is shown only if it "
    "agrees with the original on explicit expansions at several sizes (including n >= 2); it is NOT "
    "a proof for every n and must be worded 'checked for n = ...'. Bounds must be provably non-empty "
    "(declare n an integer and positive); infinite sums are refused.",
    SumsArgs,
    run_sums,
)
TOOLS["expectation"] = Tool(
    "expectation",
    "Expectation, variance and covariance of declared random variables: E[aX+bY+c], Var(X+Y) = "
    "Var X + Var Y + 2 Cov(X, Y), Cov(X, Y) = E[XY] - E[X]E[Y]. INDEPENDENCE IS NEVER ASSUMED: covariance "
    "terms stay unless the USER stated independence, and then the result HOLDS GIVEN it. Verified "
    "against explicit finite joint distributions (dependent unless independence was declared).",
    ExpectationArgs,
    run_expectation,
)
TOOLS["statistical_model"] = Tool(
    "statistical_model",
    "Probability models derived from their density: moments (mean, variance), log-likelihood and "
    "score, the maximum-likelihood estimator (with admissibility conditions, boundary cases refused, "
    "and the optimum called global only when concavity is proved), Fisher information, and the "
    "posterior for a prior (found by matching the kernel, never from a table). Ask the USER for the "
    "family, the known parameters, the sample size and WHETHER THE OBSERVATIONS ARE INDEPENDENT; "
    "never pick a prior, a parameter value or concrete data yourself. Refuses (with a reason) "
    "unsupported families, invalid parameters, missing independence, non-regular families such as "
    "Uniform(0, theta), moments that do not exist (Cauchy) and kernels outside the known families.",
    StatModelArgs,
    run_statistical_model,
)


from .statmech_tools import StatMechArgs, run_statistical_mechanics  # noqa: E402

TOOLS["statistical_mechanics"] = Tool(
    "statistical_mechanics",
    "Statistical mechanics of an explicit system in the canonical ensemble: a list of levels with "
    "degeneracies, a level family E(n) (finite or infinite), a density of states, or a separable classical "
    "Hamiltonian, optionally with N independent copies. Builds the partition function, derives free "
    "energy, internal energy, entropy, variance, heat capacity, pressure and any other partial derivative "
    "with the other state variables held fixed, and thermal averages, variances and cumulants. A formula is "
    "shown only if an independent enumeration or quadrature agreed; convergence of an infinite sum or "
    "integral is proved or the tool refuses (divergent, convergence_not_provable, not_separable). "
    "Conventions (phase-space cell, indistinguishable copies) and numbers for numerical evaluation belong "
    "to the USER: the tool asks on its card, never choose them. Also quantum systems given by an explicit "
    "Hermitian matrix (operators are never assumed to commute), the grand canonical ensemble for Bose, "
    "Fermi or classical-counting particles (chemical potential, particle-number fluctuations) and "
    "microcanonical counting of microstates. Also approximations that report assumptions, order, validity and a measured error: high- and "
    "low-temperature expansions, Stirling's ln N!, and a one-dimensional chain by the transfer matrix with "
    "its thermodynamic limit. Not supported: saddle-point and Laplace methods, mean field, Legendre "
    "transforms, interacting classical systems.",
    StatMechArgs,
    run_statistical_mechanics,
)


from .approx_tools import ApproximationArgs as ApproximationArgs  # noqa: E402
from .approx_tools import run_approximation  # noqa: E402
from .telescoping_tools import TelescopingArgs, run_creative_telescoping  # noqa: E402

TOOLS["creative_telescoping"] = Tool(
    "creative_telescoping",
    "A definite integral that depends on a parameter, I(a) = integral of f(x, a) dx, when `integrate` "
    "cannot give a verified value. SEARCHES (it does not look anything up) for a differential operator "
    "L in the parameter and a certificate G with L f = dG/dx, proves that identity exactly, checks "
    "convergence, differentiation under the integral sign and the boundary terms from the DECLARED "
    "assumptions only (declare the parameter real and positive when the problem says so), derives the "
    "differential equation of I(a) and solves it with independently verified initial values. "
    "Supported integrands: products of rational functions, powers (exponents free of x) and "
    "exponentials of rational functions, in the variable and the parameter. Reports a PARTIAL result "
    "(certificate or equation only) when the value cannot be established, and refuses sums of such "
    "terms, trigonometric functions, x**a, divergent integrals and poles inside the interval. A value "
    "is confirmed by numerical quadrature: call it 'checked numerically', never proved. Never invent "
    "an operator, certificate, equation or constant.",
    TelescopingArgs,
    run_creative_telescoping,
)


TOOLS["approximation"] = Tool(
    "approximation",
    "Approximate an expression or number by a simpler one in the regime the USER names: Taylor, Laurent (poles), "
    "asymptotic (large x), perturbation in a small parameter (also combinations such as v/c), several variables, "
    "leading behaviour, limits, Pade [m/n], Chebyshev, Legendre, Fourier (periodic), piecewise polynomials, "
    "continued fractions of a number, Euler-Maclaurin for a large sum, and a comparison of methods on an interval "
    "with an accuracy target. method `auto` chooses from what the user said. SymPy only proposes: a result is "
    "shown only if a check that does not use the route that built it agreed. Errors are measured on sampled points "
    "and are NOT bounds; an asymptotic expansion is not a convergent series. The interval, point, period, "
    "tolerance and degrees are the user's: pass them only if stated, otherwise the tool asks.",
    ApproximationArgs,
    run_approximation,
)
