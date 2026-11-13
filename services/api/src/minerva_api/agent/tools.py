"""Tools the agent may call. Each is a pure, sandboxed computation or a read-only handbook lookup.

A tool returns two things: a `Card` (rendered by Minerva from the tool's own output, never by the
model) and a compact `for_model` text (what the model gets to read). Tools have no side effects, so a
confused or manipulated model can at worst produce a poor explanation.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from minerva_engine import (
    CompareRequest,
    DiffVariable,
    DistributionInput,
    Engine,
    ExpressionInput,
    GeometryInput,
    MatrixCalculusInput,
    MatrixDecl,
    MotionRequest,
    Pins,
    QuantitySpec,
    TensorInput,
    TensorSpec,
)
from minerva_engine.geometry import QUANTITIES, FunctionDecl, TensorField
from minerva_knowledge import KnowledgeBase, SearchQuery
from minerva_knowledge.identities import (
    TransformResponse,
    constant_candidates,
    find_identities,
    simplify_expression,
    summarize,
)
from pydantic import BaseModel, Field, ValidationError, model_validator

from ..diagnostics import log_distribution
from ..geometry_log import log_geometry
from ..search_log import log_search_events
from .events import Card, Evidence
from .evidence import evidence_for
from .segments import normalize
from .tracing import category_for

MAX_INPUT = 2000


@dataclass
class ToolContext:
    engine: Engine
    kb: KnowledgeBase


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


class DifferentiateArgs(SimplifyArgs):
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
        description="Expression in the declared matrices. Write exponentials as \\exp(tA), never as e^{tA} "
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
    action: Literal["evaluate", "simplify", "differentiate", "integrate", "evolve"] = Field(
        default="simplify",
        description="evaluate: explicit matrices only. simplify: merge what is provably mergeable. "
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
        description="The metric components g_{mu nu} as an n by n grid of LaTeX strings, one row per "
        'coordinate, e.g. [["-(1-2M/r)", "0"], ["0", "(1-2M/r)^{-1}"]]. Use \\sin^2\\theta, '
        "not \\sin 2\\theta. Ask the user for the metric; never invent one.",
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
        "covariant_derivative, raise_index, lower_index. The last three need `tensor`.",
    )
    tensor: TensorField | None = Field(
        default=None,
        description="A tensor field given by variance (up/down per index) and its n**rank components, "
        "row-major, as LaTeX strings. Only for covariant_derivative, raise_index and lower_index.",
    )
    tensor_index: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Which index (0-based) raise_index or lower_index acts on",
    )


class DistributionArgs(SimplifyArgs):
    latex: str = Field(
        max_length=MAX_INPUT,
        description="Expression with Dirac delta or Heaviside functions, written with explicit names: "
        "\\operatorname{DiracDelta}(x-a) and \\operatorname{Heaviside}(x). \\delta(x) and \\theta(x) are "
        "NOT read as distributions. For integrate give only the integrand (no \\int, no dx).",
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
    latex: str = Field(
        max_length=MAX_INPUT, description="The integrand, in LaTeX, e.g. x\\cos x (no \\int or dx)"
    )
    variable: str = Field(
        min_length=1, max_length=32, description="The variable of integration, e.g. x"
    )
    lower: str | None = Field(
        default=None,
        max_length=200,
        description="Lower bound of a definite integral: a real number such as 0 or \\pi, or -oo "
        "for minus infinity. Give both bounds or neither.",
    )
    upper: str | None = Field(
        default=None,
        max_length=200,
        description="Upper bound of a definite integral: a real number, or oo for infinity.",
    )


class IdentitiesArgs(SimplifyArgs):
    query: str = Field(default="", max_length=200, description="Optional extra search words")


class TensorArgs(BaseModel):
    expression: str = Field(
        max_length=MAX_INPUT, description="e.g. S(a,b)*A(-a,-b); a leading - lowers an index"
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
    time: str = Field(default="t", max_length=16, description="Name of the time symbol")
    assumptions: dict[str, list[str]] = Field(default_factory=dict)


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
    time: str = Field(default="t", max_length=16, description="Name of the time symbol")
    assumptions: dict[str, list[str]] = Field(default_factory=dict)


class DiscoveryArgs(BaseModel):
    lagrangian: str | None = Field(
        default=None,
        max_length=MAX_INPUT,
        description=r"L(q, qdot) in LaTeX, using \dot{q} for velocities. Give exactly one of `lagrangian` and `hamiltonian`.",
    )
    hamiltonian: str | None = Field(
        default=None,
        max_length=MAX_INPUT,
        description="H(q, p) in LaTeX, with the canonical momenta as symbols such as p_x or p_\\theta.",
    )
    coordinates: list[str] = Field(
        min_length=1,
        max_length=3,
        description="Names of the generalised coordinates (at most 3), e.g. ['r', 'theta']",
    )
    momenta: list[str] | None = Field(
        default=None,
        max_length=3,
        description="Hamiltonian only: the momentum symbols in the order of the coordinates. Default p_<coordinate>.",
    )
    angles: list[str] = Field(
        default_factory=list,
        max_length=3,
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
    time: str = Field(default="t", max_length=16, description="Name of the time symbol")
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
        min_length=1, max_length=16, description="Unit or kind of every symbol in the equation."
    )


class SearchArgs(BaseModel):
    query: str = Field(max_length=200, description="Words describing the formula or method")
    chapter: str | None = Field(default=None, max_length=64)
    kind: str | None = Field(default=None, max_length=32)
    limit: int = Field(default=6, ge=1, le=10)


class EntryArgs(BaseModel):
    id: str = Field(max_length=80, description="Handbook entry id, e.g. trig.pythagorean")


def _expr(args: SimplifyArgs) -> ExpressionInput:
    return ExpressionInput(input=args.latex, format="latex", assumptions=args.assumptions)


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
            f"points; symbolic_zero={v.symbolic_zero}"
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
        data=result.model_dump(mode="json"),
    )
    v = result.verification
    head = (
        f"status={result.status.value}; goal={args.goal}; input_as_read=${result.input_latex}$; "
        + (
            f"result=${result.result_latex}$; "
            if result.status.value == "ok"
            else "result=none (the input is unchanged: no rewrite was applied, so there is no "
            "simplified form to report); "
        )
        + (
            f"end to end: numeric {v.numeric_points_agreed}/{v.numeric_points_checked}, "
            f"symbolic_zero={v.symbolic_zero} "
            f"({'symbolic and numeric evidence' if v.symbolic_zero else 'numeric evidence only, not a proof'}); "
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
    return ToolOutput(
        card, "\n".join([head, *lines]), known, result.status.value, quoted_latex=quoted
    )


def run_differentiate(args: DifferentiateArgs, ctx: ToolContext) -> ToolOutput:
    result = ctx.engine.differentiate(_expr(args), args.variables)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="differentiate",
        type="differentiate",
        title="Differentiate",
        data=result.model_dump(mode="json"),
    )
    v = result.verification
    text = (
        f"status={result.status.value}; input_as_read=${result.input_latex}$; "
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


def run_series(args: SeriesArgs, ctx: ToolContext) -> ToolOutput:
    result = ctx.engine.series(_expr(args), args.variable, args.point, args.order)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="series",
        type="series",
        title="Series",
        data=result.model_dump(mode="json"),
    )
    v = result.verification
    text = (
        f"status={result.status.value}; input_as_read=${result.input_latex}$; "
        f"about {result.variable}=${result.point_latex}$, order {result.order}; "
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
    return ToolOutput(card, text, known, result.status.value)


def run_integrate(args: IntegrateArgs, ctx: ToolContext) -> ToolOutput:
    result = ctx.engine.integrate(_expr(args), args.variable, args.lower, args.upper)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="integrate",
        type="integrate",
        title="Integrate",
        data=result.model_dump(mode="json"),
    )
    v = result.verification
    text = (
        f"status={result.status.value}; input_as_read=${result.input_latex}$; "
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
    )
    known = [
        x for x in (result.input_latex, result.notation_latex, result.result_latex) if x
    ] + result.conditions_latex
    return ToolOutput(card, text, known, result.status.value)


def geometry_request(args: GeometryArgs) -> GeometryInput:
    return GeometryInput(**args.model_dump(), format="latex")


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


def run_matrix_calculus(args: MatrixCalculusArgs, ctx: ToolContext) -> ToolOutput:
    request = MatrixCalculusInput(
        expression=_expr(args),
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
        f"status={result.status.value}; action={result.action}; input_as_read=${result.input_latex}$; "
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
            + "; "
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
            "request": request.model_dump(mode="json", exclude={"expression"}),
        },
    )
    shown = result.status.value == "ok"
    head = f"status={result.status.value}; action={result.action}; "
    if result.reason:
        head += f"reason={result.reason}; "
    if result.input_latex:
        head += f"input_as_read=${result.input_latex}$; "
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
    lines = [
        f"status={result.status.value}; input_as_read=${result.input_latex}$; {result.message}"
    ]
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


def run_lagrangian(args: LagrangianArgs, ctx: ToolContext) -> ToolOutput:
    report = ctx.engine.analyze_lagrangian(
        args.lagrangian, args.coordinates, args.time, args.assumptions
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
            # What was analysed, so the card's numerical-solution form can resend exactly this.
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
    for e in report.equations:
        lines.append(f"equation of motion ({e.coordinate}): ${e.latex}$")
        known.append(e.latex)
    for m in report.momenta:
        lines.append(f"momentum: ${m.latex}$")
        known.append(m.latex)
    for a in report.accelerations:
        lines.append(f"acceleration: ${a.latex}$")
        known.append(a.latex)
    for c in report.conserved:
        flag = "verified on the equations of motion" if c.verified else "NOT verified"
        lines.append(f"conserved ({c.kind}, {flag}): {c.name}: ${c.expression_latex}$; {c.reason}")
        if c.verified:
            known.append(c.expression_latex)
        else:
            quoted.append((c.expression_latex, NOT_VERIFIED))
    lines.extend(f"tested, not found: {t}" for t in report.tested)
    lines.extend(f"note: {n}" for n in report.notes)
    if report.accelerations:
        needed = ", ".join(report.parameters) or "none"
        lines.append(
            "next step: integrate_motion solves these numerically and verifies the run. Ask the user for "
            f"values of the parameters ({needed}), the initial positions and velocities and the time span; "
            "do not choose them. One parameter may be swept over 2-6 values. The card has a form for this."
        )
    if method_data:
        lines.append(
            "handbook methods: "
            + "; ".join(f"{m['id']} ({m['title']}, {m['verification']})" for m in method_data)
        )
        known.extend(m["statement_latex"] for m in method_data)
    return ToolOutput(card, "\n".join(lines), known, report.status.value, quoted_latex=quoted)


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
    report = ctx.engine.discover_invariants(
        args.hamiltonian if args.hamiltonian is not None else str(args.lagrangian),
        args.coordinates,
        system,
        args.momenta,
        args.angles,
        args.budget,
        pins,
        args.time,
        args.assumptions,
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
    for c in report.constants:
        label = "already known" if c.kind == "known" else "new, verified on the equations of motion"
        lines.append(f"{c.id} ({label}; {c.independence}): ${c.expression_latex}$")
        known.append(c.expression_latex)
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
    return ToolOutput(card, "\n".join(lines), known, report.status.value)


def run_invariants(args: InvariantArgs, ctx: ToolContext) -> ToolOutput:
    report = ctx.engine.search_invariants(
        args.lagrangian,
        args.coordinates,
        args.angles,
        args.degree,
        args.power_bound,
        args.harmonics,
        args.time,
        args.assumptions,
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
    if report.ansatz:
        lines.append(f"searched: {report.ansatz}")
    if report.known:
        lines.append("already known and factored out: " + "; ".join(report.known))
    for c in report.found:
        lines.append(
            f"new constant of motion (verified on the equations of motion): ${c.expression_latex}$"
        )
        known.append(c.expression_latex)
    lines.extend(f"note: {n}" for n in report.notes)
    if report.found:
        lines.append(
            "These are constants in the searched family only. Naming them (for example as the "
            "Laplace-Runge-Lenz vector) is a recognition you may offer, not something this result proves."
        )
    return ToolOutput(card, "\n".join(lines), known, report.status.value)


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


def run_motion(args: MotionArgs, ctx: ToolContext) -> ToolOutput:
    report = ctx.engine.integrate_motion(args)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="integrate_motion",
        type="motion",
        title="Numerical solution",
        data={"report": report.model_dump(mode="json"), "request": args.model_dump(mode="json")},
    )
    lines = [f"status={report.status.value}; {report.message}"]
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


def run_units_check(args: UnitsCheckArgs, ctx: ToolContext) -> ToolOutput:
    report = ctx.engine.check_units(args.equation, args.quantities)
    methods = ctx.kb.search(
        SearchQuery(features=["dimensional:units-check"], chapter="physics.methods", limit=3)
    )
    method_data = [summarize(ctx.kb, e).model_dump(mode="json") for e in methods]
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="check_units",
        type="units",
        title="Unit check",
        data={"report": report.model_dump(mode="json"), "methods": method_data},
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
        return ToolOutput(card, "no handbook entries matched; try other words", [], "no_results")
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
            "distribution",
            "Calculus with Dirac delta and Heaviside (step) functions: simplify a delta of a polynomial "
            "(scaling, translation, several simple roots, a continuous factor taking its value), integrate "
            "against deltas over given limits, or differentiate steps (d/dx of a step is a delta). Equality "
            "is as distributions, never pointwise. Each step cites a handbook identity with the conditions "
            "that were established; the result is shown only if independent test-function pairings agree. "
            "Declare the variable and every symbol in a delta or step real. Refuses (with a reason) repeated "
            "roots, unknown signs or zero scales, a delta on an integration limit without a convention the "
            "USER chose, products of distributions and derivatives of deltas.",
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
            LagrangianArgs,
            run_lagrangian,
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
            "means nothing within them. Not for square roots (Cartesian Kepler) or more than 3 coordinates.",
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
            "Never choose the numbers yourself: ask the user.",
            MotionArgs,
            run_motion,
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


def schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in TOOLS.values()]


def execute(name: str, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutput:
    """Validate and run one tool call. Never raises: problems become an error result the model can read."""
    tool = TOOLS.get(name)
    if tool is None:
        return _error(
            name,
            f"unknown tool {name!r}; available: {', '.join(TOOLS)}",
            "unknown_tool",
            arguments_valid=False,
        )
    try:
        args = tool.args_model(**arguments)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        return _error(
            name,
            f"invalid arguments ({where}): {first['msg']}",
            "invalid_arguments",
            arguments_valid=False,
        )
    except TypeError:
        return _error(name, "invalid arguments", "invalid_arguments", arguments_valid=False)
    try:
        output = tool.run(args, ctx)
    except ValidationError as exc:
        return _error(
            name,
            f"invalid arguments: {exc.errors()[0]['msg']}",
            "invalid_arguments",
            arguments_valid=False,
        )
    except Exception:  # a bug in one tool must not end the turn; the text is fixed, never the cause
        return _error(name, "the tool failed unexpectedly")
    # What the engine established, read from its own result: the model has no say in it.
    with contextlib.suppress(Exception):
        output.evidence = evidence_for(name, output.status, output.card.data)
    category = category_for(output.status)
    output.error_category = category.value if category else None
    _classify_known(name, output)
    return output


# Fields of a result that hold the user's own input as Minerva read it (not something it computed).
# `equation_latex` is an input for solve and check_units, but a computed result for matrix_calculus.
_INPUT_FIELDS = ("input_latex", "lagrangian_latex", "notation_latex", "metric_latex")
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
        tool.args_model(**arguments)
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
