"""Tools the agent may call. Each is a pure, sandboxed computation or a read-only handbook lookup.

A tool returns two things: a `Card` (rendered by Minerva from the tool's own output, never by the
model) and a compact `for_model` text (what the model gets to read). Tools have no side effects, so a
confused or manipulated model can at worst produce a poor explanation.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from minerva_engine import (
    CompareRequest,
    DiffVariable,
    Engine,
    ExpressionInput,
    MotionRequest,
    QuantitySpec,
    TensorInput,
    TensorSpec,
)
from minerva_knowledge import KnowledgeBase, SearchQuery
from minerva_knowledge.identities import (
    TransformResponse,
    constant_candidates,
    find_identities,
    simplify_expression,
    summarize,
)
from pydantic import BaseModel, Field, ValidationError

from .events import Card
from .segments import normalize

MAX_INPUT = 2000


@dataclass
class ToolContext:
    engine: Engine
    kb: KnowledgeBase


@dataclass
class ToolOutput:
    card: Card
    for_model: str
    known_latex: list[str] = field(default_factory=list)
    status: str = "ok"  # the tool's own status, for the debug trace


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


def _error(tool: str, message: str) -> ToolOutput:
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool=tool,
        type="error",
        title="Could not run",
        data={"message": message},
    )
    return ToolOutput(card=card, for_model=f"error: {message}", status="error")


# --- tool implementations ----------------------------------------------------------------------
MAX_STEPS_FOR_MODEL = 6


def _chain_lines(result: TransformResponse) -> tuple[list[str], list[str]]:
    """What the model may quote about a chain: each step's identity, forms and evidence."""
    lines: list[str] = []
    known: list[str] = []
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
            known.append(entry.statement_latex)
    for alt in result.alternatives:
        lines.append(f"- other form: ${alt.result_latex}$ ({len(alt.steps)} steps)")
        known.append(alt.result_latex)
    return lines, known


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
    lines, known_steps = _chain_lines(result)
    known = [x for x in (result.input_latex, result.result_latex) if x] + known_steps
    return ToolOutput(card, "\n".join([head, *lines]), known, result.status.value)


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
    return ToolOutput(card, text, known)


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
        known.append(c.expression_latex)
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
    return ToolOutput(card, "\n".join(lines), known, report.status.value)


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
    if report.equilibrium:
        lines.append(
            "equilibrium: " + ", ".join(f"{e.name} = ${e.latex}$" for e in report.equilibrium)
        )
    for e in report.equations:
        ok = e.verification is not None and e.verification.passed
        lines.append(
            f"expansion ({e.coordinate}, {'verified' if ok else 'NOT verified'}): ${e.full_latex}$"
        )
        known.append(e.full_latex)
        known.extend(t.latex for t in e.terms)
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
    return ToolOutput(card, "\n".join(lines), known, report.status.value)


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


def _evidence_clause(result: Any) -> str:
    v = result.verification
    if v is None:
        return ""
    return (
        f"checked numerically at {v.numeric_points_agreed}/{v.numeric_points_checked} points; "
        f"symbolic_zero={v.symbolic_zero}"
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
    head = (
        f"status={result.status.value}; input_as_read=${result.input_latex}$; action={args.action}"
    )
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
        return _error(name, f"unknown tool {name!r}; available: {', '.join(TOOLS)}")
    try:
        args = tool.args_model(**arguments)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        return _error(name, f"invalid arguments ({where}): {first['msg']}")
    except TypeError:
        return _error(name, "invalid arguments")
    try:
        return tool.run(args, ctx)
    except ValidationError as exc:
        return _error(name, f"invalid arguments: {exc.errors()[0]['msg']}")
    except Exception:  # a bug in one tool must not end the turn; the text is fixed, never the cause
        return _error(name, "the tool failed unexpectedly")


def known_formulas(outputs: list[ToolOutput]) -> set[str]:
    return {normalize(x) for o in outputs for x in o.known_latex if x}


def describe_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False)[:400]
