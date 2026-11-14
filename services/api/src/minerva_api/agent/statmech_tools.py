"""Agent tool for statistical mechanics of explicit systems.

One tool with an `action`, not one tool per quantity. The model describes the system the USER gave and
chooses an action; the engine builds the partition function, derives thermodynamic quantities with an
explicit held-fixed set, and shows a formula only if an independent numerical evaluation agreed.

Four things are the user's and are NOT arguments of this tool, so the model cannot supply them: the
phase-space cell of a classical system, whether copies are indistinguishable, the numbers for a
numerical evaluation, and a sweep. They reach the engine only through `ToolContext.statmech_answers`,
which the direct-post route fills from a card's buttons and form.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

from minerva_engine import StatMechInput, StatMechResult
from minerva_engine.statmech.models import (
    BaseQuantity,
    Coupling,
    DosSystem,
    FamilySystem,
    LevelsSystem,
    OccupationSystem,
    PhaseVariable,
    QuantumSystem,
    Statistic,
)
from pydantic import BaseModel, Field

from .events import Card
from .tools import ToolContext, ToolOutput


class AgentClassical(BaseModel):
    """A classical Hamiltonian of one copy. The phase-space cell is the user's choice (not here)."""

    kind: Literal["classical"] = "classical"
    hamiltonian: str = Field(
        max_length=400,
        description="H as a sum of one-variable terms, e.g. p**2/(2*m) + m*omega**2*x**2/2. "
        "A term that couples two variables is refused (not separable).",
    )
    variables: list[PhaseVariable] = Field(
        min_length=1,
        max_length=8,
        description="Every phase-space variable with its domain, e.g. a box 0..L for a position and "
        "-oo..oo for a momentum. A position without a finite domain makes the integral diverge.",
    )


AgentSystem = Annotated[
    LevelsSystem | FamilySystem | DosSystem | AgentClassical | QuantumSystem | OccupationSystem,
    Field(discriminator="kind"),
]


class Partial(BaseModel):
    of: BaseQuantity = Field(description="Z, lnZ, F, U, S or Var_E")
    variable: str = Field(max_length=32, description="beta, T, or a declared parameter")
    held_fixed: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="EXACTLY the other state variables: the temperature variable and the declared "
        "parameters except the one differentiated. Anything else is refused.",
    )
    order: int = Field(default=1, ge=1, le=2)
    scale: str = Field(default="1", max_length=64, description="a factor, e.g. -1 for p = -dF/dV")
    label: str = Field(
        default="", max_length=64, description="the name of the result, e.g. C_V or p"
    )


class StatMechArgs(BaseModel):
    action: Literal["analyze", "partition", "derive", "expectation", "evaluate"] = Field(
        default="partition",
        description="analyze: which methods apply and which Minerva implements. partition: the "
        "partition function (and level probabilities). derive: Z, lnZ, F, U, S, Var_E and partial "
        "derivatives with a held-fixed set (heat capacity, pressure, susceptibility ...). expectation: "
        "moments, variance, cumulants of the energy or of an observable given per level. evaluate: "
        "numbers; it needs values only the USER can give, so it works through the card.",
    )
    system: AgentSystem = Field(
        description="levels: explicit energies and degeneracies (with optional observables per "
        "level). level_family: E(n), g(n) over an integer index, finite or infinite (oo). "
        "density_of_states: rho(En) over [lower, upper]; call the energy En, because E means Euler's "
        "number in text syntax. classical: a separable Hamiltonian over phase space. quantum: an "
        "explicit Hermitian matrix H (at most 4x4) and optional operator matrices; they are never "
        "assumed to commute, and Z = Tr exp(-beta H). occupation: independent particles in "
        "single-particle levels with a statistic (bose, fermi or boltzmann) for the grand canonical "
        "ensemble; Bose needs mu declared below every energy (for example mu negative, energies "
        "nonnegative)."
    )
    ensemble: Literal["canonical", "grand_canonical", "microcanonical"] = Field(
        default="canonical",
        description="canonical (fixed temperature, the default), grand_canonical (fixed temperature "
        "and chemical potential: needs an occupation system, and only that) or microcanonical (fixed "
        "total energy: needs levels, copies_count and total_energy). A system does not change its "
        "ensemble silently: a mismatch is refused.",
    )
    total_energy: str | None = Field(
        default=None,
        max_length=32,
        description="Microcanonical only: the fixed total energy of the N copies, a rational number "
        "the user gave.",
    )
    copies_count: str | None = Field(
        default=None,
        max_length=32,
        description="N independent, non-interacting copies of the system (an integer or an integer "
        "positive symbol). Whether they are indistinguishable is the USER's choice: do not guess; "
        "the tool asks.",
    )
    temperature: Literal["beta", "T"] = Field(
        default="beta",
        description="The variable in which results are written. beta and T are positive by "
        "definition; k_B is positive too.",
    )
    parameters: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Other state variables that can be differentiated and held fixed (V, h, ...).",
    )
    assumptions: dict[str, list[str]] = Field(
        default_factory=dict,
        description='Per symbol, e.g. {"epsilon": ["positive"], "N": ["integer", "positive"]}. '
        "Energies must be provably real; degeneracies provably positive integers.",
    )
    quantities: list[BaseQuantity] = Field(
        default_factory=list, max_length=6, description="Base quantities to derive."
    )
    partials: list[Partial] = Field(default_factory=list, max_length=6)
    observable: str = Field(default="E", max_length=32)
    statistic: Statistic = "mean"
    order: int = Field(default=1, ge=1, le=4)
    other_observable: str | None = Field(default=None, max_length=32)
    coupling: Coupling | None = Field(
        default=None,
        description="Declares that E_i = E_i(0) - h A_i for a parameter h and an observable A, which "
        "makes the cumulants of A derivatives of ln Z with respect to h.",
    )


def build_request(args: StatMechArgs, answers: dict[str, Any]) -> StatMechInput:
    """The engine request: the model's arguments plus what the USER answered (never the model)."""
    system = args.system.model_dump()
    if system["kind"] == "classical":
        system["phase_space_cell"] = answers.get("phase_space_cell")
    copies = None
    if args.copies_count:
        copies = {"count": args.copies_count, "identical": answers.get("identical")}
    return StatMechInput(
        ensemble=args.ensemble,
        total_energy=args.total_energy,
        system=system,
        copies=copies,
        temperature=args.temperature,
        parameters=args.parameters,
        assumptions=args.assumptions,
        action=args.action,
        quantities=list(args.quantities),
        partials=[p.model_dump() for p in args.partials],
        observable=args.observable,
        statistic=args.statistic,
        order=args.order,
        other_observable=args.other_observable,
        coupling=args.coupling.model_dump() if args.coupling else None,
        values=dict(answers.get("values") or {}),
        sweep=answers.get("sweep"),
    )


def _head(result: StatMechResult) -> str:
    parts = [f"status={result.status.value}", f"action={result.action}"]
    if result.reason:
        parts.append(f"reason={result.reason}")
    return "; ".join(parts) + ". "


def run_statistical_mechanics(args: StatMechArgs, ctx: ToolContext) -> ToolOutput:
    answers = dict(ctx.statmech_answers)
    request = build_request(args, answers)
    result = ctx.engine.statmech(request)
    lines = [_head(result) + result.message]
    for q in result.quantities:
        checks = ", ".join(
            f"{c.name}:{'ok' if c.passed else 'undecided' if c.passed is None else 'FAILED'}"
            for c in q.checks
        )
        fixed = f"; held fixed: {', '.join(q.held_fixed)}" if q.held_fixed else ""
        lines.append(f"{q.name} [{q.result_kind}] = {q.text}{fixed} (checks: {checks})")
    for c in result.conditions:
        lines.append(
            f"condition [{'established' if c.established else 'NOT established'}]: {c.how}"
        )
    for m in result.methods:
        status = "implemented" if m.implemented else "NOT implemented"
        lines.append(
            f"method {m.id}: {'applies' if m.applies else 'does not apply'}, {status}: {m.why}"
        )
    if result.not_supported:
        lines.append("not supported: " + "; ".join(result.not_supported))
    for note in result.notes:
        lines.append(f"note: {note}")
    if result.table is not None:
        lines.append(f"table: {len(result.table.rows)} rows over {result.table.variable}")
    if result.questions:
        lines.append(
            "ASK THE USER (do not answer these yourself; the card has the buttons and fields): "
            + " ".join(result.questions)
        )
    if result.verification.passed:
        lines.append(
            f"verification: {result.verification.points_agreed}/{result.verification.points_checked} "
            "points agreed; numerical agreement is evidence, not a proof. Say 'checked numerically', "
            "'proved' only where symbolic_zero is true."
        )
    inputs = [{"input_latex": s.latex} for s in result.steps if s.name == "levels" and s.latex]
    known: list[str] = [q.latex for q in result.quantities if q.latex]
    # the levels step is the user's own input: the classifier turns it into a quoted, not verified, formula
    known += [s.latex for s in result.steps if s.latex]
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="statistical_mechanics",
        type="statmech",
        title="Statistical mechanics",
        data={
            "result": result.model_dump(mode="json"),
            "args": args.model_dump(mode="json"),
            "answers": answers,
            "inputs": inputs,
        },
    )
    return ToolOutput(card, "\n".join(lines), known, result.status.value)
