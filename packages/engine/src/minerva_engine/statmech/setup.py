"""Parse and validate a statistical-mechanics problem.

Everything the later stages rely on is decided here, with an enumerated reason when it fails:
degeneracies must be provably positive integers, energies and observables provably real, the
temperature symbols are positive by definition and must not appear in the energies, and the
conventions only the user can choose (phase-space cell, indistinguishability) must be given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from ..distributions.models import Condition
from ..errors import MinervaMathError
from ..models import ExpressionInput, Limits
from ..ops import parse_expression
from .models import (
    MAX_LEVELS,
    LevelsSystem,
    Reason,
    StatMechInput,
    StatStep,
)


class Refusal(MinervaMathError):
    """The problem cannot be treated; ``reason`` is one of the enumerated codes."""

    def __init__(self, reason: Reason, message: str, questions: list[str] | None = None) -> None:
        super().__init__(message)
        self.reason: Reason = reason
        self.questions: list[str] = questions or []


@dataclass
class Problem:
    request: StatMechInput
    limits: Limits
    assumptions: dict[str, list[str]]
    beta: sp.Symbol
    T: sp.Symbol
    k_B: sp.Symbol
    params: list[sp.Symbol]
    display_var: sp.Symbol
    kind: str = "levels"
    energies: list[sp.Expr] = field(default_factory=list)
    degeneracies: list[sp.Expr] = field(default_factory=list)
    observables: dict[str, list[sp.Expr]] = field(default_factory=dict)  # levels only; "E" first
    # Closed-form partition function of ONE copy (levels build it from the sum themselves).
    Z1: sp.Expr | None = None
    copies_count: sp.Expr | None = None
    identical: bool = False
    sample_symbols: list[sp.Symbol] = field(default_factory=list)
    setup_steps: list[StatStep] = field(default_factory=list)
    conditions: list[Condition] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)  # kind-specific data for the oracle
    # Symbolic moments of E (one copy), when the system supplies them: used for the symbolic leg.
    moments1: list[sp.Expr] | None = None

    @property
    def state_variables(self) -> list[sp.Symbol]:
        return [self.display_var, *self.params]

    def parse(self, text: str) -> sp.Expr:
        return _parse(self.request, self.limits, self.assumptions, text, self.request.format)

    def parse_text(self, text: str) -> sp.Expr:
        return _parse(self.request, self.limits, self.assumptions, text, "text")


def _parse(
    request: StatMechInput,
    limits: Limits,
    assumptions: dict[str, list[str]],
    text: str,
    fmt: str,
) -> sp.Expr:
    data = ExpressionInput(input=text, format=fmt, assumptions=assumptions)
    try:
        return parse_expression(data, limits)
    except Refusal:
        raise
    except MinervaMathError as exc:
        raise Refusal("parse_error", str(exc)) from exc


def _positive(assumptions: dict[str, list[str]], name: str) -> None:
    current = list(assumptions.get(name, []))
    if "positive" not in current:
        current.append("positive")
    assumptions[name] = current


def build_problem(request: StatMechInput, limits: Limits) -> Problem:
    names = [request.beta, request.T, request.k_B, *request.parameters]
    if len(set(names)) != len(names):
        raise Refusal("parse_error", "the temperature, k_B and parameter names must be distinct")

    assumptions = {k: list(v) for k, v in request.assumptions.items()}
    for name in (request.beta, request.T, request.k_B):
        _positive(assumptions, name)

    def parse(text: str) -> sp.Expr:
        return _parse(request, limits, assumptions, text, request.format)

    def symbol(text: str) -> sp.Symbol:
        expr = parse(text)
        if not isinstance(expr, sp.Symbol):
            raise Refusal(
                "parse_error", "temperature, k_B and parameter names must be plain symbols"
            )
        return expr

    beta, T, k_B = symbol(request.beta), symbol(request.T), symbol(request.k_B)
    params = [symbol(p) for p in request.parameters]
    problem = Problem(
        request=request,
        limits=limits,
        assumptions=assumptions,
        beta=beta,
        T=T,
        k_B=k_B,
        params=params,
        display_var=beta if request.temperature == "beta" else T,
    )
    system = request.system
    problem.kind = system.kind
    free: set[sp.Symbol] = set(params)
    if isinstance(system, LevelsSystem):
        free |= _fill_levels(problem, system, parse)
    elif system.kind == "level_family":
        from .systems import fill_family

        free |= fill_family(problem, system)
    elif system.kind == "density_of_states":
        from .systems import fill_dos

        free |= fill_dos(problem, system)
    elif system.kind == "quantum":
        from .quantum import fill_quantum

        free |= fill_quantum(problem, system)
    else:
        from .systems import fill_classical

        free |= fill_classical(problem, system)
    free |= _fill_copies(problem, parse)
    problem.sample_symbols = sorted(free | {beta, k_B}, key=lambda s: s.name)
    return problem


def _fill_copies(problem: Problem, parse: Any) -> set[sp.Symbol]:
    copies = problem.request.copies
    if copies is None:
        return set()
    count = parse(copies.count)
    if count.is_integer is not True or count.is_positive is not True:
        raise Refusal(
            "assumptions_insufficient",
            "the number of copies must be a positive integer (declare a symbolic one integer and positive)",
        )
    if copies.identical is None:
        raise Refusal(
            "convention_unconfirmed",
            "Whether the copies are indistinguishable is a convention only you can choose.",
            [
                "Are the copies indistinguishable (divide the partition function by N!, the "
                "Boltzmann counting of the classical limit) or distinguishable (no division)?"
            ],
        )
    problem.copies_count = count
    problem.identical = copies.identical
    if copies.identical:
        problem.notes.append(
            "Indistinguishable copies use Z = Z_1^N / N!, the classical-limit counting; it is not "
            "exact Bose or Fermi statistics."
        )
    return {s for s in count.free_symbols if isinstance(s, sp.Symbol)}


def _fill_levels(problem: Problem, system: LevelsSystem, parse: Any) -> set[sp.Symbol]:
    levels = system.levels
    if len(levels) > MAX_LEVELS:
        raise Refusal("resource_limit", f"at most {MAX_LEVELS} levels")
    beta, T, k_B = problem.beta, problem.T, problem.k_B
    energies: list[sp.Expr] = []
    degeneracies: list[sp.Expr] = []
    values: dict[str, list[sp.Expr]] = {}
    observable_names = {name for lv in levels for name in lv.observables}
    if "E" in observable_names:
        raise Refusal(
            "unknown_observable", "E is the energy; it cannot be redefined as an observable"
        )
    for name in observable_names:
        if any(name not in lv.observables for lv in levels):
            raise Refusal("unknown_observable", f"observable {name} needs a value on every level")
    for lv in levels:
        e = parse(lv.energy)
        g = parse(lv.degeneracy)
        if e.is_real is not True:
            raise Refusal(
                "assumptions_insufficient",
                "energies must be provably real: declare their symbols real or positive",
            )
        if g.is_integer is not True or g.is_positive is not True:
            raise Refusal(
                "invalid_degeneracy",
                "a degeneracy must be a positive integer (declare a symbolic one integer and positive)",
            )
        if {beta, T, k_B} & (e.free_symbols | g.free_symbols):
            raise Refusal(
                "unsupported_system", "energies and degeneracies must not depend on the temperature"
            )
        energies.append(e)
        degeneracies.append(g)
    values["E"] = list(energies)
    for name in sorted(observable_names):
        column = [parse(lv.observables[name]) for lv in levels]
        if any(a.is_real is not True for a in column):
            raise Refusal(
                "assumptions_insufficient",
                f"observable {name} must be provably real on every level",
            )
        if {beta, T, k_B} & {s for a in column for s in a.free_symbols}:
            raise Refusal("unsupported_system", "observables must not depend on the temperature")
        values[name] = column
    problem.energies, problem.degeneracies, problem.observables = energies, degeneracies, values
    free: set[sp.Symbol] = set()
    for expr in [*energies, *degeneracies, *(a for col in values.values() for a in col)]:
        free |= {s for s in expr.free_symbols if isinstance(s, sp.Symbol)}
    return free
