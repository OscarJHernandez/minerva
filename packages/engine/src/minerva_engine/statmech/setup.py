"""Parse and validate a statistical-mechanics problem.

Everything the later stages rely on is decided here, with an enumerated reason when it fails:
degeneracies must be provably positive integers, energies and observables provably real, the
temperature symbols are positive by definition and must not appear in the energies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from ..errors import MinervaMathError
from ..models import ExpressionInput, Limits
from ..ops import parse_expression
from .models import MAX_LEVELS, Reason, StatMechInput


class Refusal(MinervaMathError):
    """The problem cannot be treated; ``reason`` is one of the enumerated codes."""

    def __init__(self, reason: Reason, message: str) -> None:
        super().__init__(message)
        self.reason: Reason = reason


@dataclass
class Problem:
    request: StatMechInput
    limits: Limits
    assumptions: dict[str, list[str]]
    beta: sp.Symbol
    T: sp.Symbol
    k_B: sp.Symbol
    params: list[sp.Symbol]
    energies: list[sp.Expr]
    degeneracies: list[sp.Expr]
    observables: dict[str, list[sp.Expr]]  # "E" is always present
    display_var: sp.Symbol
    sample_symbols: list[sp.Symbol] = field(default_factory=list)

    @property
    def state_variables(self) -> list[sp.Symbol]:
        return [self.display_var, *self.params]

    def parse(self, text: str) -> sp.Expr:
        return _parse(self.request, self.limits, self.assumptions, text)


def _parse(
    request: StatMechInput, limits: Limits, assumptions: dict[str, list[str]], text: str
) -> sp.Expr:
    data = ExpressionInput(input=text, format=request.format, assumptions=assumptions)
    return parse_expression(data, limits)


def _positive(assumptions: dict[str, list[str]], name: str) -> None:
    current = list(assumptions.get(name, []))
    for flag in ("positive",):
        if flag not in current:
            current.append(flag)
    assumptions[name] = current


def build_problem(request: StatMechInput, limits: Limits) -> Problem:
    names = [request.beta, request.T, request.k_B, *request.parameters]
    if len(set(names)) != len(names):
        raise Refusal("parse_error", "the temperature, k_B and parameter names must be distinct")
    levels = request.system.levels
    if len(levels) > MAX_LEVELS:
        raise Refusal("resource_limit", f"at most {MAX_LEVELS} levels")

    assumptions = {k: list(v) for k, v in request.assumptions.items()}
    for name in (request.beta, request.T, request.k_B):
        _positive(assumptions, name)

    def parse(text: str) -> sp.Expr:
        try:
            return _parse(request, limits, assumptions, text)
        except MinervaMathError as exc:
            if isinstance(exc, Refusal):
                raise
            raise Refusal("parse_error", str(exc)) from exc

    def symbol(text: str) -> sp.Symbol:
        expr = parse(text)
        if not isinstance(expr, sp.Symbol):
            raise Refusal(
                "parse_error", "temperature, k_B and parameter names must be plain symbols"
            )
        return expr

    beta, T, k_B = symbol(request.beta), symbol(request.T), symbol(request.k_B)
    param_symbols = [symbol(p) for p in request.parameters]
    display = beta if request.temperature == "beta" else T

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

    free: set[sp.Symbol] = set(param_symbols)
    for expr in [*energies, *degeneracies, *(a for col in values.values() for a in col)]:
        free |= {s for s in expr.free_symbols if isinstance(s, sp.Symbol)}
    return Problem(
        request=request,
        limits=limits,
        assumptions=assumptions,
        beta=beta,
        T=T,
        k_B=k_B,
        params=param_symbols,
        energies=energies,
        degeneracies=degeneracies,
        observables=values,
        display_var=display,
        sample_symbols=sorted(free | {beta, k_B}, key=lambda s: s.name),
    )
