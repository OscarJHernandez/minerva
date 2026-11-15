"""What the boundary terms of the first variation do under the declared endpoint conditions.

Each end value `η_i^(j)(end)` is an independent quantity: for smooth perturbations any jet of data
can be prescribed. So a term is removed only by a *fixed* condition on that very derivative at that
very end (fixing `x` does not fix `x'`), a *free* one leaves it and yields the natural boundary
condition `coefficient = 0`, and an unspecified one is kept and decided by nobody.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .firstvar import FirstVariationData
from .models import BoundaryEffect, ConditionState, End
from .setup import Problem


@dataclass(frozen=True)
class Entry:
    end: End
    function: int
    order: int
    coefficient: sp.Expr
    state: ConditionState
    effect: BoundaryEffect


def classify(problem: Problem, data: FirstVariationData) -> list[Entry]:
    entries: list[Entry] = []
    for (end, i, j), coefficient in data.coefficients.items():
        cond = problem.conditions.get((end, i, j))
        state: ConditionState = "unspecified" if cond is None else cond.kind
        if coefficient == 0:
            effect: BoundaryEffect = "vanishes_identically"
        elif state == "fixed":
            effect = "removed_by_fixed"
        elif state == "free":
            effect = "natural_condition"
        else:
            effect = "kept_unspecified"
        entries.append(Entry(end, i, j, coefficient, state, effect))
    return entries


def notes(problem: Problem, data: FirstVariationData, entries: list[Entry]) -> list[str]:
    jets = problem.jets
    out: list[str] = []
    if data.null:
        out.append(
            "The Euler-Lagrange expressions vanish identically: the integrand is a total "
            "derivative and the first variation consists of boundary terms only."
        )
    for i, name in enumerate(jets.names):
        full = 2 * jets.orders[i]
        actual = data.el_orders[i]
        if jets.order_in(problem.lagrangian, i) >= 0 and 0 <= actual < full and not data.null:
            out.append(
                f"The equation for {name} has order {actual}, below the generic order {full}: "
                "the integrand is linear in its highest derivative or the highest terms cancel."
            )
    shown = {(e.end, e.function, e.order) for e in entries}
    for (end, i, j), cond in problem.conditions.items():
        if (end, i, j) not in shown and cond.kind == "fixed":
            out.append(
                f"Fixing {jets.names[i]}^({j}) at the {end} end removes no boundary term of the "
                "first variation: it is an extra condition beyond those the variation produces."
            )
    unspecified = [e for e in entries if e.effect == "kept_unspecified"]
    if unspecified:
        out.append(
            f"{len(unspecified)} boundary term(s) are kept because no condition was declared for "
            "that derivative at that end; nothing is assumed about it."
        )
    return out
