"""Explicit instances of expressions with finite sums, products and indexed data.

`explicit` replaces every `Sum` and `Product` over numeric bounds by the terms it stands for (never
through SymPy's summation, whose closed forms and Karr convention are exactly what is being checked)
and every `x[k]` by a symbol of its own. The ordinary pointwise checks are then valid, because the data
of different elements are independent symbols.

Several sizes are used on purpose, and always a size of at least 2: `sum x_i^2 = (sum x_i)^2 / n` is true
for n = 1.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import sympy as sp

from ..differentiate import DerivativeVerification, check_derivative
from ..errors import MinervaMathError
from ..verify import verify_equal
from .models import InstanceVerification

SIZE_CANDIDATES = (2, 3, 5)
MAX_TERMS = 5000


class InstanceError(MinervaMathError):
    """The expression cannot be turned into explicit instances (a bound or index is not numeric)."""


def fresh_symbol(base: sp.IndexedBase, position: int) -> sp.Symbol:
    """The symbol for element `position` of `base`; it carries the base's assumptions.

    The double underscore cannot appear in a name the parsers accept, so it never collides.
    """
    assumptions = {k: v for k, v in base.label.assumptions0.items() if isinstance(v, bool)}
    return sp.Symbol(f"{base.label.name}__{position}", **assumptions)


def explicit(
    expr: sp.Basic, fresh: Callable[[sp.IndexedBase, int], sp.Symbol] = fresh_symbol
) -> sp.Basic:
    """Expand sums and products over numeric bounds and turn indexed elements into symbols."""
    budget = [MAX_TERMS]

    def go(node: sp.Basic) -> sp.Basic:
        if isinstance(node, sp.Sum | sp.Product):
            return big(node)
        if isinstance(node, sp.Indexed):
            (position,) = node.indices
            if not position.is_Integer:
                raise InstanceError("an index is not a number once the sizes are fixed")
            return fresh(node.base, int(position))
        if not node.args:
            return node
        return node.func(*[go(a) for a in node.args])

    def big(node: sp.Basic) -> sp.Basic:
        function, *limits = node.args
        if not limits:
            return go(function)
        index, lower, upper = limits[-1]  # the outermost limit is the last one
        if not (lower.is_Integer and upper.is_Integer):
            raise InstanceError("a bound is not a number once the sizes are fixed")
        if upper < lower - 1:
            raise InstanceError("a range has negative length at this size")
        maker = type(node)
        inner = maker(function, *limits[:-1]) if limits[:-1] else function
        budget[0] -= int(upper - lower + 1)
        if budget[0] < 0:
            raise InstanceError("the expansion is too large")
        terms = [go(inner.subs(index, k)) for k in range(int(lower), int(upper) + 1)]
        return sp.Add(*terms) if maker is sp.Sum else sp.Mul(*terms)

    return go(expr)


def _bound_indices(exprs: Iterable[sp.Basic]) -> set[sp.Symbol]:
    return {v for e in exprs for node in e.atoms(sp.Sum, sp.Product) for v in node.variables}


def size_symbols(*exprs: sp.Basic) -> list[sp.Symbol]:
    """The symbols that fix how many terms the sums have (those in bounds, except bound indices)."""
    bound = _bound_indices(exprs)
    found: set[sp.Symbol] = set()
    for e in exprs:
        for node in e.atoms(sp.Sum, sp.Product):
            for _, lower, upper in node.limits:
                found |= {s for s in (lower.free_symbols | upper.free_symbols) if s not in bound}
    return sorted((s for s in found if isinstance(s, sp.Symbol)), key=lambda s: s.name)


def _values(symbol: sp.Symbol) -> list[int]:
    smallest = 1 if symbol.is_positive else 0 if symbol.is_nonnegative else None
    values = [] if smallest is None else [smallest]
    values += [v for v in SIZE_CANDIDATES if v not in values]
    return values


def size_assignments(*exprs: sp.Basic) -> list[dict[sp.Symbol, int]]:
    """Several assignments of sizes, the smallest admissible one and sizes of at least 2 among them."""
    symbols = size_symbols(*exprs)
    if not symbols:
        return [{}]
    table = [_values(s) for s in symbols]
    count = max(len(v) for v in table)
    return [
        {s: table[k][(j + k) % len(table[k])] for k, s in enumerate(symbols)} for j in range(count)
    ]


def instances(*exprs: sp.Basic) -> list[tuple[dict[sp.Symbol, int], list[sp.Basic]]]:
    out = []
    for assignment in size_assignments(*exprs):
        concrete = [explicit(e.subs(assignment)) for e in exprs]
        out.append((assignment, concrete))
    return out


def verify_equal_indexed(original: sp.Expr, candidate: sp.Expr) -> InstanceVerification:
    """Do the two expressions agree on every explicit instance?"""
    sizes: list[dict[str, int]] = []
    agreed = exact = 0
    notes: list[str] = []
    try:
        built = instances(original, candidate)
    except InstanceError as exc:
        return InstanceVerification(passed=False, note=str(exc))
    for assignment, (a, b) in built:
        sizes.append({s.name: v for s, v in assignment.items()})
        result = verify_equal(a, b)
        agreed += result.passed
        exact += bool(result.symbolic_zero)
        if not result.passed:
            notes.append(f"differs at {sizes[-1] or 'the instance'}")
    return InstanceVerification(
        sizes=sizes,
        instances_checked=len(built),
        instances_agreed=agreed,
        exact_zero_instances=exact,
        passed=bool(built) and agreed == len(built),
        note="; ".join(notes),
    )


def check_derivative_indexed(
    f: sp.Expr, var: sp.Symbol, candidate: sp.Expr
) -> tuple[DerivativeVerification, list[dict[str, int]]]:
    """Is `candidate` the derivative of `f` in `var`, on every explicit instance?"""
    try:
        built = instances(f, candidate)
    except InstanceError as exc:
        return DerivativeVerification(passed=False, note=str(exc)), []
    checked = agreed = 0
    sizes: list[dict[str, int]] = []
    passed = True
    for assignment, (a, b) in built:
        sizes.append({s.name: v for s, v in assignment.items()})
        result = check_derivative(a, var, b)
        checked += result.points_checked
        agreed += result.points_agreed
        passed = passed and result.passed
    return DerivativeVerification(
        points_checked=checked, points_agreed=agreed, passed=passed
    ), sizes
