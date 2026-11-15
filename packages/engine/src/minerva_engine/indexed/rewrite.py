"""Rewrites of finite sums and products: linearity, constants, factoring, log of a product, derivatives.

Each function builds a candidate; none of them is trusted. `operation.sums` verifies every candidate
against the original on explicit instances before showing it. SymPy's `Sum.doit` and `simplify` are not
used on sums with data (they merge ranges and follow the Karr convention for empty ranges); `summation`
is used only for a summand without data, and its answer is checked on instances like everything else.
"""

from __future__ import annotations

import sympy as sp

from .randoms import RANDOM_TYPES

BigOperator = (sp.Sum, sp.Product)


def canonical_indices(expr: sp.Basic) -> sp.Basic:
    """Rename bound indices so that sums that differ only in the name of the index are equal."""
    representatives: list[sp.Basic] = []
    mapping: dict[sp.Basic, sp.Basic] = {}
    for node in sp.preorder_traversal(expr):
        if not isinstance(node, BigOperator) or node in mapping:
            continue
        for known in representatives:
            if type(known) is type(node) and known != node and node.dummy_eq(known):
                mapping[node] = known
                break
        else:
            representatives.append(node)
    return expr.xreplace(mapping) if mapping else expr


def _depends(term: sp.Basic, indices: set[sp.Symbol]) -> bool:
    return any(term.has(i) for i in indices)


def _linear_sum(function: sp.Expr, limit: tuple[sp.Symbol, sp.Expr, sp.Expr]) -> sp.Expr:
    """sum over one limit of an expanded function: split over +, pull out what does not depend on i."""
    index, lower, upper = limit
    total: list[sp.Expr] = []
    for term in sp.Add.make_args(sp.expand(function)):
        constant = [f for f in sp.Mul.make_args(term) if not f.has(index)]
        dependent = [f for f in sp.Mul.make_args(term) if f.has(index)]
        rest = sp.Mul(*dependent)
        count = sp.Mul(*constant)
        total.append(count * (upper - lower + 1) if rest == 1 else count * sp.Sum(rest, limit))
    return sp.Add(*total)


def expand_sums(expr: sp.Expr) -> sp.Expr:
    """Linearity of finite sums, bottom up: sum(f + g) = sum f + sum g, sum(c f) = c sum f, sum c = n c."""

    def rec(node: sp.Basic) -> sp.Basic:
        if isinstance(node, sp.Sum):
            function, *limits = node.args
            result: sp.Basic = rec(function)
            for limit in limits:  # the first limit is the innermost
                result = _linear_sum(result, tuple(limit))
            return result
        if not node.args:
            return node
        return node.func(*[rec(a) for a in node.args])

    return sp.expand(canonical_indices(rec(canonical_indices(expr))))


def factor_sums(expr: sp.Expr) -> sp.Expr:
    """A sum of a product whose factors each depend on one index is a product of sums."""

    def one(node: sp.Sum) -> sp.Expr | None:
        function, *limits = node.args
        indices = [limit[0] for limit in limits]
        bounds = set().union(*[(limit[1].free_symbols | limit[2].free_symbols) for limit in limits])
        if bounds & set(indices):
            return None  # not a rectangular range
        groups: dict[sp.Symbol | None, list[sp.Expr]] = {}
        for factor in sp.Mul.make_args(function):
            used = [i for i in indices if factor.has(i)]
            if len(used) > 1:
                return None
            groups.setdefault(used[0] if used else None, []).append(factor)
        result: sp.Expr = sp.Mul(*groups.get(None, []))
        for index, lower, upper in (tuple(limit) for limit in limits):
            members = groups.get(index, [])
            result *= (
                sp.Sum(sp.Mul(*members), (index, lower, upper)) if members else upper - lower + 1
            )
        return result

    def rec(node: sp.Basic) -> sp.Basic:
        if isinstance(node, sp.Sum):
            function, *limits = node.args
            rebuilt = sp.Sum(rec(function), *limits)
            replacement = one(rebuilt)
            return rebuilt if replacement is None else replacement
        if not node.args:
            return node
        return node.func(*[rec(a) for a in node.args])

    return canonical_indices(rec(expr))


def evaluate_constant_sums(expr: sp.Expr) -> sp.Expr:
    """Closed forms for sums whose summand has no data: n c, n(n+1)/2, ... (checked on instances)."""

    def rec(node: sp.Basic) -> sp.Basic:
        if isinstance(node, sp.Sum):
            function, *limits = node.args
            function = rec(function)
            if function.has(sp.Indexed, *RANDOM_TYPES) or function.has(sp.Sum, sp.Product):
                return sp.Sum(function, *limits)
            try:
                closed = sp.summation(function, *limits)
            except Exception:
                return sp.Sum(function, *limits)
            if closed.has(sp.Sum, sp.Piecewise):
                return sp.Sum(function, *limits)
            return closed
        if not node.args:
            return node
        return node.func(*[rec(a) for a in node.args])

    return rec(expr)


def log_of_product(expr: sp.Expr) -> tuple[sp.Expr, list[sp.Expr]]:
    """log(prod f) = sum log f where every factor is provably positive; the unproved factors."""
    unproved: list[sp.Expr] = []

    def rec(node: sp.Basic) -> sp.Basic:
        if isinstance(node, sp.log) and isinstance(node.args[0], sp.Product):
            product = node.args[0]
            if product.function.is_positive is True:
                return rec(sp.Sum(sp.log(product.function), *product.limits))
            unproved.append(product.function)
            return node
        if not node.args:
            return node
        return node.func(*[rec(a) for a in node.args])

    return rec(expr), unproved


def differentiate_sums(expr: sp.Expr, variable: sp.Symbol) -> sp.Expr:
    """d/dv of a finite sum is the sum of the derivatives (the bounds must not depend on v)."""
    return sp.diff(expr, variable)
