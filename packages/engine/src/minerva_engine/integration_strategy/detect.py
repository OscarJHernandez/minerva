"""Which techniques look promising for an integrand: a pure function of its structure.

Detection never integrates. It looks at the expression tree (and, for symmetry, one cheap symbolic
test) and returns an ordered list of `Suggestion`s; the exploration decides whether a technique
actually produces a verified result. It does not depend on any language model.
"""

from __future__ import annotations

import sympy as sp

from ..creative_telescoping.integrand import analyse
from ..creative_telescoping.models import Refusal
from .models import Suggestion

SIMPLE_OPS = 80  # symbolic tests are only tried on expressions smaller than this
TRANSCENDENTAL = (sp.exp, sp.sin, sp.cos, sp.log, sp.atan, sp.sinh, sp.cosh)
TRIG = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)


def _is_linear_in(arg: sp.Expr, x: sp.Symbol) -> bool:
    return bool(sp.Poly(arg, x).degree() <= 1) if arg.is_polynomial(x) else False


def has_composite(f: sp.Expr, x: sp.Symbol) -> bool:
    """A function or power applied to something that is not linear in x (substitution candidate)."""
    for node in sp.preorder_traversal(f):
        if isinstance(node, sp.Pow):
            base, power = node.as_base_exp()
            if isinstance(base, sp.Function) and base.args and _is_linear_in(base.args[0], x):
                continue
            if (
                x in base.free_symbols
                and base != x
                and not _is_linear_in(base, x)
                and power != 1
                and power.is_number
            ):
                return True
        elif isinstance(node, sp.Function) and node.args and x in node.free_symbols:
            arg = node.args[0]
            if arg != x and not _is_linear_in(arg, x):
                return True
    return False


def polynomial_times_transcendental(f: sp.Expr, x: sp.Symbol) -> bool:
    factors = sp.Mul.make_args(sp.together(f))
    poly = any(
        x in fac.free_symbols and fac.is_polynomial(x) and sp.degree(fac, x) >= 1 for fac in factors
    )
    trans = any(
        isinstance(sub, TRANSCENDENTAL) and x in sub.free_symbols
        for sub in sp.preorder_traversal(f)
    )
    return poly and trans


def trigonometric(f: sp.Expr, x: sp.Symbol) -> bool:
    trig_nodes = [
        n for n in sp.preorder_traversal(f) if isinstance(n, TRIG) and x in n.free_symbols
    ]
    if not trig_nodes:
        return False
    powered = any(
        isinstance(n, sp.Pow) and n.exp.is_number and n.exp != 1 and n.base.has(*TRIG)
        for n in sp.preorder_traversal(f)
    )
    product = (
        len(sp.Mul.make_args(f)) >= 2 and sum(1 for a in sp.Mul.make_args(f) if a.has(*TRIG)) >= 1
    )
    return powered or (product and len(trig_nodes) >= 2)


def rational_reducible(f: sp.Expr, x: sp.Symbol) -> bool:
    if not f.is_rational_function(x) or f.is_polynomial(x):
        return False
    denominator = sp.denom(sp.together(f))
    _, factors = sp.factor_list(denominator, x)
    return len(factors) >= 2 or any(m >= 2 for _, m in factors)


def parity(f: sp.Expr, x: sp.Symbol, lower: sp.Expr, upper: sp.Expr) -> int | None:
    """+1 if even, -1 if odd, on a symmetric interval; None otherwise or if undecided cheaply."""
    if lower != -upper or sp.count_ops(f) > SIMPLE_OPS:
        return None
    flipped = f.subs(x, -x)
    try:
        if sp.simplify(flipped - f) == 0:
            return 1
        if sp.simplify(flipped + f) == 0:
            return -1
    except Exception:
        return None
    return None


def reflection(f: sp.Expr, x: sp.Symbol, lower: sp.Expr, upper: sp.Expr) -> sp.Expr | None:
    """The average (f(x) + f(a+b-x))/2 when it is strictly simpler than f; else None."""
    if not (lower.is_finite and upper.is_finite) or sp.count_ops(f) > SIMPLE_OPS:
        return None
    try:
        mirrored = f.subs(x, lower + upper - x)
        average = sp.simplify((f + mirrored) / 2)
    except Exception:
        return None
    if average != f and sp.count_ops(average) < sp.count_ops(f):
        return average
    return None


def frullani_pair(f: sp.Expr, x: sp.Symbol) -> tuple[sp.Expr, sp.Expr, sp.Expr] | None:
    """(g, c1, c2) with x f = g(c1 x) - g(c2 x), where g is a function of x alone; else None."""
    try:
        numerator = sp.expand(f * x)
    except Exception:
        return None
    terms = sp.Add.make_args(numerator)
    if len(terms) != 2:
        return None
    first, second = terms

    def scales(term: sp.Expr) -> list[sp.Expr]:
        found: list[sp.Expr] = []
        for node in sp.preorder_traversal(term):
            if isinstance(node, sp.Mul) and x in node.args:
                c = sp.Mul(*(a for a in node.args if a != x))
                if x not in c.free_symbols and c != 1:
                    found.extend(v for v in (c, -c) if v not in found)
        # g(c x) with a provably positive scale first: exp(-a x) is g(a x) for g = exp(-t)
        found.sort(key=lambda c: 0 if c.is_positive else 1)
        return found

    for c1 in scales(first) or [sp.Integer(1)]:
        g = first.subs(x, x / c1)
        for c2 in scales(second) or [sp.Integer(1)]:
            if c1 == c2:
                continue
            h = second.subs(x, x / c2)
            if sp.simplify(g + h) == 0:
                return g, c1, c2
    return None


def schwinger_factor(f: sp.Expr, x: sp.Symbol) -> tuple[sp.Expr, sp.Expr, sp.Expr] | None:
    """(N, A, n) with f = N * A**(-n), A containing x and not a bare power of x; else None."""
    for factor in sp.Mul.make_args(f):
        if isinstance(factor, sp.Pow):
            base, power = factor.as_base_exp()
            if x in base.free_symbols and not base.is_Symbol and power.is_number and power < 0:
                rest = sp.simplify(f / factor)
                return rest, base, -power
    return None


def parameters(f: sp.Expr, x: sp.Symbol) -> list[sp.Symbol]:
    return sorted((s for s in f.free_symbols if s != x), key=lambda s: s.name)


def telescopable(f: sp.Expr, x: sp.Symbol) -> list[sp.Symbol]:
    out = []
    for p in parameters(f, x):
        try:
            analyse(f, x, p)
        except Refusal:
            continue
        out.append(p)
    return out


def detect(
    f: sp.Expr, x: sp.Symbol, lower: sp.Expr | None, upper: sp.Expr | None
) -> list[Suggestion]:
    """Ordered suggestions. Specific, cheap-to-confirm techniques come first."""
    out: list[Suggestion] = []
    definite = lower is not None and upper is not None

    def add(technique: str, why: str) -> None:
        out.append(Suggestion(technique=technique, why=why))

    if definite:
        assert lower is not None and upper is not None  # noqa: S101
        if lower == 0 and upper is sp.oo and frullani_pair(f, x) is not None:
            add("frullani", "difference_at_two_scalings_over_x")
        sign = parity(f, x, lower, upper)
        if sign == -1:
            add("parity", "odd_on_symmetric_interval")
        elif sign == 1:
            add("parity", "even_on_symmetric_interval")
        if reflection(f, x, lower, upper) is not None:
            add("reflection", "reflection_simplifies")
        if schwinger_factor(f, x) is not None:
            add("schwinger", "negative_power_of_a_positive_base")
    if rational_reducible(f, x):
        add("partial_fractions", "rational_with_reducible_denominator")
    if trigonometric(f, x):
        add("trigonometric", "trigonometric_powers_or_products")
    if has_composite(f, x):
        add("substitution", "composite_with_inner_derivative")
    if polynomial_times_transcendental(f, x):
        add("parts", "polynomial_times_transcendental")
    if definite and telescopable(f, x):
        add("parameter_differentiation", "free_parameter")
    if definite and not parameters(f, x):
        add("numerical", "no_closed_form_for_a_number")
    return out
