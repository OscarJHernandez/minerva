"""Where a transformation changed the set of points at which an expression is defined.

`(x^2-1)/(x-1)` and `x+1` agree everywhere except `x = 1`; the parser even cancels `(x-1)/(x-1)`
to `1` while reading. Neither the numeric nor the symbolic check can see that, so the denominators
the parser met are recorded and compared with the denominators of the result.

This reports; it never rewrites or verifies. The check is syntactic (zeros of denominators): square
roots, logarithms and branch cuts are not covered here, and a denominator that cannot be analysed
is reported as `d != 0` rather than dropped.
"""

from __future__ import annotations

from collections.abc import Iterable

import sympy as sp

from .indexed.guard import contains_indexed

_MAX_OPS = 300


def _factors(denominator: sp.Expr) -> list[sp.Expr]:
    """Distinct non-constant factors of a denominator."""
    found: list[sp.Expr] = []
    for factor in sp.Mul.make_args(sp.factor(denominator)):
        base = factor.base if factor.is_Pow and factor.exp.is_number else factor
        if base.free_symbols and base not in found:
            found.append(base)
    return found


def _denominator(expr: sp.Expr) -> sp.Expr:
    return sp.fraction(sp.together(expr))[1]


def _still_excluded(factor: sp.Expr, expr: sp.Expr) -> bool:
    """True if `factor = 0` is already a pole of `expr` (the factor divides its denominator)."""
    quotient = sp.cancel(_denominator(expr) / factor)
    return bool(sp.fraction(quotient)[1] == 1 and quotient != 0)


def _undefined_wherever_zero(factor: sp.Expr, after: sp.Expr) -> bool:
    """True if `after` is undefined at every root of `factor` (so nothing was lost there).

    `1/(1+cos x) + ...` simplifies to `2/sin x`: the factor 1+cos x vanishes only where sin x does.
    Undecided (several symbols, no roots found) means False: report the restriction.
    """
    symbols = list(factor.free_symbols)
    if len(symbols) != 1:
        return False
    symbol = symbols[0]
    try:
        roots = sp.solve(factor, symbol)
    except Exception:
        return False
    if not roots or len(roots) > 8:
        return False
    for root in roots:
        value = after.subs(symbol, root)
        if not value.has(sp.zoo, sp.oo, sp.nan, -sp.oo):
            return False
    return True


def _describe(factor: sp.Expr) -> list[str]:
    """`x \\neq 1` for a polynomial in one variable with all roots found, else `f \\neq 0`."""
    symbols = list(factor.free_symbols)
    if len(symbols) == 1 and factor.is_polynomial(symbols[0]):
        symbol = symbols[0]
        degree = sp.degree(factor, symbol)
        roots = sp.roots(sp.Poly(factor, symbol))
        if degree >= 1 and sum(roots.values()) == degree:
            ordered = sorted(roots, key=sp.default_sort_key)
            return [rf"{sp.latex(symbol)} \neq {sp.latex(r)}" for r in ordered]
    return [rf"{sp.latex(factor)} \neq 0"]


def domain_restrictions(
    before: sp.Expr, after: sp.Expr, recorded: Iterable[sp.Expr] = ()
) -> tuple[list[str], bool]:
    """Conditions `after` needs that `before` (as written) did not, as LaTeX, and whether complete.

    Two sources: denominators of `before` (plus those recorded while parsing, which the parser may
    have cancelled already) that `after` no longer excludes; and denominators `after` introduces.
    A factor that the declared assumptions prove nonzero is not reported.
    """
    if contains_indexed(before) or contains_indexed(after):
        return [], False  # zeros of denominators containing sums are not analysed
    try:
        if sp.count_ops(before) > _MAX_OPS or sp.count_ops(after) > _MAX_OPS:
            return [], False
        candidates = _factors(_denominator(before))
        for extra in recorded:
            for factor in _factors(extra):
                if factor not in candidates:
                    candidates.append(factor)
        lost = [
            f
            for f in candidates
            if not _still_excluded(f, after) and not _undefined_wherever_zero(f, after)
        ]
        introduced = [f for f in _factors(_denominator(after)) if not _still_excluded(f, before)]
        notes: list[str] = []
        for factor in [*lost, *[f for f in introduced if f not in lost]]:
            if factor.is_zero is False:
                continue
            for text in _describe(factor):
                if text not in notes:
                    notes.append(text)
        return notes, True
    except Exception:
        return [], False
