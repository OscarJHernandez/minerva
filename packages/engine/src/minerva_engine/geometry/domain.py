"""Signature and the places where the coordinate expressions break down.

Signature is reported only for a diagonal metric whose entries have a sign the declared assumptions decide;
otherwise it is "not determined". Nothing here assumes a metric is positive definite.
"""

from __future__ import annotations

import sympy as sp

from ..latexprint import to_latex
from .setup import Geometry

_BRANCHY = (sp.log, sp.Abs, sp.asin, sp.acos, sp.atan, sp.acot, sp.asinh, sp.acosh, sp.atanh)


def signature(geo: Geometry) -> tuple[str | None, str]:
    n, g = geo.n, geo.g
    if any(g[i, j] != 0 for i in range(n) for j in range(n) if i != j):
        return None, (
            "The metric is not diagonal; its signature is not determined by this analysis "
            "(it is not assumed to be positive definite or Lorentzian)."
        )
    signs = []
    for i in range(n):
        entry = g[i, i]
        if entry.is_positive:
            signs.append("+")
        elif entry.is_negative:
            signs.append("-")
        else:
            return None, (
                f"The sign of g_({geo.coords[i].name},{geo.coords[i].name}) is not decided by the "
                "declared assumptions, so the signature is not determined (it may change across "
                "regions)."
            )
    return "(" + ", ".join(signs) + ")", "Decided from the declared assumptions."


def restrictions(geo: Geometry, det: sp.Expr, ginv: sp.Matrix) -> tuple[list[str], bool]:
    """Factors that vanish where `det g`, a metric entry or an inverse-metric entry is singular.

    The check is syntactic (zeros of numerators of the determinant and denominators). Square-root,
    logarithm and branch-cut domains are not analysed; when the metric contains them, or a factor
    cannot be analysed, the list is marked incomplete.
    """
    factors: dict[str, str] = {}
    complete = True

    def collect(expr: sp.Expr, numerator: bool) -> None:
        nonlocal complete
        try:
            num, den = sp.fraction(sp.together(expr))
            parts = [den] + ([num] if numerator else [])
            for part in parts:
                for factor, _ in sp.factor_list(part)[1]:
                    if factor.free_symbols and factor.is_zero is not False:
                        factors[str(factor)] = f"{to_latex(factor)} \\neq 0"
        except (sp.PolynomialError, NotImplementedError, TypeError, ValueError):
            complete = False

    collect(det, numerator=True)
    for matrix in (geo.g, ginv):
        for entry in matrix:
            if entry != 0:
                collect(entry, numerator=False)
    for entry in geo.g:
        if (
            entry.has(*_BRANCHY)
            or any(not p.exp.is_Integer for p in entry.atoms(sp.Pow) if p.exp.free_symbols == set())
            or any(p.exp.free_symbols for p in entry.atoms(sp.Pow))
        ):
            complete = False
    return sorted(factors.values()), complete
