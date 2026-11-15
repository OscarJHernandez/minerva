"""An exact zero test for the lowered components of a difference.

`sp.simplify` often fails to reduce a trigonometric residual (the curl of a curl in spherical
coordinates, say) to a literal 0 even though it is identically zero. The test here is exact and
narrow: the numerator is rewritten in `sin` and `cos` of plain coordinate symbols, which become
polynomial variables with the relations `s^2 + c^2 = 1`; those relations have coprime leading terms, so
they are a Groebner basis and a numerator that reduces to 0 modulo them is the zero function. A residual
that cannot be put in that form is not declared zero (never the other way round).
"""

from __future__ import annotations

import sympy as sp


def regularise(e: sp.Expr) -> sp.Expr:
    """Write tan, cot, sec and csc with sin and cos, so that `1/tan(theta)` (defined at pi/2) is not
    reported as excluding the zeros of tan(theta): only the zeros of the denominator that remains count."""
    return (
        e.replace(sp.tan, lambda a: sp.sin(a) / sp.cos(a))
        .replace(sp.cot, lambda a: sp.cos(a) / sp.sin(a))
        .replace(sp.sec, lambda a: 1 / sp.cos(a))
        .replace(sp.csc, lambda a: 1 / sp.sin(a))
    )


def exact_zero(expr: sp.Expr) -> bool:
    """True only if `expr` is identically zero (as a function of its symbols)."""
    expr = sp.sympify(expr)
    if expr == 0:
        return True
    try:
        numerator, _ = sp.fraction(sp.together(regularise(sp.expand_trig(expr))))
        numerator = sp.expand(numerator)
        if numerator == 0:
            return True
        angles: set[sp.Basic] = set()
        for node in numerator.atoms(sp.sin, sp.cos):
            if not isinstance(node.args[0], sp.Symbol):
                return False
            angles.add(node.args[0])
        mapping: dict[sp.Basic, sp.Basic] = {}
        sines: list[sp.Dummy] = []
        for angle in sorted(angles, key=str):
            s, c = sp.Dummy(f"s_{angle}"), sp.Dummy(f"c_{angle}")
            mapping[sp.sin(angle)], mapping[sp.cos(angle)] = s, c
            sines.append(s)
        remainder = sp.expand(numerator.xreplace(mapping))
        for s in sines:
            c = next(
                v
                for k, v in mapping.items()
                if isinstance(k, sp.cos) and mapping[sp.sin(k.args[0])] == s
            )
            remainder = sp.expand(sp.rem(remainder, s**2 + c**2 - 1, s))
        return bool(remainder == 0)
    except (sp.PolynomialError, ValueError, TypeError, NotImplementedError, RecursionError):
        return False
