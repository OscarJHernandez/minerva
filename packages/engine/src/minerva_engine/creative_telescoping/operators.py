"""Linear differential operators in the parameter, with polynomial coefficients.

`DifferentialOperator.kind` is `"differential"` today; a shift operator (for recurrences and
hypergeometric summation) would be a second kind with the same coefficient list. Nothing else in
the package depends on the kind, which is what keeps that extension local.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sympy as sp


@dataclass(frozen=True)
class DifferentialOperator:
    """`L = sum_k coefficients[k] * d^k/dvar^k`."""

    var: sp.Symbol
    coefficients: tuple[sp.Expr, ...]
    kind: Literal["differential"] = "differential"

    @property
    def order(self) -> int:
        for k in range(len(self.coefficients) - 1, -1, -1):
            if self.coefficients[k] != 0:
                return k
        return 0

    @property
    def is_zero(self) -> bool:
        return all(c == 0 for c in self.coefficients)

    def apply(self, f: sp.Expr) -> sp.Expr:
        """`L f`, by plain differentiation (never through the search's own recurrences)."""
        return sp.Add(*(c * sp.diff(f, self.var, k) for k, c in enumerate(self.coefficients)))

    def latex(self) -> str:
        """`p_0 + p_1 \\partial_a + ...`, each coefficient written to the left of its derivative."""
        from ..latexprint import to_latex

        d = r"\partial_{" + to_latex(self.var) + "}"
        terms: list[str] = []
        for k, c in enumerate(self.coefficients):
            if c == 0:
                continue
            text = to_latex(c)
            if isinstance(c, sp.Add):
                text = f"\\left({text}\\right)"
            power = "" if k == 0 else d if k == 1 else f"{d}^{{{k}}}"
            terms.append(text if k == 0 else (power if c == 1 else f"{text}\\,{power}"))
        return " + ".join(terms) if terms else "0"


def normalise(
    var: sp.Symbol, coefficients: list[sp.Expr], parameters: list[sp.Symbol]
) -> list[sp.Expr]:
    """Clear denominators and the common factor: primitive polynomial coefficients.

    The overall sign is fixed so that the leading nonzero coefficient is not a negative number.
    """
    cs = [sp.cancel(sp.together(c)) for c in coefficients]
    denominators = [sp.denom(c) for c in cs if c != 0]
    common = sp.lcm_list(denominators) if denominators else sp.Integer(1)
    cs = [sp.cancel(c * common) for c in cs]
    nonzero = [c for c in cs if c != 0]
    if nonzero:
        g = sp.gcd_list(nonzero, *parameters) if parameters else sp.gcd_list(nonzero)
        if g not in (0, 1):
            cs = [sp.cancel(c / g) for c in cs]
        lead = next(c for c in reversed(cs) if c != 0)
        if (
            sp.Poly(lead, *parameters).LC() < 0
            if parameters and lead.is_polynomial(*parameters)
            else False
        ):
            cs = [-c for c in cs]
    return [sp.expand(c) for c in cs]
