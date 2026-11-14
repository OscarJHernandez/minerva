"""LaTeX for vector expressions: bold vectors, \\nabla, partial derivatives.

Only vector output uses this printer; `latexprint.to_latex` is unchanged (a noncommutative symbol
there is a matrix and prints plain). Fields print without their arguments (`A_{x}`, not
`A_{x}(x, y, z)`), derivatives of fields as partial derivatives.
"""

from __future__ import annotations

from typing import Any

import sympy as sp
from sympy.printing.latex import LatexPrinter

from ..latexprint import MinervaLatexPrinter
from .expressions import Adv, Cross, Curl, Div, Dot, Grad, Lap, Pd, Pos, Unit


class VectorLatexPrinter(MinervaLatexPrinter):
    def _print_Symbol(self, expr: sp.Symbol, style: str = "plain") -> str:
        text = str(super()._print_Symbol(expr, style=style))
        if expr.is_commutative is False:
            return rf"\mathbf{{{text}}}"
        return text

    # Applied undefined functions are fields: print the name only.
    def _print_Function(self, expr: sp.Basic, exp: Any = None) -> str:
        if isinstance(type(expr), sp.core.function.UndefinedFunction):
            tex = str(self._print_Symbol(sp.Symbol(type(expr).__name__)))
            return str(self._do_exponent(tex, exp))
        return str(super()._print_Function(expr, exp=exp))

    def _print_Derivative(self, expr: sp.Derivative) -> str:
        return str(LatexPrinter._print_Derivative(self, expr))

    def _wrap(self, e: sp.Basic) -> str:
        text = str(self._print(e))
        if e.is_Atom or isinstance(e, (Pos, Unit)):
            return text
        if isinstance(e, sp.Function) and not isinstance(e, Div | Curl | Grad | Lap | Adv | Pd):
            return text
        if isinstance(e, sp.Derivative):
            return text
        return rf"\left({text}\right)"

    def _print_Grad(self, expr: Grad) -> str:
        return rf"\nabla {self._wrap(expr.args[0])}"

    def _print_Div(self, expr: Div) -> str:
        return rf"\nabla \cdot {self._wrap(expr.args[0])}"

    def _print_Curl(self, expr: Curl) -> str:
        return rf"\nabla \times {self._wrap(expr.args[0])}"

    def _print_Lap(self, expr: Lap) -> str:
        return rf"\nabla^{{2}} {self._wrap(expr.args[0])}"

    def _print_Dot(self, expr: Dot) -> str:
        return rf"{self._wrap(expr.args[0])} \cdot {self._wrap(expr.args[1])}"

    def _print_Cross(self, expr: Cross) -> str:
        return rf"{self._wrap(expr.args[0])} \times {self._wrap(expr.args[1])}"

    def _print_Adv(self, expr: Adv) -> str:
        return rf"\left({self._print(expr.args[0])} \cdot \nabla\right) {self._wrap(expr.args[1])}"

    def _print_Pd(self, expr: Pd) -> str:
        # Collect nested derivatives: pd(pd(X, x), y) is d/dy d/dx X.
        variables: list[sp.Basic] = []
        inner: sp.Basic = expr
        while isinstance(inner, Pd):
            variables.append(inner.args[1])
            inner = inner.args[0]
        counts: dict[str, int] = {}
        for v in variables:
            counts[str(self._print(v))] = counts.get(str(self._print(v)), 0) + 1
        order = len(variables)
        denominator = " ".join(
            rf"\partial {name}" + (f"^{{{n}}}" if n > 1 else "") for name, n in counts.items()
        )
        top = r"\partial" + (f"^{{{order}}}" if order > 1 else "")
        return rf"\frac{{{top} {self._wrap(inner)}}}{{{denominator}}}"

    def _print_Pos(self, expr: Pos) -> str:
        return r"\mathbf{r}"

    def _print_Unit(self, expr: Unit) -> str:
        return rf"\hat{{{self._print(expr.args[0])}}}"


def vector_latex(expr: Any) -> str:
    """LaTeX of a vector expression (bold vectors, \\nabla) or of one of its components."""
    return str(VectorLatexPrinter({"imaginary_unit": "ri", "inv_trig_style": "full"}).doprint(expr))
