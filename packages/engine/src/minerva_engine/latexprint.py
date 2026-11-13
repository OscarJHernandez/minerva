"""LaTeX output for every expression Minerva shows.

One printer, one convention: the constants are upright (``\\mathrm{e}``, ``\\mathrm{i}``) and a bare
``e`` or ``i`` in output is always a user symbol. ``latexparse`` reads ``\\mathrm{e}`` and
``\\mathrm{i}`` back as the constants, so printed output survives a round trip. Inverse
trigonometric functions use the full ``\\arcsin`` family the parser accepts, not ``\\operatorname{asin}``.
"""

from __future__ import annotations

from typing import Any

import sympy as sp
from sympy.printing.latex import LatexPrinter


class MinervaLatexPrinter(LatexPrinter):  # type: ignore[misc]
    def _print(self, expr: Any, **kwargs: Any) -> str:
        # `Exp1` carries its own `_latex` method, which the base printer prefers over `_print_Exp1`.
        if expr is sp.E:
            return r"\mathrm{e}"
        return str(super()._print(expr, **kwargs))

    # Applied undefined functions print as f(r), their derivatives as f'(r), f''(r), f^{(n)}(r).
    # The parser reads f(r) back when f is declared (geometry); other operations never produce them.
    def _print_Function(self, expr: sp.Basic, exp: Any = None) -> str:
        if not isinstance(type(expr), sp.core.function.UndefinedFunction):
            return str(super()._print_Function(expr, exp=exp))
        args = ", ".join(str(self._print(a)) for a in expr.args)
        tex = rf"{self._print_Symbol(sp.Symbol(type(expr).__name__))}\left({args}\right)"
        return str(self._do_exponent(tex, exp))

    def _print_Derivative(self, expr: sp.Derivative) -> str:
        function = expr.expr
        if (
            isinstance(function, sp.Basic)
            and isinstance(type(function), sp.core.function.UndefinedFunction)
            and len(function.args) == 1
            and len(expr.variable_count) == 1
            and expr.variable_count[0][0] == function.args[0]
        ):
            order = int(expr.variable_count[0][1])
            name = str(self._print_Symbol(sp.Symbol(type(function).__name__)))
            args = str(self._print(function.args[0]))
            mark = "'" * order if order <= 3 else rf"^{{({order})}}"
            return rf"{name}{mark}\left({args}\right)"
        return str(super()._print_Derivative(expr))

    def _print_ExpBase(self, expr: sp.Basic, exp: Any = None) -> str:
        tex = rf"\mathrm{{e}}^{{{self._print(expr.args[0])}}}"
        return str(self._do_exponent(tex, exp))


_DEFAULTS: dict[str, Any] = {"imaginary_unit": "ri", "inv_trig_style": "full"}


def to_latex(expr: Any, **settings: Any) -> str:
    """Print ``expr`` as LaTeX with Minerva's constant convention."""
    return str(MinervaLatexPrinter({**_DEFAULTS, **settings}).doprint(expr))
