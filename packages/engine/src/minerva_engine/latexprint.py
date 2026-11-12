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

    def _print_ExpBase(self, expr: sp.Basic, exp: Any = None) -> str:
        tex = rf"\mathrm{{e}}^{{{self._print(expr.args[0])}}}"
        return str(self._do_exponent(tex, exp))


_DEFAULTS: dict[str, Any] = {"imaginary_unit": "ri", "inv_trig_style": "full"}


def to_latex(expr: Any, **settings: Any) -> str:
    """Print ``expr`` as LaTeX with Minerva's constant convention."""
    return str(MinervaLatexPrinter({**_DEFAULTS, **settings}).doprint(expr))
