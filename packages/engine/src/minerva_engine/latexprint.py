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
from sympy.printing.str import StrPrinter


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

    # Distributions print with the explicit names the parsers read back (\delta and \theta are
    # ordinary symbols in Minerva's input, so the SymPy defaults would not survive a round trip).
    def _print_DiracDelta(self, expr: sp.Basic, exp: Any = None) -> str:
        if len(expr.args) > 1:  # derivative order: not supported as input, shown honestly
            tex = rf"\operatorname{{DiracDelta}}\left({self._print(expr.args[0])}, {self._print(expr.args[1])}\right)"
        else:
            tex = rf"\operatorname{{DiracDelta}}\left({self._print(expr.args[0])}\right)"
        return str(self._do_exponent(tex, exp))

    def _print_Heaviside(self, expr: sp.Basic, exp: Any = None) -> str:
        # SymPy stores the value at 0 (default 1/2) as a second argument; show it only if it differs.
        if len(expr.args) > 1 and expr.args[1] != sp.S.Half:
            tex = rf"\operatorname{{Heaviside}}\left({self._print(expr.args[0])}, {self._print(expr.args[1])}\right)"
        else:
            tex = rf"\operatorname{{Heaviside}}\left({self._print(expr.args[0])}\right)"
        return str(self._do_exponent(tex, exp))

    # Indexed data prints as x_{i}; sums and products as nested single-limit operators (SymPy's
    # multi-limit form is not something the parsers read), the term of an Add in parentheses.
    def _print_Indexed(self, expr: sp.Indexed) -> str:
        label = str(self._print(expr.base.label))
        return rf"{label}_{{{', '.join(str(self._print(i)) for i in expr.indices)}}}"

    def _print_IndexedBase(self, expr: sp.IndexedBase) -> str:
        return str(self._print(expr.label))

    def _big_operator(self, expr: sp.Basic, operator: str) -> str:
        body = expr.function
        text = str(self._print(body))
        if body.is_Add:
            text = rf"\left({text}\right)"
        for index, lower, upper in expr.limits:
            head = rf"{operator}_{{{self._print(index)}={self._print(lower)}}}^{{{self._print(upper)}}}"
            text = f"{head} {text}"
        return text

    def _print_Sum(self, expr: sp.Basic) -> str:
        return self._big_operator(expr, r"\sum")

    def _print_Product(self, expr: sp.Basic) -> str:
        return self._big_operator(expr, r"\prod")

    def _print_Expectation(self, expr: sp.Basic) -> str:
        return rf"\mathbb{{E}}\left[{self._print(expr.args[0])}\right]"

    def _print_Variance(self, expr: sp.Basic) -> str:
        return rf"\operatorname{{Var}}\left({self._print(expr.args[0])}\right)"

    def _print_Covariance(self, expr: sp.Basic) -> str:
        return rf"\operatorname{{Cov}}\left({self._print(expr.args[0])}, {self._print(expr.args[1])}\right)"


class MinervaTextPrinter(StrPrinter):  # type: ignore[misc]
    """Plain-text output that `textparse` reads back; expectations print as E[...], Var(...), Cov(...)."""

    def _print_Expectation(self, expr: sp.Basic) -> str:
        return f"E[{self._print(expr.args[0])}]"

    def _print_Variance(self, expr: sp.Basic) -> str:
        return f"Var({self._print(expr.args[0])})"

    def _print_Covariance(self, expr: sp.Basic) -> str:
        return f"Cov({self._print(expr.args[0])}, {self._print(expr.args[1])})"


_DEFAULTS: dict[str, Any] = {"imaginary_unit": "ri", "inv_trig_style": "full"}


def to_latex(expr: Any, **settings: Any) -> str:
    """Print ``expr`` as LaTeX with Minerva's constant convention."""
    return str(MinervaLatexPrinter({**_DEFAULTS, **settings}).doprint(expr))


def to_text(expr: Any) -> str:
    """Print ``expr`` in the plain-text syntax (``Sum(f, (i, 1, n))``, ``x[i]``, ``E[X]``)."""
    return str(MinervaTextPrinter().doprint(expr))
