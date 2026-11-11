"""Guarded construction of SymPy expressions.

Both parsers (plain text and LaTeX) build trees exclusively through this module, so the
function whitelist and the magnitude guards live in one place. Nothing here evaluates
user text as Python.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

import sympy as sp

from .errors import LimitError, ParseError
from .models import ALLOWED_ASSUMPTIONS, SYMBOL_NAME_MAX, Limits

# Whitelisted functions: canonical name -> constructor taking one argument.
FUNCTIONS: dict[str, Callable[[sp.Expr], sp.Expr]] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "cot": sp.cot,
    "sec": sp.sec,
    "csc": sp.csc,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "coth": sp.coth,
    "arcsinh": sp.asinh,
    "arccosh": sp.acosh,
    "arctanh": sp.atanh,
    "exp": sp.exp,
    "ln": sp.log,
    "log": sp.log,
    "abs": sp.Abs,
    "gamma": sp.gamma,
    "factorial": sp.factorial,
}

# Aliases accepted by the text parser.
TEXT_FUNCTION_ALIASES: dict[str, str] = {
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "asinh": "arcsinh",
    "acosh": "arccosh",
    "atanh": "arctanh",
    "Abs": "abs",
    "Gamma": "gamma",
}

# Constants available by name in text; in LaTeX they come from commands.
CONSTANTS: dict[str, sp.Expr] = {"pi": sp.pi, "E": sp.E, "I": sp.I, "oo": sp.oo}
RESERVED_NAMES = (
    frozenset(FUNCTIONS)
    | frozenset(TEXT_FUNCTION_ALIASES)
    | frozenset(CONSTANTS)
    | {
        "sqrt",
        "root",
    }
)

_SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class Builder:
    """Creates symbols (with the user's assumptions) and guarded compound expressions."""

    def __init__(
        self, assumptions: dict[str, list[str]] | None = None, limits: Limits | None = None
    ) -> None:
        self.limits = limits or Limits()
        self._assumptions: dict[str, dict[str, bool]] = {}
        for name, items in (assumptions or {}).items():
            unknown = set(items) - ALLOWED_ASSUMPTIONS
            if unknown:
                raise ParseError(f"unknown assumption(s): {sorted(unknown)}")
            self._assumptions[name] = {item: True for item in items}
        self.symbols: dict[str, sp.Symbol] = {}

    # -- atoms -------------------------------------------------------------------------
    def integer(self, digits: str) -> sp.Expr:
        if len(digits) > self.limits.max_int_digits:
            raise LimitError("number has too many digits")
        return sp.Integer(int(digits))

    def decimal(self, text: str) -> sp.Expr:
        if len(text) > self.limits.max_int_digits:
            raise LimitError("number has too many digits")
        # Exact rational from the decimal string; never a binary float.
        try:
            return sp.Rational(text)
        except (ValueError, TypeError) as exc:
            raise ParseError("invalid number") from exc

    def symbol(self, name: str) -> sp.Symbol:
        if len(name) > SYMBOL_NAME_MAX or not _SYMBOL_RE.match(name) or "__" in name:
            raise ParseError(f"invalid symbol name {name!r}")
        if name in RESERVED_NAMES:
            raise ParseError(f"{name!r} is a reserved function or constant name")
        if name not in self.symbols:
            try:
                self.symbols[name] = sp.Symbol(name, **self._assumptions.get(name, {}))
            except ValueError as exc:  # sympy's InconsistentAssumptions is a ValueError
                raise ParseError(f"the assumptions for {name!r} contradict each other") from exc
        return self.symbols[name]

    # -- compound ----------------------------------------------------------------------
    def power(self, base: sp.Expr, exponent: sp.Expr) -> sp.Expr:
        # Reject explosive numeric powers *before* SymPy evaluates them.
        if base.is_Number and exponent.is_Number:
            if abs(exponent) > self.limits.max_pow_exponent:
                raise LimitError("exponent too large")
            if base.is_Integer and abs(base) > 1:
                # Bound the digit count of the result without computing it.
                digits = abs(int(exponent)) * len(str(abs(int(base)))) if exponent.is_Integer else 0
                if digits > 10_000:
                    raise LimitError("result of power would be too large")
        return sp.Pow(base, exponent)

    def function(self, name: str, args: Iterable[sp.Expr]) -> sp.Expr:
        canonical = TEXT_FUNCTION_ALIASES.get(name, name)
        if canonical not in FUNCTIONS:
            raise ParseError(f"unsupported function {name!r}")
        argv = list(args)
        if len(argv) != 1:
            raise ParseError(f"{name} takes exactly one argument")
        return FUNCTIONS[canonical](argv[0])

    def log_base(self, argument: sp.Expr, base: sp.Expr) -> sp.Expr:
        return sp.log(argument, base)

    def root(self, radicand: sp.Expr, index: sp.Expr | None = None) -> sp.Expr:
        if index is None:
            return sp.sqrt(radicand)
        return self.power(radicand, sp.Integer(1) / index)

    def divide(self, numerator: sp.Expr, denominator: sp.Expr) -> sp.Expr:
        if denominator == 0:
            raise ParseError("division by zero")
        return numerator / denominator

    # -- final guards ------------------------------------------------------------------
    def check_tree(self, expr: sp.Basic) -> None:
        nodes = sum(1 for _ in sp.preorder_traversal(expr))
        if nodes > self.limits.max_tree_nodes:
            raise LimitError("expression is too large")
        if _depth(expr) > self.limits.max_tree_depth:
            raise LimitError("expression is nested too deeply")


def _depth(expr: Any) -> int:
    # Iterative to avoid recursion limits on adversarial input.
    deepest = 0
    stack: list[tuple[Any, int]] = [(expr, 1)]
    while stack:
        node, level = stack.pop()
        deepest = max(deepest, level)
        for child in getattr(node, "args", ()):
            stack.append((child, level + 1))
    return deepest
