"""Plain-text expression parser.

Uses Python's `ast` module only to *parse* a tiny expression grammar; the tree is then
walked by an explicit whitelist and converted to SymPy objects through `Builder`.
No `eval`, `exec`, `compile` or `sympify` is involved, so no text from the user is ever
executed.

Accepted: numbers, names (symbols), `pi`, `E`, `I`, `+ - * / ** ^`, parentheses, and the
functions listed in `build.FUNCTIONS` (plus `sqrt` and `root(x, n)`).
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import Any

import sympy as sp

from .build import CONSTANTS, FUNCTIONS, TEXT_FUNCTION_ALIASES, Builder, NotationLog
from .errors import LimitError, ParseError
from .models import Limits

_MAX_AST_DEPTH = 100


def parse_text(
    text: str,
    assumptions: dict[str, list[str]] | None = None,
    limits: Limits | None = None,
    denominators: list[sp.Expr] | None = None,
    noncommutative: Iterable[str] = (),
    functions: dict[str, tuple[str, ...]] | None = None,
    strict_calls: bool = False,
    notation: NotationLog | None = None,
    *,
    allow_noncommutative: bool = False,
) -> sp.Expr:
    limits = limits or Limits()
    if len(text) > limits.max_input_chars:
        raise LimitError("input is too long")
    builder = Builder(
        assumptions,
        limits,
        noncommutative,
        functions,
        strict_calls,
        notation,
        allow_noncommutative=allow_noncommutative,
    )
    if denominators is not None:
        builder.denominators = denominators
    source = text.strip().replace("^", "**")
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise ParseError("could not parse the expression") from exc
    expr = _convert(tree.body, builder, 0)
    builder.check_tree(expr)
    return expr


def _convert(node: ast.AST, b: Builder, depth: int) -> sp.Expr:
    if depth > _MAX_AST_DEPTH:
        raise LimitError("expression is nested too deeply")
    d = depth + 1

    if isinstance(node, ast.Constant):
        value: Any = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ParseError("only numbers are allowed as literals")
        # Use the source text form so floats become exact rationals.
        return b.integer(str(value)) if isinstance(value, int) else b.decimal(repr(value))

    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        return b.symbol(node.id)

    if isinstance(node, ast.UnaryOp):
        operand = _convert(node.operand, b, d)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ParseError("unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left = _convert(node.left, b, d)
        right = _convert(node.right, b, d)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return b.divide(left, right)
        if isinstance(node.op, ast.Pow):
            return b.power(left, right)
        raise ParseError("unsupported operator")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ParseError("unsupported function call")
        name = node.func.id
        args = [_convert(a, b, d) for a in node.args]
        if name in b.functions:
            return b.declared_call(name, args)
        if len(args) == 1:
            reading = b.distribution_reading(name)
            if reading in ("dirac_delta", "heaviside"):
                return b.distribution_function(reading, args[0])
            if reading == "unconfirmed":
                # Placeholder so that parsing can finish: the operation asks before using it.
                return b.symbol(name) * args[0]
        if name == "sqrt" and len(args) == 1:
            return b.root(args[0])
        if name == "root" and len(args) == 2:
            if args[1] == 0:
                raise ParseError("root index cannot be zero")
            return b.root(args[0], args[1])
        if name == "log" and len(args) == 2:
            return b.log_base(args[0], args[1])
        canonical = TEXT_FUNCTION_ALIASES.get(name, name)
        if b.strict_calls and canonical not in FUNCTIONS:
            raise ParseError(f"{name}(…) is not a known function: declare {name} as a function")
        return b.function(canonical, args)

    raise ParseError(f"unsupported syntax: {type(node).__name__}")
