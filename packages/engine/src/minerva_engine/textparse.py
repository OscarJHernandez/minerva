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
    allow_sums: bool = False,
    random_variables: Iterable[str] = (),
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
        allow_sums=allow_sums,
        random_variables=random_variables,
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

    if isinstance(node, ast.Subscript):
        return _subscript(node, b, d)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ParseError("unsupported function call")
        name = node.func.id
        if name in ("sum", "prod", "product") and name not in b.functions:
            raise ParseError(
                f"write {name.capitalize() if name != 'prod' else 'Product'}(…) with a capital letter, "
                "as in Sum(x[i], (i, 1, n))"
            )
        if b.allow_sums and name in ("Sum", "Product"):
            return _big_operator(node, b, d)
        if b.random and name in ("Var", "Cov"):
            args = [_convert(a, b, d) for a in node.args]
            if len(args) != (1 if name == "Var" else 2):
                raise ParseError(
                    f"{name} takes {'one argument' if name == 'Var' else 'two arguments'}"
                )
            return b.variance(args[0]) if name == "Var" else b.covariance(args[0], args[1])
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


def _subscript(node: ast.Subscript, b: Builder, depth: int) -> sp.Expr:
    """`x[i]` (an element of indexed data) and `E[expression]` (an expectation)."""
    if not isinstance(node.value, ast.Name):
        raise ParseError("unsupported syntax: Subscript")
    name = node.value.id
    if isinstance(node.slice, ast.Slice | ast.Tuple):
        raise ParseError("an index must be a single integer expression")
    inner = _convert(node.slice, b, depth)
    if name == "E" and b.random:
        return b.expectation(inner)
    if not b.allow_sums:
        raise ParseError("unsupported syntax: Subscript")
    return b.indexed(name, inner)


def _big_operator(node: ast.Call, b: Builder, depth: int) -> sp.Expr:
    """`Sum(body, (i, a, n))`; with several limits the first one is the innermost, as in SymPy."""
    if len(node.args) < 2 or len(node.args) > 1 + 2:
        raise ParseError(
            "Sum and Product take a summand and one or two (index, lower, upper) limits"
        )
    limit_nodes = node.args[1:]
    opened: list[tuple[sp.Symbol, sp.Expr, sp.Expr]] = []
    # The outermost limit is the last one: read its bounds first, outside every scope.
    try:
        for limit in reversed(limit_nodes):
            if (
                not isinstance(limit, ast.Tuple)
                or len(limit.elts) != 3
                or not isinstance(limit.elts[0], ast.Name)
            ):
                raise ParseError("a limit is written (index, lower, upper)")
            lower = _convert(limit.elts[1], b, depth)
            upper = _convert(limit.elts[2], b, depth)
            index = b.open_sum(limit.elts[0].id, lower)
            opened.append((index, lower, upper))
        body = _convert(node.args[0], b, depth)
        kind = "sum" if node.func.id == "Sum" else "product"  # type: ignore[attr-defined]
        result = body  # `opened` holds the outermost limit first: wrap from the inside out
        for index, lower, upper in reversed(opened):
            result = b.summation(kind, result, index, lower, upper)
        return result
    finally:
        for index, _, _ in opened:
            b.close_sum(index.name)
