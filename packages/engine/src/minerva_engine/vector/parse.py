"""Parsers for vector expressions (LaTeX and plain text).

The LaTeX parser extends the strict scalar parser (`latexparse._Parser`) with the notation of vector
calculus; the scalar parser itself is untouched. Like it, this parser never guesses:

* `\\nabla f`, `\\nabla\\cdot A`, `\\nabla\\times A`, `\\nabla^2 f` act on the *one factor* that follows
  (like `\\sin x`); `\\nabla f g` is rejected as ambiguous, write `\\nabla(fg)` or `(\\nabla f)g`.
* `\\cdot` between two vectors is a dot product and `\\times` between two vectors a cross product. The
  implicit product of two vectors, and `\\times` between two scalars that are not numerals, are errors.
* `\\mathbf{A}`, `\\vec{A}`, `\\boldsymbol{A}` name a declared vector field; `\\mathbf{r}` is the position
  vector when no field `r` is declared; `\\hat{x}` is a unit vector of the coordinate system.
* `\\partial_t X`, `\\frac{\\partial X}{\\partial t}`, `\\frac{\\partial}{\\partial x} X`,
  `\\frac{\\partial^2 X}{\\partial x \\partial y}` are partial derivatives with respect to a coordinate or `t`.
* `(\\mathbf A \\cdot \\nabla)\\mathbf B` is the directional derivative.

Text syntax: `grad(f)`, `div(A)`, `curl(A)`, `lap(f)`, `dot(A, B)`, `cross(A, B)`, `pd(X, x, y)`,
`adv(A, B)`, `pos()`, `unit(x)`, with `*` for scalar products only.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import Any

import sympy as sp

from ..build import CONSTANTS, FUNCTIONS, TEXT_FUNCTION_ALIASES, Builder
from ..errors import LimitError, ParseError
from ..latexparse import _GREEK, _GREEK_NAME, Token, _Parser, _tokenize
from ..models import Limits
from .expressions import (
    Adv,
    AdvOperator,
    Curl,
    Div,
    Grad,
    Lap,
    Pd,
    Pos,
    Unit,
    VectorTypeError,
    cross,
    dot,
    kind_of,
    product,
)

# `E` and `I` are the constants e and i in text syntax and reserved symbol names elsewhere; a field
# called E (the electric field) or I (a current) is the normal case here, so this builder allows them.
_ALLOWED_RESERVED = frozenset({"E", "I"})


class _VectorBuilder(Builder):
    def symbol(self, name: str) -> sp.Symbol:
        if name in _ALLOWED_RESERVED:
            if name not in self.symbols:
                if name in self.noncommutative:
                    self.symbols[name] = sp.Symbol(name, commutative=False)
                else:
                    self.symbols[name] = sp.Symbol(name, **self._assumptions.get(name, {}))
            return self.symbols[name]
        return super().symbol(name)


_BOLD = {"mathbf", "vec", "boldsymbol", "bm", "overrightarrow"}
_MAX_AST_DEPTH = 100


class _VectorParser(_Parser):
    def __init__(
        self,
        tokens: list[Token],
        builder: Builder,
        vectors: frozenset[str],
        directions: frozenset[str],
    ) -> None:
        super().__init__(tokens, builder)
        self.vectors = vectors
        self.directions = directions  # unit vector names of the coordinate system
        self._operator_end = -1  # token index just after the last operator application

    # -- products ------------------------------------------------------------------------
    def term(self) -> sp.Expr:
        value = self.factor()
        while True:
            tok = self.cur
            if tok.kind == "cmd" and tok.text in {"cdot", "times"}:
                self.advance()
                if tok.text == "cdot" and self._is_advective_nabla():
                    self.advance()  # \nabla
                    value = AdvOperator(value)
                    continue
                rhs = self.factor()
                value = dot(value, rhs) if tok.text == "cdot" else cross(value, rhs)
            elif self.accept_op("*"):
                value = product(value, self.factor())
            elif self.accept_op("/"):
                value = self._divide(value, self.factor())
            elif tok.kind == "cmd" and tok.text == "div":
                self.advance()
                value = self._divide(value, self.factor())
            elif self._starts_factor():
                if self.i == self._operator_end:
                    raise ParseError(
                        "ambiguous: an operator such as \\nabla acts on the one factor after it; "
                        "write \\nabla(fg) or (\\nabla f)g"
                    )
                value = product(value, self.power())
            else:
                return value

    def _divide(self, numerator: sp.Expr, denominator: sp.Expr) -> sp.Expr:
        if kind_of(denominator) != "scalar":
            raise VectorTypeError("cannot divide by a vector")
        return self.b.divide(numerator, denominator)

    def _is_advective_nabla(self) -> bool:
        """`... \\cdot \\nabla )`: the operator `(A \\cdot \\nabla)`."""
        nxt, after = self.t[self.i], self.t[self.i + 1]
        return (
            nxt.kind == "cmd" and nxt.text == "nabla" and after.kind == "op" and after.text == ")"
        )

    def power(self) -> sp.Expr:
        value = super().power()
        if isinstance(value, AdvOperator):
            return Adv(value.args[0], super().power())
        return value

    def _atom(self) -> sp.Expr:
        value = super()._atom()
        if isinstance(value, sp.Abs) and kind_of(value.args[0]) == "vector":
            return sp.sqrt(dot(value.args[0], value.args[0]))  # |A| is the magnitude
        return value

    # -- commands ------------------------------------------------------------------------
    def _command(self, name: str) -> sp.Expr:
        if name == "nabla":
            self.advance()
            return self._nabla()
        if name in _BOLD:
            self.advance()
            return self._vector_symbol(name)
        if name == "hat":
            self.advance()
            return self._unit_vector()
        if name == "partial":
            self.advance()
            return self._partial_prefix()
        if name in {"frac", "dfrac", "tfrac"} and self._partial_fraction_ahead():
            self.advance()  # \\frac
            return self._partial_fraction()
        return super()._command(name)

    def _operand(self) -> sp.Expr:
        return self.power()

    def _finish(self, value: sp.Expr) -> sp.Expr:
        self._operator_end = self.i
        kind_of(value)  # type-check the operator application now, for a precise message
        return value

    def _nabla(self) -> sp.Expr:
        tok = self.cur
        if tok.kind == "cmd" and tok.text == "cdot":
            self.advance()
            return self._finish(Div(self._operand()))
        if tok.kind == "cmd" and tok.text == "times":
            self.advance()
            return self._finish(Curl(self._operand()))
        if tok.kind == "op" and tok.text == "^":
            self.advance()
            exponent = self.script()
            if exponent != 2:
                raise ParseError("only \\nabla^2 (the Laplacian) is supported")
            return self._finish(Lap(self._operand()))
        return self._finish(Grad(self._operand()))

    def _name_text(self) -> str:
        """The name inside `\\mathbf{...}`: a letter or Greek letter with an optional subscript."""
        braced = self.accept_op("{")
        tok = self.advance()
        if tok.kind == "letter":
            name = tok.text
        elif tok.kind == "cmd" and tok.text in _GREEK:
            name = _GREEK_NAME.get(tok.text, tok.text)
        else:
            raise ParseError("a vector name must be a letter")
        if self.accept_op("_"):
            name = f"{name}_{self._subscript_text()}"
        if braced:
            self.expect_op("}")
        return name

    def _vector_symbol(self, command: str) -> sp.Expr:
        name = self._name_text()
        if name in self.vectors:
            return self.b.symbol(name)
        if name == "r":
            return Pos(sp.Integer(0))
        raise ParseError(f"\\{command}{{{name}}}: {name} is not a declared vector field")

    def _unit_vector(self) -> sp.Expr:
        name = self._name_text()
        if name not in self.directions:
            raise ParseError(
                f"\\hat{{{name}}} is not a unit vector of this coordinate system "
                f"({', '.join(sorted(self.directions))})"
            )
        return Unit(sp.Symbol(name))

    # -- partial derivatives ---------------------------------------------------------------
    def _variable(self) -> sp.Symbol:
        tok = self.advance()
        if tok.kind == "letter":
            name = tok.text
        elif tok.kind == "cmd" and tok.text in _GREEK:
            name = _GREEK_NAME.get(tok.text, tok.text)
        else:
            raise ParseError("expected a coordinate after \\partial")
        return self.b.symbol(name)

    def _partial_prefix(self) -> sp.Expr:
        """`\\partial_t X` and `\\partial_x^2 X`."""
        if not self.accept_op("_"):
            raise ParseError("write \\partial_x X or \\frac{\\partial X}{\\partial x}")
        braced = self.accept_op("{")
        var = self._variable()
        if braced:
            self.expect_op("}")
        order = 1
        if self.accept_op("^"):
            order = self._order(self.script())
        operand = self._operand()
        return self._finish(self._derivative(operand, [var] * order))

    @staticmethod
    def _order(value: sp.Expr) -> int:
        if not (value.is_Integer and 1 <= int(value) <= 4):
            raise ParseError("the order of a derivative must be an integer from 1 to 4")
        return int(value)

    def _derivative(self, operand: sp.Expr, variables: list[sp.Symbol]) -> sp.Expr:
        result = operand
        for var in variables:
            result = Pd(result, var)
        return result

    def _partial_fraction_ahead(self) -> bool:
        a, b = self.t[self.i + 1], self.t[self.i + 2]  # self.cur is the \\frac token
        return a.kind == "op" and a.text == "{" and b.kind == "cmd" and b.text == "partial"

    def _partial_fraction(self) -> sp.Expr:
        self.expect_op("{")
        self.advance()  # \partial
        order: int | None = None
        if self.accept_op("^"):
            order = self._order(self.script())
        operand: sp.Expr | None = None
        if not (self.cur.kind == "op" and self.cur.text == "}"):
            operand = self.expr()
        self.expect_op("}")
        self.expect_op("{")
        variables: list[sp.Symbol] = []
        while self.cur.kind == "cmd" and self.cur.text == "partial":
            self.advance()
            braced = self.accept_op("{")
            var = self._variable()
            if braced:
                self.expect_op("}")
            count = 1
            if self.accept_op("^"):
                count = self._order(self.script())
            variables.extend([var] * count)
        self.expect_op("}")
        if not variables or len(variables) > 4:
            raise ParseError("expected \\partial x (up to four derivatives) in the denominator")
        if order is not None and order != len(variables):
            raise ParseError("the order of the derivative does not match its denominator")
        if operand is None:
            operand = self._operand()
        return self._finish(self._derivative(operand, variables[::-1]))


def parse_vector_latex(
    source: str,
    *,
    vectors: Iterable[str],
    directions: Iterable[str],
    assumptions: dict[str, list[str]],
    limits: Limits,
    denominators: list[sp.Expr] | None = None,
) -> sp.Expr:
    if len(source) > limits.max_input_chars:
        raise LimitError("input is too long")
    names = frozenset(vectors)
    builder = _VectorBuilder(assumptions, limits, names, None, False, None)
    if denominators is not None:
        builder.denominators = denominators
    parser = _VectorParser(_tokenize(source), builder, names, frozenset(directions))
    try:
        expr = parser.parse()
    except RecursionError as exc:
        raise LimitError("expression is nested too deeply") from exc
    builder.check_tree(expr)
    kind_of(expr)
    return expr


# -- text ------------------------------------------------------------------------------------
def parse_vector_text(
    text: str,
    *,
    vectors: Iterable[str],
    names: Iterable[str] = (),
    directions: Iterable[str],
    assumptions: dict[str, list[str]],
    limits: Limits,
    denominators: list[sp.Expr] | None = None,
) -> sp.Expr:
    if len(text) > limits.max_input_chars:
        raise LimitError("input is too long")
    names = frozenset(vectors)
    builder = _VectorBuilder(assumptions, limits, names, None, False, None)
    if denominators is not None:
        builder.denominators = denominators
    try:
        tree = ast.parse(text.strip().replace("^", "**"), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise ParseError("could not parse the expression") from exc
    expr = _convert(tree.body, builder, frozenset(directions), frozenset(names), 0)
    builder.check_tree(expr)
    kind_of(expr)
    return expr


def _convert(
    node: ast.AST, b: Builder, directions: frozenset[str], names: frozenset[str], depth: int
) -> sp.Expr:
    if depth > _MAX_AST_DEPTH:
        raise LimitError("expression is nested too deeply")
    d = depth + 1

    def conv(n: ast.AST) -> sp.Expr:
        return _convert(n, b, directions, names, d)

    if isinstance(node, ast.Constant):
        value: Any = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ParseError("only numbers are allowed as literals")
        return b.integer(str(value)) if isinstance(value, int) else b.decimal(repr(value))
    if isinstance(node, ast.Name):
        if node.id in CONSTANTS and node.id not in names:
            return CONSTANTS[node.id]
        return b.symbol(node.id)
    if isinstance(node, ast.UnaryOp):
        operand = conv(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ParseError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left, right = conv(node.left), conv(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return product(left, right)
        if isinstance(node.op, ast.Div):
            if kind_of(right) != "scalar":
                raise VectorTypeError("cannot divide by a vector")
            return b.divide(left, right)
        if isinstance(node.op, ast.Pow):
            return b.power(left, right)
        raise ParseError("unsupported operator")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ParseError("unsupported function call")
        return _call(node.func.id, node.args, b, directions, names, d)
    raise ParseError(f"unsupported syntax: {type(node).__name__}")


def _call(
    name: str,
    raw: list[ast.expr],
    b: Builder,
    directions: frozenset[str],
    names: frozenset[str],
    depth: int,
) -> sp.Expr:
    def arg(n: ast.AST) -> sp.Expr:
        return _convert(n, b, directions, names, depth)

    def need(count: int) -> None:
        if len(raw) != count:
            raise ParseError(f"{name} takes {count} argument(s)")

    if name == "grad":
        need(1)
        return Grad(arg(raw[0]))
    if name == "div":
        need(1)
        return Div(arg(raw[0]))
    if name == "curl":
        need(1)
        return Curl(arg(raw[0]))
    if name == "lap":
        need(1)
        return Lap(arg(raw[0]))
    if name == "dot":
        need(2)
        left, right = arg(raw[0]), arg(raw[1])
        if kind_of(left) != "vector" or kind_of(right) != "vector":
            raise VectorTypeError("dot needs two vectors")
        return dot(left, right)
    if name == "cross":
        need(2)
        left, right = arg(raw[0]), arg(raw[1])
        if kind_of(left) != "vector" or kind_of(right) != "vector":
            raise VectorTypeError("cross needs two vectors")
        return cross(left, right)
    if name == "adv":
        need(2)
        return Adv(arg(raw[0]), arg(raw[1]))
    if name == "pos":
        need(0)
        return Pos(sp.Integer(0))
    if name == "unit":
        need(1)
        target = raw[0]
        if not isinstance(target, ast.Name) or target.id not in directions:
            raise ParseError(f"unit(...) needs one of {', '.join(sorted(directions))}")
        return Unit(sp.Symbol(target.id))
    if name == "pd":
        if len(raw) < 2 or len(raw) > 5:
            raise ParseError("pd(expression, variable, ...) takes up to four variables")
        result = arg(raw[0])
        for var in raw[1:]:
            if not isinstance(var, ast.Name):
                raise ParseError("pd differentiates with respect to a coordinate name")
            result = Pd(result, b.symbol(var.id))
        return result
    if name == "dt":
        need(1)
        return Pd(arg(raw[0]), b.symbol("t"))
    if name in {"sqrt"}:
        need(1)
        return b.root(arg(raw[0]))
    canonical = TEXT_FUNCTION_ALIASES.get(name, name)
    if canonical not in FUNCTIONS:
        raise ParseError(f"unsupported function {name!r}")
    return b.function(canonical, [arg(a) for a in raw])
