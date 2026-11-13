"""Strict LaTeX expression parser (the subset MathLive produces for ordinary formulas).

SymPy's built-in LaTeX parser returned ambiguous or wrong trees for common inputs while
this was written (for example `\\sin^2 x + \\cos^2 x`), so Minerva parses LaTeX itself with a
small recursive-descent grammar that builds SymPy objects through `Builder`. Anything
outside the grammar is rejected with a `ParseError`; the parser never guesses.

Conventions (documented in docs/architecture.md):
* `e` and `i` are ordinary symbols; use `\\exponentialE` / `\\mathrm{e}` and
  `\\imaginaryI` / `\\mathrm{i}` for the constants. `\\pi` is pi.
* `\\sin x` takes a single factor (with an optional power) as its argument; a bare numeric
  argument such as `\\sin 2x` is rejected as ambiguous, use parentheses.
* `\\sin^{-1}` is rejected as ambiguous; use `\\arcsin`.
* Unbraced exponents and subscripts are a single token, as in TeX (`x^23` is `x^2 \\cdot 3`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import sympy as sp

from .build import FUNCTIONS, Builder, NotationLog
from .errors import LimitError, ParseError
from .models import Limits

_GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta",
    "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "rho", "sigma", "tau",
    "upsilon", "phi", "varphi", "chi", "psi", "omega",
    "Delta", "Theta", "Lambda", "Xi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
}  # fmt: skip
_GREEK_NAME = {"varepsilon": "epsilon", "vartheta": "theta", "varphi": "phi"}

_FUNCTION_COMMANDS = {
    "sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "coth",
    "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh", "exp", "ln", "log",
}  # fmt: skip
_SPACING = {",", ";", ":", "!", " ", "quad", "qquad", "thinspace", "medspace", "thickspace"}
_IGNORED_SIZERS = {"left", "right", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr"}

_TOKEN_RE = re.compile(
    r"""
    (?P<num>\d+(?:\.\d+)?|\.\d+)
  | (?P<cmd>\\(?:[A-Za-z]+|.))
  | (?P<letter>[A-Za-z])
  | (?P<op>[-+*/^_(){}\[\]|=,<>!])
  | (?P<space>\s+)
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    kind: str  # num | cmd | letter | op | end
    text: str


def _tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(source):
        m = _TOKEN_RE.match(source, pos)
        if not m:
            raise ParseError(f"unexpected character {source[pos]!r}")
        pos = m.end()
        kind = m.lastgroup or ""
        text = m.group()
        if kind == "space":
            continue
        if kind == "cmd":
            name = text[1:]
            if name in _SPACING:
                continue
            if name in _IGNORED_SIZERS:
                continue
            tokens.append(Token("cmd", name))
        else:
            tokens.append(Token(kind, text))
    tokens.append(Token("end", ""))
    return tokens


class _Parser:
    def __init__(self, tokens: list[Token], builder: Builder) -> None:
        self.t = tokens
        self.i = 0
        self.b = builder
        self.abs_depth = 0
        self.depth = 0

    # -- token helpers -----------------------------------------------------------------
    @property
    def cur(self) -> Token:
        return self.t[self.i]

    def advance(self) -> Token:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def accept_op(self, text: str) -> bool:
        if self.cur.kind == "op" and self.cur.text == text:
            self.i += 1
            return True
        return False

    def expect_op(self, text: str) -> None:
        if not self.accept_op(text):
            raise ParseError(f"expected {text!r}")

    def enter(self) -> None:
        self.depth += 1
        if self.depth > self.b.limits.max_tree_depth:
            raise LimitError("expression is nested too deeply")

    def leave(self) -> None:
        self.depth -= 1

    # -- grammar -----------------------------------------------------------------------
    def parse(self) -> sp.Expr:
        expr = self.expr()
        if self.cur.kind != "end":
            raise ParseError(self._unexpected())
        return expr

    def _unexpected(self) -> str:
        tok = self.cur
        shown = f"\\{tok.text}" if tok.kind == "cmd" else tok.text
        return f"unexpected {shown!r}"

    def expr(self) -> sp.Expr:
        self.enter()
        sign = 1
        if self.cur.kind == "op" and self.cur.text in "+-":
            sign = -1 if self.advance().text == "-" else 1
        total = sign * self.term()
        while self.cur.kind == "op" and self.cur.text in "+-":
            op = self.advance().text
            rhs = self.term()
            total = total + rhs if op == "+" else total - rhs
        self.leave()
        return total

    def _starts_factor(self) -> bool:
        tok = self.cur
        if tok.kind in ("num", "letter"):
            return True
        if tok.kind == "op":
            if tok.text == "|":
                return self.abs_depth == 0
            return tok.text in "({"
        if tok.kind == "cmd":
            return tok.text not in {"cdot", "times", "div", "right", "le", "ge", "neq", "to"}
        return False

    def term(self) -> sp.Expr:
        value = self.factor()
        while True:
            if self.cur.kind == "cmd" and self.cur.text in {"cdot", "times"}:
                self.advance()
                value = value * self.factor()
            elif self.accept_op("*"):
                value = value * self.factor()
            elif self.accept_op("/"):
                value = self.b.divide(value, self.factor())
            elif self.cur.kind == "cmd" and self.cur.text == "div":
                self.advance()
                value = self.b.divide(value, self.factor())
            elif self._starts_factor():
                value = value * self.power()
            else:
                return value

    def factor(self) -> sp.Expr:
        if self.cur.kind == "op" and self.cur.text in "+-":
            neg = self.advance().text == "-"
            inner = self.factor()
            return -inner if neg else inner
        return self.power()

    def power(self) -> sp.Expr:
        base = self.atom()
        if self.accept_op("^"):
            exponent = self.script()
            if self.cur.kind == "op" and self.cur.text == "^":
                raise ParseError("ambiguous double exponent, use braces")
            return self.b.power(base, exponent)
        return base

    def script(self) -> sp.Expr:
        """Exponent/subscript argument: a braced expression or one TeX token."""
        if self.accept_op("{"):
            value = self.expr()
            self.expect_op("}")
            return value
        tok = self.cur
        if tok.kind == "num":
            self.advance()
            if not tok.text[0].isdigit():
                raise ParseError("invalid exponent, use braces")
            # TeX takes only the first digit of an unbraced script.
            if len(tok.text) > 1:
                self.t[self.i : self.i] = [Token("num", tok.text[1:])]
            return self.b.integer(tok.text[0])
        if tok.kind == "letter":
            self.advance()
            return self.b.symbol(tok.text)
        if tok.kind == "cmd":
            return self.atom()
        raise ParseError("expected an exponent")

    def atom(self) -> sp.Expr:
        self.enter()
        try:
            return self._atom()
        finally:
            self.leave()

    def _atom(self) -> sp.Expr:
        tok = self.cur
        if tok.kind == "num":
            self.advance()
            return self.b.decimal(tok.text) if "." in tok.text else self.b.integer(tok.text)
        if tok.kind == "letter":
            self.advance()
            return self._symbol_with_subscript(tok.text)
        if tok.kind == "op":
            if tok.text == "(":
                self.advance()
                value = self.expr()
                self.expect_op(")")
                return value
            if tok.text == "{":
                self.advance()
                value = self.expr()
                self.expect_op("}")
                return value
            if tok.text == "|":
                self.advance()
                self.abs_depth += 1
                value = self.expr()
                self.abs_depth -= 1
                self.expect_op("|")
                return sp.Abs(value)
            raise ParseError(self._unexpected())
        if tok.kind == "cmd":
            return self._command(tok.text)
        raise ParseError("unexpected end of input")

    def _symbol_with_subscript(self, name: str) -> sp.Expr:
        if self.accept_op("_"):
            name = f"{name}_{self._subscript_text()}"
        if name in self.b.functions:
            return self._declared_call(name)
        if self.cur.kind == "op" and self.cur.text == "(":
            reading = self.b.distribution_reading(name)
            if reading in ("dirac_delta", "heaviside"):
                self.advance()
                argument = self.expr()
                self.expect_op(")")
                return self.b.distribution_function(reading, argument)
        if self.b.strict_calls and self.cur.kind == "op" and self.cur.text == "(":
            raise ParseError(
                f"{name}(…) is ambiguous: declare {name} as a function, "
                f"or write {name}\\cdot(…) for a product"
            )
        return self.b.symbol(name)

    def _declared_call(self, name: str) -> sp.Expr:
        if not self.accept_op("("):
            args = ", ".join(self.b.functions[name])
            raise ParseError(f"{name!r} is a declared function; write {name}({args})")
        values = [self.expr()]
        while self.accept_op(","):
            values.append(self.expr())
        self.expect_op(")")
        return self.b.declared_call(name, values)

    def _subscript_text(self) -> str:
        if self.accept_op("{"):
            parts: list[str] = []
            while not self.accept_op("}"):
                tok = self.advance()
                if tok.kind in ("num", "letter"):
                    parts.append(tok.text)
                elif tok.kind == "cmd" and tok.text in _GREEK:
                    parts.append(_GREEK_NAME.get(tok.text, tok.text))
                else:
                    raise ParseError("unsupported subscript")
            if not parts:
                raise ParseError("empty subscript")
            return "".join(parts)
        tok = self.advance()
        if tok.kind in ("num", "letter") and len(tok.text) >= 1:
            if tok.kind == "num" and len(tok.text) > 1:
                self.t[self.i : self.i] = [Token("num", tok.text[1:])]
                return tok.text[0]
            return tok.text
        if tok.kind == "cmd" and tok.text in _GREEK:
            return _GREEK_NAME.get(tok.text, tok.text)
        raise ParseError("unsupported subscript")

    def _frac_argument(self) -> sp.Expr:
        """A braced group, or one token as in TeX (`\\frac12` is one half)."""
        if self.cur.kind == "op" and self.cur.text == "{":
            return self._braced()
        return self.script()

    def _braced(self) -> sp.Expr:
        self.expect_op("{")
        value = self.expr()
        self.expect_op("}")
        return value

    def _command(self, name: str) -> sp.Expr:
        self.advance()
        if name == "pi":
            return sp.pi
        if name in {"exponentialE"}:
            return sp.E
        if name in {"imaginaryI"}:
            return sp.I
        if name in {"mathrm", "operatorname"}:
            return self._mathrm(name)
        if name in {"dot", "ddot"}:
            return self._time_derivative(name)
        if name in _GREEK:
            return self._symbol_with_subscript_name(_GREEK_NAME.get(name, name))
        if name == "Gamma":
            return self._apply("gamma", None)
        if name == "frac" or name in {"dfrac", "tfrac"}:
            num = self._frac_argument()
            den = self._frac_argument()
            return self.b.divide(num, den)
        if name == "sqrt":
            index: sp.Expr | None = None
            if self.accept_op("["):
                index = self.expr()
                self.expect_op("]")
                if index == 0:
                    raise ParseError("root index cannot be zero")
            return self.b.root(self._braced(), index)
        if name in _FUNCTION_COMMANDS:
            return self._apply(name, None)
        if name in {"int", "sum", "prod", "lim", "partial", "nabla", "infty", "frac"}:
            raise ParseError(f"\\{name} is not supported in this operation")
        raise ParseError(f"unsupported LaTeX command \\{name}")

    def _time_derivative(self, name: str) -> sp.Expr:
        """`\\dot{q}` and `\\ddot{q}` are the symbols `q_dot` and `q_ddot` (velocity, acceleration)."""
        if self.accept_op("{"):
            inner = self.expr()
            self.expect_op("}")
        else:
            inner = self.atom()
        if not isinstance(inner, sp.Symbol):
            raise ParseError(f"\\{name} applies to a single symbol")
        base = inner.name
        if self.accept_op("_"):  # \\dot{\\theta}_1 is the velocity of theta_1
            base = f"{base}_{self._subscript_text()}"
        return self.b.symbol(f"{base}_{'dot' if name == 'dot' else 'ddot'}")

    def _symbol_with_subscript_name(self, name: str) -> sp.Expr:
        return self._symbol_with_subscript(name)

    def _mathrm(self, cmd: str) -> sp.Expr:
        self.expect_op("{")
        parts: list[str] = []
        while not self.accept_op("}"):
            tok = self.advance()
            if tok.kind in ("letter", "num"):
                parts.append(tok.text)
            else:
                raise ParseError(f"unsupported content in \\{cmd}")
        word = "".join(parts)
        if cmd == "mathrm" and word == "e":
            return sp.E
        if cmd == "mathrm" and word == "i":
            return sp.I
        if word in FUNCTIONS:
            return self._apply(word, None)
        raise ParseError(f"unsupported \\{cmd}{{{word}}}")

    def _apply(self, name: str, _unused: None) -> sp.Expr:
        """Parse `<fn>[_base][^power] argument`."""
        base: sp.Expr | None = None
        power: sp.Expr | None = None
        if self.cur.kind == "op" and self.cur.text == "_":
            if name not in {"log"}:
                raise ParseError(f"\\{name} does not take a subscript")
            self.advance()
            base = self.script()
        if self.accept_op("^"):
            power = self.script()
            if power == -1 and name not in {"exp", "ln", "log"}:
                raise ParseError(f"\\{name}^{{-1}} is ambiguous; use the arc-function or 1/")
        argument = self._function_argument(name)
        if base is not None:
            result = self.b.log_base(argument, base)
        else:
            result = self.b.function(name, [argument])
        return self.b.power(result, power) if power is not None else result

    def _function_argument(self, name: str) -> sp.Expr:
        tok = self.cur
        if tok.kind == "op" and tok.text == "(":
            self.advance()
            value = self.expr()
            self.expect_op(")")
            return value
        if tok.kind == "num":
            value = self.power()
            if self._starts_factor():
                raise ParseError(f"ambiguous argument of \\{name}: use parentheses")
            return value
        if tok.kind == "end" or (tok.kind == "op" and tok.text in "+-*/^=)}]|,"):
            raise ParseError(f"missing argument of \\{name}")
        return self.power()


def parse_latex(
    source: str,
    assumptions: dict[str, list[str]] | None = None,
    limits: Limits | None = None,
    denominators: list[sp.Expr] | None = None,
    noncommutative: Iterable[str] = (),
    functions: dict[str, tuple[str, ...]] | None = None,
    strict_calls: bool = False,
    notation: NotationLog | None = None,
) -> sp.Expr:
    limits = limits or Limits()
    if len(source) > limits.max_input_chars:
        raise LimitError("input is too long")
    builder = Builder(assumptions, limits, noncommutative, functions, strict_calls, notation)
    if denominators is not None:
        builder.denominators = denominators
    parser = _Parser(_tokenize(source), builder)
    try:
        expr = parser.parse()
    except RecursionError as exc:
        raise LimitError("expression is nested too deeply") from exc
    builder.check_tree(expr)
    return expr
