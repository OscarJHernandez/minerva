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

from .build import FUNCTIONS, Builder, JetBinding, NotationLog
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


def _unreadable(source: str, pos: int) -> str:
    """Why a character cannot be read, naming the notation it belongs to when that is clear."""
    char = source[pos]
    if char == "'":
        return (
            "a prime (as in f') is derivative notation for an unknown function; it is read only by the "
            "tools for functionals (variation) and for differential equations, not here"
        )
    if char == "." and source[:pos].rstrip().endswith("\\left"):
        return "an evaluation bar (\\left. ... \\right|_{...}) is not read; substitute the value instead"
    return f"unexpected character {char!r}"


def _tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(source):
        m = _TOKEN_RE.match(source, pos)
        if not m:
            raise ParseError(_unreadable(source, pos))
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
            name = {"{": "lbrace", "}": "rbrace"}.get(name, name)
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
        # Token index just after the last compound command (`\\frac{1}{2}`, `\\sqrt{x}`...): a bracket
        # group may follow it by juxtaposition, whereas after a bare letter or number it may not.
        self._compound_end = -1
        # Bases that are subscripted by a summation index somewhere in the text: `x_1` is then `x[1]`
        # everywhere, not the symbol "x_1" (see _prescan_indexed).
        self._indexed_bases = self._prescan_indexed() if builder.allow_sums else frozenset()

    # -- indexed data ------------------------------------------------------------------
    @staticmethod
    def _name_of(tok: Token) -> str | None:
        if tok.kind == "letter":
            return tok.text
        if tok.kind == "cmd" and tok.text in _GREEK:
            return _GREEK_NAME.get(tok.text, tok.text)
        return None

    def _prescan_indexed(self) -> frozenset[str]:
        """Names of bases subscripted by a summation index, found before parsing."""
        toks = self.t
        indices: set[str] = set()
        for k, tok in enumerate(toks):
            if tok.kind == "cmd" and tok.text in ("sum", "prod"):
                j = k + 1
                if toks[j].kind == "op" and toks[j].text == "_":
                    j += 1
                    if toks[j].kind == "op" and toks[j].text == "{":
                        j += 1
                    name = self._name_of(toks[j])
                    if name:
                        indices.add(name)
        bases: set[str] = set()
        for k, tok in enumerate(toks[:-2]):
            base = self._name_of(tok)
            nxt = toks[k + 1]
            if base and nxt.kind == "op" and nxt.text == "_":
                j, content = k + 2, []
                if toks[j].kind == "op" and toks[j].text == "{":
                    j += 1
                    while j < len(toks) and not (toks[j].kind == "op" and toks[j].text == "}"):
                        content.append(self._name_of(toks[j]))
                        j += 1
                else:
                    content.append(self._name_of(toks[j]))
                if indices & {c for c in content if c}:
                    bases.add(base)
        return frozenset(bases)

    def _index_script(self) -> sp.Expr:
        """The subscript of an indexed variable: one integer expression, `i`, `i+1`, `2i`, `n-i`."""
        if self.accept_op("{"):
            index = self.expr()
            if not self.accept_op("}"):
                raise ParseError("an index must be a single expression (x_{i,j} is not supported)")
            bound = self.b.bound_names
            bound_factors = [
                f for f in sp.Mul.make_args(index) if isinstance(f, sp.Symbol) and f.name in bound
            ]
            if len(bound_factors) >= 2:
                raise ParseError("x_{ij} is ambiguous and x_{i,j} is not supported: use one index")
            return index
        return self.script()

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
            if tok.text == "[":
                # `x[2]` would silently be x*2: a bracket group follows a letter or a number only
                # through an explicit \\cdot. After a compound command (`\\frac12 [a+b]`), a closing
                # parenthesis or an operator it groups.
                prev = self.t[self.i - 1] if self.i > 0 else None
                return (
                    prev is None
                    or prev.kind not in ("letter", "num")
                    or self._compound_end == self.i
                )
            return tok.text in "({"
        if tok.kind == "cmd":
            return tok.text not in {
                "cdot", "times", "div", "right", "rbrace", "le", "ge", "neq", "to",
            }  # fmt: skip
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

    def _derivative_order(self) -> int | None:
        """`^{(k)}` after a dependent function: the order k (consumed), else None (nothing consumed)."""
        t, i = self.t, self.i
        shape = ("^", "{", "(", None, ")", "}")
        if i + len(shape) > len(t):
            return None
        for offset, want in enumerate(shape):
            tok = t[i + offset]
            if want is None:
                if tok.kind != "num" or not tok.text.isdigit():
                    return None
            elif tok.kind != "op" or tok.text != want:
                return None
        order = int(t[i + 3].text)
        self.i = i + len(shape)
        return order

    def power(self) -> sp.Expr:
        base = self.atom()
        jets = self.b.jets
        if jets is not None and isinstance(base, sp.Symbol) and base.name in jets.dependent:
            order = self._derivative_order()
            if order is not None:
                if self.cur.kind == "op" and self.cur.text == "(":
                    self._variable_call(base.name)  # x^{(k)}(t): the derivative at t
                return jets.derivative(base.name, order)
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
            if tok.text == "[":
                self.advance()
                value = self.expr()
                if self.cur.kind == "op" and self.cur.text == ",":
                    raise ParseError(
                        "a comma inside square brackets is ambiguous (a commutator or an interval): "
                        "write the expression alone"
                    )
                self.expect_op("]")
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
            value = self._command(tok.text)
            if not isinstance(value, sp.Symbol):
                self._compound_end = self.i
            return value
        raise ParseError("unexpected end of input")

    def _symbol_with_subscript(self, name: str) -> sp.Expr:
        if self.accept_op("_"):
            if name in self._indexed_bases:
                return self.b.indexed(name, self._index_script())
            name = f"{name}_{self._subscript_text()}"
        if name in self.b.functions:
            return self._declared_call(name)
        jets = self.b.jets
        if (
            jets is not None
            and name in jets.dependent
            and self.cur.kind == "op"
            and self.cur.text == "("
        ):
            self._variable_call(name)  # x(t) is the function x itself
            return jets.symbols[name]
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

    def _variable_call(self, name: str) -> None:
        """Consume `(t)` after a dependent function: it must be exactly the independent variable."""
        jets = self.b.jets
        if jets is None:
            raise ParseError("unexpected '('")
        self.expect_op("(")
        argument = self.expr()
        self.expect_op(")")
        if argument != jets.symbols[jets.variable]:
            raise ParseError(
                f"{name} is a function of {jets.variable}: write {name}({jets.variable})"
            )

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

    # -- Leibniz derivative notation ----------------------------------------------------
    @staticmethod
    def _is_op(tok: Token, text: str) -> bool:
        return tok.kind == "op" and tok.text == text

    @staticmethod
    def _is_d(tok: Token) -> bool:
        return tok.kind == "letter" and tok.text == "d"

    @staticmethod
    def _name_of_token(tok: Token) -> str | None:
        if tok.kind == "letter":
            return tok.text
        if tok.kind == "cmd" and tok.text in _GREEK:
            return _GREEK_NAME.get(tok.text, tok.text)
        return None

    def _group_end(self, start: int) -> int | None:
        """Index of the `}` that closes the `{` at `start`."""
        depth = 0
        for k in range(start, len(self.t)):
            if self._is_op(self.t[k], "{"):
                depth += 1
            elif self._is_op(self.t[k], "}"):
                depth -= 1
                if depth == 0:
                    return k
        return None

    def _script_order(self, toks: list[Token]) -> tuple[int, list[Token]] | None:
        """`^k` or `^{k}` at the start of `toks`: (k, the tokens after it); none: (1, toks)."""
        if not toks or not self._is_op(toks[0], "^"):
            return 1, toks
        if len(toks) >= 2 and toks[1].kind == "num" and toks[1].text.isdigit():
            return int(toks[1].text), toks[2:]
        if (
            len(toks) >= 4
            and self._is_op(toks[1], "{")
            and toks[2].kind == "num"
            and toks[2].text.isdigit()
            and self._is_op(toks[3], "}")
        ):
            return int(toks[2].text), toks[4:]
        return None

    def _leibniz(self) -> sp.Expr | None:
        """`\\frac{dX}{dV}`, `\\frac{d^kX}{dV^k}` and the operator form `\\frac{d}{dV}`.

        Read as plain fractions the d's cancel (`\\frac{dx}{dy}` would be x/y), a wrong answer that
        verifies perfectly. Outside a functional (jets off) the notation is refused; with jets the
        derivative of a declared unknown function with respect to the independent variable is that
        derivative's jet, and every other shape is refused. A caller that declared a symbol `d`
        (assumptions) gets ordinary fractions. None: not this notation, nothing consumed.
        """
        if self.b.declares("d"):
            return None
        t, i = self.t, self.i
        if not self._is_op(t[i], "{"):
            return None
        n_end = self._group_end(i)
        if n_end is None or not self._is_op(t[n_end + 1], "{"):
            return None
        d_end = self._group_end(n_end + 1)
        if d_end is None:
            return None
        num, den = t[i + 1 : n_end], t[n_end + 2 : d_end]
        if not num or not den or not self._is_d(num[0]) or not self._is_d(den[0]):
            return None
        if len(den) < 2 or self._name_of_token(den[1]) is None:
            return None  # `\\frac{d}{d+1}`: an ordinary fraction
        rest = num[1:]
        ordered = self._script_order(rest)
        if ordered is None:
            return None
        n_order, rest = ordered
        if rest and self._name_of_token(rest[0]) is None:
            return None
        den_ordered = self._script_order(den[2:])
        den_order, den_rest = den_ordered if den_ordered is not None else (0, den[2:])
        jets = self.b.jets
        unknown = self._name_of_token(rest[0]) if len(rest) == 1 else None
        variable = self._name_of_token(den[1])
        if (
            jets is not None
            and unknown is not None
            and unknown in jets.dependent
            and variable == jets.variable
            and not den_rest
            and den_order == n_order
        ):
            self.i = d_end + 1
            return jets.derivative(unknown, n_order)
        if jets is not None:
            raise ParseError(
                "derivative notation: write the derivative of an unknown function with respect to "
                f"the independent variable as \\frac{{dx}}{{d{jets.variable}}} (or "
                "\\dot{x}, \\ddot{x}, x^{(k)}) with the same order above and below; other uses of "
                "d are not read"
            )
        raise ParseError(
            "derivative notation (\\frac{dx}{dy}, \\frac{d}{dx}) is not read here: as letters the d's "
            "would cancel and give a different expression. A derivative of an expression is the job of "
            "`differentiate` (give the variable), of an unknown function in an equation the differential "
            "equation tools, in a functional `variation`, along a motion `time_derivative`; to use d as "
            "an ordinary symbol, declare it"
        )

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
        if name == "lbrace":  # \\{ ... \\} (also \\lbrace, \\rbrace) groups like parentheses
            value = self.expr()
            if not (self.cur.kind == "cmd" and self.cur.text == "rbrace"):
                raise ParseError("expected '\\}'")
            self.advance()
            return value
        if name in {"exponentialE"}:
            return sp.E
        if name in {"imaginaryI"}:
            return sp.I
        if name in {"mathrm", "operatorname"}:
            return self._mathrm(name)
        if name in {"dot", "ddot"} or (name in {"dddot", "ddddot"} and self.b.jets is not None):
            return self._time_derivative(name)
        if name in _GREEK:
            return self._symbol_with_subscript_name(_GREEK_NAME.get(name, name))
        if name == "Gamma":
            return self._apply("gamma", None)
        if name == "ell":  # a letter, as in the semi-latus rectum l = h^2 / (G M)
            return self._symbol_with_subscript_name("ell")
        if name == "frac" or name in {"dfrac", "tfrac"}:
            leibniz = self._leibniz()
            if leibniz is not None:
                return leibniz
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
        if name in {"sum", "prod"} and self.b.allow_sums:
            return self._big_operator(name)
        if name == "mathbb" and self.b.random:
            return self._moment("E")
        if name == "int":
            raise ParseError(
                "an integral (\\int) is read only by the tools that take an integral as their input "
                "(`integrate` for an explicit integrand, `variation` for a functional of an unknown "
                "function), not inside this operation"
            )
        if name in {"sum", "prod", "lim", "partial", "nabla", "infty", "frac"}:
            raise ParseError(f"\\{name} is not supported in this operation")
        raise ParseError(f"unsupported LaTeX command \\{name}")

    def _moment(self, word: str) -> sp.Expr:
        """`\\mathbb{E}[...]`, `\\operatorname{Var}(...)`, `\\operatorname{Cov}(..., ...)`."""
        if word == "E" and self.cur.kind == "op" and self.cur.text == "{":  # after \\mathbb
            self.advance()
            tok = self.advance()
            self.expect_op("}")
            if tok.kind != "letter" or tok.text != "E":
                raise ParseError("only \\mathbb{E} (expectation) is supported")
        if not (self.cur.kind == "op" and self.cur.text in "(["):
            raise ParseError(f"{word} needs its argument in brackets, as in {word}[X]")
        closer = ")" if self.advance().text == "(" else "]"
        first = self.expr()
        second = None
        if word == "Cov":
            self.expect_op(",")
            second = self.expr()
        self.expect_op(closer)
        if word == "E":
            return self.b.expectation(first)
        if word == "Var":
            return self.b.variance(first)
        if second is None:
            raise ParseError("Cov needs two arguments")
        return self.b.covariance(first, second)

    def _big_operator(self, name: str) -> sp.Expr:
        """`\\sum_{i=a}^{b} term` and `\\prod`; the term is one product (up to the next + or -)."""
        if not (self.accept_op("_") and self.accept_op("{")):
            raise ParseError(f"write the bounds as \\{name}_{{i=a}}^{{b}}")
        index_name = self._name_of(self.advance())
        if index_name is None:
            raise ParseError("the index must be a single letter")
        self.expect_op("=")
        lower = self.expr()
        self.expect_op("}")
        if not self.accept_op("^"):
            raise ParseError(f"write the bounds as \\{name}_{{i=a}}^{{b}}")
        upper = self.script()
        index = self.b.open_sum(index_name, lower)
        try:
            body = self.term()
        finally:
            self.b.close_sum(index_name)
        return self.b.summation("sum" if name == "sum" else "product", body, index, lower, upper)

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
        jets = self.b.jets
        if jets is not None:
            if base not in jets.dependent:
                raise ParseError(f"\\{name} applies to a declared dependent function, not {base!r}")
            return jets.derivative(base, {"dot": 1, "ddot": 2, "dddot": 3, "ddddot": 4}[name])
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
        if self.b.random and word in ("E", "Var", "Cov"):
            return self._moment(word)
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


def _primes_to_dots(source: str, jets: JetBinding) -> str:
    """In a functional, `x'` and `x''` after a declared unknown function are `\\dot{x}` and `\\ddot{x}`
    (higher orders `x^{(k)}`). Anywhere else a prime is still an unexpected character."""
    names = "|".join(sorted((re.escape(n) for n in jets.dependent if len(n) == 1), reverse=True))
    if not names:
        return source

    def spell(match: re.Match[str]) -> str:
        name, primes = match.group(1), len(match.group(2))
        if primes == 1:
            return f"\\dot{{{name}}}"
        if primes == 2:
            return f"\\ddot{{{name}}}"
        return f"{name}^{{({primes})}}"

    return re.sub(rf"(?<![A-Za-z\\])({names})('{{1,4}})", spell, source)


def parse_latex(
    source: str,
    assumptions: dict[str, list[str]] | None = None,
    limits: Limits | None = None,
    denominators: list[sp.Expr] | None = None,
    noncommutative: Iterable[str] = (),
    functions: dict[str, tuple[str, ...]] | None = None,
    strict_calls: bool = False,
    notation: NotationLog | None = None,
    *,
    allow_noncommutative: bool = False,
    jets: JetBinding | None = None,
    allow_sums: bool = False,
    random_variables: Iterable[str] = (),
) -> sp.Expr:
    limits = limits or Limits()
    if len(source) > limits.max_input_chars:
        raise LimitError("input is too long")
    builder = Builder(
        assumptions,
        limits,
        noncommutative,
        functions,
        strict_calls,
        notation,
        allow_noncommutative=allow_noncommutative,
        jets=jets,
        allow_sums=allow_sums,
        random_variables=random_variables,
    )
    if denominators is not None:
        builder.denominators = denominators
    if jets is not None:
        source = _primes_to_dots(source, jets)
    parser = _Parser(_tokenize(source), builder)
    try:
        expr = parser.parse()
    except RecursionError as exc:
        raise LimitError("expression is nested too deeply") from exc
    builder.check_tree(expr)
    return expr
