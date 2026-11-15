"""Guarded construction of SymPy expressions.

Both parsers (plain text and LaTeX) build trees exclusively through this module, so the
function whitelist and the magnitude guards live in one place. Nothing here evaluates
user text as Python.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from .errors import LimitError, ParseError, RangeNotProvable, UnsupportedInput
from .indexed.guard import SUMS_REFUSAL
from .indexed.randoms import Covariance, Expectation, Variance
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
    # Distributions: only the `distribution` operation may work with these (see distributions/).
    "DiracDelta": sp.DiracDelta,
    "Heaviside": sp.Heaviside,
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

MAX_FUNCTION_ARGS = 4
MAX_SUM_DEPTH = 2  # nested sums and products
MAX_NUMERIC_RANGE = 64  # explicit numeric bounds, so that instance checks stay small

# Names that people write for a Dirac delta or a step function but that are ordinary symbols
# everywhere else: `\\delta(x)` is delta times x. Only the distribution operation may read them
# as distributions, and only for a reading that the user confirmed (`NotationLog.readings`).
DISTRIBUTION_CANDIDATES: dict[str, str] = {
    "delta": "dirac_delta",
    "theta": "heaviside",
    "Theta": "heaviside",
}
_READING_FUNCTION = {"dirac_delta": "DiracDelta", "heaviside": "Heaviside"}


@dataclass
class NotationLog:
    """How the user said `delta(...)`, `theta(...)` and `Theta(...)` are to be read, and what happened.

    `readings` comes from the user ("dirac_delta", "heaviside" or "symbol"); the parsers fill in
    `unconfirmed` (a name followed by an argument that nobody has answered for) and `used`.
    """

    readings: dict[str, str] = field(default_factory=dict)
    unconfirmed: dict[str, str] = field(default_factory=dict)
    used: dict[str, str] = field(default_factory=dict)


_SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class Builder:
    """Creates symbols (with the user's assumptions) and guarded compound expressions."""

    def __init__(
        self,
        assumptions: dict[str, list[str]] | None = None,
        limits: Limits | None = None,
        noncommutative: Iterable[str] = (),
        functions: dict[str, tuple[str, ...]] | None = None,
        strict_calls: bool = False,
        notation: NotationLog | None = None,
        *,
        allow_noncommutative: bool = False,
        allow_sums: bool = False,
        random_variables: Iterable[str] = (),
    ) -> None:
        self.limits = limits or Limits()
        # Finite sums, products and indexed variables exist only in the operations that check them
        # with instances (indexed/); everywhere else they are refused when the parser meets them.
        self.allow_sums = allow_sums
        # Names of random variables (expectation, variance, covariance); empty: E[...] is refused.
        self.random = frozenset(random_variables)
        for name in self.random:
            if not _SYMBOL_RE.match(name) or name in RESERVED_NAMES or len(name) > SYMBOL_NAME_MAX:
                raise ParseError(f"invalid random variable name {name!r}")
        self._scope: dict[str, sp.Symbol] = {}  # bound index name -> symbol, outermost first
        self._bases: dict[str, sp.IndexedBase] = {}
        # Off (None) for every operation except `distribution`.
        self.notation = notation
        # Names that stand for matrices: their products keep their order (matrix calculus only).
        self.noncommutative = frozenset(noncommutative)
        # Declared real functions of named symbols, e.g. {"f": ("r",)} (differential geometry only).
        self.functions: dict[str, tuple[str, ...]] = {}
        for fname, fargs in (functions or {}).items():
            self._declare_function(fname, tuple(fargs))
        # When set, `name(` for a name that is not a declared function is an error instead of the
        # implicit product `name * (...)`, so an undeclared `f(r)` is never silently read as f*r.
        self.strict_calls = strict_calls
        self._assumptions: dict[str, dict[str, bool]] = {}
        for name, items in (assumptions or {}).items():
            if "noncommutative" in items:
                if not allow_noncommutative:
                    raise UnsupportedInput("noncommutative symbols are not supported here")
                if len(items) > 1:
                    raise ParseError(
                        f"{name!r} cannot be noncommutative and carry other assumptions"
                    )
                self._assumptions[name] = {"commutative": False}
                continue
            unknown = set(items) - ALLOWED_ASSUMPTIONS
            if unknown:
                raise ParseError(f"unknown assumption(s): {sorted(unknown)}")
            self._assumptions[name] = {item: True for item in items}
        self.symbols: dict[str, sp.Symbol] = {}
        # Denominators met while reading (SymPy may cancel them at once): see restrictions.py
        self.denominators: list[sp.Expr] = []

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

    def _declare_function(self, name: str, args: tuple[str, ...]) -> None:
        if len(name) > SYMBOL_NAME_MAX or not _SYMBOL_RE.match(name) or "__" in name:
            raise ParseError(f"invalid function name {name!r}")
        if name in RESERVED_NAMES:
            raise ParseError(f"{name!r} is a reserved function or constant name")
        if not 1 <= len(args) <= MAX_FUNCTION_ARGS or len(set(args)) != len(args):
            raise ParseError(f"function {name!r} needs 1 to {MAX_FUNCTION_ARGS} distinct arguments")
        if name in args:
            raise ParseError(f"{name!r} cannot be both a function and one of its arguments")
        for arg in args:
            if len(arg) > SYMBOL_NAME_MAX or not _SYMBOL_RE.match(arg) or arg in RESERVED_NAMES:
                raise ParseError(f"invalid argument name {arg!r} for function {name!r}")
        self.functions[name] = args

    def declared_call(self, name: str, args: Iterable[sp.Expr]) -> sp.Expr:
        """`name(args)` for a declared function: a real-valued undefined function of its symbols."""
        declared = self.functions[name]
        argv = list(args)
        expected = [self.symbol(a) for a in declared]
        if argv != expected:
            raise ParseError(
                f"{name} is declared as {name}({', '.join(declared)}); "
                "it can only be written with exactly those arguments"
            )
        return sp.Function(name, real=True)(*expected)

    def symbol(self, name: str) -> sp.Symbol:
        if name in self._scope:
            return self._scope[name]
        if name in self._bases:
            raise ParseError(f"{name!r} is indexed data here; write {name}[…] for an element")
        if len(name) > SYMBOL_NAME_MAX or not _SYMBOL_RE.match(name) or "__" in name:
            raise ParseError(f"invalid symbol name {name!r}")
        if name in RESERVED_NAMES:
            raise ParseError(f"{name!r} is a reserved function or constant name")
        if name in self.functions:
            args = ", ".join(self.functions[name])
            raise ParseError(f"{name!r} is a declared function; write {name}({args})")
        if name not in self.symbols:
            if name in self.noncommutative:
                if self._assumptions.get(name):
                    raise ParseError(f"{name!r} is a matrix and cannot carry scalar assumptions")
                self.symbols[name] = sp.Symbol(name, commutative=False)
                return self.symbols[name]
            try:
                self.symbols[name] = sp.Symbol(name, **self._assumptions.get(name, {}))
            except ValueError as exc:  # sympy's InconsistentAssumptions is a ValueError
                raise ParseError(f"the assumptions for {name!r} contradict each other") from exc
        return self.symbols[name]

    # -- compound ----------------------------------------------------------------------
    def power(self, base: sp.Expr, exponent: sp.Expr) -> sp.Expr:
        # Reject explosive numeric powers *before* SymPy evaluates them.
        if exponent.is_Number and abs(exponent) > self.limits.max_pow_exponent:
            raise LimitError("exponent too large")  # also stops expansion bombs like (x+1)**10**7
        # Bound the digit count of an integer power without computing it.
        if (
            base.is_Integer
            and exponent.is_Integer
            and abs(base) > 1
            and abs(int(exponent)) * len(str(abs(int(base)))) > 10_000
        ):
            raise LimitError("result of power would be too large")
        if exponent.is_Number and exponent < 0 and base.free_symbols:
            self.denominators.append(base)
        return sp.Pow(base, exponent)

    def distribution_reading(self, name: str) -> str | None:
        """How `name(` is to be read: None (not applicable), "unconfirmed", or a confirmed reading.

        Called only when `name` is directly followed by an argument list. An unanswered name is
        recorded so that the operation can ask, and the parser carries on with the ordinary reading.
        """
        log = self.notation
        if log is None or name not in DISTRIBUTION_CANDIDATES:
            return None
        reading = log.readings.get(name)
        if reading is None:
            log.unconfirmed[name] = DISTRIBUTION_CANDIDATES[name]
            return "unconfirmed"
        log.used[name] = reading
        return reading

    def distribution_function(self, reading: str, argument: sp.Expr) -> sp.Expr:
        return self.function(_READING_FUNCTION[reading], [argument])

    def function(self, name: str, args: Iterable[sp.Expr]) -> sp.Expr:
        canonical = TEXT_FUNCTION_ALIASES.get(name, name)
        if canonical not in FUNCTIONS:
            raise ParseError(f"unsupported function {name!r}")
        argv = list(args)
        if len(argv) != 1:
            raise ParseError(f"{name} takes exactly one argument")
        # SymPy evaluates factorial/gamma of integers eagerly; bound the work up front.
        if canonical in {"factorial", "gamma"} and argv[0].is_Integer and argv[0] > 3000:
            raise LimitError("argument too large")
        try:
            return FUNCTIONS[canonical](argv[0])
        except (ValueError, TypeError) as exc:  # e.g. DiracDelta(I): not a real argument
            raise ParseError(f"{name} needs a real argument") from exc

    def log_base(self, argument: sp.Expr, base: sp.Expr) -> sp.Expr:
        return sp.log(argument, base)

    def root(self, radicand: sp.Expr, index: sp.Expr | None = None) -> sp.Expr:
        if index is None:
            return sp.sqrt(radicand)
        return self.power(radicand, sp.Integer(1) / index)

    def divide(self, numerator: sp.Expr, denominator: sp.Expr) -> sp.Expr:
        if denominator == 0:
            raise ParseError("division by zero")
        if denominator.free_symbols:
            self.denominators.append(denominator)
        return numerator / denominator

    # -- finite sums, products, indexed variables --------------------------------------
    def require_sums(self) -> None:
        if not self.allow_sums:
            raise UnsupportedInput(SUMS_REFUSAL)

    @property
    def bound_names(self) -> frozenset[str]:
        return frozenset(self._scope)

    def open_sum(self, name: str, lower: sp.Expr) -> sp.Symbol:
        """Bind the index of a sum or product; the bounds were read outside its scope."""
        self.require_sums()
        if len(name) > SYMBOL_NAME_MAX or not _SYMBOL_RE.match(name) or "__" in name:
            raise ParseError(f"invalid index name {name!r}")
        if name in RESERVED_NAMES:
            raise ParseError(f"{name!r} is a reserved function or constant name")
        if name in self._scope:
            raise ParseError(f"the index {name!r} is already bound by an enclosing sum")
        if len(self._scope) >= MAX_SUM_DEPTH:
            raise ParseError(
                f"sums and products nested deeper than {MAX_SUM_DEPTH} are not supported"
            )
        if (
            name in self.symbols
            or name in self._bases
            or name in self.random
            or name in self.functions
            or name in self._assumptions
        ):
            raise ParseError(f"{name!r} is already used as a variable; choose another index")
        flags: dict[str, bool] = {"integer": True}
        if lower.is_positive:
            flags["positive"] = True
        elif lower.is_nonnegative:
            flags["nonnegative"] = True
        self._scope[name] = sp.Symbol(name, **flags)
        return self._scope[name]

    def close_sum(self, name: str) -> None:
        del self._scope[name]

    def summation(
        self, kind: str, body: sp.Expr, index: sp.Symbol, lower: sp.Expr, upper: sp.Expr
    ) -> sp.Expr:
        """`Sum` or `Product` over `index` from `lower` to `upper`, never evaluated."""
        self.require_sums()
        if lower.has(sp.oo, -sp.oo, sp.zoo) or upper.has(sp.oo, -sp.oo, sp.zoo):
            raise UnsupportedInput("infinite sums and products are not supported")
        if index in lower.free_symbols or index in upper.free_symbols:
            raise ParseError("the bounds of a sum cannot depend on its own index")
        for bound in (lower, upper):
            if bound.is_integer is not True:
                if bound.free_symbols:
                    raise RangeNotProvable(
                        "cannot show that the bounds are integers: declare the symbols in them "
                        "integer (and the upper one positive or nonnegative)"
                    )
                raise ParseError("the bounds of a sum must be integers")
        if lower.is_Integer and upper.is_Integer:
            if lower > upper:
                raise ParseError("the range of this sum is empty (lower bound above upper bound)")
            if upper - lower + 1 > MAX_NUMERIC_RANGE:
                raise LimitError(f"a numeric range is limited to {MAX_NUMERIC_RANGE} terms")
        elif (upper - lower + 1).is_nonnegative is not True:
            raise RangeNotProvable(
                "cannot show that the range is non-empty: declare the upper bound positive "
                "(or nonnegative) and an integer"
            )
        maker = sp.Sum if kind == "sum" else sp.Product
        return maker(body, (index, lower, upper))

    def base(self, name: str) -> sp.IndexedBase:
        self.require_sums()
        if len(name) > SYMBOL_NAME_MAX or not _SYMBOL_RE.match(name) or "__" in name:
            raise ParseError(f"invalid name {name!r}")
        if name in RESERVED_NAMES:
            raise ParseError(f"{name!r} is a reserved function or constant name")
        if name in self.symbols or name in self._scope or name in self.functions:
            raise ParseError(f"{name!r} is used both as a variable and as indexed data")
        if name not in self._bases:
            flags = dict(self._assumptions.get(name, {}))
            if flags.get("commutative") is False:
                raise ParseError(f"{name!r} cannot be both a matrix and indexed data")
            try:
                self._bases[name] = sp.IndexedBase(name, **flags)
            except ValueError as exc:
                raise ParseError(f"the assumptions for {name!r} contradict each other") from exc
        return self._bases[name]

    def indexed(self, name: str, index: sp.Expr) -> sp.Expr:
        """The element `name[index]`; the index is an integer expression."""
        base = self.base(name)
        if index.is_integer is not True:
            raise ParseError(
                f"the index of {name} must be an integer expression; declare its symbols integer"
            )
        return base[index]

    def check_scopes(self, expr: sp.Basic) -> None:
        """No index of a sum or product is also used outside it."""
        bound = {v.name for node in expr.atoms(sp.Sum, sp.Product) for v in node.variables}
        free = {s.name for s in expr.free_symbols if isinstance(s, sp.Symbol)}
        clash = sorted(bound & free)
        if clash:
            raise ParseError(f"the index {clash[0]!r} is also used outside its sum")

    # -- random variables ----------------------------------------------------------------
    def _random_arguments(self, what: str, *args: sp.Expr) -> None:
        if not self.random:
            raise UnsupportedInput(SUMS_REFUSAL)
        for arg in args:
            names = {s.name for s in arg.free_symbols if isinstance(s, sp.Symbol)}
            names |= {b.name for b in arg.atoms(sp.IndexedBase)}
            if not names & self.random:
                raise ParseError(
                    f"{what} contains no random variable declared for this problem "
                    f"({', '.join(sorted(self.random))})"
                )

    def expectation(self, arg: sp.Expr) -> sp.Expr:
        self._random_arguments("E[…]", arg)
        return Expectation(arg)

    def variance(self, arg: sp.Expr) -> sp.Expr:
        self._random_arguments("Var(…)", arg)
        return Variance(arg)

    def covariance(self, a: sp.Expr, b: sp.Expr) -> sp.Expr:
        self._random_arguments("Cov(…)", a, b)
        return Covariance(a, b)

    # -- final guards ------------------------------------------------------------------
    def check_tree(self, expr: sp.Basic) -> None:
        nodes = sum(1 for _ in sp.preorder_traversal(expr))
        if nodes > self.limits.max_tree_nodes:
            raise LimitError("expression is too large")
        if _depth(expr) > self.limits.max_tree_depth:
            raise LimitError("expression is nested too deeply")
        self.check_scopes(expr)


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
