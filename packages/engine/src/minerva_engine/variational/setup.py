"""Reading a functional: parsing, declared functions, bounds, endpoint terms and conditions.

Everything is read through the engine's strict parsers with the jet binding switched on, so:
* only declared dependent functions can be differentiated (`\\dot{x}`, `x^{(4)}`, `x_d3`), and the
  reading is echoed back to the user;
* a given function such as `V(x)` must be declared (an undeclared `V(x)` is an error, never `V*x`);
* nothing is executed, evaluated or sympified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sympy as sp

from ..build import RESERVED_NAMES
from ..errors import LimitError, ParseError, UnsupportedInput
from ..latexparse import parse_latex
from ..latexprint import to_latex
from ..models import SYMBOL_NAME_MAX, Limits
from ..textparse import parse_text
from .jets import MAX_DEPTH, MAX_ORDER, JetSpace
from .models import End, EndpointCondition, FunctionalInput

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

_IMPLIES_REAL = {
    "real",
    "positive",
    "negative",
    "nonnegative",
    "nonpositive",
    "integer",
    "rational",
}

# Integrands the whole construction is not valid for (non-smooth, non-local or complex-conjugated).
_NOT_SMOOTH = (sp.Abs, sp.sign, sp.Piecewise, sp.conjugate, sp.re, sp.im, sp.floor, sp.ceiling)


@dataclass
class Problem:
    jets: JetSpace
    lagrangian: sp.Expr
    lower: sp.Expr
    upper: sp.Expr
    lower_term: sp.Expr
    upper_term: sp.Expr
    conditions: dict[tuple[End, int, int], EndpointCondition]  # (end, function index, order)
    parameters: tuple[sp.Symbol, ...]
    given: dict[str, tuple[str, ...]]
    assumptions: dict[str, list[str]]
    limits: Limits
    reading: list[str] = field(default_factory=list)
    names: dict[sp.Symbol, str] = field(default_factory=dict)

    @property
    def variable(self) -> sp.Symbol:
        return self.jets.variable

    def latex(self, expr: sp.Expr) -> str:
        return to_latex(expr, symbol_names=self.names)

    @property
    def bounds_numeric(self) -> bool:
        return bool(self.lower.is_number and self.upper.is_number)

    @property
    def has_endpoint_terms(self) -> bool:
        return bool(self.lower_term != 0 or self.upper_term != 0)


def _parse(source: str, fmt: str, problem_args: dict[str, object]) -> sp.Expr:
    if fmt == "text":
        return parse_text(source, **problem_args)  # type: ignore[arg-type]
    return parse_latex(source, **problem_args)  # type: ignore[arg-type]


def read_problem(data: FunctionalInput, limits: Limits) -> Problem:
    variable = data.variable
    names = tuple(data.functions)
    for name in (variable, *names):
        if not _NAME.match(name) or name in RESERVED_NAMES or len(name) > SYMBOL_NAME_MAX:
            raise ParseError(f"{name!r} is not a usable name")
    if variable in names:
        raise ParseError("the independent variable cannot also be an unknown function")

    given: dict[str, tuple[str, ...]] = {}
    for decl in data.given_functions:
        if decl.name in names or decl.name == variable:
            raise ParseError(f"{decl.name!r} cannot be both a given and an unknown function")
        if decl.name in given:
            raise ParseError(f"{decl.name!r} is declared twice")
        for arg in decl.args:
            if arg != variable and arg not in names:
                raise ParseError(
                    f"{decl.name} may depend only on {variable} and on the unknown functions"
                )
        given[decl.name] = tuple(decl.args)

    provisional = JetSpace.build(variable, names, tuple(0 for _ in names), MAX_DEPTH)
    assumptions = {k: list(v) for k, v in data.assumptions.items() if k not in (variable, *names)}
    common: dict[str, object] = {
        "assumptions": assumptions,
        "limits": limits,
        "functions": given,
        "strict_calls": True,
        "jets": provisional.binding(),
    }

    def read(source: str) -> sp.Expr:
        return sp.sympify(_parse(source, data.format, common))

    # A symbol in a bound is a real number: declare it so before anything else is read, so that
    # the same name in the integrand is the same (real) symbol.
    for source in (data.lower, data.upper):
        if source is not None:
            for symbol in read(source).free_symbols:
                items = assumptions.setdefault(symbol.name, [])
                if not set(items) & _IMPLIES_REAL:
                    items.append("real")

    integrand = read(data.integrand)
    if isinstance(integrand, sp.Basic) and integrand.atoms(sp.Derivative) - {
        d
        for d in integrand.atoms(sp.Derivative)
        if isinstance(d.expr, sp.core.function.AppliedUndef)
    }:
        raise UnsupportedInput("derivatives of the integrand's own terms are not supported")
    _refuse_unsupported(integrand)

    lower_term = read(data.lower_term) if data.lower_term else sp.Integer(0)
    upper_term = read(data.upper_term) if data.upper_term else sp.Integer(0)
    _refuse_unsupported(lower_term)
    _refuse_unsupported(upper_term)

    orders = tuple(max(provisional.order_in(integrand, i), 0) for i in range(len(names)))
    if any(o > MAX_ORDER for o in orders):  # pragma: no cover - the parser cannot produce this
        raise LimitError(f"derivatives of order above {MAX_ORDER} are not supported")
    depth = max(orders) + max(orders)
    jets = JetSpace.build(variable, names, orders, depth)

    def clash(expr: sp.Expr) -> bool:
        return any(
            s.name in ("a", "b") for s in expr.free_symbols - jets.state_symbols - {jets.variable}
        )

    if (data.lower is None or data.upper is None) and (
        clash(integrand) or clash(lower_term) or clash(upper_term)
    ):
        raise ParseError(
            "the integrand uses a or b: give the interval explicitly, or rename those symbols"
        )
    lower = read(data.lower) if data.lower is not None else sp.Symbol("a", real=True)
    upper = read(data.upper) if data.upper is not None else sp.Symbol("b", real=True)
    for bound in (lower, upper):
        if bound.free_symbols & (jets.state_symbols | {jets.variable}):
            raise ParseError(
                "the bounds must not depend on the unknown functions or the variable "
                "(moving endpoints are not supported)"
            )
        if bound.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
            raise UnsupportedInput("infinite bounds are not supported")
        if bound.is_number and not bound.is_real:
            raise ParseError("the bounds must be real")
    if lower.is_number and upper.is_number and not bool(lower < upper):
        raise ParseError("the lower bound must be smaller than the upper bound")
    if lower == upper:
        raise ParseError("the bounds must differ")

    conditions: dict[tuple[End, int, int], EndpointCondition] = {}
    for cond in data.conditions:
        if cond.function not in names:
            raise ParseError(f"{cond.function!r} is not one of the unknown functions")
        key = (cond.end, names.index(cond.function), cond.order)
        if key in conditions:
            raise ParseError(
                f"{cond.function}^({cond.order}) at the {cond.end} end is declared twice"
            )
        conditions[key] = cond

    free = (
        integrand.free_symbols
        | lower_term.free_symbols
        | upper_term.free_symbols
        | lower.free_symbols
        | upper.free_symbols
    )
    parameters = tuple(sorted(free - jets.state_symbols - {jets.variable}, key=lambda s: s.name))

    problem = Problem(
        jets=jets,
        lagrangian=integrand,
        lower=lower,
        upper=upper,
        lower_term=lower_term,
        upper_term=upper_term,
        conditions=conditions,
        parameters=parameters,
        given=given,
        assumptions=assumptions,
        limits=limits,
        names=jets.symbol_names(),
    )
    problem.reading = _reading(problem)
    return problem


def _refuse_unsupported(expr: sp.Expr) -> None:
    if expr.has(*_NOT_SMOOTH):
        raise UnsupportedInput(
            "the integrand must be smooth: absolute values, sign, piecewise definitions, "
            "conjugates and real/imaginary parts are not supported"
        )
    if expr.has(sp.DiracDelta, sp.Heaviside):
        raise UnsupportedInput("distributions are not supported in a functional")


def _reading(problem: Problem) -> list[str]:
    jets = problem.jets
    lines: list[str] = []
    t = jets.variable.name
    for i, name in enumerate(jets.names):
        n = jets.orders[i]
        spellings = [f"\\dot{{{name}}}", f"\\ddot{{{name}}}", f"{name}^{{(k)}}"]
        spellings += [f"{name}_dot", f"{name}_ddot", f"{name}_d3"]
        used = f"up to order {n}" if n else "none"
        lines.append(
            f"{name} is an unknown function of {t}; derivatives are written "
            f"{', '.join(spellings)} (derivatives used by the integrand: {used})"
        )
    for name, args in problem.given.items():
        lines.append(f"{name}({', '.join(args)}) is a given function, not unknown")
    if problem.parameters:
        lines.append(
            "parameters: "
            + ", ".join(p.name for p in problem.parameters)
            + " (complex unless declared otherwise)"
        )
    lines.append(f"the variable {t} and the unknown functions are real")
    return lines
