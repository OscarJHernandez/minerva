"""Turning the user's numbers into a `Numerics` object: parameter values, definitions, paths."""

from __future__ import annotations

from typing import Any

import sympy as sp

from ..errors import MinervaMathError, ParseError, UnsupportedInput
from ..latexparse import parse_latex
from ..textparse import parse_text
from .models import TrialPath, VariationInput
from .numerics import JetField, Numerics, _to_mp
from .paths import AnalyticField, SampledField, numeric_value, parse_in_variable, to_sympy
from .setup import Problem


class MissingValues(MinervaMathError):
    """The user has not supplied something a numerical action needs (never guessed)."""


def prepare(problem: Problem, data: VariationInput, seed: int) -> Numerics:
    values: dict[sp.Symbol, Any] = {}
    missing: list[str] = []
    known = {p.name for p in problem.parameters}
    for name in data.values:
        if name not in known and (data.sweep is None or name != data.sweep.name):
            raise ParseError(f"{name!r} is not a parameter of this functional")
    for p in problem.parameters:
        source = data.values.get(p.name)
        if source is None:
            missing.append(p.name)
            continue
        number = numeric_value(problem, source, data.format)
        if number.is_real is False and "real" in problem.assumptions.get(p.name, []):
            raise ParseError(f"{p.name} is declared real but its value is not")
        values[p] = _to_mp(number)

    definitions: dict[Any, sp.Lambda] = {}
    for name, args in problem.given.items():
        source = data.definitions.get(name)
        if source is None:
            missing.append(f"a definition of {name}({', '.join(args)})")
            continue
        definitions[sp.Function(name, real=True)] = _definition(
            problem, name, args, source, data, values
        )
    unknown = set(data.definitions) - set(problem.given)
    if unknown:
        raise ParseError(f"{sorted(unknown)[0]!r} is not a declared given function")
    if missing:
        raise MissingValues("Numerical values are needed for: " + ", ".join(missing))
    return Numerics(problem, seed, values=values, definitions=definitions)


def _definition(
    problem: Problem,
    name: str,
    args: tuple[str, ...],
    source: str,
    data: VariationInput,
    values: dict[sp.Symbol, Any],
) -> sp.Lambda:
    assumptions = {k: list(v) for k, v in problem.assumptions.items()}
    for a in args:
        assumptions[a] = ["real"]
    kwargs: dict[str, Any] = {"assumptions": assumptions, "limits": problem.limits}
    expr = sp.sympify(
        parse_text(source, **kwargs) if data.format == "text" else parse_latex(source, **kwargs)
    )
    symbols = tuple(sp.Symbol(a, real=True) for a in args)
    extra = expr.free_symbols - set(symbols)
    names = {p.name for p in problem.parameters}
    stray = sorted(s.name for s in extra if s.name not in names)
    if stray:
        raise ParseError(f"the definition of {name} uses {stray[0]!r}, which is not a parameter")
    named = {p.name: to_sympy(v) for p, v in values.items()}
    expr = expr.xreplace({s: named[s.name] for s in extra if s.name in named})
    return sp.Lambda(symbols, expr)


def path_field(
    num: Numerics,
    problem: Problem,
    path: TrialPath,
    data: VariationInput,
    *,
    depth: int,
    sweep_value: Any = None,
) -> tuple[JetField, str]:
    names = problem.jets.names
    given = set(path.analytic) | set(path.sampled)
    if given != set(names) or set(path.analytic) & set(path.sampled):
        raise ParseError("a path must give every unknown function exactly once")
    if path.analytic and path.sampled:
        raise UnsupportedInput("a path is either all analytic or all sampled")
    if path.sampled:
        field = SampledField(
            [path.sampled[n] for n in names],
            depth,
            [max(problem.jets.orders[i], 0) for i in range(len(names))],
            float(num.lower),
            float(num.upper),
        )
        return field, "sampled"
    named: dict[str, Any] = {p.name: v for p, v in num.params.items()}
    if data.sweep is not None and sweep_value is not None:
        named[data.sweep.name] = sweep_value
    exprs = [parse_in_variable(problem, path.analytic[n], data.format) for n in names]
    mapping: dict[sp.Symbol, Any] = {}
    for expr in exprs:
        for symbol in expr.free_symbols:
            if symbol.name in named:
                mapping[symbol] = named[symbol.name]
    return (
        AnalyticField(problem, exprs, mapping, depth=depth, lower=num.lower, upper=num.upper),
        "analytic",
    )
