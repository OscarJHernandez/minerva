"""Which integrands are in the supported class, and their logarithmic derivatives.

A function is *hyperexponential* in `x` and `a` when `f_x / f` and `f_a / f` are rational
functions of `x` and `a` (and any further symbols, which are treated as generic constants).
Products of rational functions, powers with exponents free of `x`, and exponentials of rational
functions qualify; sums of such terms, trigonometric functions and `x**a` do not. The class is
decided here, once, from the tree of the parsed integrand: the user's text is never evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from .models import Refusal


@dataclass(frozen=True)
class Hyperexponential:
    f: sp.Expr
    x: sp.Symbol
    a: sp.Symbol
    r1: sp.Expr  # f_x / f
    r2: sp.Expr  # f_a / f
    exponent: sp.Expr  # sum of the arguments of exp(...) that depend on x
    powers: tuple[tuple[sp.Expr, sp.Expr], ...]  # (base, exponent) with a base depending on x
    constants: sp.Expr = field(default=sp.Integer(1))  # the factor that does not depend on x


def _is_rational(expr: sp.Expr, *symbols: sp.Symbol) -> bool:
    return bool(expr.is_rational_function(*symbols)) and not expr.atoms(sp.Function)


def _logarithmic_derivative(f: sp.Expr, var: sp.Symbol) -> sp.Expr:
    ratio = sp.diff(f, var) / f
    ratio = sp.powsimp(sp.expand(ratio))
    return sp.cancel(sp.together(sp.simplify(ratio)))


def split_factors(
    f: sp.Expr, x: sp.Symbol
) -> tuple[sp.Expr, list[tuple[sp.Expr, sp.Expr]], sp.Expr]:
    """(sum of exponents of exp(...), [(base, exponent)], x-free factor)."""
    exponent: sp.Expr = sp.Integer(0)
    powers: list[tuple[sp.Expr, sp.Expr]] = []
    constants: sp.Expr = sp.Integer(1)
    for factor in sp.Mul.make_args(f):
        if x not in factor.free_symbols:
            constants *= factor
        elif isinstance(factor, sp.exp):
            exponent += factor.args[0]
        elif isinstance(factor, sp.Pow):
            base, power = factor.as_base_exp()
            if x in power.free_symbols:
                if base == sp.E:
                    exponent += power
                    continue
                raise Refusal(
                    "not_hyperexponential", "A power with x in its exponent is not supported."
                )
            powers.append((base, power))
        elif isinstance(factor, sp.Add | sp.Symbol):
            powers.append((factor, sp.Integer(1)))
        else:
            raise Refusal(
                "not_hyperexponential",
                "Only products of rational functions, powers and exponentials of rational "
                "functions are supported (no sums of such terms, no trigonometric functions).",
            )
    return exponent, powers, constants


def analyse(f: sp.Expr, x: sp.Symbol, a: sp.Symbol) -> Hyperexponential:
    """Decide eligibility. Raises `Refusal` with the reason; otherwise the decomposition."""
    if x not in f.free_symbols:
        raise Refusal(
            "variable_missing", "The integrand does not depend on the integration variable."
        )
    if a not in f.free_symbols:
        raise Refusal("parameter_missing", "The integrand does not depend on the parameter.")
    if f.has(sp.Integral, sp.Sum, sp.Derivative) or f.has(sp.zoo, sp.oo, sp.nan):
        raise Refusal("not_hyperexponential", "The integrand contains an unevaluated operator.")
    exponent, powers, constants = split_factors(f, x)
    if not _is_rational(exponent, x):
        raise Refusal("not_hyperexponential", "An exponent is not a rational function of x.")
    for base, power in powers:
        if not _is_rational(base, x) or x in power.free_symbols:
            raise Refusal("not_hyperexponential", "A power's base is not a rational function of x.")
    r1 = _logarithmic_derivative(f, x)
    r2 = _logarithmic_derivative(f, a)
    symbols = [s for s in (r1.free_symbols | r2.free_symbols) if s != x]
    if not _is_rational(r1, x, *symbols) or not _is_rational(r2, x, *symbols):
        raise Refusal(
            "not_hyperexponential",
            "f_x/f and f_a/f must both be rational functions: the parameter may not appear "
            "in an exponent of x (as in x**a) or inside a logarithm.",
        )
    return Hyperexponential(f, x, a, r1, r2, exponent, tuple(powers), constants)
