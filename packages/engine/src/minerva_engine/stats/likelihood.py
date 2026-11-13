"""Likelihood and log-likelihood of observations from a family.

The likelihood of n observations is a product of densities **only under independence**, which the user
must state (`iid`); without it nothing is computed. The logarithm is expanded with `expand_log(force=True)`,
which is valid because every factor of these densities is positive on the support and in the parameter
space; the expansion is also checked numerically against the sum of log densities (`check_loglik`).

Sums over the observations are abstracted to symbols (`abstract_sums`) so that differentiation and solving
are ordinary algebra; every result is restored to sums and checked on explicit data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import mpmath as mp
import sympy as sp

from ..indexed.instantiate import explicit
from ..indexed.rewrite import expand_sums
from ..models import ALLOWED_ASSUMPTIONS, ExpressionInput, Limits, Status
from ..ops import parse_expression
from .families import Family
from .statmodel_types import Reading, Refusal

INDEX_NAMES = "ijkl"


@dataclass
class Observations:
    name: str
    x: sp.Basic  # IndexedBase for several observations, a Symbol for one
    n: sp.Expr | None  # the number of observations (a symbol, an integer) or None for one
    index: sp.Symbol | None
    values: list[sp.Rational] | None  # concrete data, if given
    flags: dict[str, bool]


@dataclass
class Likelihood:
    reading: Reading
    obs: Observations
    theta: list[sp.Symbol]  # the parameters to estimate, as the family's symbols
    element: sp.Expr  # one observation: x[i] or x
    density: sp.Expr  # f(element; parameters)
    logf: sp.Expr  # expanded log density of one observation
    likelihood: sp.Expr
    loglik: sp.Expr
    premises: list[str]


def _flags(family: Family, user: list[str] | None) -> dict[str, bool]:
    flags = dict(family.observation)
    for item in user or []:
        if item in ALLOWED_ASSUMPTIONS:
            flags[item] = True
    return flags


def observation_setup(data: object, family: Family, limits: Limits) -> Observations:
    from .models import StatModelInput

    assert isinstance(data, StatModelInput)  # noqa: S101 - narrowing
    name = data.observation or "x"
    flags = _flags(family, data.assumptions.get(name))
    if data.data is not None:
        values = []
        for text in data.data:
            value = parse_expression(ExpressionInput(input=text, format="text"), limits)
            if value.free_symbols or not value.is_real:
                raise Refusal(
                    Status.PARSE_ERROR, "parse_error", "observations must be real numbers"
                )
            values.append(sp.nsimplify(value))
        count = len(values)
        i = sp.Symbol(INDEX_NAMES[0], integer=True, positive=True)
        return Observations(
            name, sp.IndexedBase(name, **flags), sp.Integer(count), i, values, flags
        )
    if data.sample_size is None:
        return Observations(name, sp.Symbol(name, **flags), None, None, None, flags)
    n_flags = {"integer": True, "positive": True}
    for item in data.assumptions.get(data.sample_size, []):
        if item in ALLOWED_ASSUMPTIONS:
            n_flags[item] = True
    n = sp.Symbol(data.sample_size, **n_flags)
    used = {name, data.sample_size}
    index_name = next((c for c in INDEX_NAMES if c not in used), "k")
    return Observations(
        name,
        sp.IndexedBase(name, **flags),
        n,
        sp.Symbol(index_name, integer=True, positive=True),
        None,
        flags,
    )


def build(data: object, reading: Reading, limits: Limits) -> Likelihood:
    from .models import StatModelInput

    assert isinstance(data, StatModelInput)  # noqa: S101 - narrowing
    obs = observation_setup(data, reading.family, limits)
    several = obs.n is not None
    premises: list[str] = []
    if several:
        if data.iid is not True:
            raise Refusal(
                Status.CONDITIONS_NOT_SATISFIED,
                "independence_not_declared",
                "a likelihood of several observations is a product of densities only if they are "
                "independent; state that the observations are independent and identically distributed",
            )
        premises.append("the observations are independent and identically distributed")
    element = obs.x[obs.index] if several else obs.x
    density = reading.family.density(reading.exprs, element)
    logf = sp.expand(sp.expand_log(sp.log(density), force=True))
    if several:
        assert obs.index is not None and obs.n is not None  # noqa: S101
        likelihood = sp.Product(density, (obs.index, 1, obs.n))
        loglik = sp.Sum(logf, (obs.index, 1, obs.n))
    else:
        likelihood, loglik = density, logf
    theta = [reading.family.symbols[name] for name in data.estimate]
    return Likelihood(reading, obs, theta, element, density, logf, likelihood, loglik, premises)


# --------------------------------------------------------------------------- abstraction of sums


@dataclass
class Abstraction:
    expr: sp.Expr
    atoms: dict[sp.Symbol, sp.Basic]  # symbol -> the Sum it stands for

    def restore(self, expr: sp.Expr) -> sp.Expr:
        return expr.xreplace(self.atoms)


def abstract_sums(expr: sp.Expr, obs: Observations) -> Abstraction:
    """Expand sums by linearity and replace every remaining sum over the data by a symbol."""
    expanded = expand_sums(expr) if expr.has(sp.Sum) else expr
    atoms: dict[sp.Symbol, sp.Basic] = {}
    mapping: dict[sp.Basic, sp.Symbol] = {}
    nonnegative_data = bool(obs.flags.get("nonnegative") or obs.flags.get("positive"))
    other = 0
    for node in sorted(expanded.atoms(sp.Sum), key=sp.default_sort_key):
        function = node.function
        base, power = function.as_base_exp()
        bound = node.limits[0][0]
        if base == obs.x[bound] and power.is_Integer and power > 0:
            even = int(power) % 2 == 0
            flags = (
                {"real": True, "nonnegative": True}
                if (even or nonnegative_data)
                else {"real": True}
            )
            symbol = sp.Symbol(f"S{int(power)}", **flags)
        else:
            other += 1
            symbol = sp.Symbol(f"T{other}", real=True)
        mapping[node] = symbol
        atoms[symbol] = node
    return Abstraction(expanded.xreplace(mapping), atoms)


# --------------------------------------------------------------------------- explicit data


def draw_values(family: Family, count: int, rng: random.Random) -> list[sp.Rational]:
    """Observations that lie in the support of the family."""
    out: list[sp.Rational] = []
    for _ in range(count):
        flags = family.observation
        if family.id == "bernoulli":
            out.append(sp.Integer(rng.randint(0, 1)))
        elif flags.get("integer"):
            out.append(sp.Integer(rng.randint(0, 6)))
        elif family.unit_interval_data:
            out.append(sp.Rational(rng.randint(1, 9), 10))
        elif flags.get("positive"):
            out.append(sp.Rational(rng.randint(1, 16), 4))
        else:
            out.append(sp.Rational(rng.randint(-12, 12), 4))
    return out


def at_instance(expr: sp.Expr, lk_obs: Observations, values: list[sp.Rational]) -> sp.Expr:
    """`expr` with the observations replaced by concrete values (and sums expanded)."""
    e = expr
    if lk_obs.n is not None and isinstance(lk_obs.n, sp.Symbol):
        e = e.subs(lk_obs.n, len(values))
    if isinstance(lk_obs.x, sp.IndexedBase):
        return explicit(e, fresh=lambda base, k: values[k - 1])
    return e.subs(lk_obs.x, values[0])


def numeric(expr: sp.Expr, point: dict[sp.Symbol, sp.Rational], digits: int = 30) -> mp.mpf:
    value = sp.N(expr.subs(point), digits)
    if not value.is_real:
        raise ValueError("not real")
    return mp.mpf(str(value))
