"""The probability families Minerva knows: parameters, support and density, nothing else.

Moments, likelihoods, scores, information and posteriors are *derived* from these densities (by
integration, summation and differentiation) and checked independently; no estimator or posterior is
stored here. Densities are plain SymPy expressions (not `sympy.stats` objects, whose route is hidden and
slow). The parameter symbols carry the assumptions of the standard parameter space; `0 < p < 1` cannot be
expressed as an assumption and is decided by `domain`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import sympy as sp

x = sp.Symbol("x", real=True)  # the variable of a density


@dataclass(frozen=True)
class Param:
    name: str
    constraint: str  # see domain.decide
    default: str | None = None  # a value used when the parameter is not given (standard form)


@dataclass(frozen=True)
class Family:
    id: str
    title: str
    parametrization: dict[str, tuple[Param, ...]]  # first entry is the default
    discrete: bool
    # support as (lower, upper) in terms of the family's parameter symbols; None: unbounded
    support: Callable[[dict[str, sp.Symbol]], tuple[sp.Expr, sp.Expr]]
    density: Callable[[dict[str, sp.Symbol], sp.Symbol], sp.Expr]
    # False when the support depends on a parameter: differentiation under the integral sign is not allowed
    regular: bool = True
    symbols: dict[str, sp.Symbol] = field(default_factory=dict)
    # assumptions that hold for every observation of the family (its support), e.g. nonnegative integers
    observation: dict[str, bool] = field(default_factory=lambda: {"real": True})
    # the observations are in [0, 1]: n - sum(x) is a sum of nonnegative terms (used in second-order proofs)
    unit_interval_data: bool = False

    def params(self, parametrization: str | None) -> tuple[Param, ...]:
        key = parametrization or next(iter(self.parametrization))
        if key not in self.parametrization:
            raise KeyError(key)
        return self.parametrization[key]


def _symbols(*specs: tuple[str, dict[str, bool]]) -> dict[str, sp.Symbol]:
    return {name: sp.Symbol(name, **flags) for name, flags in specs}


_REAL = {"real": True}
_POS = {"positive": True}
_POSINT = {"positive": True, "integer": True}

FAMILIES: dict[str, Family] = {}


def _register(f: Family) -> None:
    FAMILIES[f.id] = f


_register(
    Family(
        id="normal",
        title="Normal",
        parametrization={
            "sd": (Param("mu", "real"), Param("sigma", "positive")),
            "variance": (Param("mu", "real"), Param("v", "positive")),
        },
        discrete=False,
        support=lambda s: (-sp.oo, sp.oo),
        density=lambda s, t: (
            sp.exp(-((t - s["mu"]) ** 2) / (2 * s["sigma"] ** 2))
            / (s["sigma"] * sp.sqrt(2 * sp.pi))
            if "sigma" in s
            else sp.exp(-((t - s["mu"]) ** 2) / (2 * s["v"])) / sp.sqrt(2 * sp.pi * s["v"])
        ),
        symbols=_symbols(("mu", _REAL), ("sigma", _POS), ("v", _POS)),
    )
)
_register(
    Family(
        id="bernoulli",
        title="Bernoulli",
        parametrization={"default": (Param("p", "unit_open"),)},
        discrete=True,
        support=lambda s: (sp.Integer(0), sp.Integer(1)),
        density=lambda s, t: s["p"] ** t * (1 - s["p"]) ** (1 - t),
        symbols=_symbols(("p", _POS)),
        observation={"integer": True, "nonnegative": True},
        unit_interval_data=True,
    )
)
_register(
    Family(
        id="binomial",
        title="Binomial",
        parametrization={"default": (Param("n", "positive_integer"), Param("p", "unit_open"))},
        discrete=True,
        support=lambda s: (sp.Integer(0), s["n"]),
        density=lambda s, t: sp.binomial(s["n"], t) * s["p"] ** t * (1 - s["p"]) ** (s["n"] - t),
        symbols=_symbols(("n", _POSINT), ("p", _POS)),
        observation={"integer": True, "nonnegative": True},
    )
)
_register(
    Family(
        id="poisson",
        title="Poisson",
        parametrization={"default": (Param("lam", "positive"),)},
        discrete=True,
        support=lambda s: (sp.Integer(0), sp.oo),
        density=lambda s, t: sp.exp(-s["lam"]) * s["lam"] ** t / sp.factorial(t),
        symbols=_symbols(("lam", _POS)),
        observation={"integer": True, "nonnegative": True},
    )
)
_register(
    Family(
        id="beta",
        title="Beta",
        parametrization={"default": (Param("alpha", "positive"), Param("beta", "positive"))},
        discrete=False,
        support=lambda s: (sp.Integer(0), sp.Integer(1)),
        density=lambda s, t: (
            t ** (s["alpha"] - 1) * (1 - t) ** (s["beta"] - 1) / sp.beta(s["alpha"], s["beta"])
        ),
        symbols=_symbols(("alpha", _POS), ("beta", _POS)),
        observation={"positive": True},
        unit_interval_data=True,
    )
)
_register(
    Family(
        id="gamma",
        title="Gamma (shape and rate)",
        parametrization={"default": (Param("alpha", "positive"), Param("beta", "positive"))},
        discrete=False,
        support=lambda s: (sp.Integer(0), sp.oo),
        density=lambda s, t: (
            s["beta"] ** s["alpha"]
            * t ** (s["alpha"] - 1)
            * sp.exp(-s["beta"] * t)
            / sp.gamma(s["alpha"])
        ),
        symbols=_symbols(("alpha", _POS), ("beta", _POS)),
        observation={"positive": True},
    )
)
_register(
    Family(
        id="cauchy",
        title="Cauchy",
        parametrization={
            "default": (Param("loc", "real", default="0"), Param("scale", "positive", default="1"))
        },
        discrete=False,
        support=lambda s: (-sp.oo, sp.oo),
        density=lambda s, t: s["scale"] / (sp.pi * (s["scale"] ** 2 + (t - s["loc"]) ** 2)),
        symbols=_symbols(("loc", _REAL), ("scale", _POS)),
    )
)
_register(
    Family(
        id="uniform_0_theta",
        title="Uniform on (0, theta)",
        parametrization={"default": (Param("theta", "positive"),)},
        discrete=False,
        support=lambda s: (sp.Integer(0), s["theta"]),
        density=lambda s, t: 1 / s["theta"],
        regular=False,
        symbols=_symbols(("theta", _POS)),
    )
)
