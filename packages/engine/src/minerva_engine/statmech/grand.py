"""Grand canonical ensemble of non-interacting particles in single-particle levels.

State variables are the temperature, the chemical potential and the declared parameters; every quantity
comes from derivatives of ln Xi with the other state variables held fixed (``N = (1/beta) d ln Xi/d mu``,
``U - mu N = -d ln Xi/d beta`` at fixed mu, ``Var N = (1/beta^2) d^2 ln Xi/d mu^2``). Nothing is a stored
formula for Bose or Fermi systems: the ln Xi of each statistic is the only input.

The independent oracle never sees those formulas. Each single-particle *state* is enumerated over its
occupation numbers (0 or 1 for fermions, 0, 1, 2, ... until the weight is negligible for bosons, a Poisson
weight for classical counting); the mean and variance of N, the energy and the Gibbs entropy of the
occupation distribution come from those distributions directly.

A Bose system needs mu below every single-particle energy; that must be provable from the declared
assumptions, otherwise the request is refused instead of the series being summed anyway.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import mpmath as mp
import sympy as sp

from ..distributions.models import Condition
from ..latexprint import to_latex
from .canonical import (
    TITLES,
    Canonical,
    _enumerated,
    _evaluator,
    _nonnegative,
    _partial_quantity,
    compare,
    make_quantity,
    tidy,
)
from .models import (
    GRAND_QUANTITIES,
    OccupationSystem,
    StatCheck,
    StatMechInput,
    StatQuantity,
    StatStep,
)
from .oracle import BaseOracle, Point, cumulants_from_raw, sample_points
from .setup import Problem, Refusal

GRAND_TITLES = {
    "Xi": "Grand partition function",
    "lnXi": "Logarithm of the grand partition function",
    "Omega": "Grand potential",
    "N": "Mean particle number",
    "Var_N": "Particle-number variance",
}
TITLES.update(GRAND_TITLES)

SYMBOL = {
    "Xi": r"\Xi",
    "lnXi": r"\ln \Xi",
    "Omega": r"\Omega",
    "N": r"\langle N \rangle",
    "U": "U",
    "S": "S",
    "Var_N": r"\operatorname{Var}(N)",
}
STATISTICS_WORDS = {
    "bose": "Bose-Einstein",
    "fermi": "Fermi-Dirac",
    "boltzmann": "classical (Boltzmann) counting",
}


def fill_occupation(problem: Problem, s: OccupationSystem) -> set[sp.Symbol]:
    request = problem.request
    if request.copies is not None:
        raise Refusal(
            "unsupported_system",
            "copies do not apply to an occupation system: the particle number is the variable",
        )
    beta, T, k_B = problem.beta, problem.T, problem.k_B
    current = set(problem.assumptions.get(s.mu, []))
    problem.assumptions[s.mu] = sorted(current | {"real"})
    mu = problem.parse(s.mu)
    if not isinstance(mu, sp.Symbol):
        raise Refusal("parse_error", "the chemical potential must be a plain symbol")
    if mu in {beta, T, k_B, *problem.params}:
        raise Refusal("parse_error", "the chemical potential needs a name of its own")
    energies: list[sp.Expr] = []
    degeneracies: list[sp.Expr] = []
    for lv in s.levels:
        e, g = problem.parse(lv.energy), problem.parse(lv.degeneracy)
        if e.is_real is not True:
            raise Refusal(
                "assumptions_insufficient",
                "energies must be provably real: declare their symbols real or positive",
            )
        if g.is_integer is not True or g.is_positive is not True:
            raise Refusal(
                "invalid_degeneracy",
                "a degeneracy must be a positive integer (declare a symbolic one integer and positive)",
            )
        if {beta, T, k_B, mu} & (e.free_symbols | g.free_symbols):
            raise Refusal(
                "unsupported_system",
                "energies and degeneracies must not depend on the temperature or the chemical potential",
            )
        energies.append(e)
        degeneracies.append(g)
    conditions: list[Condition] = []
    if s.statistics == "bose":
        if any((e - mu).is_positive is not True for e in energies):
            raise Refusal(
                "convergence_not_provable",
                "Bose-Einstein statistics needs mu below every single-particle energy, and the "
                "assumptions do not imply it (declare mu negative and the energies nonnegative)",
            )
        conditions.append(
            Condition(
                latex=r"\mu < \varepsilon_i",
                established=True,
                how="provable from the declared assumptions, so every geometric series over occupation "
                "numbers converges",
            )
        )
    elif s.statistics == "fermi":
        conditions.append(
            Condition(
                latex=r"n_i \in \{0, 1\}",
                established=True,
                how="exclusion: each single-particle state holds at most one particle, so every sum is finite",
            )
        )
    else:
        conditions.append(
            Condition(
                latex=r"\Xi = \exp\left(\mathrm{e}^{\beta\mu} Z_1\right)",
                established=True,
                how="classical counting: the occupation of each state is Poisson distributed; it is the "
                "limit of low occupation, not an exact quantum statistic",
            )
        )
    problem.kind = "occupation"
    problem.params = [mu, *problem.params]
    problem.energies, problem.degeneracies = energies, degeneracies
    problem.observables = {"E": list(energies)}
    problem.extra = {"mu": mu, "statistics": s.statistics}
    problem.conditions = conditions
    problem.notes.append(
        f"{STATISTICS_WORDS[s.statistics]}: independent particles; each of the g_i states of a level is "
        "counted separately."
    )
    rows = [
        rf"\varepsilon_{{{i}}} = {to_latex(e)},\ g_{{{i}}} = {to_latex(g)}"
        for i, (e, g) in enumerate(zip(energies, degeneracies, strict=True), 1)
    ]
    problem.setup_steps = [
        StatStep(
            name="levels",
            title=f"Single-particle levels ({STATISTICS_WORDS[s.statistics]})",
            latex=r"\qquad ".join(rows) if len(rows) <= 8 else rf"\text{{{len(rows)} levels}}",
        )
    ]
    return {
        x for e in (*energies, *degeneracies) for x in e.free_symbols if isinstance(x, sp.Symbol)
    } | {mu}


class GrandCanonical(Canonical):
    """ln Xi of the chosen statistic and everything derived from it, in beta, mu and the parameters."""

    def __init__(self, problem: Problem) -> None:
        self.p = problem
        b, kB = problem.beta, problem.k_B
        mu: sp.Symbol = problem.extra["mu"]
        stat: str = problem.extra["statistics"]
        ln_xi: sp.Expr = sp.Integer(0)
        occupations: list[sp.Expr] = []
        for e, g in zip(problem.energies, problem.degeneracies, strict=True):
            x = b * (e - mu)
            if stat == "bose":
                ln_xi += -g * sp.log(1 - sp.exp(-x))
                occupations.append(1 / (sp.exp(x) - 1))
            elif stat == "fermi":
                ln_xi += g * sp.log(1 + sp.exp(-x))
                occupations.append(1 / (sp.exp(x) + 1))
            else:
                ln_xi += g * sp.exp(-x)
                occupations.append(sp.exp(-x))
        self.lnXi = ln_xi
        self.occupations = occupations
        n = tidy(sp.diff(ln_xi, mu) / b)
        u = tidy(-sp.diff(ln_xi, b) + mu * n)
        self.base = {
            "Xi": sp.exp(ln_xi),
            "lnXi": ln_xi,
            "Omega": -ln_xi / b,
            "N": n,
            "U": u,
            "S": tidy(kB * (ln_xi + b * (u - mu * n))),
            "Var_N": tidy(sp.diff(ln_xi, mu, 2) / b**2),
        }
        self.probabilities = []


class GrandOracle(BaseOracle):
    """Direct enumeration of occupation numbers, one single-particle state at a time."""

    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        self.mu: sp.Symbol = problem.extra["mu"]
        self.stat: str = problem.extra["statistics"]
        self._symbols = sorted(
            {s for e in (*problem.energies, *problem.degeneracies) for s in e.free_symbols},
            key=lambda s: s.name,
        )
        self._energy = [sp.lambdify(self._symbols, e, modules="mpmath") for e in problem.energies]
        self._degeneracy = [
            sp.lambdify(self._symbols, g, modules="mpmath") for g in problem.degeneracies
        ]

    def levels(self, point: Point) -> list[tuple[Any, Any]]:
        args = [point[s] for s in self._symbols]
        return [(e(*args), g(*args)) for e, g in zip(self._energy, self._degeneracy, strict=True)]

    def occupation_weights(self, x: Any) -> list[Any]:
        """Weights of the occupation numbers 0, 1, 2, ... of one state with x = beta (eps - mu)."""
        if self.stat == "fermi":
            return [mp.mpf(1), mp.exp(-x)]
        if x <= 0 and self.stat == "bose":
            raise Refusal(
                "convergence_not_provable", "the occupation series diverges at this point"
            )
        out = [mp.mpf(1)]
        n = 1
        while n < 5000:
            w = mp.exp(-n * x) / (mp.factorial(n) if self.stat == "boltzmann" else 1)
            out.append(w)
            if n >= 3 and w < mp.mpf(10) ** -(self.precision + 6) * sum(out):
                break
            n += 1
        return out

    def state_moments(self, x: Any, upto: int = 4) -> list[Any]:
        weights = self.occupation_weights(x)
        return [sum(w * n**k for n, w in enumerate(weights)) for k in range(upto + 1)]

    def _per_level(self, point: Point, beta: Any, mu: Any) -> list[tuple[Any, Any, list[Any]]]:
        return [(e, g, self.occupation_weights(beta * (e - mu))) for e, g in self.levels(point)]

    def quantity(self, name: str, point: Point, *, beta: Any | None = None) -> Any:
        p = point if beta is None else {**point, self.problem.beta: beta}
        b, mu, kB = p[self.problem.beta], p[self.mu], p[self.problem.k_B]
        with mp.workdps(self.precision):
            levels = self._per_level(p, b, mu)
            ln_xi = sum(g * mp.log(sum(w)) for _, g, w in levels)
            match name:
                case "lnXi":
                    return ln_xi
                case "Xi":
                    return mp.exp(ln_xi)
                case "Omega":
                    return -ln_xi / b
                case "N":
                    return sum(
                        g * sum(n * w for n, w in enumerate(ws)) / sum(ws) for _, g, ws in levels
                    )
                case "U":
                    return sum(
                        g * e * sum(n * w for n, w in enumerate(ws)) / sum(ws)
                        for e, g, ws in levels
                    )
                case "Var_N":
                    total = mp.mpf(0)
                    for _, g, ws in levels:
                        z = sum(ws)
                        m1 = sum(n * w for n, w in enumerate(ws)) / z
                        m2 = sum(n * n * w for n, w in enumerate(ws)) / z
                        total += g * (m2 - m1 * m1)
                    return total
                case "S":
                    # Gibbs entropy of the occupation distribution of every state. With classical
                    # counting the weight of n particles carries 1/n!, which is not a Boltzmann factor of
                    # the configuration, so the entropy is -sum p ln(p n!) (the Gibbs correction).
                    classical = self.stat == "boltzmann"
                    total = mp.mpf(0)
                    for _, g, ws in levels:
                        z = sum(ws)
                        total += g * -sum(
                            (w / z)
                            * (mp.log(w / z) + (mp.log(mp.factorial(n)) if classical else 0))
                            for n, w in enumerate(ws)
                            if w > 0
                        )
                    return kB * total
        raise ValueError(name)

    def number_cumulant(self, point: Point, order: int) -> Any:
        b, mu = point[self.problem.beta], point[self.mu]
        with mp.workdps(self.precision):
            total = mp.mpf(0)
            for e, g in self.levels(point):
                raw = self.state_moments(b * (e - mu))
                total += g * cumulants_from_raw(raw, order)[order - 1]
            return total

    def occupation(self, point: Point, index: int) -> Any:
        b, mu = point[self.problem.beta], point[self.mu]
        e, _ = self.levels(point)[index]
        with mp.workdps(self.precision):
            raw = self.state_moments(b * (e - mu), 1)
            return raw[1] / raw[0]


def _occupation_of(oracle: GrandOracle, index: int) -> Callable[[Point], Any]:
    return lambda pt: oracle.occupation(pt, index)


def _equation(name: str, canon: GrandCanonical, expr: sp.Expr) -> str:
    b = to_latex(canon.p.beta)
    mu = to_latex(canon.p.extra["mu"])
    route = {
        "Omega": rf"-\frac{{1}}{{{b}}}\ln \Xi",
        "N": rf"\frac{{1}}{{{b}}}\frac{{\partial \ln \Xi}}{{\partial {mu}}}",
        "U": rf"-\frac{{\partial \ln \Xi}}{{\partial {b}}} + {mu} N",
        "S": rf"k_B\left(\ln \Xi + {b}(U - {mu} N)\right)",
        "Var_N": rf"\frac{{1}}{{{b}^{{2}}}}\frac{{\partial^2 \ln \Xi}}{{\partial {mu}^{{2}}}}",
    }.get(name)
    shown = canon.latex(expr)
    return rf"{SYMBOL[name]} = {route} = {shown}" if route else rf"{SYMBOL[name]} = {shown}"


def _checks(
    canon: GrandCanonical, oracle: GrandOracle, names: Sequence[str], points: list[Point]
) -> dict[str, list[StatCheck]]:
    p = canon.p
    mu_name = oracle.mu.name
    out: dict[str, list[StatCheck]] = {n: [] for n in names}
    for n in names:
        expr = canon.shown(canon.base[n])
        out[n].append(
            compare(
                "enumeration",
                n,
                points,
                _evaluator(oracle, expr),
                _enumerated(oracle, n),
                "occupation numbers of every single-particle state enumerated directly",
                oracle.tol * 10**3,
            )
        )
        if n == "N":
            out[n].append(
                compare(
                    "finite_difference",
                    n,
                    points,
                    _evaluator(oracle, expr),
                    lambda pt: -oracle.derivative_of("Omega", pt, mu_name, 1),
                    "minus the numerical derivative of the enumerated grand potential with respect to mu",
                    oracle.diff_tol * 10,
                )
            )
        elif n == "S":
            out[n].append(
                compare(
                    "finite_difference",
                    n,
                    points,
                    _evaluator(oracle, expr),
                    lambda pt: -oracle.derivative_of("Omega", pt, p.T.name, 1),
                    "minus the numerical derivative of the enumerated grand potential with respect to T "
                    "at fixed mu",
                    oracle.diff_tol * 10,
                )
            )
        elif n == "U":
            out[n].append(
                compare(
                    "finite_difference",
                    n,
                    points,
                    _evaluator(oracle, expr),
                    lambda pt: (
                        -oracle.derivative_of("lnXi", pt, p.beta.name, 1)
                        + pt[oracle.mu] * oracle.quantity("N", pt)
                    ),
                    "U - mu N = -d ln Xi / d beta at fixed mu, differentiated numerically",
                    oracle.diff_tol * 10,
                )
            )
        elif n == "Var_N":
            out[n].append(
                compare(
                    "finite_difference",
                    n,
                    points,
                    _evaluator(oracle, expr),
                    lambda pt: oracle.derivative_of("lnXi", pt, mu_name, 2) / pt[p.beta] ** 2,
                    "the second numerical derivative of ln Xi with respect to mu, divided by beta^2",
                    oracle.diff_tol * 10,
                )
            )
            out[n].append(_nonnegative(expr, oracle, points))
    if {"Omega", "U", "S", "N"} <= set(names):
        out["Omega"].append(
            compare(
                "free_energy_identity",
                "Omega",
                points,
                lambda pt: oracle.quantity("Omega", pt),
                lambda pt: (
                    oracle.quantity("U", pt)
                    - oracle.quantity("S", pt) / (pt[p.k_B] * pt[p.beta])
                    - pt[oracle.mu] * oracle.quantity("N", pt)
                ),
                "Omega = U - T S - mu N on the enumerated values",
                oracle.tol * 10**3,
            )
        )
    return out


def _setup_steps(problem: Problem, canon: GrandCanonical) -> list[StatStep]:
    b, mu = to_latex(problem.beta), to_latex(problem.extra["mu"])
    kind = problem.extra["statistics"]
    generic = {
        "bose": rf"\ln\Xi = -\sum_i g_i \ln\left(1 - \mathrm{{e}}^{{-{b}(\varepsilon_i - {mu})}}\right)",
        "fermi": rf"\ln\Xi = \sum_i g_i \ln\left(1 + \mathrm{{e}}^{{-{b}(\varepsilon_i - {mu})}}\right)",
        "boltzmann": rf"\ln\Xi = \sum_i g_i\, \mathrm{{e}}^{{-{b}(\varepsilon_i - {mu})}}",
    }[kind]
    explicit = canon.latex(canon.lnXi) if len(problem.energies) <= 4 else ""
    steps = [step.model_copy(deep=True) for step in problem.setup_steps]
    steps.append(
        StatStep(
            name="partition_function",
            title="Grand partition function",
            latex=generic + (rf" = {explicit}" if explicit else ""),
            conditions=[
                Condition(
                    latex=rf"{b} > 0,\ k_B > 0",
                    established=True,
                    how="positive by the definition of an ensemble temperature",
                ),
                *problem.conditions,
            ],
        )
    )
    return steps


def derive_grand(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition]]:
    canon = GrandCanonical(problem)
    oracle = GrandOracle(problem)
    points = sample_points(problem)
    names = list(request.quantities)
    if not names and not request.partials:
        names = ["lnXi", "Omega", "N", "U", "S"]
    for n in names:
        if n not in GRAND_QUANTITIES:
            raise Refusal(
                "unsupported_system",
                f"{n} is not a grand canonical quantity (use Xi, lnXi, Omega, N, U, S or Var_N)",
            )
    checks = _checks(canon, oracle, names, points)
    steps = _setup_steps(problem, canon)
    quantities: list[StatQuantity] = []
    others = sorted(s.name for s in problem.params)
    routes = {
        "Xi": "exponential of ln Xi",
        "lnXi": "sum over single-particle states of the logarithm of their occupation sum",
        "Omega": "Omega = -(1/beta) ln Xi",
        "N": "N = (1/beta) d ln Xi / d mu",
        "U": "U = -d ln Xi / d beta + mu N",
        "S": "S = k_B (ln Xi + beta (U - mu N))",
        "Var_N": "Var(N) = (1/beta^2) d^2 ln Xi / d mu^2",
    }
    for n in names:
        expr = canon.base[n]
        held = others if n in {"N", "Var_N"} else [s for s in others if s != oracle.mu.name]
        if n in {"U", "S"}:
            held = others  # mu is held fixed in the beta derivative
        q = make_quantity(
            canon,
            n,
            expr,
            title=TITLES[n],
            route=routes[n],
            held_fixed=held if n not in {"Xi", "lnXi", "Omega"} else [],
            checks=checks[n],
        )
        quantities.append(q)
        steps.append(
            StatStep(
                name=n,
                title=TITLES[n],
                latex=_equation(n, canon, expr),
                text=q.text,
            )
        )
    for spec in request.partials:
        q = _partial_quantity(canon, oracle, points, spec)
        quantities.append(q)
        steps.append(StatStep(name="partial", title=q.title, latex=q.latex, text=q.text))
    return steps, quantities, _conditions(problem)


def _conditions(problem: Problem) -> list[Condition]:
    return [
        Condition(
            latex=rf"{to_latex(problem.beta)} > 0,\ k_B > 0",
            established=True,
            how="positive by the definition of an ensemble temperature",
        ),
        *problem.conditions,
    ]


def partition_grand(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition]]:
    """ln Xi and the mean occupation of every level, each checked against the enumeration."""
    steps, quantities, conditions = derive_grand(
        problem, request.model_copy(update={"quantities": ["lnXi", "Xi"], "partials": []}), deadline
    )
    canon = GrandCanonical(problem)
    oracle = GrandOracle(problem)
    points = sample_points(problem)
    items, checks = [], []
    for i, occupation in enumerate(canon.occupations[:8]):
        shown = canon.shown(occupation)
        items.append(rf"\langle n\rangle_{{{i + 1}}} = {to_latex(shown)}")
        checks.append(
            compare(
                "enumeration",
                f"n_{i + 1}",
                points,
                _evaluator(oracle, shown),
                _occupation_of(oracle, i),
                "mean occupation of one state, enumerated over its occupation numbers",
                oracle.tol * 10**3,
            )
        )
    latex = r",\quad ".join(items)
    steps.append(
        StatStep(name="occupations", title="Mean occupation of a state in each level", latex=latex)
    )
    quantities.append(
        StatQuantity(
            name="occupations",
            title="Mean occupation numbers",
            latex=latex,
            text="",
            route="the derivative of each state's occupation sum with respect to its fugacity",
            checks=checks,
        )
    )
    return steps, quantities, conditions


def expectation_grand(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition]]:
    """Cumulants of the particle number: kappa_n(N) = beta^-n d^n ln Xi / d mu^n."""
    from .canonical import _energy_statistic

    if request.observable != "N" or request.statistic == "covariance":
        raise Refusal(
            "unknown_observable",
            "in the grand canonical ensemble the statistics available are those of the particle number N",
        )
    canon = GrandCanonical(problem)
    oracle = GrandOracle(problem)
    points = sample_points(problem)
    b, mu = problem.beta, problem.extra["mu"]
    from .canonical import _order_needed

    needed = _order_needed(request.statistic, request.order)
    k = {  # only the orders asked for are differentiated and simplified
        n: tidy(sp.diff(canon.lnXi, mu, n) / b**n) if n <= needed else sp.Integer(0)
        for n in (1, 2, 3, 4)
    }
    expr, kn, ref = _energy_statistic(k, request.statistic, request.order)
    expr = tidy(expr)
    shown = canon.shown(expr)
    checks = [
        compare(
            "enumeration",
            request.statistic,
            points,
            _evaluator(oracle, shown),
            lambda pt: ref([oracle.number_cumulant(pt, j) for j in (1, 2, 3, 4)]),
            "cumulants of N from the enumerated occupation distributions",
            oracle.tol * 10**3,
        )
    ]
    if request.statistic in {"variance"} or (
        request.statistic == "cumulant" and request.order == 2
    ):
        checks.append(_nonnegative(shown, oracle, points))
    if request.statistic in {"mean", "variance", "cumulant"}:
        checks.append(
            compare(
                "fluctuation_response",
                f"kappa_{kn}(N)",
                points,
                _evaluator(oracle, shown),
                lambda pt: oracle.derivative_of("lnXi", pt, mu.name, kn) / pt[b] ** kn,
                f"cumulant as beta^-{kn} times the {kn}-th numerical derivative of ln Xi with respect to mu",
                oracle.diff_tol * 10,
            )
        )
    conditions = _conditions(problem)
    q = make_quantity(
        canon,
        f"{request.statistic}_N",
        expr,
        title=f"{request.statistic.replace('_', ' ').capitalize()} of N",
        route="cumulants of N as derivatives of ln Xi with respect to mu at fixed beta",
        held_fixed=sorted(s.name for s in problem.params if s != mu),
        checks=checks,
        order=request.order if request.statistic == "cumulant" else None,
    )
    steps = [
        *_setup_steps(problem, canon),
        StatStep(
            name="expectation", title=q.title, latex=q.latex, text=q.text, conditions=conditions
        ),
    ]
    return steps, [q], conditions
