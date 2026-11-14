"""Canonical ensemble of an explicit finite spectrum: partition function, thermodynamic derivations,
expectation values and fluctuation-response relations.

One generic route produces every thermodynamic quantity: build ``ln Z``, differentiate with a declared
held-fixed set. Nothing here is a formula for one particular system. Every displayed formula is then
compared with the independent enumeration in ``oracle.py`` at sampled points; a formula that does not
agree is dropped (``verification_failed``), never shown.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import mpmath as mp
import sympy as sp

from ..distributions.models import Condition
from ..latexprint import to_latex, to_text
from .models import (
    BASE_QUANTITIES,
    Partial,
    StatCheck,
    StatMechInput,
    StatQuantity,
    StatStep,
)
from .oracle import DIFF_TOL, DPS, MIN_POINTS, Oracle, Point, close, sample_points
from .setup import Problem, Refusal

TIDY_MAX_OPS = 160
SYMBOLIC_LEVELS = 6  # the symbolic-zero leg is attempted only for small spectra

TITLES = {
    "Z": "Partition function",
    "lnZ": "Logarithm of the partition function",
    "F": "Helmholtz free energy",
    "U": "Internal energy",
    "S": "Entropy",
    "Var_E": "Energy variance",
}


def tidy(expr: sp.Expr) -> sp.Expr:
    """Simplify small expressions; the numeric comparison later covers whatever this does."""
    if sp.count_ops(expr) > TIDY_MAX_OPS:
        return expr
    try:
        return sp.simplify(expr)
    except Exception:
        return expr


class Canonical:
    """Symbolic quantities in the inverse temperature, displayed in the user's variable."""

    def __init__(self, problem: Problem) -> None:
        self.p = problem
        b, kB = problem.beta, problem.k_B
        terms = [
            g * sp.exp(-b * e) for e, g in zip(problem.energies, problem.degeneracies, strict=True)
        ]
        self.Z: sp.Expr = sp.Add(*terms)
        self.lnZ: sp.Expr = sp.log(self.Z)
        U = tidy(-sp.diff(self.lnZ, b))
        self.base: dict[str, sp.Expr] = {
            "Z": self.Z,
            "lnZ": self.lnZ,
            "F": -self.lnZ / b,
            "U": U,
            "S": tidy(kB * (self.lnZ + b * U)),
            "Var_E": tidy(sp.diff(self.lnZ, b, 2)),
        }
        self.probabilities: list[sp.Expr] = [t / self.Z for t in terms]

    # -- display -------------------------------------------------------------------------
    def shown(self, expr: sp.Expr) -> sp.Expr:
        """The expression in the user's temperature variable."""
        if self.p.display_var == self.p.beta:
            return expr
        return expr.subs(self.p.beta, 1 / (self.p.k_B * self.p.T))

    def latex(self, expr: sp.Expr) -> str:
        return to_latex(self.shown(expr))

    def text(self, expr: sp.Expr) -> str:
        return to_text(self.shown(expr))

    # -- generic partial derivative -------------------------------------------------------
    def partial(self, spec: Partial) -> tuple[sp.Expr, list[str]]:
        """d^n Q / dv^n with the other state variables held fixed (validated), as an expression."""
        p = self.p
        names = {s.name for s in p.state_variables}
        temp = p.display_var.name
        alias = {p.beta.name, p.T.name}

        def norm(n: str) -> str:
            return temp if n in alias else n

        var = norm(spec.variable)
        if var not in names:
            raise Refusal(
                "ensemble_mismatch",
                f"{spec.variable} is not a state variable of this canonical ensemble; the state "
                f"variables are {', '.join(sorted(names))}",
            )
        held = [norm(n) for n in spec.held_fixed]
        expected = names - {var}
        if len(set(held)) != len(held) or var in held or set(held) != expected:
            fixed = ", ".join(sorted(expected)) or "nothing else"
            raise Refusal(
                "ensemble_mismatch",
                f"a derivative with respect to {spec.variable} in the canonical ensemble holds "
                f"exactly {fixed} fixed; derivatives at fixed energy, entropy or other quantities "
                "need a change of variables, which is not supported",
            )
        q = self.base[spec.of]
        b, kB, T = p.beta, p.k_B, p.T
        if var == temp and spec.variable in alias:
            if spec.variable == p.T.name:
                d = sp.diff(q.subs(b, 1 / (kB * T)), T, spec.order).subs(T, 1 / (kB * b))
            else:
                d = sp.diff(q, b, spec.order)
        elif var == temp:  # the display variable itself
            d = sp.diff(q, b, spec.order) if p.display_var == b else self._d_dT(q, spec.order)
        else:
            symbol = next(s for s in p.params if s.name == var)
            d = sp.diff(q, symbol, spec.order)
        return tidy(d), sorted(expected)

    def _d_dT(self, q: sp.Expr, order: int) -> sp.Expr:
        b, kB, T = self.p.beta, self.p.k_B, self.p.T
        return sp.diff(q.subs(b, 1 / (kB * T)), T, order).subs(T, 1 / (kB * b))


# -- numerical comparison helpers ---------------------------------------------------------
def _scale_expr(problem: Problem, text: str) -> sp.Expr:
    try:
        return problem.parse(text)
    except Exception as exc:
        raise Refusal("parse_error", "the scale of a derivative is not a valid expression") from exc


def _diff_at(
    oracle: Oracle,
    point: Point,
    name: str,
    variable: str,
    order: int,
) -> Any:
    """mpmath's numerical derivative of an enumerated base quantity (independent of sympy)."""
    p = oracle.problem
    with mp.workdps(DPS):
        if variable in {p.beta.name, p.T.name}:
            if variable == p.beta.name:
                return mp.diff(lambda x: oracle.quantity(name, point, beta=x), point[p.beta], order)
            return mp.diff(
                lambda x: oracle.quantity(name, point, beta=1 / (point[p.k_B] * x)),
                point[p.T],
                order,
            )
        symbol = next(s for s in p.params if s.name == variable)
        return mp.diff(lambda x: oracle.quantity(name, {**point, symbol: x}), point[symbol], order)


def _evaluator(oracle: Oracle, expr: sp.Expr) -> Callable[[Point], Any]:
    return lambda pt: oracle.eval_expr(expr, pt)


def _enumerated(oracle: Oracle, name: str) -> Callable[[Point], Any]:
    return lambda pt: oracle.quantity(name, pt)


def _level_probability(oracle: Oracle, index: int) -> Callable[[Point], Any]:
    return lambda pt: oracle.probabilities(pt)[index]


def compare(
    name: str,
    subject: str,
    points: list[Point],
    shown: Callable[[Point], Any],
    reference: Callable[[Point], Any],
    detail: str,
    tol: Any,
) -> StatCheck:
    agreed = 0
    with mp.workdps(DPS):  # the oracle's numbers and the tolerance are 50-digit quantities
        for point in points:
            try:
                if close(shown(point), reference(point), tol):
                    agreed += 1
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
    ok = len(points) >= MIN_POINTS and agreed == len(points)
    return StatCheck(
        name=name, subject=subject, passed=ok, points=len(points), agreed=agreed, detail=detail
    )


def numeric_checks_base(
    canon: Canonical, oracle: Oracle, names: Sequence[str], points: list[Point]
) -> dict[str, list[StatCheck]]:
    """Enumeration (and, where a derivative produced the formula, finite-difference) checks."""
    p = canon.p
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
                "direct sum over the levels, log-sum-exp",
                mp.mpf(10) ** -18,
            )
        )
        if n == "U":
            out[n].append(
                compare(
                    "finite_difference",
                    n,
                    points,
                    _evaluator(oracle, expr),
                    lambda pt: -_diff_at(oracle, pt, "lnZ", p.beta.name, 1),
                    "minus the numerical derivative of the enumerated ln Z with respect to beta",
                    DIFF_TOL,
                )
            )
        elif n == "Var_E":
            out[n].append(
                compare(
                    "finite_difference",
                    n,
                    points,
                    _evaluator(oracle, expr),
                    lambda pt: _diff_at(oracle, pt, "lnZ", p.beta.name, 2),
                    "the second numerical derivative of the enumerated ln Z with respect to beta",
                    DIFF_TOL,
                )
            )
            out[n].append(_nonnegative(expr, oracle, points))
        elif n == "S":
            out[n].append(
                compare(
                    "finite_difference",
                    n,
                    points,
                    _evaluator(oracle, expr),
                    lambda pt: -_diff_at(oracle, pt, "F", p.T.name, 1),
                    "minus the numerical derivative of the enumerated F with respect to T",
                    DIFF_TOL,
                )
            )
    if {"F", "U", "S"} <= set(names):
        out["F"].append(
            compare(
                "free_energy_identity",
                "F",
                points,
                lambda pt: oracle.quantity("F", pt),
                lambda pt: (
                    oracle.quantity("U", pt) - oracle.quantity("S", pt) / (pt[p.k_B] * pt[p.beta])
                ),
                "F = U - T S on the enumerated values",
                mp.mpf(10) ** -18,
            )
        )
    return out


def _nonnegative(expr: sp.Expr, oracle: Oracle, points: list[Point]) -> StatCheck:
    with mp.workdps(DPS):
        ok = all(oracle.eval_expr(expr, pt) >= -(mp.mpf(10) ** -30) for pt in points)
    return StatCheck(
        name="nonnegative_variance",
        subject="Var_E",
        passed=ok and len(points) >= MIN_POINTS,
        points=len(points),
        agreed=len(points) if ok else 0,
        detail="a variance cannot be negative",
    )


def symbolic_zero(subject: str, derived: sp.Expr, direct: sp.Expr, deadline: float) -> StatCheck:
    """Extra evidence: the derived formula minus the direct moment simplifies to zero."""
    zero: bool | None = None
    if time.monotonic() < deadline:
        try:
            zero = True if sp.simplify(derived - direct) == 0 else None
        except Exception:
            zero = None
    return StatCheck(
        name="symbolic_zero",
        subject=subject,
        passed=zero,
        detail="the derived formula minus the direct moment simplifies to 0"
        if zero
        else "SymPy could not decide; the numeric checks above are the evidence",
    )


def make_quantity(
    canon: Canonical,
    name: str,
    expr: sp.Expr,
    *,
    title: str,
    route: str,
    held_fixed: list[str],
    checks: list[StatCheck],
    variable: str | None = None,
) -> StatQuantity:
    return StatQuantity(
        name=name,
        title=title,
        latex=canon.latex(expr),
        text=canon.text(expr),
        held_fixed=held_fixed,
        variable=variable,
        route=route,
        checks=checks,
    )


# -- the actions --------------------------------------------------------------------------
def levels_step(problem: Problem, canon: Canonical) -> StatStep:
    rows = []
    for i, (e, g) in enumerate(zip(problem.energies, problem.degeneracies, strict=True), 1):
        rows.append(rf"E_{{{i}}} = {to_latex(e)},\ g_{{{i}}} = {to_latex(g)}")
    return StatStep(
        name="levels",
        title="Energy levels and degeneracies",
        latex=r"\quad".join(rows) if len(rows) <= 8 else r"\text{" + f"{len(rows)} levels" + "}",
    )


def partition_steps(problem: Problem, canon: Canonical) -> list[StatStep]:
    b = to_latex(problem.beta)
    return [
        levels_step(problem, canon),
        StatStep(
            name="partition_function",
            title="Partition function",
            latex=rf"Z = \sum_i g_i\, e^{{-{b} E_i}} = {canon.latex(canon.Z)}",
            text=canon.text(canon.Z),
            conditions=[
                Condition(
                    latex=rf"{b} > 0,\ k_B > 0",
                    established=True,
                    how="positive by the definition of an ensemble temperature",
                ),
                Condition(
                    latex=r"\text{finite sum}",
                    established=True,
                    how="a finite sum of exponentials always converges; derivatives pass through it exactly",
                ),
            ],
        ),
    ]


def probabilities_step(canon: Canonical) -> StatStep:
    problem = canon.p
    if len(problem.energies) <= 8:
        items = [
            rf"P_{{{i}}} = {canon.latex(tidy(pr))}" for i, pr in enumerate(canon.probabilities, 1)
        ]
        latex = r",\quad ".join(items)
    else:
        latex = rf"P_i = g_i\, e^{{-{to_latex(problem.beta)} E_i}} / Z"
    return StatStep(
        name="probabilities",
        title="Probability of each level",
        latex=latex,
        text="P_i = g_i exp(-beta E_i) / Z",
    )


def probability_checks(canon: Canonical, oracle: Oracle, points: list[Point]) -> list[StatCheck]:
    checks: list[StatCheck] = []
    shown = [canon.shown(pr) for pr in canon.probabilities]
    for i, expr in enumerate(shown):
        if i >= 8:
            break
        checks.append(
            compare(
                "enumeration",
                f"P_{i + 1}",
                points,
                _evaluator(oracle, expr),
                _level_probability(oracle, i),
                "level probability by log-sum-exp",
                mp.mpf(10) ** -18,
            )
        )
    checks.append(
        compare(
            "normalisation",
            "P",
            points,
            lambda pt: sum(oracle.eval_expr(e, pt) for e in shown),
            lambda pt: mp.mpf(1),
            "the level probabilities sum to 1",
            mp.mpf(10) ** -18,
        )
    )
    return checks


def derive(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition]]:
    canon = Canonical(problem)
    oracle = Oracle(problem)
    points = sample_points(problem)
    names = list(request.quantities)
    if not names and not request.partials:
        names = ["Z", "lnZ", "F", "U", "S"]
    for n in names:
        if n not in BASE_QUANTITIES:
            raise Refusal("unsupported_system", "unknown quantity")
    checks = numeric_checks_base(canon, oracle, names, points)
    steps = partition_steps(problem, canon)
    quantities: list[StatQuantity] = []
    routes = {
        "Z": "sum of Boltzmann weights over the levels",
        "lnZ": "logarithm of Z",
        "F": "F = -(1/beta) ln Z",
        "U": "U = -d ln Z / d beta",
        "S": "S = k_B (ln Z + beta U)",
        "Var_E": "Var(E) = d^2 ln Z / d beta^2",
    }
    fixed = sorted(s.name for s in problem.params)
    for n in names:
        expr = canon.base[n]
        q = make_quantity(
            canon,
            n,
            expr,
            title=TITLES[n],
            route=routes[n],
            held_fixed=fixed if n in {"U", "S", "Var_E"} else [],
            checks=checks[n],
        )
        quantities.append(q)
        steps.append(
            StatStep(
                name=n,
                title=TITLES[n],
                latex=_equation(n, canon, expr),
                text=q.text,
                conditions=[
                    Condition(
                        latex=r"\text{held fixed: }"
                        + (", ".join(fixed) if fixed else r"\text{nothing else}"),
                        established=True,
                        how="every state variable other than the temperature is held fixed",
                    )
                ]
                if n in {"U", "S", "Var_E"}
                else [],
            )
        )
    if problem.energies and len(problem.energies) <= SYMBOLIC_LEVELS:
        _attach_symbolic(canon, quantities, deadline)
    for spec in request.partials:
        quantities.append(_partial_quantity(canon, oracle, points, spec))
        steps.append(
            StatStep(
                name="partial",
                title=quantities[-1].title,
                latex=quantities[-1].latex,
                text=quantities[-1].text,
            )
        )
    return steps, quantities, partition_steps(problem, canon)[1].conditions


def _equation(name: str, canon: Canonical, expr: sp.Expr) -> str:
    b = to_latex(canon.p.beta)
    lhs = {
        "Z": "Z",
        "lnZ": r"\ln Z",
        "F": "F",
        "U": "U",
        "S": "S",
        "Var_E": r"\operatorname{Var}(E)",
    }[name]
    route = {
        "F": rf"-\frac{{1}}{{{b}}}\ln Z",
        "U": rf"-\frac{{\partial \ln Z}}{{\partial {b}}}",
        "S": rf"k_B\left(\ln Z + {b} U\right)",
        "Var_E": rf"\frac{{\partial^2 \ln Z}}{{\partial {b}^2}}",
    }.get(name)
    shown = canon.latex(expr)
    return rf"{lhs} = {route} = {shown}" if route else rf"{lhs} = {shown}"


def _attach_symbolic(canon: Canonical, quantities: list[StatQuantity], deadline: float) -> None:
    """Symbolic-zero leg: U and Var_E against their direct moments."""
    p = canon.p
    w = canon.probabilities
    u_direct = sp.Add(*[wi * e for wi, e in zip(w, p.energies, strict=True)])
    var_direct = sp.Add(*[wi * (e - u_direct) ** 2 for wi, e in zip(w, p.energies, strict=True)])
    for q in quantities:
        if q.name == "U":
            q.checks.append(symbolic_zero("U", canon.base["U"], u_direct, deadline))
        elif q.name == "Var_E":
            q.checks.append(symbolic_zero("Var_E", canon.base["Var_E"], var_direct, deadline))


def _partial_quantity(
    canon: Canonical, oracle: Oracle, points: list[Point], spec: Partial
) -> StatQuantity:
    problem = canon.p
    expr, fixed = canon.partial(spec)
    scale = _scale_expr(problem, spec.scale)
    if scale.free_symbols - set(problem.sample_symbols) - {problem.T}:
        raise Refusal("parse_error", "the scale may use only declared symbols")
    total = tidy(scale * expr)
    var = spec.variable
    scale_shown = scale
    shown_total = canon.shown(total)
    check = compare(
        "finite_difference",
        spec.label or f"d{spec.of}/d{var}",
        points,
        lambda pt: oracle.eval_expr(shown_total, pt),
        lambda pt: (
            oracle.eval_expr(scale_shown, pt) * _diff_at(oracle, pt, spec.of, var, spec.order)
        ),
        "numerical derivative of the enumerated quantity, the other state variables fixed",
        DIFF_TOL,
    )
    d = "d" if spec.order == 1 else f"d^{spec.order}"
    label = spec.label or f"{d}{spec.of}/d{var}"
    return make_quantity(
        canon,
        label,
        total,
        title=f"Partial derivative of {TITLES[spec.of].lower()} with respect to {var}",
        route=f"{d}{spec.of}/d{var} at fixed {', '.join(fixed) or 'nothing else'}, times {spec.scale}",
        held_fixed=fixed,
        checks=[check],
        variable=var,
    )


# -- expectation --------------------------------------------------------------------------
def expectation(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition]]:
    canon = Canonical(problem)
    oracle = Oracle(problem)
    name = request.observable
    if name not in problem.observables:
        raise Refusal(
            "unknown_observable", f"{name} has no values on the levels (give them per level)"
        )
    values = problem.observables[name]
    other = request.other_observable
    if request.statistic == "covariance" and (other is None or other not in problem.observables):
        raise Refusal("unknown_observable", "a covariance needs a second observable with values")
    points = sample_points(problem)
    P = canon.probabilities

    def mean(vals: list[sp.Expr]) -> sp.Expr:
        return sp.Add(*[pi * v for pi, v in zip(P, vals, strict=True)])

    m = mean(values)
    order = request.order
    stat = request.statistic
    shown_order = {"mean": 1, "variance": 2, "covariance": 2}.get(stat, order)

    def central(k: int) -> sp.Expr:
        return sp.Add(*[pi * (v - m) ** k for pi, v in zip(P, values, strict=True)])

    if stat == "mean":
        expr = m
    elif stat == "moment":
        expr = sp.Add(*[pi * v**order for pi, v in zip(P, values, strict=True)])
    elif stat == "central_moment":
        expr = central(order)
    elif stat == "variance":
        expr = central(2)
    elif stat == "cumulant":
        expr = {1: m, 2: central(2), 3: central(3), 4: central(4) - 3 * central(2) ** 2}[order]
    else:
        o = problem.observables[other or ""]
        expr = mean([a * c for a, c in zip(values, o, strict=True)]) - m * mean(o)
    expr = tidy(expr)

    def reference(pt: Point) -> Any:
        v = oracle.observable(name, pt) if name != "E" else oracle.energies(pt)
        match stat:
            case "mean":
                return oracle.mean(v, pt)
            case "moment":
                return oracle.raw(v, order, pt)
            case "central_moment":
                return oracle.central(v, order, pt)
            case "variance":
                return oracle.central(v, 2, pt)
            case "cumulant":
                if order <= 3:
                    return oracle.mean(v, pt) if order == 1 else oracle.central(v, order, pt)
                return oracle.central(v, 4, pt) - 3 * oracle.central(v, 2, pt) ** 2
            case _:
                o = oracle.observable(other or "", pt) if other != "E" else oracle.energies(pt)
                prod = [x * y for x, y in zip(v, o, strict=True)]
                return oracle.mean(prod, pt) - oracle.mean(v, pt) * oracle.mean(o, pt)

    shown_expr = canon.shown(expr)
    checks = [
        compare(
            "enumeration",
            stat,
            points,
            lambda pt: oracle.eval_expr(shown_expr, pt),
            reference,
            "direct weighted sum over the levels, log-sum-exp",
            mp.mpf(10) ** -15,
        )
    ]
    conditions = [
        Condition(
            latex=r"\text{finite sum}",
            established=True,
            how="a finite sum of exponentials always converges",
        )
    ]
    even_central = stat in {"variance"} or (stat == "cumulant" and order == 2)
    if even_central or (stat == "central_moment" and order % 2 == 0):
        checks.append(_nonnegative(canon.shown(expr), oracle, points))
    cumulant_order = shown_order if stat in {"mean", "variance", "cumulant"} else None
    if cumulant_order is not None and cumulant_order <= 4:
        checks.extend(
            _response_checks(
                problem, request, canon, oracle, points, expr, cumulant_order, conditions, deadline
            )
        )
    q = make_quantity(
        canon,
        f"{stat}_{name}",
        expr,
        title=f"{stat.replace('_', ' ').capitalize()} of {name}",
        route="weighted sum over the levels with the Boltzmann probabilities",
        held_fixed=sorted(s.name for s in problem.params),
        checks=checks,
    )
    steps = [*partition_steps(problem, canon), probabilities_step(canon)]
    steps.append(
        StatStep(
            name="expectation", title=q.title, latex=q.latex, text=q.text, conditions=conditions
        )
    )
    return steps, [q], conditions


def _response_checks(
    problem: Problem,
    request: StatMechInput,
    canon: Canonical,
    oracle: Oracle,
    points: list[Point],
    expr: sp.Expr,
    n: int,
    conditions: list[Condition],
    deadline: float,
) -> list[StatCheck]:
    """Second route: cumulants as derivatives of ln Z (energy: beta; coupled observable: the field)."""
    out: list[StatCheck] = []
    shown = canon.shown(expr)
    p = problem
    if request.observable == "E":
        out.append(
            compare(
                "fluctuation_response",
                f"kappa_{n}(E)",
                points,
                lambda pt: oracle.eval_expr(shown, pt),
                lambda pt: (-1) ** n * _diff_at(oracle, pt, "lnZ", p.beta.name, n),
                "cumulant of E as (-1)^n times the n-th derivative of the enumerated ln Z with respect to beta",
                DIFF_TOL * 10,
            )
        )
        conditions.append(
            Condition(
                latex=rf"\kappa_{n}(E) = (-1)^{n}\,\partial_\beta^{n}\ln Z",
                established=True,
                how="the levels do not depend on beta, and a finite sum can be differentiated term by term",
            )
        )
        return out
    coupling = request.coupling
    if coupling is None or coupling.observable != request.observable:
        return out
    lam = next((s for s in p.params if s.name == coupling.parameter), None)
    if lam is None:
        raise Refusal("coupling_not_linear", "the coupling parameter must be a declared parameter")
    values = p.observables[request.observable]
    linear = all(
        sp.simplify(sp.diff(e, lam) + a) == 0 and not a.has(lam)
        for e, a in zip(p.energies, values, strict=True)
    )
    if not linear:
        raise Refusal(
            "coupling_not_linear",
            "the fluctuation-response relation needs dE_i/d(parameter) = -A_i on every level, with "
            "A_i independent of the parameter",
        )
    conditions.append(
        Condition(
            latex=rf"\partial E_i/\partial {to_latex(lam)} = -{request.observable}_i",
            established=True,
            how="checked symbolically on every level",
        )
    )
    b = p.beta
    out.append(
        compare(
            "fluctuation_response",
            f"kappa_{n}({request.observable})",
            points,
            lambda pt: oracle.eval_expr(shown, pt),
            lambda pt: _diff_at(oracle, pt, "lnZ", lam.name, n) / pt[b] ** n,
            f"cumulant as beta^-{n} times the {n}-th derivative of the enumerated ln Z with respect to "
            f"{lam.name}",
            DIFF_TOL * 10,
        )
    )
    response = tidy(sp.diff(canon.lnZ, lam, n) / b**n)
    if len(p.energies) <= SYMBOLIC_LEVELS:
        out.append(symbolic_zero(f"kappa_{n}", response, expr, deadline))
    return out
