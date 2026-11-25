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
from .oracle import DPS, MIN_POINTS, BaseOracle, Oracle, Point, close, sample_points
from .setup import Problem, Refusal

TIDY_MAX_OPS = 100
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


def make_oracle(problem: Problem) -> BaseOracle:
    """The independent numerical oracle for the kind of system."""
    if problem.kind == "levels":
        return Oracle(problem)
    if problem.kind == "quantum":
        from .quantum import QuantumOracle

        return QuantumOracle(problem)
    if problem.kind == "occupation":
        from .grand import GrandOracle

        return GrandOracle(problem)
    if problem.kind == "chain":
        from .lattice import ChainOracle

        return ChainOracle(problem)
    from .systems import ClassicalOracle, DosOracle, FamilyOracle

    if problem.kind == "level_family":
        return FamilyOracle(problem)
    if problem.kind == "density_of_states":
        return DosOracle(problem)
    return ClassicalOracle(problem)


class Canonical:
    """Symbolic quantities in the inverse temperature, displayed in the user's variable."""

    def __init__(self, problem: Problem) -> None:
        self.p = problem
        b, kB = problem.beta, problem.k_B
        terms: list[sp.Expr] = []
        if problem.kind in {"levels", "quantum"}:
            terms = [
                g * sp.exp(-b * e)
                for e, g in zip(problem.energies, problem.degeneracies, strict=True)
            ]
            z1: sp.Expr = sp.Add(*terms)
        elif problem.Z1 is None:
            raise Refusal(
                "unsupported_system",
                "no closed form of the partition function was found, so it cannot be differentiated "
                "symbolically; numerical evaluation (action evaluate) is available",
            )
        else:
            z1 = problem.Z1
        n = problem.copies_count
        if n is None:
            self.Z: sp.Expr = z1
            self.lnZ: sp.Expr = sp.log(z1)
        elif problem.identical:
            self.Z = z1**n / sp.factorial(n)
            self.lnZ = n * sp.log(z1) - sp.log(sp.factorial(n))
        else:
            self.Z = z1**n
            self.lnZ = n * sp.log(z1)
        U = tidy(-sp.diff(self.lnZ, b))
        self.base: dict[str, sp.Expr] = {
            "Z": self.Z,
            "lnZ": self.lnZ,
            "F": -self.lnZ / b,
            "U": U,
            "S": tidy(kB * (self.lnZ + b * U)),
            "Var_E": tidy(sp.diff(self.lnZ, b, 2)),
        }
        self.probabilities: list[sp.Expr] = (
            [t / self.Z for t in terms]
            if problem.kind in {"levels", "quantum"} and n is None
            else []
        )

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


def _evaluator(oracle: BaseOracle, expr: sp.Expr) -> Callable[[Point], Any]:
    return lambda pt: oracle.eval_expr(expr, pt)


def _enumerated(oracle: BaseOracle, name: str) -> Callable[[Point], Any]:
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
    min_points: int = MIN_POINTS,
) -> StatCheck:
    agreed = 0
    with mp.workdps(DPS):  # the oracle's numbers and the tolerance are 50-digit quantities
        for point in points:
            try:
                if close(shown(point), reference(point), tol):
                    agreed += 1
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
    ok = len(points) >= min_points and agreed == len(points)
    return StatCheck(
        name=name, subject=subject, passed=ok, points=len(points), agreed=agreed, detail=detail
    )


def numeric_checks_base(
    canon: Canonical, oracle: BaseOracle, names: Sequence[str], points: list[Point]
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
                "direct numerical evaluation (sum, quadrature or log-sum-exp)",
                oracle.tol,
            )
        )
        if n == "U":
            out[n].append(
                compare(
                    "finite_difference",
                    n,
                    points,
                    _evaluator(oracle, expr),
                    lambda pt: -oracle.derivative_of("lnZ", pt, p.beta.name, 1),
                    "minus the numerical derivative of the enumerated ln Z with respect to beta",
                    oracle.diff_tol,
                )
            )
        elif n == "Var_E":
            out[n].append(
                compare(
                    "finite_difference",
                    n,
                    points,
                    _evaluator(oracle, expr),
                    lambda pt: oracle.derivative_of("lnZ", pt, p.beta.name, 2),
                    "the second numerical derivative of the enumerated ln Z with respect to beta",
                    oracle.diff_tol,
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
                    lambda pt: -oracle.derivative_of("F", pt, p.T.name, 1),
                    "minus the numerical derivative of the enumerated F with respect to T",
                    oracle.diff_tol,
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
                oracle.tol,
            )
        )
    return out


def _nonnegative(expr: sp.Expr, oracle: BaseOracle, points: list[Point]) -> StatCheck:
    valid = 0
    nonnegative = True
    with mp.workdps(DPS):
        for pt in points:
            try:
                value = oracle.eval_expr(expr, pt)
            except (ValueError, ZeroDivisionError, OverflowError):
                continue  # a point where the formula is undefined is skipped, never counted as passing
            valid += 1
            if mp.re(value) < -(mp.mpf(10) ** -30):
                nonnegative = False
    return StatCheck(
        name="nonnegative_variance",
        subject="Var_E",
        passed=nonnegative and valid >= MIN_POINTS,
        points=valid,
        agreed=valid if nonnegative else 0,
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
    order: int | None = None,
) -> StatQuantity:
    return StatQuantity(
        name=name,
        title=title,
        latex=canon.latex(expr),
        text=canon.text(expr),
        held_fixed=held_fixed,
        variable=variable,
        order=order,
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
        latex=r"\qquad ".join(rows) if len(rows) <= 8 else r"\text{" + f"{len(rows)} levels" + "}",
    )


def partition_conditions(problem: Problem) -> list[Condition]:
    b = to_latex(problem.beta)
    base = Condition(
        latex=rf"{b} > 0,\ k_B > 0",
        established=True,
        how="positive by the definition of an ensemble temperature",
    )
    if problem.kind == "levels":
        return [
            base,
            Condition(
                latex=r"\text{finite sum}",
                established=True,
                how="a finite sum of exponentials always converges; derivatives pass through it exactly",
            ),
        ]
    return [base, *problem.conditions]


def partition_steps(problem: Problem, canon: Canonical) -> list[StatStep]:
    b = to_latex(problem.beta)
    if problem.kind == "levels":
        steps = [
            levels_step(problem, canon),
            StatStep(
                name="partition_function",
                title="Partition function",
                latex=rf"Z = \sum_i g_i\, e^{{-{b} E_i}} = {canon.latex(canon.Z)}",
                text=canon.text(canon.Z),
                conditions=partition_conditions(problem),
            ),
        ]
    else:
        steps = [step.model_copy(deep=True) for step in problem.setup_steps]
        for step in steps:
            if step.name == "partition_function":
                step.conditions = partition_conditions(problem)
    n = problem.copies_count
    if n is not None:
        steps.append(
            StatStep(
                name="copies",
                title="Independent copies",
                latex=(
                    rf"Z = \frac{{Z_1^{{{to_latex(n)}}}}}{{{to_latex(n)}!}} = {canon.latex(canon.Z)}"
                    if problem.identical
                    else rf"Z = Z_1^{{{to_latex(n)}}} = {canon.latex(canon.Z)}"
                ),
                text=canon.text(canon.Z),
            )
        )
    return steps


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


def make_canonical(problem: Problem) -> Canonical:
    """The symbolic quantities of the problem's ensemble."""
    if problem.kind == "occupation":
        from .grand import GrandCanonical

        return GrandCanonical(problem)
    return Canonical(problem)


def derive(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition]]:
    if problem.kind == "occupation":
        from .grand import derive_grand

        return derive_grand(problem, request, deadline)
    canon = Canonical(problem)
    oracle = make_oracle(problem)
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
        "Z": "partition function of the system (single copy raised to the number of copies)",
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
        if n == "Z":
            continue  # the partition-function step above already shows it
        steps.append(
            StatStep(
                name=n,
                title=TITLES[n],
                latex=_equation(n, canon, expr),
                text=q.text,
                conditions=[_held_fixed_condition(fixed)] if n in {"U", "S", "Var_E"} else [],
            )
        )
    _attach_symbolic(canon, quantities, deadline)
    for spec in request.partials:
        quantities.append(_partial_quantity(canon, oracle, points, spec))
        steps.append(
            StatStep(
                name="partial",
                title=quantities[-1].title,
                latex=_partial_equation(canon, spec, quantities[-1]),
                text=quantities[-1].text,
                conditions=[_held_fixed_condition(quantities[-1].held_fixed)],
            )
        )
    return steps, quantities, partition_conditions(problem)


def _held_fixed_condition(fixed: list[str]) -> Condition:
    names = ", ".join(to_latex(sp.Symbol(n)) for n in fixed)
    return Condition(
        latex=rf"\text{{held fixed: }}{names}" if fixed else r"\text{no other state variable}",
        established=True,
        how=(
            "every state variable other than the one differentiated is held fixed"
            if fixed
            else "the temperature is the only state variable here"
        ),
    )


_SYMBOL = {
    "Z": "Z",
    "lnZ": r"\ln Z",
    "F": "F",
    "U": "U",
    "S": "S",
    "Var_E": r"\operatorname{Var}(E)",
}


def _partial_equation(canon: Canonical, spec: Partial, q: StatQuantity) -> str:
    """label = scale * (d^n Q / dv^n)_fixed = result, as LaTeX."""
    sym = _SYMBOL[spec.of]
    var = to_latex(sp.Symbol(spec.variable))
    n = "" if spec.order == 1 else f"^{{{spec.order}}}"
    fixed = ", ".join(to_latex(sp.Symbol(name)) for name in q.held_fixed)
    sub = f"_{{{fixed}}}" if fixed else ""
    derivative = rf"\left(\frac{{\partial{n} {sym}}}{{\partial {var}{n}}}\right){sub}"
    if spec.scale.strip() not in {"", "1"}:
        derivative = rf"\left({to_latex(canon.p.parse(spec.scale))}\right) {derivative}"
    label = to_latex(sp.Symbol(spec.label)) if spec.label else ""
    head = f"{label} = " if label else ""
    return rf"{head}{derivative} = {q.latex}"


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
    """Symbolic-zero leg: U and Var_E against their direct moments, where those are available."""
    p = canon.p
    if p.copies_count is not None:
        return
    if p.kind in {"levels", "quantum"}:
        if not 0 < len(p.energies) <= SYMBOLIC_LEVELS:
            return
        w = canon.probabilities
        u_direct = sp.Add(*[wi * e for wi, e in zip(w, p.energies, strict=True)])
        var_direct = sp.Add(
            *[wi * (e - u_direct) ** 2 for wi, e in zip(w, p.energies, strict=True)]
        )
    elif p.moments1 is not None:
        m0, m1, m2 = p.moments1
        u_direct = m1 / m0
        var_direct = m2 / m0 - u_direct**2
    else:
        return
    for q in quantities:
        if q.name == "U":
            q.checks.append(symbolic_zero("U", canon.base["U"], u_direct, deadline))
        elif q.name == "Var_E":
            q.checks.append(symbolic_zero("Var_E", canon.base["Var_E"], var_direct, deadline))


def _partial_quantity(
    canon: Canonical, oracle: BaseOracle, points: list[Point], spec: Partial
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
            oracle.eval_expr(scale_shown, pt) * oracle.derivative_of(spec.of, pt, var, spec.order)
        ),
        "numerical derivative of the enumerated quantity, the other state variables fixed",
        oracle.diff_tol,
    )
    d = "d" if spec.order == 1 else f"d^{spec.order}"
    label = spec.label or f"{d}{spec.of}/d{var}"
    return make_quantity(
        canon,
        label,
        total,
        title=f"Partial derivative of {TITLES[spec.of].lower()} with respect to {var}",
        route=f"{d}{spec.of}/d{var} at fixed {', '.join(fixed) or 'no other state variable'}, times {spec.scale}",
        held_fixed=fixed,
        checks=[check],
        variable=var,
    )


# -- expectation --------------------------------------------------------------------------
def expectation(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition]]:
    if problem.kind == "occupation":
        from .grand import expectation_grand

        return expectation_grand(problem, request, deadline)
    if problem.kind == "quantum" and request.observable != "E" and problem.copies_count is None:
        return expectation_quantum(problem, request, deadline)
    if problem.kind != "levels" or problem.copies_count is not None:
        return expectation_energy(problem, request, deadline)
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
        order=order if stat == "cumulant" else None,
    )
    steps = [*partition_steps(problem, canon), probabilities_step(canon)]
    steps.append(
        StatStep(
            name="expectation", title=q.title, latex=q.latex, text=q.text, conditions=conditions
        )
    )
    return steps, [q], conditions


Cumulants = list[Any]


def _energy_statistic(
    k: dict[int, sp.Expr], stat: str, order: int
) -> tuple[sp.Expr, int, Callable[[Cumulants], Any]]:
    """The statistic of E as an expression in the cumulants, and the same from numeric cumulants."""
    if stat == "mean":
        return k[1], 1, lambda c: c[0]
    if stat == "variance":
        return k[2], 2, lambda c: c[1]
    if stat == "cumulant":
        return k[order], order, lambda c: c[order - 1]
    if stat == "central_moment":
        central: dict[int, tuple[sp.Expr, Callable[[Cumulants], Any]]] = {
            1: (sp.Integer(0), lambda c: 0),
            2: (k[2], lambda c: c[1]),
            3: (k[3], lambda c: c[2]),
            4: (k[4] + 3 * k[2] ** 2, lambda c: c[3] + 3 * c[1] ** 2),
        }
        return central[order][0], order, central[order][1]
    raw: dict[int, tuple[sp.Expr, Callable[[Cumulants], Any]]] = {
        1: (k[1], lambda c: c[0]),
        2: (k[2] + k[1] ** 2, lambda c: c[1] + c[0] ** 2),
        3: (k[3] + 3 * k[2] * k[1] + k[1] ** 3, lambda c: c[2] + 3 * c[1] * c[0] + c[0] ** 3),
        4: (
            k[4] + 4 * k[3] * k[1] + 3 * k[2] ** 2 + 6 * k[2] * k[1] ** 2 + k[1] ** 4,
            lambda c: c[3] + 4 * c[2] * c[0] + 3 * c[1] ** 2 + 6 * c[1] * c[0] ** 2 + c[0] ** 4,
        ),
    }
    return raw[order][0], order, raw[order][1]


def expectation_energy(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition]]:
    """Statistics of the energy from derivatives of ln Z (any system with a closed form).

    The cumulants are kappa_n = (-1)^n d^n ln Z / d beta^n; they are compared with the cumulants of
    the directly enumerated moments. Per-level observables exist only for an explicit list of levels.
    """
    if request.observable != "E" or request.statistic == "covariance":
        raise Refusal(
            "unknown_observable",
            "for this system only statistics of the energy E are available; observables with "
            "values per level need an explicit list of levels (without copies)",
        )
    canon = Canonical(problem)
    oracle = make_oracle(problem)
    points = sample_points(problem)
    b = problem.beta
    order, stat = request.order, request.statistic
    needed = _order_needed(stat, order)
    k = {  # only the orders asked for are differentiated and simplified
        n: tidy((-1) ** n * sp.diff(canon.lnZ, b, n)) if n <= needed else sp.Integer(0)
        for n in (1, 2, 3, 4)
    }
    expr, kn, ref = _energy_statistic(k, stat, order)
    expr = tidy(expr)
    shown = canon.shown(expr)

    def reference(pt: Point) -> Any:
        return ref([oracle.energy_cumulant(pt, j) for j in (1, 2, 3, 4)])

    checks = [
        compare(
            "enumeration",
            stat,
            points,
            _evaluator(oracle, shown),
            reference,
            "cumulants of the directly enumerated moments of E",
            oracle.tol * 10**3,
        )
    ]
    if stat in {"variance"} or (stat == "cumulant" and order == 2):
        checks.append(_nonnegative(shown, oracle, points))
    if stat in {"mean", "variance", "cumulant"} and kn <= (4 if oracle.exact else 2):
        checks.append(
            compare(
                "fluctuation_response",
                f"kappa_{kn}(E)",
                points,
                _evaluator(oracle, shown),
                lambda pt: (-1) ** kn * oracle.derivative_of("lnZ", pt, problem.beta.name, kn),
                "cumulant of E as (-1)^n times the n-th numerical derivative of ln Z with respect to beta",
                oracle.diff_tol * 10,
            )
        )
    conditions = partition_conditions(problem)
    q = make_quantity(
        canon,
        f"{stat}_E",
        expr,
        title=f"{stat.replace('_', ' ').capitalize()} of E",
        route="cumulants of E as derivatives of ln Z with respect to beta",
        held_fixed=sorted(s.name for s in problem.params),
        checks=checks,
        order=order if stat == "cumulant" else None,
    )
    steps = [
        *partition_steps(problem, canon),
        StatStep(
            name="expectation", title=q.title, latex=q.latex, text=q.text, conditions=conditions
        ),
    ]
    return steps, [q], conditions


def _order_needed(stat: str, order: int) -> int:
    return {"mean": 1, "variance": 2, "covariance": 2}.get(stat, order)


def expectation_quantum(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition]]:
    """Thermal statistics of a matrix observable: Tr(A^k exp(-beta H))/Z with every operator in place.

    The first derivative of ln Z with respect to a coupled parameter is always beta times the mean (trace
    cyclicity). Beyond it, and for a covariance, the answer depends on ordering unless the operators commute,
    so those are refused (``operators_do_not_commute``) instead of being computed under a commuting assumption.
    """
    from .quantum import commutes, is_hermitian, thermal_matrix

    name, stat, order = request.observable, request.statistic, request.order
    matrices: dict[str, sp.Matrix] = problem.extra["matrices"]
    if name not in matrices:
        raise Refusal(
            "unknown_observable", f"{name} is not an operator of this system (give its matrix)"
        )
    A = matrices[name]
    H: sp.Matrix = problem.extra["H"]
    b = problem.beta
    other: sp.Matrix | None = None
    if stat == "covariance":
        if request.other_observable not in matrices:
            raise Refusal(
                "unknown_observable", "a covariance needs a second operator with a matrix"
            )
        other = matrices[request.other_observable or ""]
        if not commutes(A, other):
            raise Refusal(
                "operators_do_not_commute",
                "the covariance of operators that do not commute depends on their ordering, so "
                "no single value is computed; ask for each ordering or a symmetrised form separately",
            )
    canon = Canonical(problem)
    oracle = make_oracle(problem)
    points = sample_points(problem)
    eu, distinct = thermal_matrix(problem)
    z = canon.Z
    kn = _order_needed(stat, order)

    def moment(k: int, op: sp.Matrix = A) -> sp.Expr:
        return tidy(((op**k) * eu).trace() / z)

    if other is not None:
        m1a, m1b = moment(1), moment(1, other)
        cov = tidy(((A * other) * eu).trace() / z - m1a * m1b)
        expr, ref = (
            cov,
            lambda pt: _quantum_covariance(oracle, name, request.other_observable or "", pt),
        )
    else:
        raw = {k: moment(k) for k in range(1, kn + 1)}
        k = {1: raw[1]}
        if kn >= 2:
            k[2] = raw[2] - raw[1] ** 2
        if kn >= 3:
            k[3] = raw[3] - 3 * raw[2] * raw[1] + 2 * raw[1] ** 3
        if kn >= 4:
            k[4] = (
                raw[4]
                - 4 * raw[3] * raw[1]
                - 3 * raw[2] ** 2
                + 12 * raw[2] * raw[1] ** 2
                - 6 * raw[1] ** 4
            )
        for j in range(1, 5):
            k.setdefault(j, sp.Integer(0))
        expr, _, cum_ref = _energy_statistic(k, stat, order)

        def ref(pt: Point) -> Any:
            return cum_ref([oracle.observable_cumulant(name, pt, j) for j in (1, 2, 3, 4)])  # type: ignore[attr-defined]

        expr = tidy(expr)
    shown = canon.shown(expr)
    checks = [
        compare(
            "enumeration",
            stat,
            points,
            _evaluator(oracle, shown),
            ref,
            "Tr(rho A^k) from mpmath.expm of the numerical matrix",
            oracle.tol * 10**3,
        ),
        compare(
            "spectral_decomposition",
            "Z",
            points,
            lambda pt: oracle.lnZ1(pt, pt[problem.beta]),
            lambda pt: oracle.lnZ_spectral(pt, pt[problem.beta]),  # type: ignore[attr-defined]
            "ln Tr exp(-beta H) from expm equals ln of the sum of exp(-beta lambda) from eigh",
            oracle.tol * 10**3,
        ),
    ]
    conditions = partition_conditions(problem)
    herm = is_hermitian(A)
    conditions.append(
        Condition(
            latex=rf"{name} = {name}^{{\dagger}}",
            established=herm,
            how="the operator is Hermitian, so its thermal average is real"
            if herm
            else "the operator is not Hermitian: its thermal average may be complex",
        )
    )
    comm = commutes(A, H)
    conditions.append(
        Condition(
            latex=rf"[{name}, H] = 0",
            established=comm,
            how="the operator commutes with H; the order of A and exp(-beta H) does not matter"
            if comm
            else "the operator does not commute with H: Tr(A exp(-beta H)) keeps A where it is",
        )
    )
    conditions.append(distinct)
    if request.coupling is not None and request.coupling.observable == name and other is None:
        lam = next((s for s in problem.params if s.name == request.coupling.parameter), None)
        if lam is None:
            raise Refusal(
                "coupling_not_linear", "the coupling parameter must be a declared parameter"
            )
        if not all(sp.simplify(e) == 0 for e in (sp.diff(H, lam) + A)) or A.has(lam):
            raise Refusal(
                "coupling_not_linear",
                "the response relation needs dH/d(parameter) = -A with A independent of the parameter",
            )
        if kn >= 2 and not comm:
            raise Refusal(
                "operators_do_not_commute",
                f"{name} does not commute with H, so the second derivative of ln Z with respect to "
                f"{lam.name} is the Kubo-Duhamel covariance, not beta^2 Var({name}); only the first "
                "derivative (the mean) is a response relation here",
            )
        conditions.append(
            Condition(
                latex=rf"\partial H/\partial {to_latex(lam)} = -{name}",
                established=True,
                how="checked symbolically; the first derivative of ln Z is beta times the mean by trace cyclicity",
            )
        )
        checks.append(
            compare(
                "fluctuation_response",
                f"kappa_{kn}({name})",
                points,
                _evaluator(oracle, shown),
                lambda pt: oracle.derivative_of("lnZ", pt, lam.name, kn) / pt[b] ** kn,
                f"cumulant as beta^-{kn} times the {kn}-th numerical derivative of ln Z with respect to {lam.name}",
                oracle.diff_tol * 10,
            )
        )
    if (stat == "variance" or (stat == "cumulant" and order == 2)) and herm:
        checks.append(_nonnegative(shown, oracle, points))
    q = make_quantity(
        canon,
        f"{stat}_{name}",
        expr,
        title=f"{stat.replace('_', ' ').capitalize()} of {name}",
        route="Tr(A^k exp(-beta H))/Z with the operators in the order given, exp(-beta H) from the spectral decomposition",
        held_fixed=sorted(s.name for s in problem.params),
        checks=checks,
        order=order if stat == "cumulant" else None,
    )
    steps = [
        *partition_steps(problem, canon),
        StatStep(
            name="expectation", title=q.title, latex=q.latex, text=q.text, conditions=conditions
        ),
    ]
    return steps, [q], conditions


def _quantum_covariance(oracle: Any, a: str, b: str, pt: Point) -> Any:
    beta = pt[oracle.problem.beta]
    ma, mb = oracle.operator(a, pt), oracle.operator(b, pt)
    return oracle.moment(pt, beta, ma * mb, 1) - oracle.moment(pt, beta, ma, 1) * oracle.moment(
        pt, beta, mb, 1
    )


def _response_checks(
    problem: Problem,
    request: StatMechInput,
    canon: Canonical,
    oracle: BaseOracle,
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
                lambda pt: (-1) ** n * oracle.derivative_of("lnZ", pt, p.beta.name, n),
                "cumulant of E as (-1)^n times the n-th derivative of the enumerated ln Z with respect to beta",
                oracle.diff_tol * 10,
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
            lambda pt: oracle.derivative_of("lnZ", pt, lam.name, n) / pt[b] ** n,
            f"cumulant as beta^-{n} times the {n}-th derivative of the enumerated ln Z with respect to "
            f"{lam.name}",
            oracle.diff_tol * 10,
        )
    )
    response = tidy(sp.diff(canon.lnZ, lam, n) / b**n)
    if len(p.energies) <= SYMBOLIC_LEVELS:
        out.append(symbolic_zero(f"kappa_{n}", response, expr, deadline))
    return out
