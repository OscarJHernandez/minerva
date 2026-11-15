"""Numerical evaluation at user-chosen numbers, and parameter sweeps.

All numbers come from the user. With a closed form, each value is the formula evaluated at the point
and is compared with the independent oracle; without one, the oracle value is shown as a numerical
result (never as an identity) after two internal consistency checks: the internal energy from the
moments against the numerical derivative of ln Z, and F = U - T S.
"""

from __future__ import annotations

from typing import Any

import mpmath as mp
import sympy as sp

from ..distributions.models import Condition
from .canonical import (
    TITLES,
    Canonical,
    compare,
    make_canonical,
    make_oracle,
    partition_conditions,
    partition_steps,
)
from .models import (
    BASE_QUANTITIES,
    GRAND_QUANTITIES,
    StatCheck,
    StatMechInput,
    StatQuantity,
    StatStep,
    StatTable,
)
from .oracle import DPS, BaseOracle, Point
from .setup import Problem, Refusal

SHOWN_DIGITS = 15


def _number(problem: Problem, text: str) -> sp.Expr:
    value = problem.parse(text)
    if value.free_symbols or value.is_real is not True:
        raise Refusal("parse_error", "values must be real numbers")
    return value


def _respects(symbol: sp.Symbol, value: sp.Expr) -> bool:
    if symbol.is_positive and not value.is_positive:
        return False
    if symbol.is_nonnegative and not value.is_nonnegative:
        return False
    if symbol.is_negative and not value.is_negative:
        return False
    return not (symbol.is_integer and not value.is_integer)


def build_point(problem: Problem, values: dict[str, sp.Expr]) -> Point:
    needed = set(problem.sample_symbols)
    if problem.request.temperature == "T":
        needed.discard(problem.beta)
        needed.add(problem.T)
    by_name = {s.name: s for s in needed}
    missing = sorted(set(by_name) - set(values))
    if missing:
        raise Refusal(
            "parse_error",
            f"give a number for each of: {', '.join(missing)} (for example k_B = 1 in reduced units)",
        )
    extra = sorted(set(values) - set(by_name))
    if extra:
        raise Refusal("parse_error", f"no such symbol in this problem: {', '.join(extra)}")
    point: Point = {}
    with mp.workdps(DPS):
        for name, symbol in by_name.items():
            v = values[name]
            if not _respects(symbol, v):
                raise Refusal(
                    "assumptions_insufficient",
                    f"the value of {name} contradicts its declared assumptions",
                )
            point[symbol] = (
                mp.mpf(v.p) / mp.mpf(v.q) if v.is_Rational else mp.mpf(str(sp.N(v, DPS)))
            )
        if problem.request.temperature == "beta":
            point[problem.T] = 1 / (point[problem.k_B] * point[problem.beta])
        else:
            point[problem.beta] = 1 / (point[problem.k_B] * point[problem.T])
    return point


def _fmt(x: Any) -> str:
    return str(mp.nstr(x, SHOWN_DIGITS))


def evaluate(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition], StatTable | None]:
    grand = problem.kind == "occupation"
    names = list(request.quantities) or (
        ["lnXi", "Omega", "N", "U", "S"] if grand else ["lnZ", "F", "U", "S", "Var_E"]
    )
    for n in names:
        if n not in (GRAND_QUANTITIES if grand else BASE_QUANTITIES):
            raise Refusal("unsupported_system", "unknown quantity")
    oracle = make_oracle(problem)
    try:
        canon: Canonical | None = make_canonical(problem)
    except Refusal:
        canon = None  # no closed form: numerical evaluation only
    values = {k: _number(problem, v) for k, v in request.values.items()}
    sweep = request.sweep
    if sweep is None:
        points = [build_point(problem, values)]
        swept_name = ""
    else:
        swept_name = sweep.variable
        lo, hi = _number(problem, sweep.start), _number(problem, sweep.stop)
        if sweep.scale == "log" and (lo <= 0 or hi <= 0):
            raise Refusal("parse_error", "a logarithmic sweep needs positive end points")
        points = []
        steps_n = sweep.points - 1
        for i in range(sweep.points):
            t = sp.Rational(i, steps_n)
            v = lo * (hi / lo) ** t if sweep.scale == "log" else lo + (hi - lo) * t
            points.append(build_point(problem, {**values, swept_name: sp.nsimplify(v)}))
    rows: list[list[str]] = []
    series: dict[str, list[Any]] = {n: [] for n in names}
    with mp.workdps(DPS):
        for pt in points:
            row = []
            for n in names:
                value = oracle.quantity(n, pt)
                series[n].append(value)
                row.append(_fmt(value))
            rows.append(row)
    check_points = [points[0], points[len(points) // 2], points[-1]] if sweep else points
    quantities: list[StatQuantity] = []
    for n in names:
        checks = _checks(n, canon, oracle, check_points, problem)
        last = series[n][-1] if sweep is None else None
        text = _fmt(last) if last is not None else f"{len(points)} values, see the table"
        quantities.append(
            StatQuantity(
                name=n,
                title=TITLES[n],
                latex=text,
                text=text,
                result_kind="numeric",
                route=(
                    "closed form evaluated at the numbers and compared with the independent numerical "
                    "evaluation"
                    if canon is not None
                    else "direct numerical evaluation (no closed form); two internal consistency checks"
                ),
                checks=checks,
            )
        )
    table = None
    if sweep is not None:
        sym = next(s for s in [*problem.sample_symbols, problem.T] if s.name == swept_name)
        xs = [_fmt(pt[sym]) for pt in points]
        table = StatTable(
            columns=[swept_name, *names],
            rows=[[x, *row] for x, row in zip(xs, rows, strict=True)],
            variable=swept_name,
        )
    steps = (
        _grand_steps(problem, canon)
        if grand and canon is not None
        else partition_steps(problem, canon)
        if canon is not None
        else [step.model_copy(deep=True) for step in problem.setup_steps]
    )
    steps.append(
        StatStep(
            name="numeric_evaluation",
            title="Numerical evaluation",
            latex=r"\text{numerical values at the numbers you gave}",
            result_kind="numeric",
        )
    )
    conditions = partition_conditions(problem)
    return steps, quantities, conditions, table


def _grand_steps(problem: Problem, canon: Canonical) -> list[StatStep]:
    from .grand import GrandCanonical, _setup_steps

    assert isinstance(canon, GrandCanonical)  # noqa: S101 - narrowing for the type checker
    return _setup_steps(problem, canon)


def _checks(
    name: str,
    canon: Canonical | None,
    oracle: BaseOracle,
    points: list[Point],
    problem: Problem,
) -> list[StatCheck]:
    checks: list[StatCheck] = []
    if canon is not None:
        expr = canon.shown(canon.base[name])
        checks.append(
            compare(
                "enumeration",
                name,
                points,
                lambda pt: oracle.eval_expr(expr, pt),
                lambda pt: oracle.quantity(name, pt),
                "closed form evaluated at the numbers versus the independent numerical evaluation",
                oracle.tol * 10**3,
                min_points=1,
            )
        )
        return checks
    if name == "U":
        checks.append(
            compare(
                "finite_difference",
                "U",
                points,
                lambda pt: oracle.quantity("U", pt),
                lambda pt: -oracle.derivative_of("lnZ", pt, problem.beta.name, 1),
                "internal energy from the moments versus minus the numerical derivative of ln Z",
                oracle.diff_tol * 10,
                min_points=1,
            )
        )
    elif name == "F":
        checks.append(
            compare(
                "free_energy_identity",
                "F",
                points,
                lambda pt: oracle.quantity("F", pt),
                lambda pt: (
                    oracle.quantity("U", pt)
                    - oracle.quantity("S", pt) / (pt[problem.k_B] * pt[problem.beta])
                ),
                "F = U - T S on the numerical values",
                oracle.tol * 10**3,
                min_points=1,
            )
        )
    elif name == "Var_E":
        checks.append(
            compare(
                "finite_difference",
                "Var_E",
                points,
                lambda pt: oracle.quantity("Var_E", pt),
                lambda pt: oracle.derivative_of("lnZ", pt, problem.beta.name, 2),
                "energy variance from the moments versus the second numerical derivative of ln Z",
                oracle.diff_tol * 10,
                min_points=1,
            )
        )
    return checks
