"""Local and perturbative approximations of a differential equation, shown only when verified.

A Taylor series about the point of the initial values, and expansions in a small parameter. Every result
is a *local* statement: the series says nothing far from its point (its radius of convergence is not
determined) and a perturbation expansion is for small values of the parameter on bounded intervals.

The coefficients are built one way (repeated total differentiation of the solved form; order by order
in the small parameter) and checked another: the truncated result is put into the equation *as the user
wrote it*, and the residual must fall at the claimed power when the step (or the parameter) is halved
twice. A wrong coefficient of any retained order makes the residual fall too slowly and is refused.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import mpmath as mp
import sympy as sp

from ..errors import MinervaMathError
from ..latexprint import to_latex
from ..models import Limits, Status
from ..series import RATIO_SLACK, TINY
from .analyze import _classify_conditions, interpretation, solve_highest
from .models import (
    ConditionCheck,
    ExpansionCheck,
    OdeDiagnostics,
    OdeExpansion,
    OdeSolution,
    OdeSolutionItem,
    OdeSolveInput,
    OdeStage,
    OdeVerification,
)
from .reading import OdeProblem, read_ode
from .verification import (
    DIGITS,
    Unevaluable,
    _Evaluator,
    _lambdify,
    _parameter_values,
)

MAX_TERM_OPS = 30_000
# The largest step of the triple h, h/2, h/4. Small enough that an error of 0.1% in the last retained
# coefficient dominates the truncation error (it would hide behind it at h = 0.04); the working
# precision (60 digits) keeps the residuals meaningful this small.
STEPS = (0.004,)
CONFIRMATIONS = 4  # parameter samples times directions that must all confirm the order
LOCAL = (
    "A local approximation about the expansion point: its radius of convergence is not determined, and "
    "it says nothing about the solution far from that point."
)


@dataclass
class _Run:
    """Timing, diagnostics and the two ways out (a refusal, a finished result)."""

    method: str
    trace_id: str | None
    started: float = field(default_factory=time.perf_counter)
    clock: float = field(default_factory=time.perf_counter)
    stages: list[OdeStage] = field(default_factory=list)
    diagnostics: OdeDiagnostics = field(default_factory=OdeDiagnostics)

    def __post_init__(self) -> None:
        self.diagnostics = OdeDiagnostics(trace_id=self.trace_id, method=self.method)

    def mark(self, name: Literal["read", "solve", "verify", "total"]) -> None:
        now = time.perf_counter()
        self.stages.append(OdeStage(name=name, ms=round((now - self.clock) * 1000, 2)))
        self.clock = now

    def finish(self, solution: OdeSolution) -> OdeSolution:
        self.stages.append(
            OdeStage(name="total", ms=round((time.perf_counter() - self.started) * 1000, 2))
        )
        self.diagnostics.stages = self.stages
        solution.diagnostics = self.diagnostics
        return solution

    def refuse(self, status: Status, message: str, reason: str, **extra: Any) -> OdeSolution:
        self.diagnostics.failure_reason = reason
        return self.finish(OdeSolution(status=status, message=message, method=self.method, **extra))


def weakest(items: list[OdeSolutionItem]) -> OdeVerification:
    """The weakest of the shown solutions' checks: fewest points, and symbolic only if every one proved it."""
    return OdeVerification(
        points_checked=min(i.verification.points_checked for i in items),
        points_agreed=min(i.verification.points_agreed for i in items),
        points_skipped=max(i.verification.points_skipped for i in items),
        max_residual=max((i.verification.max_residual or 0.0) for i in items),
        symbolic_zero=True if all(i.verification.symbolic_zero is True for i in items) else None,
        independent_constants=min((i.verification.independent_constants or 0) for i in items),
        passed=all(i.verification.passed for i in items),
    )


def _finite(expr: sp.Expr) -> bool:
    return not expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo)


def _tidy(expr: sp.Expr) -> sp.Expr:
    return sp.simplify(expr) if sp.count_ops(expr) <= 250 else sp.together(expr)


# --- Taylor series ---------------------------------------------------------------------------


def _taylor(
    problem: OdeProblem,
    rhs: list[sp.Expr],
    t0: sp.Expr,
    initial: dict[tuple[str, int], sp.Expr],
    order: int,
) -> tuple[list[sp.Expr] | None, str]:
    """The Taylor polynomial of each unknown about `t0`, by repeated total differentiation of the first-order
    system; (None, reason) at a point where a derivative is not defined."""
    t = problem.variable
    labels = [(u, k) for u in problem.unknowns for k in range(problem.orders[u])]
    states = [problem.jets[label] for label in labels]
    field_of: dict[sp.Symbol, sp.Expr] = {}
    for u in problem.unknowns:
        n = problem.orders[u]
        for k in range(n):
            field_of[problem.jets[(u, k)]] = (
                problem.jets[(u, k + 1)] if k < n - 1 else rhs[problem.unknowns.index(u)]
            )

    def derivative(g: sp.Expr) -> sp.Expr:
        return sp.diff(g, t) + sum((sp.diff(g, s) * field_of[s] for s in states), sp.Integer(0))

    at = {t: t0, **{problem.jets[label]: initial[label] for label in labels}}
    polynomials: list[sp.Expr] = []
    for u in problem.unknowns:
        g: sp.Expr = problem.jets[(u, 0)]
        terms: list[sp.Expr] = []
        for m in range(order + 1):
            if m:
                g = derivative(g)
                if sp.count_ops(g) > MAX_TERM_OPS:
                    return None, "the derivatives grow too large to continue"
            value = g.xreplace(at)
            if not _finite(value):
                return None, "a derivative is not defined at the expansion point (a singular point)"
            terms.append(_tidy(value) / sp.factorial(m) * (t - t0) ** m)
        polynomials.append(sp.Add(*terms))
    return polynomials, ""


def _check_series(
    problem: OdeProblem,
    polynomials: list[sp.Expr],
    t0: sp.Expr,
    order: int,
    kinds: list[tuple[str, int, sp.Expr, sp.Expr]],
    seed: int = 20260401,
) -> tuple[OdeVerification, list[ExpansionCheck], list[ConditionCheck]]:
    """The residual of the polynomial in the equation as written, at three steps h, h/2, h/4 on both sides
    of the point, for several parameter samples: it must fall at least as h^(order + 1 - n)."""
    mp.mp.dps = DIGITS
    t = problem.variable
    n_max = max(problem.orders.values())
    claimed = order + 1 - n_max
    args = [*problem.parameters]
    try:
        evaluator = _Evaluator(problem)
        derivatives = [
            [
                _lambdify([t, *args], sp.diff(p, t, j) if j else p)
                for j in range(problem.orders[u] + 1)
            ]
            for u, p in zip(problem.unknowns, polynomials, strict=True)
        ]
        t0_fn = _lambdify(args, t0)
    except Unevaluable:
        return OdeVerification(note="could not be evaluated numerically"), [], []
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    checked = agreed = skipped = 0
    smallest = 0.0
    for _ in range(CONFIRMATIONS // 2 + 4):
        if checked >= CONFIRMATIONS:
            break
        parameters = _parameter_values(problem, rng, wide=False)
        try:
            centre = t0_fn(*parameters)
        except (ArithmeticError, ValueError, TypeError, ZeroDivisionError):
            skipped += 1
            continue
        for direction in (1, -1):
            if checked >= CONFIRMATIONS:
                break
            confirmed = True
            residuals: list[float] = []
            try:
                sizes = [mp.mpf(STEPS[0]), mp.mpf(STEPS[0]) / 2, mp.mpf(STEPS[0]) / 4]
                reference: list[Any] = []
                row = []
                for h in sizes:
                    x = centre + direction * h
                    jets = [d(x, *parameters) for per in derivatives for d in per]
                    parts = evaluator.parts(x, jets, parameters)
                    if not reference:
                        # The scale is fixed at the largest step: terms that vanish at the point
                        # (as in y' = -2 x y at 0) must not make the residual look larger.
                        reference = [max(scale, mp.mpf(10) ** -40) for _, scale in parts]
                    row.append(
                        float(max(v / r for (v, _), r in zip(parts, reference, strict=True)))
                    )
                residuals = row
                confirmed = confirmed and _falls(row, claimed)
            except (ArithmeticError, ValueError, TypeError, ZeroDivisionError, OverflowError):
                skipped += 1
                continue
            checked += 1
            agreed += int(confirmed)
            smallest = max(smallest, residuals[-1] if residuals else 0.0)
    verification = OdeVerification(
        points_checked=checked,
        points_agreed=agreed,
        points_skipped=skipped,
        max_residual=smallest,
        passed=checked >= CONFIRMATIONS and agreed == checked,
        note=(
            f"the polynomial put into the equation as written leaves a residual that falls at least as "
            f"h^{claimed} when the step h is halved twice"
        ),
    )
    conditions = _condition_checks(problem, derivatives, t0_fn, kinds, rng)
    checks = [
        ExpansionCheck(
            name="residual_order",
            passed=verification.passed,
            value=smallest,
            note=f"claimed h^{claimed}",
        ),
        ExpansionCheck(
            name="initial_conditions",
            passed=all(c.satisfied for c in conditions) and bool(conditions),
        ),
    ]
    return verification, checks, conditions


def _falls(residuals: list[float], claimed: int) -> bool:
    """h -> h/2 -> h/4 reduces the residual by at least 2**claimed each time (up to the slack of the
    other series checks); a residual at the limit of working precision counts as zero."""
    a, b, c = (mp.mpf(r) for r in residuals)
    if a <= TINY:
        return True
    need = RATIO_SLACK * 2**claimed
    return bool((b <= TINY or a / b >= need) and (c <= TINY or b / c >= need))


def _condition_checks(
    problem: OdeProblem,
    derivatives: list[list[Any]],
    t0_fn: Any,
    kinds: list[tuple[str, int, sp.Expr, sp.Expr]],
    rng: random.Random,
) -> list[ConditionCheck]:
    from .reading import _display

    out: list[ConditionCheck] = []
    value_fns = []
    try:
        value_fns = [_lambdify(problem.parameters, value) for _, _, _, value in kinds]
    except Unevaluable:
        return out
    parameters = _parameter_values(problem, rng, wide=False)
    for (unknown, order, at, value), value_fn in zip(kinds, value_fns, strict=True):
        index = problem.unknowns.index(unknown)
        label = (
            f"{_display(unknown, order)}\\left({problem.latex(at)}\\right) = {problem.latex(value)}"
        )
        try:
            got = derivatives[index][order](t0_fn(*parameters), *parameters)
            want = value_fn(*parameters)
            residual = float(abs(got - want) / max(mp.mpf(1), abs(want)))
        except (ArithmeticError, ValueError, TypeError, ZeroDivisionError):
            out.append(ConditionCheck(latex=label, satisfied=False, residual=None))
            continue
        out.append(ConditionCheck(latex=label, satisfied=residual <= 1e-12, residual=residual))
    return out


def solve_series(data: OdeSolveInput, limits: Limits, trace_id: str | None = None) -> OdeSolution:
    run = _Run("series", trace_id)
    try:
        problem = read_ode(data, limits)
    except MinervaMathError as exc:
        from ..ops import _failure

        status, message = _failure(exc)
        run.mark("read")
        return run.refuse(status, message, status.value)
    run.mark("read")
    meaning = interpretation(problem)

    def refuse(status: Status, message: str, reason: str) -> OdeSolution:
        return run.refuse(status, message, reason, interpretation=meaning)

    solved, why = solve_highest(problem)
    if solved is None:
        return refuse(
            Status.UNSUPPORTED, f"Not solved for the highest derivative: {why}.", "not_solved"
        )
    kind, needed = _classify_conditions(problem)
    if kind != "initial_value":
        return refuse(
            Status.PARSE_ERROR,
            "A series about a point needs every initial value there, all at one point"
            + (f" (still needed: {', '.join(needed)})" if needed else "")
            + ". Minerva does not choose them.",
            "conditions",
        )
    t0 = problem.conditions[0].at
    initial = {(c.unknown, c.order): c.value for c in problem.conditions}
    polynomials, reason = _taylor(problem, solved.rhs, t0, initial, data.order)
    run.mark("solve")
    if polynomials is None:
        return refuse(Status.UNSUPPORTED, f"No series is given: {reason}.", "singular_point")
    kinds = [(c.unknown, c.order, c.at, c.value) for c in problem.conditions]
    verification, checks, conditions = _check_series(problem, polynomials, t0, data.order, kinds)
    run.mark("verify")
    run.diagnostics.order = max(problem.orders.values())
    run.diagnostics.state_dimension = sum(problem.orders.values())
    if not verification.passed or not all(c.satisfied for c in conditions):
        run.diagnostics.verified = 0
        run.diagnostics.dropped = 1
        return refuse(
            Status.NOT_VERIFIED,
            "The series did not pass the check of its residual or of the initial values, so it is not shown.",
            "not_verified",
        )
    shift = to_latex(problem.variable - t0)
    equations = [
        f"{_name(problem, u)} = {to_latex(p)} + O\\left(\\left({shift}\\right)^{{{data.order + 1}}}\\right)"
        for u, p in zip(problem.unknowns, polynomials, strict=True)
    ]
    item = OdeSolutionItem(
        kind="series",
        equations_latex=equations,
        rhs_text=[str(p) for p in polynomials],
        family="particular",
        verification=verification,
        conditions=conditions,
        expansion=OdeExpansion(
            kind="taylor",
            order=data.order,
            point_latex=f"{to_latex(problem.variable)} = {to_latex(t0)}",
            checks=checks,
            validity=LOCAL,
        ),
    )
    run.diagnostics.verified = 1
    run.diagnostics.dropped = 0
    run.diagnostics.solution_kind = "series"
    run.diagnostics.max_residual = verification.max_residual
    return run.finish(
        OdeSolution(
            status=Status.OK,
            method="series",
            interpretation=meaning,
            solutions=[item],
            verification=weakest([item]),
            notes=[LOCAL],
        )
    )


def _name(problem: OdeProblem, unknown: str) -> str:
    from .reading import _latex_name

    return f"{_latex_name(unknown)}\\left({to_latex(problem.variable)}\\right)"


# --- perturbation ----------------------------------------------------------------------------

MAX_REGULAR_ORDER = 4
EPSILONS = (mp.mpf("0.004"), mp.mpf("0.002"), mp.mpf("0.001"))
T_OFFSETS = (0.4, 0.9, 1.6)
VALID_FOR_SMALL = (
    "An expansion in the small parameter: valid for small values of it and, where a term grows with the "
    "variable, only on a bounded interval. It says nothing about large values of the parameter."
)


def _secular(terms: sp.Expr, t: sp.Symbol) -> bool:
    """A factor t (a power of t) multiplying an oscillation: the expansion breaks down for t ~ 1/eps."""
    for term in sp.Add.make_args(sp.expand(terms)):
        if term.has(sp.sin, sp.cos):
            bare = term.replace(sp.sin, lambda *_: 1).replace(sp.cos, lambda *_: 1)
            if bare.has(t):
                return True
    return False


def _check_expansion(
    problem: OdeProblem,
    candidate: sp.Expr,
    eps: sp.Symbol,
    order: int,
    kinds: list[tuple[str, int, sp.Expr, sp.Expr]],
    seed: int = 20260402,
) -> tuple[OdeVerification, list[ExpansionCheck], list[ConditionCheck]]:
    """The truncated expansion put into the equation as written, for eps, eps/2, eps/4: the residual must
    fall at least as eps^(order + 1), at several parameter samples and values of the variable."""
    mp.mp.dps = DIGITS
    t = problem.variable
    name = problem.unknowns[0]
    n = problem.orders[name]
    args = [*problem.parameters]
    position = args.index(eps)
    claimed = order + 1
    try:
        evaluator = _Evaluator(problem)
        derivatives = [
            _lambdify([t, *args], sp.diff(candidate, t, j) if j else candidate)
            for j in range(n + 1)
        ]
        t0_fn = _lambdify(args, kinds[0][2])
    except Unevaluable:
        return OdeVerification(note="could not be evaluated numerically"), [], []
    rng = random.Random(seed)  # noqa: S311
    checked = agreed = skipped = 0
    smallest = 0.0
    for _ in range(CONFIRMATIONS + 6):
        if checked >= CONFIRMATIONS:
            break
        base = _parameter_values(problem, rng, wide=False)
        for offset in T_OFFSETS:
            if checked >= CONFIRMATIONS:
                break
            row: list[float] = []
            reference: list[Any] = []
            try:
                for epsilon in EPSILONS:
                    parameters = list(base)
                    parameters[position] = epsilon
                    x = t0_fn(*parameters) + mp.mpf(offset)
                    jets = [d(x, *parameters) for d in derivatives]
                    parts = evaluator.parts(x, jets, parameters)
                    if not reference:
                        reference = [max(scale, mp.mpf(10) ** -40) for _, scale in parts]
                    row.append(
                        float(max(v / r for (v, _), r in zip(parts, reference, strict=True)))
                    )
            except (ArithmeticError, ValueError, TypeError, ZeroDivisionError, OverflowError):
                skipped += 1
                continue
            checked += 1
            agreed += int(_falls(row, claimed))
            smallest = max(smallest, row[-1])
    verification = OdeVerification(
        points_checked=checked,
        points_agreed=agreed,
        points_skipped=skipped,
        max_residual=smallest,
        passed=checked >= CONFIRMATIONS and agreed == checked,
        note=(
            f"the truncated expansion put into the equation as written leaves a residual that falls at "
            f"least as eps^{claimed} when the small parameter is halved twice"
        ),
    )
    conditions = _expansion_conditions(problem, derivatives, kinds, position, args, rng)
    checks = [
        ExpansionCheck(
            name="residual_order",
            passed=verification.passed,
            value=smallest,
            note=f"claimed eps^{claimed}",
        ),
        ExpansionCheck(
            name="initial_conditions",
            passed=bool(conditions) and all(c.satisfied for c in conditions),
        ),
    ]
    return verification, checks, conditions


def _expansion_conditions(
    problem: OdeProblem,
    derivatives: list[Any],
    kinds: list[tuple[str, int, sp.Expr, sp.Expr]],
    position: int,
    args: list[sp.Symbol],
    rng: random.Random,
) -> list[ConditionCheck]:
    from .reading import _display

    try:
        points = [_lambdify(args, at) for _, _, at, _ in kinds]
        wanted = [_lambdify(args, value) for _, _, _, value in kinds]
    except Unevaluable:
        return []
    out: list[ConditionCheck] = []
    base = _parameter_values(problem, rng, wide=False)
    for (unknown, order, at, value), point_fn, value_fn in zip(kinds, points, wanted, strict=True):
        label = (
            f"{_display(unknown, order)}\\left({problem.latex(at)}\\right) = {problem.latex(value)}"
        )
        worst = 0.0
        ok = True
        for epsilon in (mp.mpf("0.01"), mp.mpf("0.003")):
            parameters = list(base)
            parameters[position] = epsilon
            try:
                got = derivatives[order](point_fn(*parameters), *parameters)
                want = value_fn(*parameters)
                worst = max(worst, float(abs(got - want) / max(mp.mpf(1), abs(want))))
            except (ArithmeticError, ValueError, TypeError, ZeroDivisionError):
                ok = False
        out.append(ConditionCheck(latex=label, satisfied=ok and worst <= 1e-12, residual=worst))
    return out


def _dsolve_ivp(
    equation: sp.Expr, function: sp.Expr, t: sp.Symbol, t0: sp.Expr, values: list[sp.Expr]
) -> sp.Expr | None:
    """dsolve with the initial values at t0 (`values[j]` is the j-th derivative); None unless one explicit
    solution comes back."""
    ics = {
        (sp.Derivative(function, (t, j)) if j else function).subs(t, t0): v
        for j, v in enumerate(values)
    }
    try:
        result = sp.dsolve(sp.Eq(equation, 0), function, ics=ics)
    except (NotImplementedError, ValueError, TypeError, KeyError, RecursionError, AttributeError):
        return None
    if not isinstance(result, sp.Eq) or result.lhs != function or result.rhs.has(function):
        return None
    return result.rhs


def solve_perturbation(
    data: OdeSolveInput, limits: Limits, trace_id: str | None = None
) -> OdeSolution:
    from .symbolic import SolverTimeout, deadline

    run = _Run("perturbation", trace_id)
    try:
        problem = read_ode(data, limits)
    except MinervaMathError as exc:
        from ..ops import _failure

        status, message = _failure(exc)
        run.mark("read")
        return run.refuse(status, message, status.value)
    run.mark("read")
    meaning = interpretation(problem)
    run.diagnostics.order = max(problem.orders.values())
    run.diagnostics.state_dimension = sum(problem.orders.values())

    def refuse(status: Status, message: str, reason: str) -> OdeSolution:
        return run.refuse(status, message, reason, interpretation=meaning)

    if len(problem.unknowns) != 1 or len(problem.sides) != 1:
        return refuse(
            Status.UNSUPPORTED,
            "Perturbation expansions are implemented for one equation.",
            "system",
        )
    parameters = ", ".join(p.name for p in problem.parameters) or "none"
    if data.small_parameter is None:
        return refuse(
            Status.PARSE_ERROR,
            f"Name the small parameter (the parameters are: {parameters}). Minerva does not choose it.",
            "small_parameter",
        )
    eps = next((p for p in problem.parameters if p.name == data.small_parameter), None)
    if eps is None:
        return refuse(
            Status.PARSE_ERROR,
            f"{data.small_parameter!r} is not a parameter of the equation (the parameters are: {parameters}).",
            "small_parameter",
        )
    if data.equilibrium is not None:
        return _periodic(data, problem, eps, run, meaning, limits)
    if data.order > MAX_REGULAR_ORDER:
        return refuse(
            Status.UNSUPPORTED,
            f"Regular expansions are implemented to order {MAX_REGULAR_ORDER}.",
            "order",
        )
    kind, needed = _classify_conditions(problem)
    if kind != "initial_value":
        return refuse(
            Status.PARSE_ERROR,
            "A regular expansion needs every initial value, all at one point"
            + (f" (still needed: {', '.join(needed)})" if needed else "")
            + ". Minerva does not choose them.",
            "conditions",
        )
    t = problem.variable
    name = problem.unknowns[0]
    n = problem.orders[name]
    t0 = problem.conditions[0].at
    initial = {c.order: c.value for c in problem.conditions}
    order = data.order
    functions = [sp.Function(f"minerva_pt{k}")(t) for k in range(order + 1)]
    expansion = sum((eps**k * functions[k] for k in range(order + 1)), sp.Integer(0))
    try:
        with deadline(0.5 * limits.timeout_seconds):
            expr = _function_form_residual_functions(problem, expansion, name, t).doit()
            series = sp.expand(sp.series(expr, eps, 0, order + 1).removeO())
            coefficients = [series.coeff(eps, k) for k in range(order + 1)]
            solved: list[sp.Expr] = []
            for k in range(order + 1):
                equation = coefficients[k]
                for j, done in enumerate(solved):
                    equation = equation.subs(functions[j], done)
                equation = sp.simplify(equation.doit())
                values = [
                    initial.get(j, sp.Integer(0)) if k == 0 else sp.Integer(0) for j in range(n)
                ]
                found = _dsolve_ivp(equation, functions[k], t, t0, values)
                if found is None:
                    run.mark("solve")
                    which = "unperturbed" if k == 0 else f"order {k}"
                    return refuse(
                        Status.UNSUPPORTED,
                        f"The {which} problem has no closed form that SymPy could find, so the expansion "
                        "cannot be built. That does not show that none exists.",
                        "no_closed_form",
                    )
                solved.append(found)
    except SolverTimeout:
        run.mark("solve")
        return refuse(
            Status.RESOURCE_LIMIT, "The expansion did not finish within the time limit.", "timeout"
        )
    run.mark("solve")
    truncated = sum((eps**k * solved[k] for k in range(order + 1)), sp.Integer(0))
    kinds = [(c.unknown, c.order, c.at, c.value) for c in problem.conditions]
    verification, checks, conditions = _check_expansion(problem, truncated, eps, order, kinds)
    run.mark("verify")
    secular = [k for k in range(1, order + 1) if _secular(solved[k], t)]
    if not verification.passed or not all(c.satisfied for c in conditions):
        run.diagnostics.verified = 0
        run.diagnostics.dropped = 1
        return refuse(
            Status.NOT_VERIFIED,
            "The expansion did not pass the check of its residual or of the initial values, so it is not shown.",
            "not_verified",
        )
    if secular:
        checks.append(
            ExpansionCheck(
                name="boundedness",
                passed=False,
                note="terms grow with the variable at orders " + ", ".join(map(str, secular)),
            )
        )
    notes = [VALID_FOR_SMALL]
    if secular:
        notes.append(
            "Secular terms (a factor of the variable multiplying an oscillation) appear at order "
            + ", ".join(map(str, secular))
            + ": the expansion is not uniformly valid and fails when the variable is of order 1/"
            + data.small_parameter
            + ". The Poincare-Lindstedt method (with an equilibrium) removes them for periodic motion."
        )
    item = OdeSolutionItem(
        kind="perturbation",
        equations_latex=[
            f"{_name(problem, name)} = {to_latex(truncated)} + O\\left({to_latex(eps)}^{{{order + 1}}}\\right)"
        ],
        rhs_text=[str(truncated)],
        family="particular",
        verification=verification,
        conditions=conditions,
        expansion=OdeExpansion(
            kind="regular_perturbation",
            order=order,
            small_latex=to_latex(eps),
            secular_orders=secular,
            checks=checks,
            validity=VALID_FOR_SMALL,
        ),
    )
    run.diagnostics.verified = 1
    run.diagnostics.dropped = 0
    run.diagnostics.solution_kind = "perturbation"
    run.diagnostics.max_residual = verification.max_residual
    return run.finish(
        OdeSolution(
            status=Status.OK,
            method="perturbation",
            interpretation=meaning,
            solutions=[item],
            verification=weakest([item]),
            notes=notes,
        )
    )


def _function_form_residual_functions(
    problem: OdeProblem, expansion: sp.Expr, name: str, t: sp.Symbol
) -> sp.Expr:
    """The equation with the unknown replaced by an expansion in unevaluated functions."""
    replace = {
        problem.jets[(name, k)]: (sp.Derivative(expansion, (t, k)) if k else expansion)
        for k in range(problem.orders[name] + 1)
    }
    return problem.residuals[0].xreplace(replace)


def _periodic(
    data: OdeSolveInput,
    problem: OdeProblem,
    eps: sp.Symbol,
    run: _Run,
    meaning: Any,
    limits: Limits,
) -> OdeSolution:
    """Poincare-Lindstedt for y'' = a(y), through the verified Lagrangian implementation: the equation
    is the Euler-Lagrange equation of L = y'^2 / 2 + integral of a."""
    from ..perturbation import perturbation_solve
    from .reading import _latex_name
    from .symbolic import SolverTimeout, deadline

    def refuse(status: Status, message: str, reason: str) -> OdeSolution:
        return run.refuse(status, message, reason, interpretation=meaning)

    solved, why = solve_highest(problem)
    name = problem.unknowns[0]
    if solved is None or problem.orders[name] != 2:
        return refuse(
            Status.UNSUPPORTED,
            "Poincare-Lindstedt needs a single second-order equation solved for y''"
            + (f" ({why})" if why else "")
            + ".",
            "not_second_order",
        )
    y, v = problem.jets[(name, 0)], problem.jets[(name, 1)]
    acceleration = solved.rhs[0]
    if acceleration.has(v) or acceleration.has(problem.variable):
        return refuse(
            Status.UNSUPPORTED,
            "The acceleration depends on the velocity or explicitly on the variable (damping, a "
            "gyroscopic term or forcing): the periodic expansion is implemented for y'' = a(y) only.",
            "not_position_only",
        )
    try:
        with deadline(0.3 * limits.timeout_seconds):
            potential = sp.integrate(acceleration, y)
    except (SolverTimeout, NotImplementedError, ValueError, TypeError):
        potential = sp.Integral(acceleration, y)
    if potential.has(sp.Integral):
        return refuse(
            Status.UNSUPPORTED,
            "The potential (the integral of the acceleration) could not be found in closed form.",
            "no_potential",
        )
    lagrangian = v**2 / 2 + potential
    symbol_names = {y: _latex_name(name), v: f"\\dot{{{_latex_name(name)}}}"}
    latex = to_latex(lagrangian, symbol_names=symbol_names)
    assumptions = {k: list(a) for k, a in (data.assumptions or {}).items() if k != name}
    report = perturbation_solve(
        latex,
        [name],
        problem.variable.name,
        assumptions,
        eps.name,
        data.equilibrium or "0",
        min(data.order, 6),
        None,
        limits,
    )
    run.mark("solve")
    if report.status is not Status.OK:
        return refuse(
            report.status, report.message or "No periodic expansion was found.", "refused"
        )
    checks = [
        ExpansionCheck(
            name="residual_order",
            passed=bool(report.verification and report.verification.residual_passed),
            note="the truncated solution in the unexpanded acceleration; falls as eps^(order + 1)",
        ),
        ExpansionCheck(
            name="initial_conditions",
            passed=bool(report.verification and report.verification.initial_conditions_passed),
        ),
        ExpansionCheck(
            name="boundedness",
            passed=bool(report.verification and report.verification.bounded),
            note="every order is a finite trigonometric polynomial: no secular term",
        ),
    ]
    sets = report.verification.parameter_sets_checked if report.verification else 0
    verification = OdeVerification(
        points_checked=sets,
        points_agreed=sets if report.verification and report.verification.passed else 0,
        passed=bool(report.verification and report.verification.passed),
        note="verified by the Poincare-Lindstedt implementation (residual order, initial values, boundedness)",
    )
    if not verification.passed:
        return refuse(
            Status.NOT_VERIFIED, "The periodic expansion did not pass its checks.", "not_verified"
        )
    # The report already names its parts ("u(t) = ...", "\\omega = ..."); u is the displacement from the
    # equilibrium.
    equations = [e for e in (report.solution_latex, report.frequency_latex) if e]
    item = OdeSolutionItem(
        kind="perturbation",
        equations_latex=equations,
        family="particular",
        verification=verification,
        expansion=OdeExpansion(
            kind="poincare_lindstedt",
            order=report.order,
            small_latex=report.small_latex,
            checks=checks,
            validity=(
                "Periodic motion near the equilibrium, for small values of the parameter; the frequency "
                "is corrected order by order and the series is asymptotic."
            ),
        ),
    )
    run.diagnostics.verified = 1
    run.diagnostics.dropped = 0
    run.diagnostics.solution_kind = "perturbation"
    return run.finish(
        OdeSolution(
            status=Status.OK,
            method="perturbation",
            interpretation=meaning,
            solutions=[item],
            verification=weakest([item]),
            notes=[
                item.expansion.validity if item.expansion else "",
                "u is the displacement from the equilibrium; A is the amplitude (the initial displacement).",
                *report.notes,
            ],
        )
    )
