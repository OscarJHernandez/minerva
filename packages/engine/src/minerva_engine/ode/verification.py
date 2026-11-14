"""Independent checks of candidate solutions of differential equations.

Nothing here uses the route that produced a candidate. A candidate is put into the equation *as
written* after differentiating it **numerically** with mpmath (never with SymPy's derivative), at
random points consistent with the declared assumptions, and the residual is measured against the
size of the terms that cancel. A candidate is accepted only when enough points were valid and every
one of them agreed. The symbolic zero test (`symbolic_zero`) is reported separately and never decides.

Numerical agreement is evidence, not proof; the tolerance is `TOLERANCE` and tests include a
candidate that is wrong by one part in a million.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import mpmath as mp
import numpy as np
import sympy as sp

from ..verify import _sample
from .models import OdeVerification
from .reading import OdeProblem

TOLERANCE = 1e-9
DIGITS = 60
MIN_POINTS = 4
NARROW_POINTS = 6
WIDE_POINTS = 4
ATTEMPTS = 60
MAX_IMPLICIT_ORDER = 2
REFERENCE_OFFSET = mp.mpf("0.05")
_REF = sp.Symbol("minerva_ref")  # lower limit of an indefinite integral, set just below each sample
_CONSTANT = re.compile(r"C\d+")


def constants_of(*expressions: sp.Expr) -> list[sp.Symbol]:
    """The integration constants SymPy names `C1`, `C2`, ... in the given expressions."""
    found = {s for e in expressions for s in e.free_symbols if _CONSTANT.fullmatch(s.name)}
    return sorted(found, key=lambda s: int(s.name[1:]))


class Unevaluable(Exception):
    """An expression that cannot be turned into a numerical function (so it cannot be checked)."""


def _lambdify(args: Sequence[sp.Symbol], expr: sp.Expr) -> Callable[..., Any]:
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise Unevaluable
    try:
        return sp.lambdify(args, expr, modules="mpmath")  # type: ignore[no-any-return]
    except (TypeError, ValueError, NameError, AttributeError, KeyError, NotImplementedError):
        raise Unevaluable from None


def lambdify_numeric(
    args: Sequence[sp.Symbol], expr: Any, module: Literal["numpy", "math"]
) -> Callable[..., Any]:
    """A double-precision function for the numerical solvers; `Unevaluable` if SymPy cannot make one."""
    probe = expr if isinstance(expr, sp.Basic) else sp.Tuple(*expr)
    if probe.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise Unevaluable
    try:
        return sp.lambdify(args, expr, modules=module)  # type: ignore[no-any-return]
    except (TypeError, ValueError, NameError, AttributeError, KeyError, NotImplementedError):
        raise Unevaluable from None


@dataclass
class Residual:
    checked: int
    agreed: int
    skipped: int
    worst: float  # largest relative residual among the checked points

    @property
    def passed(self) -> bool:
        return self.checked >= MIN_POINTS and self.agreed == self.checked

    def verification(self, **extra: Any) -> OdeVerification:
        return OdeVerification(
            points_checked=self.checked,
            points_agreed=self.agreed,
            points_skipped=self.skipped,
            max_residual=self.worst if self.checked else None,
            passed=self.passed,
            **extra,
        )


def _to_mp(value: sp.Expr) -> mp.mpf | mp.mpc:
    number = complex(sp.N(value, 30))
    return mp.mpf(number.real) if number.imag == 0 else mp.mpc(number)


def _finite(value: Any) -> bool:
    try:
        return bool(mp.isfinite(value))
    except TypeError:
        return False


def _definite(expr: sp.Expr, variable: sp.Symbol) -> sp.Expr:
    """Indefinite integrals stand for *some* antiderivative: use the one starting just below the sample
    point (`_REF`), so that they can be evaluated and the integration never crosses a distant
    singularity. A different starting point changes the result by a constant, which the family's
    constants absorb; the derivative being checked does not depend on it."""

    def fix(node: sp.Integral) -> sp.Expr:
        if node.limits and all(len(limit) == 1 for limit in node.limits):
            result: sp.Expr = node.function
            for (symbol,) in node.limits:
                dummy = sp.Dummy("s")
                result = sp.Integral(result.xreplace({symbol: dummy}), (dummy, _REF, symbol))
            return result
        return node

    return expr.replace(lambda e: isinstance(e, sp.Integral), fix)


MAX_EXPAND_OPS = 400


def _terms(f: sp.Expr) -> sp.Tuple:
    """The terms that cancel in `f`: its expanded summands (their sizes are the scale a residual is
    judged against). An equation too large to expand is judged against its unexpanded summands."""
    if sp.count_ops(f) <= MAX_EXPAND_OPS:
        expanded = sp.expand(f)
        if sp.count_ops(expanded) <= 2 * MAX_EXPAND_OPS:
            return sp.Tuple(*sp.Add.make_args(expanded))
    return sp.Tuple(*sp.Add.make_args(f))


class _Evaluator:
    """Lambdified residuals and the sizes of their terms, for one problem."""

    def __init__(self, problem: OdeProblem) -> None:
        self.problem = problem
        self.jets = [
            problem.jets[(u, k)] for u in problem.unknowns for k in range(problem.orders[u] + 1)
        ]
        args = [problem.variable, *self.jets, *problem.parameters]
        self.residual = [_lambdify(args, f) for f in problem.residuals]
        self.terms = [_lambdify(args, _terms(f)) for f in problem.residuals]

    def relative(self, x: Any, jet_values: Sequence[Any], parameter_values: Sequence[Any]) -> float:
        worst = 0.0
        for residual, terms in zip(self.residual, self.terms, strict=True):
            value = abs(residual(x, *jet_values, *parameter_values))
            scale = sum(abs(term) for term in terms(x, *jet_values, *parameter_values))
            if not _finite(value) or not _finite(scale):
                raise ArithmeticError("not finite")
            worst = max(worst, float(value / scale) if scale > mp.mpf(10) ** -40 else float(value))
        return worst


def _points(rng: random.Random, narrow: bool) -> mp.mpf:
    if narrow:
        return mp.mpf(rng.uniform(-3.0, 3.0))
    return mp.mpf(rng.choice([-1, 1]) * rng.uniform(3.0, 12.0))


def _constant_values(rng: random.Random, constants: Sequence[sp.Symbol]) -> list[mp.mpf]:
    return [mp.mpf(rng.choice([-1, 1]) * rng.uniform(0.3, 2.5)) for _ in constants]


def _parameter_values(problem: OdeProblem, rng: random.Random, wide: bool) -> list[mp.mpf | mp.mpc]:
    spread = max(1, len(problem.parameters))
    return [_to_mp(_sample(p, rng, wide=wide, spread=spread)) for p in problem.parameters]


def residual_check(
    problem: OdeProblem,
    outputs: Sequence[sp.Expr],
    constants: Sequence[sp.Symbol] = (),
    *,
    seed: int = 20260301,
    unevaluated: bool = False,
) -> Residual:
    """Put `outputs` (one expression per unknown, in the order of `problem.unknowns`) into the
    equations. The derivatives of the outputs are numerical (`mpmath.diff`)."""
    checked = agreed = skipped = 0
    worst = 0.0
    if len(outputs) != len(problem.unknowns):
        return Residual(0, 0, 0, 0.0)
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    mp.mp.dps = DIGITS
    candidate = [_definite(o, problem.variable) if unevaluated else o for o in outputs]
    args = [problem.variable, *problem.parameters, *constants] + ([_REF] if unevaluated else [])
    try:
        functions = [_lambdify(args, c) for c in candidate]
        evaluator = _Evaluator(problem)
    except Unevaluable:
        return Residual(0, 0, 0, 0.0)
    wanted = (NARROW_POINTS, WIDE_POINTS) if not unevaluated else (5, 0)
    for narrow, limit in ((True, wanted[0]), (False, wanted[1])):
        got = 0
        for _ in range(ATTEMPTS):
            if got >= limit:
                break
            x0 = _points(rng, narrow)
            parameters = _parameter_values(problem, rng, wide=not narrow)
            consts = _constant_values(rng, constants)
            try:
                jet_values: list[Any] = []
                for fn, unknown in zip(functions, problem.unknowns, strict=True):
                    bound = _bind(
                        fn, parameters, consts, x0 - REFERENCE_OFFSET if unevaluated else None
                    )
                    for order in range(problem.orders[unknown] + 1):
                        jet_values.append(bound(x0) if order == 0 else mp.diff(bound, x0, order))
                if not all(_finite(v) for v in jet_values):
                    raise ArithmeticError("not finite")
                relative = evaluator.relative(x0, jet_values, parameters)
            except (ArithmeticError, ValueError, TypeError, ZeroDivisionError, OverflowError):
                skipped += 1
                continue
            checked += 1
            got += 1
            worst = max(worst, relative)
            if relative <= TOLERANCE:
                agreed += 1
    return Residual(checked, agreed, skipped, worst)


def _bind(
    fn: Callable[..., Any],
    parameters: Sequence[Any],
    consts: Sequence[Any],
    reference: Any = None,
) -> Callable[[Any], Any]:
    extra = () if reference is None else (reference,)

    def bound(x: Any) -> Any:
        return fn(x, *parameters, *consts, *extra)

    return bound


# --- are the constants independent? --------------------------------------------------------


def independent_constants(
    problem: OdeProblem,
    outputs: Sequence[sp.Expr],
    constants: Sequence[sp.Symbol],
    *,
    seed: int = 20260302,
    unevaluated: bool = False,
) -> int | None:
    """Numerical rank of d(state)/d(constants): the number of constants that really are free. A family
    is called general only when this equals the order of the system."""
    if not constants:
        return 0
    rng = random.Random(seed)  # noqa: S311
    mp.mp.dps = DIGITS
    args = [problem.variable, *problem.parameters, *constants] + ([_REF] if unevaluated else [])
    best = 0
    try:
        derivative = [
            [
                _lambdify(
                    args,
                    _definite(sp.diff(o, c), problem.variable) if unevaluated else sp.diff(o, c),
                )
                for c in constants
            ]
            for o in outputs
        ]
    except Unevaluable:
        return None
    for _ in range(3):
        x0 = _points(rng, True)
        parameters = _parameter_values(problem, rng, wide=False)
        consts = _constant_values(rng, constants)
        rows: list[list[complex]] = []
        try:
            for per_unknown, unknown in zip(derivative, problem.unknowns, strict=True):
                for order in range(problem.orders[unknown]):
                    row = []
                    for fn in per_unknown:
                        bound = _bind(
                            fn, parameters, consts, x0 - REFERENCE_OFFSET if unevaluated else None
                        )
                        value = bound(x0) if order == 0 else mp.diff(bound, x0, order)
                        row.append(complex(value))
                    rows.append(row)
        except (ArithmeticError, ValueError, TypeError, ZeroDivisionError, OverflowError):
            continue
        matrix = np.array(rows, dtype=complex)
        if not np.all(np.isfinite(matrix)) or matrix.size == 0:
            continue
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular.size and singular[0] > 0:
            best = max(best, int(np.sum(singular > 1e-9 * singular[0])))
    return best


# --- implicit solutions G(x, y; C) = 0 ---------------------------------------------------


def _find_root(
    g: Callable[..., Any],
    x: Any,
    start: Any,
    parameters: Sequence[Any],
    consts: Sequence[Any],
) -> Any:
    """A root `y` of g(x, y, ...) near `start`, to nearly full working precision, else ArithmeticError."""
    found = mp.findroot(
        lambda value: g(x, value, *parameters, *consts),
        start,
        tol=mp.mpf(10) ** -(DIGITS - 8),
        maxsteps=60,
    )
    if abs(g(x, found, *parameters, *consts)) > mp.mpf(10) ** -(DIGITS - 15):
        raise ArithmeticError("not a root")
    return found


def implicit_check(
    problem: OdeProblem,
    relation: sp.Expr,
    constants: Sequence[sp.Symbol] = (),
    *,
    seed: int = 20260303,
) -> Residual:
    """A scalar implicit solution `relation = 0` in `x` and the unknown. At each sample, the real or
    complex roots `y` of the relation are found numerically (several starting values), each branch is
    followed to its neighbours by root finding, and `y'`, `y''` come from finite differences of those
    values: the slope is not obtained by implicit differentiation. Every branch found must satisfy
    the equation."""
    unknown = problem.unknowns[0]
    order = problem.orders[unknown]
    if len(problem.unknowns) != 1 or order > MAX_IMPLICIT_ORDER:
        return Residual(0, 0, 0, 0.0)
    y = problem.jets[(unknown, 0)]
    rng = random.Random(seed)  # noqa: S311
    mp.mp.dps = DIGITS
    args = [problem.variable, y, *problem.parameters, *constants]
    try:
        g = _lambdify(args, relation)
        evaluator = _Evaluator(problem)
    except Unevaluable:
        return Residual(0, 0, 0, 0.0)
    checked = agreed = skipped = 0
    worst = 0.0
    starts = [mp.mpf(v) for v in (-2, -0.7, 0.4, 1.3, 2.6)] + [mp.mpc(0.5, 0.9), mp.mpc(-0.8, -1.1)]
    for narrow, limit in ((True, NARROW_POINTS), (False, WIDE_POINTS)):
        got = 0
        for _ in range(ATTEMPTS):
            if got >= limit:
                break
            x0 = _points(rng, narrow)
            parameters = _parameter_values(problem, rng, wide=not narrow)
            consts = _constant_values(rng, constants)

            roots: list[Any] = []
            for start in starts:
                try:
                    candidate = _find_root(g, x0, start, parameters, consts)
                except (ArithmeticError, ValueError, ZeroDivisionError, OverflowError):
                    continue
                if _finite(candidate) and all(
                    abs(candidate - r) > mp.mpf(10) ** -20 for r in roots
                ):
                    roots.append(candidate)
                if len(roots) == 2:
                    break
            if not roots:
                skipped += 1
                continue
            for branch in roots:
                try:
                    h = mp.mpf(10) ** -12 if order <= 1 else mp.mpf(10) ** -9
                    plus = _find_root(g, x0 + h, branch, parameters, consts)
                    minus = _find_root(g, x0 - h, branch, parameters, consts)
                    values: list[Any] = [branch]
                    if order >= 1:
                        values.append((plus - minus) / (2 * h))
                    if order >= 2:
                        values.append((plus - 2 * branch + minus) / h**2)
                    if abs(plus - branch) > mp.mpf(10) ** -6 or not all(_finite(v) for v in values):
                        raise ArithmeticError("branches collide")
                    relative = evaluator.relative(x0, values, parameters)
                except (ArithmeticError, ValueError, ZeroDivisionError, OverflowError, TypeError):
                    skipped += 1
                    continue
                checked += 1
                got += 1
                worst = max(worst, relative)
                if relative <= TOLERANCE:
                    agreed += 1
    return Residual(checked, agreed, skipped, worst)


# --- the symbolic leg --------------------------------------------------------------------


def symbolic_residual_is_zero(
    problem: OdeProblem, outputs: Sequence[sp.Expr], max_ops: int = 400
) -> bool | None:
    """True if the equations simplify to zero exactly with `outputs` substituted, None if that was not
    decided (never False: failing to simplify is not a refutation)."""
    if any(sp.count_ops(o) > max_ops for o in outputs):
        return None
    replace: dict[sp.Symbol, sp.Expr] = {}
    for unknown, out in zip(problem.unknowns, outputs, strict=True):
        for order in range(problem.orders[unknown] + 1):
            replace[problem.jets[(unknown, order)]] = (
                sp.diff(out, problem.variable, order) if order else out
            )
    try:
        for f in problem.residuals:
            from .analyze import is_zero

            if not is_zero(f.xreplace(replace)):
                return None
    except (TypeError, ValueError, RecursionError, NotImplementedError):
        return None
    return True


# --- conditions ---------------------------------------------------------------------------


def condition_residual(
    problem: OdeProblem,
    outputs: Sequence[sp.Expr],
    constants: Sequence[sp.Symbol],
    unknown: str,
    order: int,
    at: sp.Expr,
    value: sp.Expr,
    *,
    seed: int = 20260304,
) -> float | None:
    """|d^order u(at) - value| relative to max(1, |value|), the derivative taken numerically; the
    worst over several parameter samples, None if it could not be evaluated."""
    rng = random.Random(seed)  # noqa: S311
    mp.mp.dps = DIGITS
    index = problem.unknowns.index(unknown)
    out = outputs[index]
    args = [problem.variable, *problem.parameters, *constants]
    try:
        fn = _lambdify(args, out)
        at_fn = _lambdify(problem.parameters, at)
        value_fn = _lambdify(problem.parameters, value)
    except Unevaluable:
        return None
    worst = 0.0
    done = 0
    for _ in range(ATTEMPTS):
        if done >= 3:
            break
        parameters = _parameter_values(problem, rng, wide=False)
        consts = _constant_values(rng, constants)
        try:
            point = at_fn(*parameters)
            target = value_fn(*parameters)
            bound = _bind(fn, parameters, consts)
            got = bound(point) if order == 0 else mp.diff(bound, point, order)
            difference = abs(got - target) / max(mp.mpf(1), abs(target))
            if not _finite(difference):
                raise ArithmeticError("not finite")
        except (ArithmeticError, ValueError, TypeError, ZeroDivisionError, OverflowError):
            continue
        done += 1
        worst = max(worst, float(difference))
    return worst if done else None
