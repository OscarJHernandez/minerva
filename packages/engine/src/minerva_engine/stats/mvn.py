"""The bivariate Normal distribution: density, moments, marginal, conditional and affine image (#229).

A model is a mean vector and a covariance matrix the user gives, in plain-text syntax. Before anything is
computed the covariance must be 2 x 2, exactly symmetric (never symmetrised) and positive definite:

* provably not positive definite (a numeric matrix, or a sign the declared assumptions decide) is
  `not_positive_definite`; a determinant that is identically zero is `singular_covariance`
  (`unsupported`: no pseudoinverse and no degenerate density is ever produced);
* undecided (symbolic entries the assumptions do not decide) is an unestablished condition, which makes
  every result conditional ("holds given $\\Sigma_{11} > 0$, $\\det\\Sigma > 0$"), never silently true.

The results are exact algebra (the 2 x 2 inverse by its adjugate, the Schur complement, `A Sigma A^T`)
and are shown only if an independent numerical leg agrees at sampled admissible parameter values: SciPy
quadrature (with its error estimate) of the printed result read back, against a joint density built in
floating point from the numeric mean and covariance (NumPy's inverse and determinant, no SymPy algebra).
A quadrature whose error estimate is not small counts as no agreement. Numerical agreement is evidence,
not a proof: results are worded "numerically checked".

Only two dimensions are supported. The independent check is a nested quadrature and in three dimensions
it took minutes per integral (measured), so a third component is `dimension_unsupported`, as is one.
"""

from __future__ import annotations

import math
import random
import re
import time
import warnings
from collections.abc import Callable

import numpy as np
import sympy as sp
from scipy.integrate import IntegrationWarning, dblquad, quad

from ..distributions.models import Condition
from ..errors import MinervaMathError
from ..latexprint import to_latex, to_text
from ..models import ExpressionInput, Limits, Status
from ..ops import parse_expression
from .mvn_models import MvnAction, MvnCheck, MvnInput, MvnReason, MvnResult
from .numeric import sample_symbol

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,31}")
# a box of 12 standard deviations around the mean leaves a tail mass below 1e-32
_BOX = 12.0
REL_TOL = 1e-9
MIN_POINTS = 3
MAX_POINTS = 4
ATTEMPTS = 80
CHECK_SHARE = 0.6  # of the time limit, for the numerical checks


class MvnRefusal(MinervaMathError):
    def __init__(self, status: Status, reason: MvnReason, message: str) -> None:
        super().__init__(message)
        self.status, self.reason = status, reason


Point = dict[sp.Symbol, sp.Rational]


# --------------------------------------------------------------------------------------- reading


def _parse(text: str, data: MvnInput, limits: Limits, what: str) -> sp.Expr:
    try:
        expr = parse_expression(
            ExpressionInput(input=text, format="text", assumptions=data.assumptions), limits
        )
    except MinervaMathError as exc:
        raise MvnRefusal(Status.PARSE_ERROR, "parse_error", f"{what}: {exc}") from exc
    if expr.is_real is False:
        raise MvnRefusal(Status.PARSE_ERROR, "parse_error", f"{what} must be real")
    return expr


class _Model:
    def __init__(self, data: MvnInput, limits: Limits) -> None:
        d = len(data.mean)
        if d != 2:
            hint = (
                "a single component is the Normal family of statistical_model"
                if d == 1
                else "the independent check of a result is a nested quadrature, which is only "
                "fast enough in two dimensions"
            )
            raise MvnRefusal(
                Status.UNSUPPORTED,
                "dimension_unsupported",
                f"only a Normal distribution of two components is supported ({hint})",
            )
        if len(data.covariance) != 2 or any(len(row) != 2 for row in data.covariance):
            raise MvnRefusal(
                Status.PARSE_ERROR, "parse_error", "the covariance must be a 2 x 2 matrix"
            )
        self.data, self.limits = data, limits
        self.mu = [_parse(t, data, limits, f"mean entry {k + 1}") for k, t in enumerate(data.mean)]
        self.S = sp.Matrix(
            2,
            2,
            lambda i, j: _parse(
                data.covariance[i][j], data, limits, f"covariance entry ({i + 1}, {j + 1})"
            ),
        )
        if sp.simplify(self.S[0, 1] - self.S[1, 0]) != 0:
            raise MvnRefusal(
                Status.PARSE_ERROR,
                "not_symmetric",
                "a covariance matrix is symmetric: entries (1, 2) and (2, 1) differ (Minerva does not "
                "symmetrise it for you)",
            )
        self.conditions: list[Condition] = []
        self.det = sp.factor(sp.expand(self.S[0, 0] * self.S[1, 1] - self.S[0, 1] ** 2))
        self._positive_definite()
        self._real_entries()
        self.vars = self._variables()

    def _positive_definite(self) -> None:
        if sp.simplify(self.det) == 0:
            raise MvnRefusal(
                Status.UNSUPPORTED,
                "singular_covariance",
                "the covariance matrix is singular (its determinant is zero): the distribution has no "
                "density, and Minerva does not use a pseudoinverse",
            )
        for label, value in (("\\Sigma_{11}", self.S[0, 0]), ("\\det\\Sigma", self.det)):
            latex = f"{label} = {to_latex(value)} > 0"
            sign = sp.simplify(value).is_positive
            if sign is False:
                raise MvnRefusal(
                    Status.CONDITIONS_NOT_SATISFIED,
                    "not_positive_definite",
                    f"the covariance matrix is not positive definite: {latex} fails",
                )
            if sign is None:
                self.conditions.append(
                    Condition(
                        latex=latex,
                        established=False,
                        how="positive definiteness (leading principal minors) is not decided by the "
                        "declared assumptions",
                        parameter="covariance",
                        role="given",
                    )
                )

    def _real_entries(self) -> None:
        entries = [*self.mu, *self.S]
        if any(e.is_real is not True for e in entries):
            self.conditions.append(
                Condition(
                    latex="\\text{the entries of the mean and the covariance are real}",
                    established=False,
                    how="a symbol not declared real is complex",
                    parameter="covariance",
                    role="given",
                )
            )

    def _variables(self) -> list[sp.Symbol] | None:
        names = self.data.variables
        if names is None:
            return None
        if len(names) != 2 or len(set(names)) != 2 or not all(_NAME.fullmatch(n) for n in names):
            raise MvnRefusal(
                Status.PARSE_ERROR, "parse_error", "name the two components with two plain names"
            )
        used = {s.name for e in [*self.mu, *self.S] for s in e.free_symbols}
        if used & set(names):
            raise MvnRefusal(
                Status.PARSE_ERROR,
                "parse_error",
                "a component name is also a symbol of the mean or the covariance",
            )
        return [sp.Symbol(n, real=True) for n in names]

    def need_variables(self) -> list[sp.Symbol]:
        if self.vars is None:
            raise MvnRefusal(
                Status.PARSE_ERROR,
                "variables_missing",
                "name the two components (variables, e.g. x and y) for a density, a marginal or a "
                "conditional",
            )
        return self.vars

    def index_of(self, name: str, what: str) -> int:
        variables = self.need_variables()
        for k, v in enumerate(variables):
            if v.name == name:
                return k
        raise MvnRefusal(
            Status.PARSE_ERROR,
            "index_invalid",
            f"{what} must be one of the components ({', '.join(v.name for v in variables)})",
        )


# --------------------------------------------------------------------------------- exact algebra


def _normal_1d(t: sp.Symbol, mean: sp.Expr, var: sp.Expr) -> sp.Expr:
    return sp.exp(-((t - mean) ** 2) / (2 * var)) / sp.sqrt(2 * sp.pi * var)


def _density_2d(m: _Model, x: sp.Symbol, y: sp.Symbol) -> sp.Expr:
    dx, dy = x - m.mu[0], y - m.mu[1]
    s = m.S
    quadratic = (s[1, 1] * dx**2 - 2 * s[0, 1] * dx * dy + s[0, 0] * dy**2) / m.det
    return sp.exp(-quadratic / 2) / (2 * sp.pi * sp.sqrt(m.det))


def _correlation(m: _Model) -> sp.Expr:
    return sp.simplify(m.S[0, 1] / sp.sqrt(m.S[0, 0] * m.S[1, 1]))


def _tidy(e: sp.Expr) -> sp.Expr:
    return sp.factor(sp.cancel(sp.expand(e)))


# ------------------------------------------------------------------------------ numerical oracle


def _value(expr: sp.Expr, point: Point) -> float:
    v = complex(sp.N(expr.xreplace(point), 30))
    if abs(v.imag) > 1e-12 * max(1.0, abs(v.real)):
        raise ValueError("not real at the point")
    return v.real


def _numeric_model(m: _Model, point: Point) -> tuple[np.ndarray, np.ndarray]:
    mu = np.array([_value(e, point) for e in m.mu])
    s = np.array([[_value(m.S[i, j], point) for j in range(2)] for i in range(2)])
    return mu, s


def _joint(mu: np.ndarray, s: np.ndarray) -> Callable[[float, float], float]:
    inv = np.linalg.inv(s)
    c = 1.0 / (2.0 * math.pi * math.sqrt(np.linalg.det(s)))

    def f(x: float, y: float) -> float:
        d0, d1 = x - mu[0], y - mu[1]
        q = inv[0, 0] * d0 * d0 + 2.0 * inv[0, 1] * d0 * d1 + inv[1, 1] * d1 * d1
        return float(c * math.exp(-0.5 * q))

    return f


def _box(mu: np.ndarray, s: np.ndarray) -> tuple[float, float, float, float]:
    sx, sy = math.sqrt(s[0, 0]), math.sqrt(s[1, 1])
    return (mu[0] - _BOX * sx, mu[0] + _BOX * sx, mu[1] - _BOX * sy, mu[1] + _BOX * sy)


def _double(
    g: Callable[[float, float], float], box: tuple[float, float, float, float]
) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        v, err = dblquad(
            lambda y, x: g(x, y), box[0], box[1], box[2], box[3], epsabs=1e-13, epsrel=1e-12
        )
    return float(v), float(err)


def _single(g: Callable[[float], float], lo: float, hi: float) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        v, err = quad(g, lo, hi, epsabs=1e-14, epsrel=1e-12, limit=200)
    return float(v), float(err)


def _agrees(found: tuple[float, float], expected: float, scale: float) -> bool:
    value, err = found
    tol = REL_TOL * max(scale, abs(expected), 1e-300)
    return math.isfinite(value) and err < tol / 10 and abs(value - expected) < tol


def _read_back(expr: sp.Expr, m: _Model) -> sp.Expr:
    """The printed result, parsed again: the check is of what is shown, not of the tree that made it."""
    assumptions = dict(m.data.assumptions)
    for v in m.vars or []:
        assumptions[v.name] = ["real"]
    try:
        return parse_expression(
            ExpressionInput(input=to_text(expr), format="text", assumptions=assumptions), m.limits
        )
    except MinervaMathError as exc:
        raise MvnRefusal(
            Status.NOT_VERIFIED,
            "verification_failed",
            "the printed result could not be read back for its check; nothing is shown",
        ) from exc


def _lambda(expr: sp.Expr, args: list[sp.Symbol], point: Point) -> Callable[..., float]:
    fixed = expr.xreplace(point)
    fn = sp.lambdify(args, fixed, modules=["math"])
    return lambda *a: float(fn(*a))


# ------------------------------------------------------------------------------------- sampling


def _rng() -> random.Random:
    return random.Random(229)  # noqa: S311 - deterministic sampling, not cryptographic


def _symbols(exprs: list[sp.Expr]) -> list[sp.Symbol]:
    return sorted({s for e in exprs for s in e.free_symbols}, key=lambda s: s.name)


def _has_symbols(m: _Model, exprs: list[sp.Expr]) -> bool:
    return bool(_symbols([*m.mu, *m.S, *exprs]))


def _admissible(m: _Model, point: Point, extra: list[sp.Expr]) -> bool:
    s = m.S.xreplace(point)
    if not all(e.is_real for e in [*s, *(e.xreplace(point) for e in m.mu)]):
        return False
    if not (s[0, 0].is_positive and (s[0, 0] * s[1, 1] - s[0, 1] ** 2).is_positive):
        return False
    return all(e.xreplace(point).is_nonzero for e in extra)


def _points(
    m: _Model, exprs: list[sp.Expr], nonzero: list[sp.Expr], rng: random.Random
) -> list[Point]:
    symbols = _symbols([*m.mu, *m.S, *exprs])
    out: list[Point] = []
    for _ in range(ATTEMPTS):
        if len(out) >= MAX_POINTS:
            break
        point = {s: sample_symbol(s, rng) for s in symbols}
        if _admissible(m, point, nonzero) and point not in out:
            out.append(point)
        if not symbols:
            break
    return out


def _run_checks(
    checks: list[MvnCheck],
    points: list[Point],
    one: Callable[[Point], list[bool]],
    deadline: float,
    symbolic: bool = True,
) -> None:
    """`one` returns one verdict per check at a point; a point that cannot be evaluated is skipped.

    A model with free symbols needs `MIN_POINTS` sampled values; a fully numeric one has one point.
    """
    need = MIN_POINTS if symbolic else 1
    for point in points:
        if time.monotonic() > deadline:
            break
        try:
            verdicts = one(point)
        except (
            ValueError,
            ZeroDivisionError,
            OverflowError,
            TypeError,
            NameError,
            AttributeError,
            np.linalg.LinAlgError,
        ):
            # a point that cannot be evaluated (a function the float route lacks): skipped, never agreed
            continue
        for check, ok in zip(checks, verdicts, strict=True):
            check.points_checked += 1
            check.points_agreed += ok
    for check in checks:
        check.passed = check.points_checked >= need and check.points_agreed == check.points_checked
        if check.points_checked < need:
            check.note = f"only {check.points_checked} sampled point(s) could be checked"


# --------------------------------------------------------------------------------------- actions


def _texts(exprs: list[sp.Expr]) -> tuple[list[str], list[str]]:
    return [to_text(e) for e in exprs], [to_latex(e) for e in exprs]


def _grid(matrix: sp.Matrix) -> tuple[list[list[str]], list[list[str]]]:
    rows = [[matrix[i, j] for j in range(matrix.cols)] for i in range(matrix.rows)]
    return [[to_text(e) for e in r] for r in rows], [[to_latex(e) for e in r] for r in rows]


def _density(m: _Model, result: MvnResult, deadline: float) -> None:
    x, y = m.need_variables()
    f = _density_2d(m, x, y)
    shown = _read_back(f, m)
    result.density_text, result.density_latex = to_text(f), to_latex(f)
    norm = MvnCheck(
        kind="normalisation", method="SciPy quadrature of the printed density over the plane"
    )
    mom = MvnCheck(
        kind="moments",
        method="SciPy quadrature of the printed density: means and covariances equal the given ones",
    )

    def one(point: Point) -> list[bool]:
        mu, s = _numeric_model(m, point)
        g = _lambda(shown, [x, y], point)
        box = _box(mu, s)
        sd = math.sqrt(max(s[0, 0], s[1, 1]))
        ok_norm = _agrees(_double(g, box), 1.0, 1.0)
        ok = [
            _agrees(_double(lambda a, b: a * g(a, b), box), mu[0], abs(mu[0]) + sd),
            _agrees(_double(lambda a, b: b * g(a, b), box), mu[1], abs(mu[1]) + sd),
            _agrees(_double(lambda a, b: (a - mu[0]) ** 2 * g(a, b), box), s[0, 0], s[0, 0]),
            _agrees(
                _double(lambda a, b: (a - mu[0]) * (b - mu[1]) * g(a, b), box),
                s[0, 1],
                math.sqrt(s[0, 0] * s[1, 1]),
            ),
            _agrees(_double(lambda a, b: (b - mu[1]) ** 2 * g(a, b), box), s[1, 1], s[1, 1]),
        ]
        return [ok_norm, all(ok)]

    points = _points(m, [], [], _rng())
    _run_checks([norm, mom], points, one, deadline, _has_symbols(m, []))
    result.checks = [norm, mom]


def _moments(m: _Model, result: MvnResult, deadline: float) -> None:
    rho = _correlation(m)
    shown = _read_back(rho, m)
    result.correlation_text, result.correlation_latex = to_text(rho), to_latex(rho)
    mom = MvnCheck(
        kind="moments",
        method="SciPy quadrature of a density built in floating point from the numeric mean and "
        "covariance: its means, covariances and correlation",
    )

    def one(point: Point) -> list[bool]:
        mu, s = _numeric_model(m, point)
        f = _joint(mu, s)
        box = _box(mu, s)
        sd = math.sqrt(max(s[0, 0], s[1, 1]))
        cov = _double(lambda a, b: (a - mu[0]) * (b - mu[1]) * f(a, b), box)
        v0 = _double(lambda a, b: (a - mu[0]) ** 2 * f(a, b), box)
        v1 = _double(lambda a, b: (b - mu[1]) ** 2 * f(a, b), box)
        corr = cov[0] / math.sqrt(v0[0] * v1[0])
        ok = [
            _agrees(
                _double(lambda a, b: a * f(a, b), box), _value(m.mu[0], point), abs(mu[0]) + sd
            ),
            _agrees(
                _double(lambda a, b: b * f(a, b), box), _value(m.mu[1], point), abs(mu[1]) + sd
            ),
            _agrees(v0, _value(m.S[0, 0], point), s[0, 0]),
            _agrees(cov, _value(m.S[0, 1], point), math.sqrt(s[0, 0] * s[1, 1])),
            _agrees(v1, _value(m.S[1, 1], point), s[1, 1]),
            abs(corr - _value(shown, point)) <= REL_TOL * max(abs(corr), 1e-4),
        ]
        return [all(ok)]

    points = _points(m, [], [], _rng())
    _run_checks([mom], points, one, deadline, _has_symbols(m, []))
    result.checks = [mom]


def _one_dimensional(
    m: _Model,
    result: MvnResult,
    t: sp.Symbol,
    mean: sp.Expr,
    var: sp.Expr,
) -> sp.Expr:
    h = _normal_1d(t, mean, var)
    result.variables = [t.name]
    result.mean_text, result.mean_latex = _texts([mean])
    result.covariance_text, result.covariance_latex = _grid(sp.Matrix([[var]]))
    result.density_text, result.density_latex = to_text(h), to_latex(h)
    return _read_back(h, m)


def _marginal(m: _Model, result: MvnResult, deadline: float) -> None:
    if m.data.keep is None:
        raise MvnRefusal(
            Status.PARSE_ERROR, "index_invalid", "say which component the marginal keeps"
        )
    i = m.index_of(m.data.keep, "the component kept")
    j = 1 - i
    t = m.need_variables()[i]
    shown = _one_dimensional(m, result, t, m.mu[i], _tidy(m.S[i, i]))
    check = MvnCheck(
        kind="marginal",
        method="the printed marginal density against SciPy quadrature of the joint density over the "
        "other component, at three values, and its normalisation",
    )

    def one(point: Point) -> list[bool]:
        mu, s = _numeric_model(m, point)
        f = _joint(mu, s)
        g = _lambda(shown, [t], point)
        lo, hi = mu[j] - _BOX * math.sqrt(s[j, j]), mu[j] + _BOX * math.sqrt(s[j, j])
        sd = math.sqrt(s[i, i])
        ok = [_agrees(_single(g, mu[i] - _BOX * sd, mu[i] + _BOX * sd), 1.0, 1.0)]
        for k in (-1.3, 0.2, 2.1):
            at = mu[i] + k * sd
            joint = (lambda u, at=at: f(at, u)) if i == 0 else (lambda u, at=at: f(u, at))
            ok.append(_agrees(_single(joint, lo, hi), g(at), g(at)))
        return [all(ok)]

    points = _points(m, [], [], _rng())
    _run_checks([check], points, one, deadline, _has_symbols(m, []))
    result.checks = [check]


def _conditional(m: _Model, result: MvnResult, deadline: float) -> None:
    given = m.data.given or {}
    if len(given) != 1:
        raise MvnRefusal(
            Status.PARSE_ERROR,
            "index_invalid",
            "give the value of exactly one component to condition on",
        )
    ((name, text),) = given.items()
    j = m.index_of(name, "the component conditioned on")
    i = 1 - j
    value = _parse(text, m.data, m.limits, "the value conditioned on")
    if {v.name for v in m.need_variables()} & {sym.name for sym in value.free_symbols}:
        raise MvnRefusal(
            Status.PARSE_ERROR,
            "parse_error",
            "the value conditioned on is a number or a parameter, not a component of the distribution",
        )
    if value.is_real is not True:
        m.conditions.append(
            Condition(
                latex=f"{to_latex(value)} \\in \\mathbb{{R}}",
                established=False,
                how="the value conditioned on is not declared real",
                parameter="given",
                role="given",
            )
        )
    t = m.need_variables()[i]
    s = m.S
    mean = _tidy(m.mu[i] + s[i, j] / s[j, j] * (value - m.mu[j]))
    var = _tidy(s[i, i] - s[i, j] ** 2 / s[j, j])
    shown = _one_dimensional(m, result, t, mean, var)
    check = MvnCheck(
        kind="conditional",
        method="the printed conditional density against the joint density divided by its integral "
        "over the free component (SciPy quadrature), at three values, and its normalisation",
    )

    def one(point: Point) -> list[bool]:
        mu, sn = _numeric_model(m, point)
        f = _joint(mu, sn)
        v = _value(value, point)
        h = _lambda(shown, [t], point)
        sd = math.sqrt(sn[i, i])
        lo, hi = mu[i] - 2 * _BOX * sd, mu[i] + 2 * _BOX * sd

        def joint(u: float) -> float:
            return f(u, v) if i == 0 else f(v, u)

        mass = _single(joint, lo, hi)
        if not _agrees(mass, mass[0], mass[0]):  # the error estimate alone
            return [False]
        centre = _value(mean, point)
        csd = math.sqrt(_value(var, point))
        ok = [_agrees(_single(h, centre - _BOX * csd, centre + _BOX * csd), 1.0, 1.0)]
        for k in (-1.1, 0.3, 1.9):
            at = centre + k * csd
            ok.append(abs(joint(at) / mass[0] - h(at)) <= 1e-8 * h(at))
        return [all(ok)]

    points = _points(m, [value], [], _rng())
    _run_checks([check], points, one, deadline, _has_symbols(m, [value]))
    result.checks = [check]


def _affine(m: _Model, result: MvnResult, deadline: float) -> None:
    data = m.data
    if data.matrix is None or data.offset is None:
        raise MvnRefusal(
            Status.PARSE_ERROR, "parse_error", "an affine image needs the matrix A and the offset b"
        )
    rows = len(data.matrix)
    if rows not in (1, 2) or any(len(r) != 2 for r in data.matrix) or len(data.offset) != rows:
        raise MvnRefusal(
            Status.PARSE_ERROR,
            "parse_error",
            "A must have one or two rows of two entries, and b one entry per row of A",
        )
    a = sp.Matrix(
        rows,
        2,
        lambda r, c: _parse(data.matrix[r][c], data, m.limits, f"A entry ({r + 1}, {c + 1})"),
    )
    b = sp.Matrix(
        [_parse(t, data, m.limits, f"b entry {k + 1}") for k, t in enumerate(data.offset)]
    )
    if rows == 2:
        rank_expr = sp.factor(a.det())
        what = "A is singular (its determinant is zero)"
    else:
        rank_expr = sp.factor(a[0, 0] ** 2 + a[0, 1] ** 2)
        what = "A is the zero row"
    if sp.simplify(rank_expr) == 0:
        raise MvnRefusal(
            Status.UNSUPPORTED,
            "map_singular",
            f"{what}: the image is a degenerate distribution without a density, which is not supported",
        )
    if any(e.is_real is not True for e in [*a, *b]):
        m.conditions.append(
            Condition(
                latex="\\text{the entries of } A \\text{ and } b \\text{ are real}",
                established=False,
                how="a symbol not declared real is complex, and then the covariance of A X + b is "
                "not A Sigma A^T",
                parameter="matrix",
                role="given",
            )
        )
    if sp.simplify(rank_expr).is_nonzero is not True:
        m.conditions.append(
            Condition(
                latex=f"{to_latex(rank_expr)} \\neq 0",
                established=False,
                how="the map A has full row rank only if this holds; the declared assumptions do not "
                "decide it",
                parameter="matrix",
                role="given",
            )
        )
    mean = (a * sp.Matrix(m.mu) + b).applyfunc(_tidy)
    cov = (a * m.S * a.T).applyfunc(_tidy)
    result.mean_text, result.mean_latex = _texts(list(mean))
    result.covariance_text, result.covariance_latex = _grid(cov)
    shown_mean = [_read_back(e, m) for e in mean]
    shown_cov = [[_read_back(cov[r, c], m) for c in range(rows)] for r in range(rows)]
    check = MvnCheck(
        kind="affine",
        method="the mean and covariance of A X + b by SciPy quadrature under the original density "
        "(built in floating point from the numeric mean and covariance)",
    )

    def one(point: Point) -> list[bool]:
        mu, s = _numeric_model(m, point)
        f = _joint(mu, s)
        box = _box(mu, s)
        an = np.array([[_value(a[r, c], point) for c in range(2)] for r in range(rows)])
        bn = np.array([_value(b[r], point) for r in range(rows)])

        def comp(r: int) -> Callable[[float, float], float]:
            return lambda u, w: an[r, 0] * u + an[r, 1] * w + bn[r]

        # each component in units of a rough scale of its own (from the numeric A and Sigma), so the
        # integrator's absolute error floor is relative to that component: an entry of 1e-6 is checked
        # to 1e-15, not to the scale of the largest one
        rough = [
            math.sqrt(sum(an[r, k] ** 2 * s[k, k] for k in range(2))) or 1.0 for r in range(rows)
        ]

        def centred(r: int) -> Callable[[float, float], float]:
            g = comp(r)
            return lambda u, w: (g(u, w) - bn[r]) / rough[r] * f(u, w)

        def product(r: int, c: int, mr: float, mc: float) -> Callable[[float, float], float]:
            g, h = comp(r), comp(c)
            return lambda u, w: (g(u, w) - mr) / rough[r] * (h(u, w) - mc) / rough[c] * f(u, w)

        means: list[tuple[float, float]] = []
        for r in range(rows):
            v, err = _double(centred(r), box)
            means.append((bn[r] + v * rough[r], err * rough[r]))
        cov: dict[tuple[int, int], tuple[float, float]] = {}
        for r in range(rows):
            for c in range(r, rows):
                v, err = _double(product(r, c, means[r][0], means[c][0]), box)
                cov[(r, c)] = cov[(c, r)] = (v * rough[r] * rough[c], err * rough[r] * rough[c])
        ok = [cov[(r, r)][0] > 0 for r in range(rows)]
        for r in range(rows):
            spread = math.sqrt(max(cov[(r, r)][0], 0.0))
            ok.append(_agrees(means[r], _value(shown_mean[r], point), abs(means[r][0]) + spread))
        for r in range(rows):
            for c in range(rows):
                # each entry against its own scale, sqrt(Var_r Var_c) of the reference
                scale = math.sqrt(max(cov[(r, r)][0] * cov[(c, c)][0], 0.0))
                ok.append(_agrees(cov[(r, c)], _value(shown_cov[r][c], point), scale))
        return [all(ok)]

    nonzero = [rank_expr]
    points = _points(m, [*a, *b], nonzero, _rng())
    _run_checks([check], points, one, deadline, _has_symbols(m, [*a, *b]))
    result.checks = [check]


_ACTIONS: dict[MvnAction, Callable[[_Model, MvnResult, float], None]] = {
    "density": _density,
    "moments": _moments,
    "marginal": _marginal,
    "conditional": _conditional,
    "affine": _affine,
}


def mvn(data: MvnInput, limits: Limits) -> MvnResult:
    start = time.monotonic()
    deadline = start + CHECK_SHARE * limits.timeout_seconds
    try:
        m = _Model(data, limits)
        result = MvnResult(status=Status.OK, action=data.action)
        result.input_mean_text = [to_text(e) for e in m.mu]
        result.input_covariance_text = _grid(m.S)[0]
        if data.action in ("density", "moments"):
            result.variables = [v.name for v in m.vars or []]
            result.mean_text, result.mean_latex = _texts(m.mu)
            result.covariance_text, result.covariance_latex = _grid(m.S)
        _ACTIONS[data.action](m, result, deadline)
    except MvnRefusal as exc:
        return MvnResult(status=exc.status, action=data.action, reason=exc.reason, message=str(exc))
    failed = [c for c in result.checks if not c.passed]
    if failed:
        return MvnResult(
            status=Status.NOT_VERIFIED,
            action=data.action,
            reason="verification_failed",
            message="the result did not pass its independent numerical check ("
            + "; ".join(
                f"{c.kind}: {c.points_agreed} of {c.points_checked} points agreed" for c in failed
            )
            + "); nothing is shown",
            checks=result.checks,
            conditions=m.conditions,
        )
    result.conditions = m.conditions
    result.premises = [
        f"{c.latex} (not established by the declared assumptions)" for c in m.conditions
    ]
    result.evidence = "conditional" if result.premises else "identity"
    result.message = (
        "numerically checked: SciPy quadrature at sampled parameter values agreed; numerical "
        "agreement is evidence, not a proof"
    )
    return result
