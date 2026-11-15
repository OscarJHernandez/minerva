"""Line, surface and volume integrals of explicit fields, and the three integral theorems.

The domain is always given by the user, never invented: a curve and a surface are the three coordinates
of the chosen system as functions of one or two parameters (so the orientation is the parameter order:
the direction of increasing parameter for a curve, the normal d_u r x d_v r for a surface), a volume is a
coordinate box. For the divergence theorem the outward boundary of the box is generated here (six faces,
normals out); for Stokes' theorem the boundary of a parametrised rectangle is traversed counter-clockwise
in (u, v), the right-hand rule for the normal d_u r x d_v r.

Two independent routes meet in every integral value:

* exact: the integrand is pulled back to the parameters in the physical frame (scale factors, the
  cross product of the tangent vectors) and integrated by SymPy, within a time limit;
* numerical: the *input expression* is read at Cartesian points by the finite-difference evaluator of
  `numeric.py`, tangents and normals come from differencing the Cartesian map of the parametrisation, a
  volume element from the determinant of the differenced map, and the integral is a quadrature
  (`mpmath`) with its error estimate. No scale factor is shared with the exact route.

The two must agree or nothing is shown. When SymPy finds no closed form the value is the quadrature
alone, labelled a numerical approximation (cross-checked by a second quadrature rule). A theorem is
reported as two separately computed sides; "holds" means the sides agree, and its hypotheses are listed
with what is known: those that hold by construction are `established`, a finite scan of the domain that
found the field defined and finite is `checked` (never a proof), and a field with a pole on the domain
refuses the theorem (`singular_in_domain`), because there its hypotheses fail.
"""

from __future__ import annotations

import contextlib
import random
import signal
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import mpmath as mp
import sympy as sp

from ..models import Status
from ..textparse import parse_text
from .context import Context, Refused, tidy
from .expressions import kind_of
from .models import (
    Curve,
    Hypothesis,
    IntegralKind,
    IntegralResult,
    IntegralValue,
    Param,
    Region,
    Surface,
    VectorDiagnostics,
    VectorResult,
)
from .numeric import Evaluator, Setup, Skip
from .printing import vector_latex

DPS = 15  # the quadrature's own precision: its convergence test is at this level
INNER_DPS = 40  # the integrand is evaluated with more digits, so differencing adds no noise
SYMBOLIC_SECONDS = 4.0
SYMBOLIC_MAX_OPS = 400
TOL = 1e-8
SCAN = 9  # points per dimension of the scan for poles (the ends of every range included)
HUGE = mp.mpf("1e12")
STEP = mp.mpf("1e-9")


class IntegrationFailed(Exception):
    """The quadrature did not converge (a pole, a cusp or an oscillation it cannot resolve)."""


@contextlib.contextmanager
def _alarm(seconds: float) -> Iterator[None]:
    """Raise `TimeoutError` after `seconds` (POSIX main thread only; elsewhere no limit)."""
    usable = hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
    if not usable:
        yield
        return

    def handler(signum: int, frame: Any) -> None:
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class Task:
    """One integral: the exact integrand and ranges, and the independent numerical integrand."""

    label: str
    integrand: sp.Expr  # in the parameters and the constants
    ranges: list[tuple[sp.Symbol, sp.Expr, sp.Expr]]
    numeric: Callable[[list[Any]], Any]  # parameter values in the order of `ranges`
    symbolic_seconds: float = (
        4.0  # the exact route's own limit; `evaluate` raises it with the sandbox timeout
    )


class Builder:
    """Pulls the field and the geometry back to the parameters and builds the numerical integrands."""

    def __init__(
        self,
        ctx: Context,
        expr: sp.Expr,
        rng: random.Random | None = None,
        params: dict[sp.Symbol, sp.Rational] | None = None,
    ) -> None:
        self.ctx = ctx
        self.expr = expr
        self.kind = kind_of(expr)
        lowered = ctx.low.lower(expr)
        self.comps: list[sp.Expr] = (
            [tidy(c) for c in lowered] if isinstance(lowered, tuple) else [tidy(lowered)]
        )
        self.q = ctx.low.q
        self.h = ctx.low.h
        # sample values of the constants, for the numerical route and for the agreement check
        from .numeric import _param_value

        self.params = (
            params
            if params is not None
            else {s: _param_value(s, rng or random.Random(0)) for s in ctx.low.constants.values()}  # noqa: S311
        )
        fields = {}
        for d in ctx.low.declared.values():
            for i, comp in enumerate(d.comps):
                assert d.explicit is not None  # noqa: S101  (checked before: integrals need explicit fields)
                fields[comp] = d.explicit[i].subs(self.params)
        self.setup = Setup(ctx.low, fields, self.params)
        self.ev = Evaluator(self.setup, ctx.system, mp.mpf(0))
        self.x_of_q = ctx.system.x_of_q()
        self._numeric_expr = expr.subs(self.params)

    # -- parametrisations ------------------------------------------------------------------
    def parse_scalar(self, text: str, allowed: dict[str, sp.Symbol]) -> sp.Expr:
        ctx = self.ctx
        assumptions = {**ctx.assumptions, **{n: ["real"] for n in allowed}}
        parsed = parse_text(text, assumptions, ctx.limits)
        extra = {s.name for s in parsed.free_symbols} - set(allowed) - set(ctx.low.constants)
        if extra:
            raise Refused(
                Status.PARSE_ERROR,
                f"{text!r} may use only its parameters and declared constants (found "
                f"{', '.join(sorted(extra))})",
                "bad_declaration",
            )
        mapping = {s: allowed[s.name] for s in parsed.free_symbols if s.name in allowed}
        return parsed.xreplace(mapping)

    def param(self, p: Param, taken: set[str]) -> tuple[sp.Symbol, sp.Expr, sp.Expr]:
        forbidden = (
            set(self.ctx.system.coords) | set(self.ctx.low.declared) | set(self.ctx.low.constants)
        )
        if p.name in forbidden or p.name in taken:
            raise Refused(
                Status.PARSE_ERROR,
                f"the parameter {p.name!r} is already a coordinate, field, constant or parameter",
                "bad_declaration",
            )
        symbol = sp.Symbol(p.name, real=True)
        lo = self.parse_scalar(p.lower, {})
        hi = self.parse_scalar(p.upper, {})
        return symbol, lo, hi

    def coordinates(self, q: list[str], names: dict[str, sp.Symbol]) -> list[sp.Expr]:
        return [self.parse_scalar(text, names) for text in q]

    # -- the field -----------------------------------------------------------------------------
    def field_at(self, qvals: list[Any]) -> Any:
        """The expression evaluated at the point with coordinates `qvals`: a Cartesian vector or a
        number, by the independent finite-difference reading."""
        X = mp.matrix(list(self.x_of_q(*qvals)))
        return self.ev.value(self._numeric_expr, X)

    # -- exact integrands (physical frame) -------------------------------------------------------
    def pull(self, expr: sp.Expr, qexprs: list[sp.Expr]) -> sp.Expr:
        return sp.sympify(expr).xreplace(dict(zip(self.q, qexprs, strict=True)))

    def tangent(self, qexprs: list[sp.Expr], var: sp.Symbol) -> list[sp.Expr]:
        return [self.pull(self.h[i], qexprs) * sp.diff(qexprs[i], var) for i in range(3)]

    # -- tasks -----------------------------------------------------------------------------------
    def line(
        self, qexprs: list[sp.Expr], param: tuple[sp.Symbol, sp.Expr, sp.Expr], label: str
    ) -> Task:
        s, lo, hi = param
        t = self.tangent(qexprs, s)
        if self.kind == "vector":
            f = [self.pull(c, qexprs) for c in self.comps]
            integrand = sum((f[i] * t[i] for i in range(3)), sp.Integer(0))
        else:
            integrand = self.pull(self.comps[0], qexprs) * sp.sqrt(
                sum((c**2 for c in t), sp.Integer(0))
            )
        qfun = self.numeric_map(qexprs, [s])
        h = STEP

        def numeric(values: list[Any]) -> Any:
            sv = values[0]
            q0 = qfun([sv])
            point = [self.x_of_q(*qfun([sv + h])), self.x_of_q(*qfun([sv - h]))]
            tangent = mp.matrix([(point[0][i] - point[1][i]) / (2 * h) for i in range(3)])
            field = self.field_at(q0)
            if self.kind == "vector":
                return sum((field[i] * tangent[i] for i in range(3)), mp.mpf(0))
            return field * mp.norm(tangent)

        return Task(label, integrand, [(s, lo, hi)], numeric)

    def flux(
        self,
        qexprs: list[sp.Expr],
        u: tuple[sp.Symbol, sp.Expr, sp.Expr],
        v: tuple[sp.Symbol, sp.Expr, sp.Expr],
        label: str,
        sign: int = 1,
    ) -> Task:
        us, ulo, uhi = u
        vs, vlo, vhi = v
        ru, rv = self.tangent(qexprs, us), self.tangent(qexprs, vs)
        normal = [
            ru[1] * rv[2] - ru[2] * rv[1],
            ru[2] * rv[0] - ru[0] * rv[2],
            ru[0] * rv[1] - ru[1] * rv[0],
        ]
        f = [self.pull(c, qexprs) for c in self.comps]
        integrand = sign * sum((f[i] * normal[i] for i in range(3)), sp.Integer(0))
        qfun = self.numeric_map(qexprs, [us, vs])
        h = STEP

        def numeric(values: list[Any]) -> Any:
            a, b = values
            q0 = qfun([a, b])
            cols = []
            for k in range(2):
                up, down = [a, b], [a, b]
                up[k] += h
                down[k] -= h
                p1, p2 = self.x_of_q(*qfun(up)), self.x_of_q(*qfun(down))
                cols.append(mp.matrix([(p1[i] - p2[i]) / (2 * h) for i in range(3)]))
            n = mp.matrix(
                [
                    cols[0][1] * cols[1][2] - cols[0][2] * cols[1][1],
                    cols[0][2] * cols[1][0] - cols[0][0] * cols[1][2],
                    cols[0][0] * cols[1][1] - cols[0][1] * cols[1][0],
                ]
            )
            field = self.field_at(q0)
            return sign * sum((field[i] * n[i] for i in range(3)), mp.mpf(0))

        return Task(label, integrand, [(us, ulo, uhi), (vs, vlo, vhi)], numeric)

    def volume(self, region: list[tuple[sp.Expr, sp.Expr]], label: str) -> Task:
        ranges = [(self.q[i], region[i][0], region[i][1]) for i in range(3)]
        integrand = self.comps[0] * self.h[0] * self.h[1] * self.h[2]
        h = STEP

        def numeric(values: list[Any]) -> Any:
            q0 = list(values)
            columns = []
            for k in range(3):
                up, down = list(q0), list(q0)
                up[k] += h
                down[k] -= h
                p1, p2 = self.x_of_q(*up), self.x_of_q(*down)
                columns.append([(p1[i] - p2[i]) / (2 * h) for i in range(3)])
            jac = mp.matrix(3, 3)
            for i in range(3):
                for k in range(3):
                    jac[i, k] = columns[k][i]
            return self.field_at(q0) * abs(mp.det(jac))

        return Task(label, integrand, ranges, numeric)

    def numeric_map(
        self, qexprs: list[sp.Expr], params: list[sp.Symbol]
    ) -> Callable[[list[Any]], list[Any]]:
        fn = sp.lambdify(params, [e.subs(self.params) for e in qexprs], modules="mpmath")

        def call(values: list[Any]) -> list[Any]:
            return list(fn(*values))

        return call


# -- evaluation of a task ---------------------------------------------------------------------------
def _numeric_ranges(task: Task, params: dict[sp.Symbol, sp.Expr]) -> list[tuple[Any, Any]]:
    return [
        (mp.mpf(sp.N(lo.subs(params), 30)), mp.mpf(sp.N(hi.subs(params), 30)))
        for _, lo, hi in task.ranges
    ]


def _finite(task: Task, point: list[Any]) -> Any | None:
    """The integrand at `point`, or None if it cannot be evaluated or is not a finite real number."""
    try:
        with mp.workdps(INNER_DPS):
            value = task.numeric(point)
        if isinstance(value, mp.mpc):
            if abs(value.imag) > mp.mpf("1e-20"):
                return None
            value = value.real
        return value if mp.isfinite(value) else None
    except (ZeroDivisionError, ValueError, TypeError, OverflowError, Skip):
        return None


def _scan(task: Task, ranges: list[tuple[Any, Any]]) -> bool:
    """False if the integrand has a pole or cannot be evaluated on the domain.

    The ends of a range are often coordinate singularities of the system (r = 0, the poles of the
    sphere, the axis), where the Cartesian reading is undefined even for a smooth field, so they are not
    sampled themselves. Instead: a grid of interior points, and, for every end of every range, two
    probes close to it. A value that explodes between the probes (a pole at the end, like 1/r^2 at
    r = 0) fails the scan; a value that stays bounded passes."""
    mid = [[lo + (hi - lo) * (k + mp.mpf(1) / 2) / SCAN for k in range(SCAN)] for lo, hi in ranges]
    for point in _product(mid):
        value = _finite(task, list(point))
        if value is None or abs(value) > HUGE:
            return False
    centre = [lo + (hi - lo) / 2 for lo, hi in ranges]
    for axis, (lo, hi) in enumerate(ranges):
        if hi == lo:
            continue
        for from_lo in (True, False):
            values = []
            for fraction in (mp.mpf("1e-2"), mp.mpf("1e-4")):
                probe = list(centre)
                probe[axis] = lo + (hi - lo) * fraction if from_lo else hi - (hi - lo) * fraction
                value = _finite(task, probe)
                if value is None:
                    return False
                values.append(abs(value))
            near, close = values
            if close > 1000 and close > 50 * max(near, mp.mpf("1e-30")):
                return False
    return True


def _product(axes: list[list[Any]]) -> Iterator[tuple[Any, ...]]:
    if not axes:
        yield ()
        return
    for first in axes[0]:
        for rest in _product(axes[1:]):
            yield (first, *rest)


def _quadrature(task: Task, ranges: list[tuple[Any, Any]], method: str) -> tuple[Any, float]:
    def f(*values: Any) -> Any:
        with mp.workdps(INNER_DPS):
            return task.numeric(list(values))

    try:
        value, error = mp.quad(
            f, *[list(r) for r in ranges], method=method, error=True, maxdegree=5
        )
    except (ZeroDivisionError, ValueError, TypeError, OverflowError) as exc:
        raise IntegrationFailed from exc
    if isinstance(value, mp.mpc):
        value = value.real
    return value, float(error)


def _symbolic(task: Task) -> sp.Expr | None:
    if task.integrand.count_ops() > SYMBOLIC_MAX_OPS:
        return None
    try:
        with _alarm(task.symbolic_seconds):
            value = task.integrand
            for symbol, lo, hi in task.ranges:
                value = sp.integrate(value, (symbol, lo, hi))
                if value.has(sp.Integral) or value.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                    return None
            return tidy(value)
    except (
        TimeoutError,
        NotImplementedError,
        ValueError,
        TypeError,
        RecursionError,
        ZeroDivisionError,
    ):
        return None


@dataclass
class Evaluated:
    value: IntegralValue
    exact: sp.Expr | None
    numeric: Any  # mpf at the sample constants


def evaluate(builder: Builder, task: Task, singular_ok: bool = False) -> Evaluated:
    """Both routes for one integral. Raises `Refused` for a pole on the domain and `IntegrationFailed`."""
    params = builder.params
    if sp.simplify(task.integrand) == 0:  # a face of zero extent: exactly nothing to integrate
        zero = IntegralValue(
            label=task.label, latex="0", text="0", numeric="0", exact=True, agreed=True, points=0
        )
        return Evaluated(zero, sp.Integer(0), mp.mpf(0))
    with mp.workdps(DPS):
        ranges = _numeric_ranges(task, params)
        if not singular_ok and not _scan(task, ranges):
            raise Refused(
                Status.UNSUPPORTED,
                f"the field is not defined and finite everywhere on the domain of {task.label} "
                "(a pole, or a point where it cannot be evaluated): the integral is not computed",
                "singular_in_domain",
            )
        numeric, error = _quadrature(task, ranges, "gauss-legendre")
        task.symbolic_seconds = max(SYMBOLIC_SECONDS, 0.3 * builder.ctx.limits.timeout_seconds)
        exact = _symbolic(task)
        points = 1
        for _ in ranges:
            points *= SCAN
    if exact is None:
        # no closed form: the value is a quadrature, cross-checked by a second rule
        with mp.workdps(DPS):
            second, _ = _quadrature(task, ranges, "tanh-sinh")
        agreed = abs(second - numeric) <= TOL * max(1, abs(numeric))
        value = IntegralValue(
            label=task.label,
            numeric=mp.nstr(numeric, 15) if not builder.ctx.low.constants else None,
            exact=False,
            quadrature_error=error,
            agreed=agreed,
            points=points,
        )
        return Evaluated(value, None, numeric)
    at_sample = mp.mpf(sp.N(exact.subs(params), 30)) if not exact.has(sp.I) else mp.mpf(0)
    agreed = abs(at_sample - numeric) <= TOL * max(1, abs(numeric))
    free = sorted(s.name for s in exact.free_symbols)
    value = IntegralValue(
        label=task.label,
        latex=vector_latex(exact),
        text=str(exact),
        numeric=mp.nstr(mp.mpf(sp.N(exact, 20)), 15) if not free else None,
        exact=True,
        quadrature_error=error,
        agreed=agreed,
        points=points,
    )
    return Evaluated(value, exact, numeric)


# -- the actions ------------------------------------------------------------------------------------
def _need(value: Any, name: str, kind: IntegralKind) -> Any:
    if value is None:
        raise Refused(
            Status.PARSE_ERROR,
            f"a {kind.replace('_', ' ')} needs `{name}`: the domain is never invented",
            "missing_geometry",
        )
    return value


def _curve(
    builder: Builder, curve: Curve
) -> tuple[list[sp.Expr], tuple[sp.Symbol, sp.Expr, sp.Expr]]:
    p = builder.param(curve.parameter, set())
    return builder.coordinates(curve.q, {p[0].name: p[0]}), p


def _surface(
    builder: Builder, surface: Surface
) -> tuple[list[sp.Expr], tuple[sp.Symbol, sp.Expr, sp.Expr], tuple[sp.Symbol, sp.Expr, sp.Expr]]:
    u = builder.param(surface.u, set())
    v = builder.param(surface.v, {u[0].name})
    return builder.coordinates(surface.q, {u[0].name: u[0], v[0].name: v[0]}), u, v


def _region(builder: Builder, region: Region) -> list[tuple[sp.Expr, sp.Expr]]:
    return [
        (builder.parse_scalar(lo, {}), builder.parse_scalar(hi, {})) for lo, hi in region.bounds
    ]


def _line_label(builder: Builder, name: str = "C") -> str:
    field = vector_latex(builder.expr)
    if builder.kind == "vector":
        return rf"\int_{{{name}}} \left({field}\right) \cdot d\mathbf{{l}}"
    return rf"\int_{{{name}}} \left({field}\right) \, dl"


def _faces(
    builder: Builder, region: list[tuple[sp.Expr, sp.Expr]]
) -> list[
    tuple[
        str,
        list[sp.Expr],
        tuple[sp.Symbol, sp.Expr, sp.Expr],
        tuple[sp.Symbol, sp.Expr, sp.Expr],
        int,
    ]
]:
    """The six outward faces of a coordinate box: (name, coordinates, u, v, orientation sign).

    The frame (q1, q2, q3) is right-handed, so the face q_i = const with parameters taken in cyclic order
    (q_j, q_k) has the normal e_i; the upper face is outward as it is, the lower face needs a minus."""
    q = builder.q
    out = []
    names = builder.ctx.system.coords
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        u = sp.Symbol(f"_{names[j]}", real=True)
        v = sp.Symbol(f"_{names[k]}", real=True)
        for side, sign in (("upper", 1), ("lower", -1)):
            qexprs: list[sp.Expr] = [sp.Integer(0)] * 3
            qexprs[i] = region[i][1 if side == "upper" else 0]
            qexprs[j], qexprs[k] = u, v
            out.append(
                (
                    f"{side} face of {names[i]}",
                    qexprs,
                    (u, region[j][0], region[j][1]),
                    (v, region[k][0], region[k][1]),
                    sign,
                )
            )
    del q
    return out


def run_integral(ctx: Context, result: VectorResult, diag: VectorDiagnostics) -> None:
    inp = ctx.inp
    if inp.integral is None:
        raise Refused(Status.PARSE_ERROR, "name the integral (`integral`)", "missing_geometry")
    kind = inp.integral
    expr = ctx.parse(inp.expression)
    for s in expr.free_symbols:
        d = ctx.low.declared.get(s.name)
        if d is not None and d.explicit is None:
            raise Refused(
                Status.UNSUPPORTED,
                f"{s.name} is an abstract field: an integral needs explicit components (a field "
                "with no formula has no value to integrate)",
                "unsupported_expression",
            )
    for d in ctx.low.declared.values():
        if d.explicit is not None and any(
            sym.name == "t" for e in d.explicit for sym in e.free_symbols
        ):
            raise Refused(
                Status.UNSUPPORTED,
                "integrals of time-dependent fields are not supported (the field must not depend on t)",
                "unsupported_expression",
            )
    want_vector = kind in ("line", "flux", "divergence_theorem", "stokes_theorem")
    if (kind_of(expr) == "vector") != want_vector:
        raise Refused(
            Status.PARSE_ERROR,
            f"a {kind.replace('_', ' ')} integrates a {'vector' if want_vector else 'scalar'} "
            "expression",
            "type_error",
        )
    rng = random.Random(20260514)  # noqa: S311  (sample constants, not security)
    builder = Builder(ctx, expr, rng)
    diag.result_kind = "scalar"
    result.input_latex = vector_latex(expr)
    try:
        if kind in ("line", "line_scalar"):
            curve = _need(inp.curve, "curve", kind)
            qexprs, p = _curve(builder, curve)
            task = builder.line(qexprs, p, _line_label(builder))
            _single(builder, task, kind, result, diag, "increasing parameter")
        elif kind == "flux":
            surface = _need(inp.surface, "surface", kind)
            qexprs, u, v = _surface(builder, surface)
            field = vector_latex(builder.expr)
            task = builder.flux(qexprs, u, v, rf"\iint_S \left({field}\right) \cdot d\mathbf{{S}}")
            _single(
                builder,
                task,
                kind,
                result,
                diag,
                f"normal d_{u[0].name} r x d_{v[0].name} r (the order of the parameters is the orientation)",
            )
        elif kind == "volume":
            region = _region(builder, _need(inp.region, "region", kind))
            field = vector_latex(builder.expr)
            task = builder.volume(region, rf"\iiint_V \left({field}\right) dV")
            _single(builder, task, kind, result, diag, "none (a scalar volume integral)")
        elif kind == "gradient_theorem":
            _gradient(builder, _need(inp.curve, "curve", kind), result, diag)
        elif kind == "divergence_theorem":
            _divergence(builder, _need(inp.region, "region", kind), result, diag)
        else:
            _stokes(builder, _need(inp.surface, "surface", kind), result, diag)
    except IntegrationFailed:
        result.status = Status.RESOURCE_LIMIT
        result.message = (
            "the quadrature did not converge on this domain (a pole, a cusp or a rapid oscillation); "
            "no value is shown"
        )
        diag.reason = "integration_failed"


def _single(
    builder: Builder,
    task: Task,
    kind: IntegralKind,
    result: VectorResult,
    diag: VectorDiagnostics,
    orientation: str,
) -> None:
    evaluated = evaluate(builder, task)
    value = evaluated.value
    if evaluated.exact is not None and not value.agreed:
        result.status = Status.NOT_VERIFIED
        result.message = (
            "the exact value and the independent quadrature disagree, so no value is shown"
        )
        diag.reason = "verification_failed"
        return
    if evaluated.exact is None and not value.agreed:
        result.status = Status.NOT_VERIFIED
        result.message = "two quadrature rules disagree, so no value is shown"
        diag.reason = "verification_failed"
        return
    result.integral = IntegralResult(
        kind=kind,
        values=[value],
        orientation=orientation,
        method="symbolic" if evaluated.exact is not None else "numerical",
    )
    shown = value.latex if value.latex is not None else (value.numeric or r"\text{numerical}")
    result.result_kind = "scalar"
    result.result_latex = f"{value.label} = {shown}"
    result.result_text = value.text or value.numeric
    result.status = Status.OK
    result.message = _message(value, evaluated.exact is not None)


def _message(value: IntegralValue, exact: bool) -> str:
    if exact:
        return (
            "closed form by SymPy, checked against an independent quadrature of the input read "
            f"in Cartesian coordinates (error estimate {value.quadrature_error:.1e}); "
            "numerical agreement is evidence, not a proof"
        )
    return (
        "no closed form was found: this is a numerical approximation (two quadrature rules agree, "
        f"error estimate {value.quadrature_error:.1e}), not an exact value"
    )


def _compare(left: Evaluated, right: Evaluated) -> bool:
    if left.exact is not None and right.exact is not None:
        difference = tidy(sp.Add(left.exact, sp.Mul(-1, right.exact)))
        if difference == 0:
            return True
    return bool(
        abs(left.numeric - right.numeric) <= TOL * max(1, abs(left.numeric), abs(right.numeric))
    )


def _theorem(
    builder: Builder,
    kind: IntegralKind,
    name: str,
    left: Evaluated,
    right: Evaluated,
    hypotheses: list[Hypothesis],
    orientation: str,
    result: VectorResult,
    diag: VectorDiagnostics,
) -> None:
    for side in (left, right):
        if not side.value.agreed:
            result.status = Status.NOT_VERIFIED
            result.message = "a side of the theorem was not confirmed by its independent quadrature"
            diag.reason = "verification_failed"
            return
    holds = _compare(left, right)
    if not holds:
        result.status = Status.NOT_VERIFIED
        result.message = (
            f"the two sides of {name} differ although its hypotheses were checked: nothing is "
            "shown as a result (the field may not be smooth enough, or the domain may be mis-described)"
        )
        diag.reason = "verification_failed"
        return
    exact = left.exact is not None and right.exact is not None
    result.integral = IntegralResult(
        kind=kind,
        values=[left.value, right.value],
        theorem=name,
        holds=True,
        hypotheses=hypotheses,
        orientation=orientation,
        method="symbolic" if exact else "numerical",
    )
    shown = left.value.latex or left.value.numeric or r"\text{numerical}"
    result.result_kind = "scalar"
    result.result_latex = f"{left.value.label} = {right.value.label} = {shown}"
    result.result_text = left.value.text or left.value.numeric
    result.status = Status.OK
    result.message = (
        f"both sides of {name} were computed separately and agree"
        + ("" if exact else " (numerically: no closed form for at least one side)")
        + "; the hypotheses are listed with what is known of each. This is consistent with the theorem for "
        "this field and domain, not a proof of its hypotheses"
    )


def _gradient(
    builder: Builder, curve: Curve, result: VectorResult, diag: VectorDiagnostics
) -> None:
    """int_C grad f . dl = f(end) - f(start): the left side by the line integral of the gradient (both
    routes), the right side by evaluating f at the two ends (symbolically and numerically)."""
    from .expressions import Grad

    if kind_of(builder.expr) != "scalar":
        raise Refused(
            Status.PARSE_ERROR, "the gradient theorem needs a scalar field f", "type_error"
        )
    inner = Builder(builder.ctx, Grad(builder.expr), None, builder.params)
    qexprs, p = _curve(inner, curve)
    field = vector_latex(builder.expr)
    left = evaluate(
        inner, inner.line(qexprs, p, rf"\int_C \nabla \left({field}\right) \cdot d\mathbf{{l}}")
    )
    s, lo, hi = p
    f = builder.comps[0]
    end = f.xreplace(dict(zip(builder.q, [e.subs(s, hi) for e in qexprs], strict=True)))
    start = f.xreplace(dict(zip(builder.q, [e.subs(s, lo) for e in qexprs], strict=True)))
    exact = tidy(sp.Add(end, sp.Mul(-1, start)))
    qfun = builder.numeric_map(qexprs, [s])
    with mp.workdps(DPS):
        hi_n = mp.mpf(sp.N(hi.subs(builder.params), 30))
        lo_n = mp.mpf(sp.N(lo.subs(builder.params), 30))
        right_n = builder.field_at(qfun([hi_n])) - builder.field_at(qfun([lo_n]))
    free = sorted(x.name for x in exact.free_symbols)
    sample = mp.mpf(sp.N(exact.subs(builder.params), 30))
    right_value = IntegralValue(
        label=rf"\left({field}\right)\Big|_{{\text{{end}}}} - \left({field}\right)\Big|_{{\text{{start}}}}",
        latex=vector_latex(exact),
        text=str(exact),
        numeric=mp.nstr(mp.mpf(sp.N(exact, 20)), 15) if not free else None,
        exact=True,
        agreed=bool(abs(sample - right_n) <= TOL * max(1, abs(right_n))),
        points=2,
    )
    hypotheses = [
        Hypothesis(
            text="f is continuously differentiable on the curve",
            status="checked",
            note="the gradient is finite at every sampled point of the curve; smoothness is not proved",
        ),
        Hypothesis(
            text="the curve is piecewise smooth, traversed in the direction of increasing parameter",
            status="established",
            note="a single parametrised curve",
        ),
    ]
    _theorem(
        builder,
        "gradient_theorem",
        "the gradient theorem",
        left,
        Evaluated(right_value, exact, right_n),
        hypotheses,
        "increasing parameter",
        result,
        diag,
    )


def _divergence(
    builder: Builder, region_spec: Region, result: VectorResult, diag: VectorDiagnostics
) -> None:
    """The divergence theorem over a coordinate box: the volume integral of div F, and the flux of F
    through the six outward faces the engine generates."""
    from .expressions import Div

    region = _region(builder, region_spec)
    field = vector_latex(builder.expr)
    inner = Builder(builder.ctx, Div(builder.expr), None, builder.params)
    left = evaluate(inner, inner.volume(region, rf"\iiint_V \nabla \cdot \left({field}\right) dV"))
    faces = _faces(builder, region)
    total_exact: sp.Expr | None = sp.Integer(0)
    total_numeric = mp.mpf(0)
    points = 0
    for name, qexprs, u, v, sign in faces:
        task = builder.flux(qexprs, u, v, name, sign)
        if all(a == b for a, b in zip(u[1:], (u[1], u[2]), strict=True)) and (
            u[1] == u[2] or v[1] == v[2]
        ):
            continue  # a face of zero extent
        part = evaluate(builder, task)
        if not part.value.agreed:
            result.status = Status.NOT_VERIFIED
            result.message = f"the flux through the {name} was not confirmed by its quadrature"
            diag.reason = "verification_failed"
            return
        points += part.value.points
        total_numeric += part.numeric
        total_exact = (
            sp.Add(total_exact, part.exact)
            if total_exact is not None and part.exact is not None
            else None
        )
    free = sorted(x.name for x in total_exact.free_symbols) if total_exact is not None else []
    right_value = IntegralValue(
        label=rf"\oint_{{\partial V}} \left({field}\right) \cdot d\mathbf{{S}}",
        latex=vector_latex(tidy(total_exact)) if total_exact is not None else None,
        text=str(tidy(total_exact)) if total_exact is not None else None,
        numeric=mp.nstr(total_numeric, 15) if not builder.ctx.low.constants else None,
        exact=total_exact is not None,
        agreed=True,
        points=points,
    )
    if total_exact is not None and not free:
        right_value.numeric = mp.nstr(mp.mpf(sp.N(tidy(total_exact), 20)), 15)
    right = Evaluated(
        right_value, tidy(total_exact) if total_exact is not None else None, total_numeric
    )
    hypotheses = [
        Hypothesis(
            text="the region is a coordinate box with a piecewise smooth boundary",
            status="established",
            note="the six faces and their outward normals were generated from the bounds",
        ),
        Hypothesis(
            text="F is continuously differentiable on the closed region",
            status="checked",
            note="F and the integrand are finite at every point of a grid on the region and its "
            "boundary; smoothness is not proved",
        ),
    ]
    _theorem(
        builder, "divergence_theorem", "the divergence theorem", left, right, hypotheses,
        "outward normals on the six faces of the box", result, diag,
    )  # fmt: skip


def _stokes(
    builder: Builder, surface: Surface, result: VectorResult, diag: VectorDiagnostics
) -> None:
    """Stokes' theorem on a parametrised rectangle: the flux of curl F through S, and the circulation of
    F around the boundary traversed counter-clockwise in (u, v)."""
    from .expressions import Curl

    qexprs, u, v = _surface(builder, surface)
    field = vector_latex(builder.expr)
    inner = Builder(builder.ctx, Curl(builder.expr), None, builder.params)
    left = evaluate(
        inner,
        inner.flux(
            qexprs, u, v, rf"\iint_S \nabla \times \left({field}\right) \cdot d\mathbf{{S}}"
        ),
    )
    us, ulo, uhi = u
    vs, vlo, vhi = v
    s = sp.Symbol("_s", real=True)

    def edge(qs: list[sp.Expr], lo: sp.Expr, hi: sp.Expr) -> Evaluated:
        task = builder.line(qs, (s, lo, hi), "edge")
        return evaluate(builder, task)

    edges = [
        edge([q.subs({us: s, vs: vlo}) for q in qexprs], ulo, uhi),
        edge([q.subs({us: uhi, vs: s}) for q in qexprs], vlo, vhi),
        edge([q.subs({us: s, vs: vhi}) for q in qexprs], uhi, ulo),
        edge([q.subs({us: ulo, vs: s}) for q in qexprs], vhi, vlo),
    ]
    if any(not e.value.agreed for e in edges):
        result.status = Status.NOT_VERIFIED
        result.message = "an edge integral was not confirmed by its independent quadrature"
        diag.reason = "verification_failed"
        return
    exacts = [e.exact for e in edges]
    total_exact = sp.Add(*exacts) if all(x is not None for x in exacts) else None
    total_numeric = sum((e.numeric for e in edges), mp.mpf(0))
    free = sorted(x.name for x in total_exact.free_symbols) if total_exact is not None else []
    right_value = IntegralValue(
        label=rf"\oint_{{\partial S}} \left({field}\right) \cdot d\mathbf{{l}}",
        latex=vector_latex(tidy(total_exact)) if total_exact is not None else None,
        text=str(tidy(total_exact)) if total_exact is not None else None,
        numeric=mp.nstr(total_numeric, 15) if not builder.ctx.low.constants else None,
        exact=total_exact is not None,
        agreed=True,
        points=sum(e.value.points for e in edges),
    )
    if total_exact is not None and not free:
        right_value.numeric = mp.nstr(mp.mpf(sp.N(tidy(total_exact), 20)), 15)
    right = Evaluated(
        right_value, tidy(total_exact) if total_exact is not None else None, total_numeric
    )
    hypotheses = [
        Hypothesis(
            text="S is a piecewise smooth oriented surface whose boundary is the image of the rectangle's edges",
            status="established",
            note="the boundary is traversed counter-clockwise in (u, v): the right-hand rule for the "
            f"normal d_{us.name} r x d_{vs.name} r",
        ),
        Hypothesis(
            text="F is continuously differentiable on a neighbourhood of S",
            status="checked",
            note="F, its curl and the edge integrands are finite at every point of a grid on S and its "
            "boundary; smoothness is not proved",
        ),
    ]
    _theorem(
        builder, "stokes_theorem", "Stokes' theorem", left, right, hypotheses,
        f"normal d_{us.name} r x d_{vs.name} r; boundary counter-clockwise in ({us.name}, {vs.name})",
        result, diag,
    )  # fmt: skip
