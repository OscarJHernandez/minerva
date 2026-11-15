"""Analytical and numerical electrostatics: potentials and fields, multipoles, a boundary-value problem on a
rectangle, and a check of a candidate solution.

Scope, stated plainly: point charges and a charge density over a Cartesian box; Laplace's and Poisson's
equation on a rectangle with Dirichlet data; Cartesian coordinates. Everything is in units of
k_e = 1/(4 pi epsilon_0). No finite-element or finite-difference *solver* is offered as a method; a
finite-difference relaxation exists only as the independent check of the series solution.

Every result is two methods that must agree, never one:

* field: the Coulomb vector sum, the negative gradient of the symbolic potential, and a central
  difference of the potential evaluated as a number;
* density: the Coulomb integral for E by quadrature, and the negative difference quotient of the potential
  integral (two different integrals). The field point must be outside the source: a point in or at its
  edge makes the integrand singular and is refused (`field_point_in_source`);
* multipole: the moments are exact; the truncated expansion is compared with the exact potential along a
  ray at growing distances, and the error decay is *measured* (never a bound);
* boundary value: a separated series (sine coefficients in closed form when SymPy finds them, by quadrature
  otherwise) against a finite-difference relaxation on two grids; the result is a numerical approximation
  and is shown only if they agree within what the grids and the truncation allow;
* solution check: the exact residual of Poisson's equation and the boundary values, symbolically and by the
  independent finite-difference reading of the vector module.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

import sympy as sp

from ..models import Status
from ..textparse import parse_text
from .algebra import exact_zero
from .context import Context, Refused
from .models import (
    BoundaryProblem,
    CheckProblem,
    EMComputation,
    EMItem,
    FieldProblem,
    MultipoleProblem,
    VectorDiagnostics,
    VectorResult,
)
from .numeric import _param_value
from .printing import vector_latex

KE = sp.Symbol("k_e", positive=True)
UNITS = "k_e = 1/(4 pi epsilon_0)"
GL_LOW, GL_HIGH = 10, 16


# -- small numerical tools (plain floats: these are the independent route) --------------------------
def _legendre_rule(n: int) -> list[tuple[float, float]]:
    """Gauss-Legendre nodes and weights on [-1, 1] by Newton's method."""
    out = []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            step = p1 / dp
            x -= step
            if abs(step) < 1e-15:
                break
        out.append((x, 2 / ((1 - x * x) * dp * dp)))
    return out


_RULES: dict[int, list[tuple[float, float]]] = {}


def _rule(n: int) -> list[tuple[float, float]]:
    if n not in _RULES:
        _RULES[n] = _legendre_rule(n)
    return _RULES[n]


def _integrate_box(
    f: Callable[[float, float, float], float], lo: list[float], hi: list[float], n: int
) -> float:
    rule = _rule(n)
    total = 0.0
    half = [(h - low) / 2 for low, h in zip(lo, hi, strict=True)]
    mid = [(h + low) / 2 for low, h in zip(lo, hi, strict=True)]
    for xi, wi in rule:
        x = mid[0] + half[0] * xi
        for yj, wj in rule:
            y = mid[1] + half[1] * yj
            for zk, wk in rule:
                total += wi * wj * wk * f(x, y, mid[2] + half[2] * zk)
    return total * half[0] * half[1] * half[2]


class EMProblem:
    """A request's parsing helpers: text to expressions, with sampled values for the constants."""

    def __init__(self, ctx: Context, seed: int = 20260514) -> None:
        self.ctx = ctx
        rng = random.Random(seed)  # noqa: S311  (sample constants, not security)
        self.sample = {s: _param_value(s, rng) for s in ctx.low.constants.values()}

    def expr(self, text: str, extra: tuple[str, ...] = ()) -> sp.Expr:
        ctx = self.ctx
        assumptions = {**ctx.assumptions, **{n: ["real"] for n in extra}}
        parsed = parse_text(text, assumptions, ctx.limits)
        stray = {s.name for s in parsed.free_symbols} - set(ctx.low.constants) - set(extra)
        if stray:
            raise Refused(
                Status.PARSE_ERROR,
                f"{text!r} may use only {', '.join(extra) + ', ' if extra else ''}declared constants "
                f"(found {', '.join(sorted(stray))})",
                "bad_declaration",
            )
        return parsed

    def number(self, e: sp.Expr) -> float:
        return float(sp.N(e.subs(self.sample), 30))


def _latex(e: sp.Expr) -> str:
    return vector_latex(e)


def _fmt(x: float) -> str:
    return f"{x:.12g}"


# -- field ----------------------------------------------------------------------------------------------
def _field(ctx: Context, prob: FieldProblem, result: VectorResult, diag: VectorDiagnostics) -> None:
    ep = EMProblem(ctx)
    if not prob.charges and prob.density is None:
        raise Refused(
            Status.PARSE_ERROR, "give point charges and/or a charge density", "missing_geometry"
        )
    xs = sp.symbols("x y z", real=True)
    point = [ep.expr(c) for c in prob.point]
    items: list[EMItem] = []
    checks: list[str] = []
    worst = 0.0
    approximate = False
    if prob.charges:
        phi = sp.Integer(0)
        direct = [sp.Integer(0)] * 3
        for ch in prob.charges:
            q = ep.expr(ch.q)
            pos = [ep.expr(c) for c in ch.position]
            if all(sp.simplify(point[i] - pos[i]) == 0 for i in range(3)):
                raise Refused(
                    Status.UNSUPPORTED,
                    "the field point is at a point charge: the potential and field are infinite there",
                    "field_point_in_source",
                )
            dist = sp.sqrt(sum((xs[i] - pos[i]) ** 2 for i in range(3)))
            phi += KE * q / dist
            rel = [point[i] - pos[i] for i in range(3)]
            norm3 = sp.sqrt(sum(r**2 for r in rel)) ** 3
            direct = [direct[i] + KE * q * rel[i] / norm3 for i in range(3)]
        at = dict(zip(xs, point, strict=True))
        phi_p = sp.simplify(phi.xreplace(at))
        e_grad = [sp.simplify(-sp.diff(phi, v).xreplace(at)) for v in xs]
        e_direct = [sp.simplify(d) for d in direct]
        agree = all(sp.simplify(e_grad[i] - e_direct[i]) == 0 for i in range(3))
        if not agree:
            result.status = Status.NOT_VERIFIED
            result.message = "the Coulomb sum and the gradient of the potential disagree"
            diag.reason = "verification_failed"
            return
        # the third method: a central difference of the potential as a number
        f = sp.lambdify(xs, phi.subs({KE: 1}).subs(ep.sample), modules="math")
        p0 = [ep.number(c) for c in point]
        h = 1e-5 * max(1.0, max(abs(c) for c in p0))
        fd = []
        for i in range(3):
            up, down = list(p0), list(p0)
            up[i] += h
            down[i] -= h
            fd.append(-(f(*up) - f(*down)) / (2 * h))
        exact_num = [ep.number(e.subs(KE, 1)) for e in e_direct]
        scale = max(1.0, *[abs(v) for v in exact_num])
        worst = max(abs(a - b) for a, b in zip(fd, exact_num, strict=True)) / scale
        if worst > 1e-6:
            result.status = Status.NOT_VERIFIED
            result.message = "the field disagrees with a central difference of the potential"
            diag.reason = "solver_disagreement"
            return
        items.append(EMItem(label="Potential at the point", latex=_latex(phi_p)))
        items.append(
            EMItem(
                label="Electric field at the point",
                latex=r"\mathbf{E} = \left(" + ", ".join(_latex(c) for c in e_direct) + r"\right)",
            )
        )
        checks.append(
            "the Coulomb vector sum equals the negative gradient of the symbolic potential, and a central "
            f"difference of the potential as a number agrees (relative {worst:.1e})"
        )
    if prob.density is not None:
        approximate = True
        if prob.box is None:
            raise Refused(
                Status.PARSE_ERROR, "a density needs `box`, its support", "missing_geometry"
            )
        rho = ep.expr(prob.density, ("x", "y", "z"))
        lo = [ep.number(ep.expr(a)) for a, _ in prob.box.bounds]
        hi = [ep.number(ep.expr(b)) for _, b in prob.box.bounds]
        p0 = [ep.number(c) for c in point]
        size = max(h - low for low, h in zip(lo, hi, strict=True))
        for i in range(3):
            if lo[i] - 0.05 * size <= p0[i] <= hi[i] + 0.05 * size:
                inside = all(lo[j] - 0.05 * size <= p0[j] <= hi[j] + 0.05 * size for j in range(3))
                if inside:
                    raise Refused(
                        Status.UNSUPPORTED,
                        "the field point is inside the charge distribution (or at its edge): the "
                        "Coulomb integrand is singular there, so nothing is computed",
                        "field_point_in_source",
                    )
        g = sp.lambdify(sp.symbols("x y z", real=True), rho.subs(ep.sample), modules="math")

        def kernel_phi(px: float, py: float, pz: float) -> Callable[[float, float, float], float]:
            def f(x: float, y: float, z: float) -> float:
                return float(g(x, y, z)) / math.dist((px, py, pz), (x, y, z))

            return f

        def phi_at(point3: list[float], n: int) -> float:
            return _integrate_box(kernel_phi(*point3), lo, hi, n)

        def e_direct_at(n: int) -> list[float]:
            out = []
            for i in range(3):

                def f(x: float, y: float, z: float, i: int = i) -> float:
                    r = math.dist(tuple(p0), (x, y, z))
                    return float(g(x, y, z)) * (p0[i] - (x, y, z)[i]) / r**3

                out.append(_integrate_box(f, lo, hi, n))
            return out

        phi_high = phi_at(p0, GL_HIGH)
        e_low, e_high = e_direct_at(GL_LOW), e_direct_at(GL_HIGH)
        h = 1e-3 * size
        e_fd = []
        for i in range(3):
            up, down = list(p0), list(p0)
            up[i] += h
            down[i] -= h
            e_fd.append(-(phi_at(up, GL_HIGH) - phi_at(down, GL_HIGH)) / (2 * h))
        scale = max(1.0, *[abs(v) for v in e_high])
        quad_gap = max(abs(a - b) for a, b in zip(e_low, e_high, strict=True)) / scale
        method_gap = max(abs(a - b) for a, b in zip(e_fd, e_high, strict=True)) / scale
        worst = max(worst, method_gap)
        if method_gap > 1e-4 or quad_gap > 1e-6:
            result.status = Status.NOT_VERIFIED
            result.message = (
                "the Coulomb integral for E and the gradient of the potential integral disagree "
                f"(gap {method_gap:.1e}, quadrature rules {quad_gap:.1e}): nothing is shown"
            )
            diag.reason = "solver_disagreement"
            return
        items.append(
            EMItem(
                label="Potential at the point (density)", numeric=_fmt(phi_high), note="quadrature"
            )
        )
        items.append(
            EMItem(
                label="Electric field at the point (density)",
                numeric="(" + ", ".join(_fmt(c) for c in e_high) + ")",
                note="quadrature",
            )
        )
        checks.append(
            "the Coulomb integral for E agrees with the negative difference quotient of the potential "
            f"integral (relative {method_gap:.1e}); two Gauss-Legendre rules agree ({quad_gap:.1e})"
        )
    result.em = EMComputation(
        kind="field",
        items=items,
        method="Coulomb's law",
        checks=checks,
        approximate=approximate,
        units=UNITS,
        agreement=worst,
    )
    _ok(result, "numerical approximation" if approximate else "closed form", checks)


# -- multipole -------------------------------------------------------------------------------------------
def _multipole(
    ctx: Context, prob: MultipoleProblem, result: VectorResult, diag: VectorDiagnostics
) -> None:
    ep = EMProblem(ctx)
    qs = [ep.expr(c.q) for c in prob.charges]
    pos = [[ep.expr(c) for c in ch.position] for ch in prob.charges]
    total = sp.simplify(sum(qs, sp.Integer(0)))
    dipole = [
        sp.simplify(sum((q * p[i] for q, p in zip(qs, pos, strict=True)), sp.Integer(0)))
        for i in range(3)
    ]
    quad = [
        [
            sp.simplify(
                sum(
                    (
                        q * (3 * p[i] * p[j] - (sum(c**2 for c in p) if i == j else 0))
                        for q, p in zip(qs, pos, strict=True)
                    ),
                    sp.Integer(0),
                )
            )
            for j in range(3)
        ]
        for i in range(3)
    ]
    items = [
        EMItem(label="Monopole moment Q", latex=_latex(total)),
        EMItem(
            label="Dipole moment p",
            latex=r"\mathbf{p} = \left(" + ", ".join(_latex(c) for c in dipole) + r"\right)",
        ),
        EMItem(
            label="Quadrupole moments Q_ij (traceless: 3 x_i x_j - r^2 delta_ij)",
            latex=r"Q_{ij} = " + _latex(sp.Matrix(quad)),
        ),
    ]
    # the truncated expansion against the exact potential along a ray
    direction = [ep.number(ep.expr(c)) for c in prob.direction]
    norm = math.sqrt(sum(d * d for d in direction))
    if norm == 0:
        raise Refused(
            Status.PARSE_ERROR, "the direction of the ray is the zero vector", "bad_declaration"
        )
    n = [d / norm for d in direction]
    qn = [ep.number(q) for q in qs]
    pn = [[ep.number(c) for c in p] for p in pos]
    extent = max(math.sqrt(sum(c * c for c in p)) for p in pn) or 1.0
    moment_q = sum(qn)
    moment_p = [sum(q * p[i] for q, p in zip(qn, pn, strict=True)) for i in range(3)]
    moment_quad = [
        [
            sum(
                q * (3 * p[i] * p[j] - (sum(c * c for c in p) if i == j else 0.0))
                for q, p in zip(qn, pn, strict=True)
            )
            for j in range(3)
        ]
        for i in range(3)
    ]

    def exact(r: float) -> float:
        point = [r * c for c in n]
        return sum(q / math.dist(point, p) for q, p in zip(qn, pn, strict=True))

    def truncated(r: float) -> float:
        phi = moment_q / r
        if prob.order >= 1:
            phi += sum(moment_p[i] * n[i] for i in range(3)) / r**2
        if prob.order >= 2:
            phi += (
                0.5
                * sum(moment_quad[i][j] * n[i] * n[j] for i in range(3) for j in range(3))
                / r**3
            )
        return phi

    radii = [4 * extent * 2**k for k in range(4)]
    errors = [abs(exact(r) - truncated(r)) for r in radii]
    exponents = [
        math.log(errors[k] / errors[k + 1]) / math.log(2)
        for k in range(3)
        if errors[k] > 1e-300 and errors[k + 1] > 1e-300
    ]
    for r, err in zip(radii, errors, strict=True):
        items.append(
            EMItem(
                label=f"Error of the order-{prob.order} expansion at distance {_fmt(r)}",
                numeric=f"{err:.3e}",
                note=f"exact potential {_fmt(exact(r))}",
            )
        )
    expected = (
        prob.order + 2
    )  # the next multipole is order + 1: its potential falls like r^-(order + 2)
    if exponents and min(exponents) < expected - 0.35:
        result.status = Status.NOT_VERIFIED
        result.message = (
            f"the truncated expansion does not converge like the next multipole (measured decay "
            f"exponents {', '.join(f'{e:.2f}' for e in exponents)}, expected at least {expected}): "
            "nothing is shown"
        )
        diag.reason = "solver_disagreement"
        return
    check = (
        f"the error of the truncated expansion along the ray falls with measured exponents "
        f"{', '.join(f'{e:.2f}' for e in exponents) or 'no error (an exact expansion)'}; the next "
        f"multipole predicts {expected}. A measurement at these distances, never a bound"
    )
    result.em = EMComputation(
        kind="multipole",
        items=items,
        method="multipole moments of point charges",
        checks=[check],
        approximate=False,
        units=UNITS,
        agreement=max(errors) if errors else 0.0,
    )
    _ok(result, "exact moments", [check])


# -- boundary value problem on a rectangle ----------------------------------------------------------------
def _sor(
    nx: int,
    ny: int,
    a: float,
    b: float,
    edge: dict[str, Callable[[float], float]],
    source: Callable[[float, float], float] | None,
) -> list[list[float]]:
    """Gauss-Seidel with over-relaxation for laplacian(phi) = -source on a grid with Dirichlet data:
    plain double-precision arithmetic, the independent check of the series."""
    hx, hy = a / nx, b / ny
    phi = [[0.0] * (ny + 1) for _ in range(nx + 1)]
    for i in range(nx + 1):
        phi[i][0] = edge["bottom"](i * hx)
        phi[i][ny] = edge["top"](i * hx)
    for j in range(ny + 1):
        phi[0][j] = edge["left"](j * hy)
        phi[nx][j] = edge["right"](j * hy)
    omega = 2 / (1 + math.sin(math.pi / max(nx, ny)))
    hx2, hy2 = hx * hx, hy * hy
    denom = 2 * (hx2 + hy2)
    for _ in range(20000):
        change = 0.0
        for i in range(1, nx):
            for j in range(1, ny):
                rhs = hy2 * (phi[i + 1][j] + phi[i - 1][j]) + hx2 * (phi[i][j + 1] + phi[i][j - 1])
                if source is not None:
                    rhs += hx2 * hy2 * source(i * hx, j * hy)
                new = rhs / denom
                delta = omega * (new - phi[i][j])
                phi[i][j] += delta
                change = max(change, abs(delta))
        if change < 1e-11:
            break
    return phi


def _grid_value(phi: list[list[float]], a: float, b: float, x: float, y: float) -> float:
    nx, ny = len(phi) - 1, len(phi[0]) - 1
    fx, fy = x / a * nx, y / b * ny
    i, j = min(int(fx), nx - 1), min(int(fy), ny - 1)
    tx, ty = fx - i, fy - j
    return (
        (1 - tx) * (1 - ty) * phi[i][j]
        + tx * (1 - ty) * phi[i + 1][j]
        + (1 - tx) * ty * phi[i][j + 1]
        + tx * ty * phi[i + 1][j + 1]
    )


def _boundary(
    ctx: Context, prob: BoundaryProblem, result: VectorResult, diag: VectorDiagnostics
) -> None:
    ep = EMProblem(ctx)
    a_sym, b_sym = ep.expr(prob.a), ep.expr(prob.b)
    a, b = ep.number(a_sym), ep.number(b_sym)
    if a <= 0 or b <= 0:
        raise Refused(Status.PARSE_ERROR, "the rectangle needs positive sides", "bad_declaration")
    x, y = sp.symbols("x y", real=True)
    n = sp.Symbol("n", integer=True, positive=True)
    edges_sym = {
        "bottom": ep.expr(prob.bottom, ("x",)),
        "top": ep.expr(prob.top, ("x",)),
        "left": ep.expr(prob.left, ("y",)),
        "right": ep.expr(prob.right, ("y",)),
    }
    for name in edges_sym:
        edges_sym[name] = edges_sym[name].xreplace(
            {
                s: (x if name in ("bottom", "top") else y)
                for s in edges_sym[name].free_symbols
                if s.name in ("x", "y")
            }
        )
    source_sym = ep.expr(prob.source, ("x", "y")) if prob.source is not None else None
    if source_sym is not None and any(not exact_zero(e) for e in edges_sym.values()):
        raise Refused(
            Status.UNSUPPORTED,
            "Poisson's equation is solved here with zero boundary values only (a nonzero boundary needs "
            "Laplace's solution added: pose it as two problems)",
            "unsupported_expression",
        )
    edge_fn: dict[str, Callable[[float], float]] = {}
    for name, e in edges_sym.items():
        var = x if name in ("bottom", "top") else y
        fn = sp.lambdify(var, e.subs(ep.sample), modules="math")
        edge_fn[name] = (lambda f: lambda t: float(f(t)))(fn)
    source_fn = (
        (lambda f: lambda px, py: float(f(px, py)))(
            sp.lambdify((x, y), source_sym.subs(ep.sample), modules="math")
        )
        if source_sym is not None
        else None
    )
    terms = prob.terms
    rule = _rule(48)

    def fourier(fn: Callable[[float], float], length: float, k: int) -> float:
        half = length / 2
        return (
            (2 / length)
            * half
            * sum(
                w * fn(half * (xi + 1)) * math.sin(k * math.pi * half * (xi + 1) / length)
                for xi, w in rule
            )
        )

    # separated coefficients: closed form in n when SymPy finds one
    closed: dict[str, sp.Expr | None] = {}
    for name, e in edges_sym.items():
        if exact_zero(e):
            continue
        var, length = (x, a_sym) if name in ("bottom", "top") else (y, b_sym)
        try:
            coef = sp.simplify(
                (2 / length) * sp.integrate(e * sp.sin(n * sp.pi * var / length), (var, 0, length))
            )
            closed[name] = None if coef.has(sp.Integral) else coef
        except Exception:  # SymPy gave up: the coefficients come from quadrature
            closed[name] = None
    coef_num: dict[str, list[float]] = {}
    for name in edges_sym:
        if exact_zero(edges_sym[name]):
            continue
        length = a if name in ("bottom", "top") else b
        if closed.get(name) is not None:
            f = sp.lambdify(n, closed[name].subs(ep.sample), modules="math")  # type: ignore[union-attr]
            coef_num[name] = [float(f(k)) for k in range(1, terms + 1)]
        else:
            coef_num[name] = [fourier(edge_fn[name], length, k) for k in range(1, terms + 1)]

    def series(px: float, py: float, count: int) -> float:
        total = 0.0
        for k in range(1, count + 1):
            sx, sy = k * math.pi / a, k * math.pi / b
            if "top" in coef_num:
                total += (
                    coef_num["top"][k - 1]
                    * math.sin(sx * px)
                    * math.sinh(sx * py)
                    / math.sinh(sx * b)
                )
            if "bottom" in coef_num:
                total += (
                    coef_num["bottom"][k - 1]
                    * math.sin(sx * px)
                    * math.sinh(sx * (b - py))
                    / math.sinh(sx * b)
                )
            if "right" in coef_num:
                total += (
                    coef_num["right"][k - 1]
                    * math.sin(sy * py)
                    * math.sinh(sy * px)
                    / math.sinh(sy * a)
                )
            if "left" in coef_num:
                total += (
                    coef_num["left"][k - 1]
                    * math.sin(sy * py)
                    * math.sinh(sy * (a - px))
                    / math.sinh(sy * a)
                )
        return total

    rule24 = _rule(48)
    if source_fn is not None:
        coefficients: dict[tuple[int, int], float] = {}
        for p in range(1, 25):
            for q in range(1, 25):
                total = 0.0
                for xi, wi in rule24:
                    for yj, wj in rule24:
                        px, py = a / 2 * (xi + 1), b / 2 * (yj + 1)
                        total += (
                            wi
                            * wj
                            * source_fn(px, py)
                            * math.sin(p * math.pi * px / a)
                            * math.sin(q * math.pi * py / b)
                        )
                coefficients[(p, q)] = (
                    (4 / (a * b))
                    * (a / 2)
                    * (b / 2)
                    * total
                    / ((p * math.pi / a) ** 2 + (q * math.pi / b) ** 2)
                )

        def series(px: float, py: float, count: int) -> float:
            top = min(count, 24)
            return sum(
                c * math.sin(p * math.pi * px / a) * math.sin(q * math.pi * py / b)
                for (p, q), c in coefficients.items()
                if p <= top and q <= top
            )

    points = [(ep.number(ep.expr(px)), ep.number(ep.expr(py))) for px, py in prob.points] or [
        (0.5 * a, 0.5 * b),
        (0.3 * a, 0.5 * b),
        (0.5 * a, 0.7 * b),
        (0.7 * a, 0.3 * b),
    ]
    for px, py in points:
        if not (0.08 * a <= px <= 0.92 * a and 0.08 * b <= py <= 0.92 * b):
            raise Refused(
                Status.UNSUPPORTED,
                "sample points must be inside the rectangle, away from its edges (a series near an edge "
                "converges slowly and a corner with discontinuous data not at all)",
                "bad_declaration",
            )
    coarse = _sor(24, 24, a, b, edge_fn, source_fn)
    fine = _sor(48, 48, a, b, edge_fn, source_fn)
    items: list[EMItem] = []
    worst = 0.0
    scale = max(1e-12, max(abs(series(px, py, terms)) for px, py in points))
    for px, py in points:
        value = series(px, py, terms)
        half = series(px, py, max(4, terms // 2)) if source_fn is None else series(px, py, 15)
        fd = _grid_value(fine, a, b, px, py)
        fd_coarse = _grid_value(coarse, a, b, px, py)
        fd_error = abs(fd - fd_coarse) / 3  # Richardson: second order, h halved
        tail = abs(value - half)
        allowed = 3 * fd_error + tail + 2e-3 * scale
        gap = abs(value - fd)
        worst = max(worst, gap / max(scale, 1e-12))
        if gap > allowed:
            result.status = Status.NOT_VERIFIED
            result.message = (
                f"the series and the finite-difference relaxation disagree at ({_fmt(px)}, {_fmt(py)}): "
                f"{_fmt(value)} against {_fmt(fd)} (allowed {allowed:.1e}); nothing is shown"
            )
            diag.reason = "solver_disagreement"
            return
        items.append(
            EMItem(
                label=f"phi({_fmt(px)}, {_fmt(py)})",
                numeric=_fmt(value),
                note=f"finite differences {_fmt(fd)}, grid error about {fd_error:.1e}, series tail {tail:.1e}",
            )
        )
    for name, coef in closed.items():
        if coef is not None:
            items.insert(
                0,
                EMItem(
                    label=f"Sine coefficient of the {name} edge", latex=rf"A_n = {_latex(coef)}"
                ),
            )
    method = (
        "double sine series for Poisson's equation with zero boundary values"
        if source_fn is not None
        else "separated series sum_n A_n sin(n pi x / a) sinh(n pi y / a) / sinh(n pi b / a) and its three rotations, one per edge"
    )
    check = (
        f"the series (truncated at {terms} terms) agrees with a finite-difference relaxation on two grids "
        f"at {len(points)} interior points (largest relative gap {worst:.1e}; grid error by Richardson)"
    )
    result.em = EMComputation(
        kind="boundary_value",
        items=items,
        method=method,
        checks=[check],
        approximate=True,
        units="potential in the units of the data",
        series_terms=terms,
        agreement=worst,
    )
    _ok(result, "numerical approximation", [check])


# -- a candidate solution --------------------------------------------------------------------------------
def _check(ctx: Context, prob: CheckProblem, result: VectorResult, diag: VectorDiagnostics) -> None:
    ep = EMProblem(ctx)
    phi = ep.expr(prob.potential, ("x", "y", "z"))
    source = ep.expr(prob.source, ("x", "y", "z"))
    xs = sp.symbols("x y z", real=True)
    mapping = {
        s: xs[["x", "y", "z"].index(s.name)]
        for s in phi.free_symbols | source.free_symbols
        if s.name in ("x", "y", "z")
    }
    phi, source = phi.xreplace(mapping), source.xreplace(mapping)
    items: list[EMItem] = []
    holds = True
    # exact residual of laplacian(phi) + source
    residual = sp.simplify(sum((sp.diff(phi, v, 2) for v in xs), sp.Integer(0)) + source)
    exact_ok = exact_zero(residual)
    # the independent reading: a five-point finite-difference Laplacian of the candidate as a number
    # (step doubled to see that the value is stable), compared with minus the source at random points
    lo0 = [ep.number(ep.expr(a)) for a, _ in prob.box.bounds]
    hi0 = [ep.number(ep.expr(b)) for _, b in prob.box.bounds]
    f_phi = sp.lambdify(xs, phi.subs(ep.sample), modules="math")
    f_src = sp.lambdify(xs, source.subs(ep.sample), modules="math")
    rng0 = random.Random(11)  # noqa: S311
    numeric_ok = True
    for _ in range(6):
        pt = [
            rng0.uniform(lo0[k] + 0.2 * (hi0[k] - lo0[k]), hi0[k] - 0.2 * (hi0[k] - lo0[k]))
            for k in range(3)
        ]
        size0 = max(h - low for low, h in zip(lo0, hi0, strict=True))

        def fd_lap(h: float, pt: list[float] = pt) -> float:
            base = float(f_phi(*pt))
            total = -6 * base
            for k in range(3):
                for sign in (1, -1):
                    q = list(pt)
                    q[k] += sign * h
                    total += float(f_phi(*q))
            return total / (h * h)

        h1 = 1e-3 * size0
        lap1, lap2 = fd_lap(h1), fd_lap(2 * h1)
        target0 = -float(f_src(*pt))
        scale0 = max(1.0, abs(target0), abs(lap1))
        if abs(lap1 - lap2) > 1e-3 * scale0:
            continue  # the difference quotient is not stable here: skipped, never failed
        if abs(lap1 - target0) > 1e-3 * scale0:
            numeric_ok = False
    laplace_holds = exact_ok and numeric_ok
    holds = holds and laplace_holds
    items.append(
        EMItem(
            label="Residual of laplacian(phi) + source",
            latex=_latex(residual) if not exact_ok else "0",
            note=(
                "exactly zero, and the independent finite-difference reading agrees"
                if laplace_holds
                else "not zero: the candidate does not satisfy the equation"
            ),
        )
    )
    lo = [ep.number(ep.expr(a)) for a, _ in prob.box.bounds]
    hi = [ep.number(ep.expr(b)) for _, b in prob.box.bounds]
    bounds = {n: (ep.expr(a), ep.expr(b)) for n, (a, b) in zip("xyz", prob.box.bounds, strict=True)}
    rng = random.Random(7)  # noqa: S311
    for cond in prob.conditions:
        axis, side = cond.face[0], cond.face[2:]
        fixed = bounds[axis][0 if side == "lo" else 1]
        others = [v for v in "xyz" if v != axis]
        target = ep.expr(cond.value, tuple(others))
        target = target.xreplace(
            {s: xs["xyz".index(s.name)] for s in target.free_symbols if s.name in "xyz"}
        )
        on_face = phi.xreplace({xs["xyz".index(axis)]: fixed})
        difference = sp.simplify(on_face - target)
        symbolic_ok = exact_zero(difference)
        numeric_ok_face = True
        f = sp.lambdify(xs, (on_face - target).subs(ep.sample), modules="math")
        for _ in range(6):
            point = [rng.uniform(lo[i], hi[i]) for i in range(3)]
            point["xyz".index(axis)] = float(ep.number(fixed))
            try:
                if abs(f(*point)) > 1e-9 * max(
                    1.0,
                    abs(
                        float(ep.number(on_face.subs(dict(zip(xs, point, strict=True)))))
                        if on_face.free_symbols <= set(xs)
                        else 1.0
                    ),
                ):
                    numeric_ok_face = False
            except (ZeroDivisionError, ValueError, TypeError):
                numeric_ok_face = False
        ok = symbolic_ok and numeric_ok_face
        holds = holds and ok
        items.append(
            EMItem(
                label=f"Boundary value on {cond.face}",
                latex=_latex(difference) if not symbolic_ok else "0",
                note="matches the stated value (exact difference zero, sampled points agree)"
                if ok
                else "does not match the stated value",
            )
        )
    result.em = EMComputation(
        kind="check_solution",
        items=items,
        method="exact residual, sampled boundary values, independent finite-difference reading",
        checks=[
            "the exact residual and the boundary differences were also read by Cartesian finite differences"
        ],
        approximate=False,
        holds=holds,
    )
    result.status = Status.OK
    result.message = (
        (
            "the candidate satisfies the equation and every stated boundary value"
            if holds
            else "the candidate does NOT satisfy every requirement: see which"
        )
        + "; a check of this candidate, not a proof that it is the only solution (uniqueness needs the boundary conditions to determine it)"
    )


def _ok(result: VectorResult, kind: str, checks: list[str]) -> None:
    result.status = Status.OK
    result.message = f"{kind}; " + "; ".join(checks)


def run_em_compute(ctx: Context, result: VectorResult, diag: VectorDiagnostics) -> None:
    problem = ctx.inp.em
    if problem is None:
        raise Refused(Status.PARSE_ERROR, "give the problem in `em`", "missing_geometry")
    if ctx.system.name != "cartesian":
        raise Refused(
            Status.UNSUPPORTED,
            "these electrostatics tools work in Cartesian coordinates",
            "unsupported_expression",
        )
    diag.result_kind = "scalar"
    if isinstance(problem, FieldProblem):
        _field(ctx, problem, result, diag)
    elif isinstance(problem, MultipoleProblem):
        _multipole(ctx, problem, result, diag)
    elif isinstance(problem, BoundaryProblem):
        _boundary(ctx, problem, result, diag)
    else:
        _check(ctx, problem, result, diag)


_UNUSED: tuple[Any, ...] = ()
