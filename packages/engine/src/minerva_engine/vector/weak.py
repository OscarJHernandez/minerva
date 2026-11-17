"""Independent numerical pairings of a point-source field with test functions on R^3.

A Dirac delta is zero at every point a sampler can draw, so a distributional identity can only be tested
*against test functions*: `<div F, phi> = -int F . grad(phi) d^3x`, `<lap f, phi> = int f lap(phi) d^3x`.
This module computes those integrals by quadrature in spherical shells centred on the source, from a
*callable that reads the user's input field*, and knows nothing about how the coefficient or the source were
found (`singular_fields.match_kernel`): it cannot agree with a wrong coefficient by sharing the mistake.

The test functions are Gaussian windows times polynomials (rapidly decreasing, smooth: they stand in for
compactly supported functions, whose pairings differ by far less than the tolerance), with different values
at the source, including 0, and one centred away from it. The quadrature (composite Gauss-Legendre in the
radius, Gauss-Legendre in cos(theta) times the trapezoid rule in the azimuth) is spectrally accurate for
these integrands. Agreement is evidence on sampled constants, never a proof.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import sympy as sp

Array = npt.NDArray[np.float64]
Point = tuple[float, float, float]
VectorField = Callable[[Array, Array, Array], tuple[Array, Array, Array]]
ScalarField = Callable[[Array, Array, Array], Array]

PAIRING_TOLERANCE = 1e-7  # relative to max(1, |expected|), the same as the one-dimensional pairing
RADIAL_PANELS = 14
RADIAL_NODES = 16
THETA_NODES = 48
PHI_NODES = 64
EPS_LEVELS = (0.4, 0.2, 0.1)

X, Y, Z = sp.symbols("x y z", real=True)


@dataclass(frozen=True)
class TestFunction:
    """A smooth, rapidly decreasing function of (x, y, z); pairings use its analytic derivatives."""

    __test__ = False  # not a pytest class

    label_latex: str
    expr: sp.Expr
    center: Point
    width: float

    def value_at(self, p: Point) -> float:
        return float(self.expr.subs({X: p[0], Y: p[1], Z: p[2]}))

    def reach(self, source: Point) -> float:
        """A radius around `source` beyond which the function is below 1e-14 of its peak."""
        return math.dist(self.center, source) + 9.0 * self.width

    def gradient(self) -> tuple[ScalarField, ScalarField, ScalarField]:
        return tuple(  # type: ignore[return-value]
            _vectorised(sp.lambdify((X, Y, Z), sp.diff(self.expr, v), modules="numpy"))
            for v in (X, Y, Z)
        )

    def laplacian(self) -> ScalarField:
        lap = sum((sp.diff(self.expr, v, 2) for v in (X, Y, Z)), sp.Integer(0))
        return _vectorised(sp.lambdify((X, Y, Z), lap, modules="numpy"))

    def function(self) -> ScalarField:
        return _vectorised(sp.lambdify((X, Y, Z), self.expr, modules="numpy"))


def _vectorised(fn: Callable[..., object]) -> ScalarField:
    def call(x: Array, y: Array, z: Array) -> Array:
        return np.broadcast_to(np.asarray(fn(x, y, z), dtype=np.float64), x.shape).copy()

    return call


def standard_test_functions(source: Point) -> list[TestFunction]:
    """Four test functions for a point source at `source`, with phi(source) = 1, 2, 0.68... and 0."""
    sx, sy, sz = source
    u, v, w = X - sx, Y - sy, Z - sz
    r2 = u**2 + v**2 + w**2
    radial = sp.exp(-r2 / (2 * sp.Float(1.1) ** 2))
    skew = (
        2 + sp.Rational(1, 2) * u - sp.Rational(3, 10) * v + sp.Rational(2, 5) * u * w
    ) * sp.exp(-r2 / (2 * sp.Float(0.9) ** 2))
    c = (sx + 0.5, sy - 0.4, sz + 0.3)
    away = (1 + sp.Rational(3, 10) * (Z - c[2])) * sp.exp(
        -((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) / (2 * sp.Float(0.8) ** 2)
    )
    at_zero = u * sp.exp(-r2 / 2)
    return [
        TestFunction(r"e^{-|\mathbf r-\mathbf a|^2/2.42}", radial, source, 1.1),
        TestFunction(
            r"\left(2+\tfrac12 u-\tfrac3{10}v+\tfrac25 uw\right)e^{-|\mathbf r-\mathbf a|^2/1.62}",
            skew,
            source,
            0.9,
        ),
        TestFunction(
            r"\left(1+\tfrac3{10}(z-c_z)\right)e^{-|\mathbf r-\mathbf c|^2/1.28},\ "
            r"\mathbf c=\mathbf a+(0.5,-0.4,0.3)",
            away,
            c,
            0.8,
        ),
        TestFunction(r"u\,e^{-|\mathbf r-\mathbf a|^2/2}", at_zero, source, 1.0),
    ]


# -- quadrature nodes -------------------------------------------------------------------------------
def _gauss(n: int) -> tuple[Array, Array]:
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes.astype(np.float64), weights.astype(np.float64)


def radial_rule(edges: list[float]) -> tuple[Array, Array]:
    """Composite Gauss-Legendre nodes and weights on consecutive intervals of `edges`."""
    t, w = _gauss(RADIAL_NODES)
    nodes: list[Array] = []
    weights: list[Array] = []
    for lo, hi in itertools.pairwise(edges):
        half = (hi - lo) / 2.0
        nodes.append(lo + half * (t + 1.0))
        weights.append(half * w)
    return np.concatenate(nodes), np.concatenate(weights)


def uniform_edges(rmax: float) -> list[float]:
    return [rmax * i / RADIAL_PANELS for i in range(RADIAL_PANELS + 1)]


def sphere_rule() -> tuple[Array, Array, Array, Array]:
    """Directions on the unit sphere: (nx, ny, nz, weights), the weights summing to 4 pi."""
    ct, wt = _gauss(THETA_NODES)
    phi = (np.arange(PHI_NODES) + 0.5) * (2.0 * math.pi / PHI_NODES)
    st = np.sqrt(1.0 - ct**2)
    nx = (st[:, None] * np.cos(phi)[None, :]).ravel()
    ny = (st[:, None] * np.sin(phi)[None, :]).ravel()
    nz = np.repeat(ct, PHI_NODES)
    return nx, ny, nz, np.repeat(wt, PHI_NODES) * (2.0 * math.pi / PHI_NODES)


@dataclass(frozen=True)
class Shells:
    """Points `a + rho n` for every (rho, n) with the weights of `int g d^3x = sum g w`."""

    x: Array
    y: Array
    z: Array
    weight: Array


def shells(source: Point, edges: list[float]) -> Shells:
    rho, w_rho = radial_rule(edges)
    nx, ny, nz, wn = sphere_rule()
    r = rho[:, None]
    return Shells(
        source[0] + r * nx[None, :],
        source[1] + r * ny[None, :],
        source[2] + r * nz[None, :],
        (w_rho * rho**2)[:, None] * wn[None, :],
    )


# -- pairings ---------------------------------------------------------------------------------------
def pair_divergence(field: VectorField, phi: TestFunction, source: Point) -> float:
    """`<div F, phi> = -int F . grad(phi) d^3x`, with F read from the input."""
    grid = shells(source, uniform_edges(phi.reach(source)))
    fx, fy, fz = field(grid.x, grid.y, grid.z)
    gx, gy, gz = (g(grid.x, grid.y, grid.z) for g in phi.gradient())
    return float(-np.sum((fx * gx + fy * gy + fz * gz) * grid.weight))


def pair_laplacian(f: ScalarField, phi: TestFunction, source: Point) -> float:
    """`<lap f, phi> = int f lap(phi) d^3x`, with f read from the input."""
    grid = shells(source, uniform_edges(phi.reach(source)))
    return float(
        np.sum(f(grid.x, grid.y, grid.z) * phi.laplacian()(grid.x, grid.y, grid.z) * grid.weight)
    )


def flux_through_sphere(field: VectorField, source: Point, radius: float) -> float:
    """`oint F . n dS` over the sphere of the given radius about `source`, F read from the input."""
    nx, ny, nz, wn = sphere_rule()
    fx, fy, fz = field(source[0] + radius * nx, source[1] + radius * ny, source[2] + radius * nz)
    return float(np.sum((fx * nx + fy * ny + fz * nz) * wn) * radius**2)


def regularised_pairing(
    strength: float, source: Point, phi: TestFunction, eps: float
) -> tuple[float, float]:
    """`int phi div(F_eps) d^3x` for `F_eps = C (r-a)/(|r-a|^2 + eps^2)^(3/2)`, and the largest relative
    difference between the divergence taken by central differences of `F_eps` and the closed form
    `3 C eps^2 / (rho^2 + eps^2)^(5/2)` (a guard on the quadrature, not part of the result)."""
    rmax = phi.reach(source)
    edges = [0.0]
    step = eps / 2.0
    while edges[-1] < rmax:
        edges.append(min(edges[-1] + step, rmax))
        step *= 1.6
    grid = shells(source, edges)
    sx, sy, sz = source
    h = 1e-4 * eps

    def f_eps(x: Array, y: Array, z: Array) -> tuple[Array, Array, Array]:
        q = ((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2 + eps**2) ** 1.5
        return strength * (x - sx) / q, strength * (y - sy) / q, strength * (z - sz) / q

    div = (
        (f_eps(grid.x + h, grid.y, grid.z)[0] - f_eps(grid.x - h, grid.y, grid.z)[0])
        + (f_eps(grid.x, grid.y + h, grid.z)[1] - f_eps(grid.x, grid.y - h, grid.z)[1])
        + (f_eps(grid.x, grid.y, grid.z + h)[2] - f_eps(grid.x, grid.y, grid.z - h)[2])
    ) / (2 * h)
    rho2 = (grid.x - sx) ** 2 + (grid.y - sy) ** 2 + (grid.z - sz) ** 2
    closed = 3.0 * strength * eps**2 / (rho2 + eps**2) ** 2.5
    scale = np.maximum(np.abs(closed), 1e-300)
    deviation = float(np.max(np.abs(div - closed) / scale))
    value = float(np.sum(div * phi.function()(grid.x, grid.y, grid.z) * grid.weight))
    return value, deviation
