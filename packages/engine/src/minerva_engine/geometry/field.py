"""Electromagnetic field tensors on a metric supplied by the user.

Four quantities act on the user's tensor field (components in the coordinates, all indices *down*):

* `exterior_derivative`: `F_{mn} = d_m A_n - d_n A_m` for a 1-form, `(dF)_{lmn} = d_l F_{mn} + d_m F_{nl} +
  d_n F_{lm}` for an antisymmetric 2-form. Needs no metric or connection.
* `antisymmetric_divergence`: `J^n = nabla_m F^{mn}` for an antisymmetric `F` (Levi-Civita connection).
* `field_invariant`: `F_{mn} F^{mn}` (in flat Minkowski space with signature (-+++) and c = 1 this is
  `2 (B^2 - E^2)`).
* `hodge_dual`: `(*F)_{mn} = 1/2 eps_{mnrs} F^{rs}` in four dimensions with `eps_{0123} = +sqrt|det g|` in the
  order the coordinates were given. Needs the sign of `det g` to be decided by the declared assumptions.

The signature, the metric and the units are always the user's: nothing is assumed (no `c`, no signature).
Every quantity is compared with a numerical recomputation that shares no code with the symbolic one
(finite differences of the lambdified components, the metric inverted numerically). Identities that hold
for the result are checked as separate legs: `d(dA) = 0`, a second formula for the divergence
(`|g|^{-1/2} d_m (|g|^{1/2} F^{mn})`), and `** = sign(det g)` on two-forms in four dimensions.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

import mpmath as mp
import sympy as sp

from ..models import Status
from .compute import Workspace
from .models import Check, GeometryVerification
from .numeric import Evaluator, Point
from .setup import GeometryError
from .tensors import Tensor

FIELD_QUANTITIES = frozenset(
    {"exterior_derivative", "antisymmetric_divergence", "field_invariant", "hodge_dual"}
)

FIELD_CONVENTIONS = [
    r"Exterior derivative: (dA)_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu and "
    r"(dF)_{\lambda\mu\nu} = \partial_\lambda F_{\mu\nu} + \partial_\mu F_{\nu\lambda} + \partial_\nu F_{\lambda\mu}",
    r"Divergence of an antisymmetric tensor: J^{\nu} = \nabla_\mu F^{\mu\nu}; F^{\mu\nu} is F with both "
    "indices raised by the inverse metric you gave",
    r"Invariant F_{\mu\nu}F^{\mu\nu}; in Minkowski space with signature (-+++) and c = 1 it equals "
    r"2(B^2 - E^2)",
    r"Hodge dual in four dimensions: (\star F)_{\mu\nu} = \tfrac12 \epsilon_{\mu\nu\rho\sigma}F^{\rho\sigma} "
    r"with \epsilon_{0123} = +\sqrt{|\det g|}, the indices in the order of the coordinates you gave "
    "(the opposite orientation changes the sign)",
    "The signature, the metric and the units are yours; nothing here assumes c = 1 or a signature",
]


# -- symbolic side ------------------------------------------------------------------------------
def _require(ws: Workspace, kind: str) -> Tensor:
    geo = ws.geo
    t = geo.tensor
    if t is None:
        raise GeometryError(Status.PARSE_ERROR, f"{kind.replace('_', ' ')} needs a tensor field.")
    if any(v != "down" for v in t.variance):
        raise GeometryError(
            Status.PARSE_ERROR,
            "Give the covariant components (all indices down), for example A_mu or F_{mu nu}.",
        )
    if kind == "exterior_derivative":
        if t.rank not in (1, 2):
            raise GeometryError(Status.PARSE_ERROR, "d acts here on a 1-form or a 2-form.")
    elif t.rank != 2:
        raise GeometryError(Status.PARSE_ERROR, f"{kind.replace('_', ' ')} needs a rank-2 tensor.")
    if t.rank == 2:
        for i in range(t.n):
            for j in range(i, t.n):
                if ws.tidy(t.get((i, j)) + t.get((j, i))) != 0:
                    raise GeometryError(
                        Status.PARSE_ERROR,
                        "The tensor is not antisymmetric (F_{ij} + F_{ji} is not zero), so it is not a "
                        "field strength; nothing is computed.",
                    )
    if kind == "hodge_dual" and t.n != 4:
        raise GeometryError(
            Status.UNSUPPORTED, "The Hodge dual of a 2-form is implemented in four dimensions only."
        )
    return t


def exterior_derivative(ws: Workspace) -> Tensor:
    def produce() -> Tensor:
        t = _require(ws, "exterior_derivative")
        x, n = ws.geo.coords, t.n
        out = Tensor(n, ("down", "down") if t.rank == 1 else ("down", "down", "down"))
        if t.rank == 1:
            for m in range(n):
                for k in range(n):
                    out.set(
                        (m, k), ws.tidy(sp.diff(t.get((k,)), x[m]) - sp.diff(t.get((m,)), x[k]))
                    )
        else:
            for a in range(n):
                for b in range(n):
                    for c in range(n):
                        total = (
                            sp.diff(t.get((b, c)), x[a])
                            + sp.diff(t.get((c, a)), x[b])
                            + sp.diff(t.get((a, b)), x[c])
                        )
                        out.set((a, b, c), ws.tidy(total))
        return out

    result: Tensor = ws.stage("exterior_derivative", produce)
    return result


def _raised(ws: Workspace, t: Tensor) -> sp.Matrix:
    ginv = ws.inverse()
    f = sp.Matrix(t.n, t.n, lambda i, j: t.get((i, j)))
    return (ginv * f * ginv.T).applyfunc(ws.tidy)


def antisymmetric_divergence(ws: Workspace) -> Tensor:
    def produce() -> Tensor:
        t = _require(ws, "antisymmetric_divergence")
        n, x = t.n, ws.geo.coords
        up = _raised(ws, t)
        gamma = ws.christoffel()
        out = Tensor(n, ("up",))
        for nu in range(n):
            total = sum((sp.diff(up[m, nu], x[m]) for m in range(n)), sp.S.Zero)
            for m in range(n):
                for lam in range(n):
                    total += gamma.get((m, m, lam)) * up[lam, nu]
            out.set((nu,), ws.tidy(total))
        return out

    result: Tensor = ws.stage("antisymmetric_divergence", produce)
    return result


def field_invariant(ws: Workspace) -> sp.Expr:
    def produce() -> sp.Expr:
        t = _require(ws, "field_invariant")
        up = _raised(ws, t)
        total = sum((t.get((i, j)) * up[i, j] for i in range(t.n) for j in range(t.n)), sp.S.Zero)
        return ws.tidy(total)

    value: sp.Expr = ws.stage("field_invariant", produce)
    return value


def volume_factor(ws: Workspace) -> tuple[sp.Expr, int]:
    """`sqrt|det g|` and the sign of `det g` (+1 or -1), both decided by the declared assumptions."""
    det = ws.determinant()
    if det.is_extended_negative or det.is_nonpositive:
        return sp.sqrt(-det), -1
    if det.is_extended_positive or det.is_nonnegative:
        return sp.sqrt(det), 1
    raise GeometryError(
        Status.UNSUPPORTED,
        "The sign of det g is not decided by the declared assumptions, so sqrt|det g| (and the Hodge "
        "dual) is not defined here. Declare the coordinates and parameters so that it is.",
    )


def _levi_civita(n: int) -> dict[tuple[int, ...], int]:
    out = {}
    for perm in permutations(range(n)):
        inversions = sum(1 for i in range(n) for j in range(i + 1, n) if perm[i] > perm[j])
        out[perm] = -1 if inversions % 2 else 1
    return out


def hodge_dual(ws: Workspace) -> Tensor:
    def produce() -> Tensor:
        t = _require(ws, "hodge_dual")
        root, _ = volume_factor(ws)
        up = _raised(ws, t)
        out = Tensor(4, ("down", "down"))
        eps = _levi_civita(4)
        for m in range(4):
            for n in range(4):
                total = sp.S.Zero
                for (a, b, c, d), sign in eps.items():
                    if (a, b) == (m, n):
                        total += sign * up[c, d]
                out.set((m, n), ws.tidy(root * total / 2))
        return out

    result: Tensor = ws.stage("hodge_dual", produce)
    return result


# -- numerical side ------------------------------------------------------------------------------
class FieldVerifier:
    """Independent numerical legs; `owner` is the geometry `Verifier` (sampling, runners, evaluators)."""

    def __init__(self, owner: Any) -> None:
        self.o = owner
        self.ws: Workspace = owner.ws
        self.n: int = owner.geo.n

    def _tensor_at(self, ev: Evaluator, point: Point) -> dict[Any, Any]:
        t = self.ws.geo.tensor
        assert t is not None  # noqa: S101 - guarded by the caller
        found: dict[Any, Any] = self.o._cand(ev, "tensor", t, point)
        return found

    def _raised_num(self, ev: Evaluator, point: Point) -> mp.matrix:
        x, p = point
        n = self.n
        ginv = ev.inverse_metric(x, p)
        tc = self._tensor_at(ev, point)
        f = mp.matrix(n, n)
        for i in range(n):
            for j in range(n):
                f[i, j] = tc.get((i, j), 0)
        return ginv * f * ginv.T

    def _shifted(self, ev: Evaluator, point: Point, a: int, h: Any) -> Point:
        return (ev._shift(point[0], a, h), point[1])

    def exterior_derivative(self) -> GeometryVerification:
        t = self.ws.geo.tensor
        assert t is not None  # noqa: S101 - guarded by the caller
        result = exterior_derivative(self.ws)
        n = self.n

        def fn(ev: Evaluator, point: Point, h: Any) -> list[tuple[Any, ...]]:
            cand = self.o._cand(ev, "exterior_derivative", result, point)
            d = []
            for a in range(n):
                plus = self._tensor_at(ev, self._shifted(ev, point, a, h))
                minus = self._tensor_at(ev, self._shifted(ev, point, a, -h))
                keys = set(plus) | set(minus)
                d.append({k: (plus.get(k, 0) - minus.get(k, 0)) / (2 * h) for k in keys})
            out: list[tuple[Any, ...]] = []
            for idx in result.indices():
                if t.rank == 1:
                    m, k = idx
                    ref = d[m].get((k,), 0) - d[k].get((m,), 0)
                    size = abs(d[m].get((k,), 0)) + abs(d[k].get((m,), 0))
                else:
                    a, b, c = idx
                    ref = d[a].get((b, c), 0) + d[b].get((c, a), 0) + d[c].get((a, b), 0)
                    size = sum(
                        (
                            abs(d[a].get((b, c), 0)),
                            abs(d[b].get((c, a), 0)),
                            abs(d[c].get((a, b), 0)),
                        )
                    )
                out.append((cand.get(idx, 0), ref, size))
            return out

        def closed(ev: Evaluator, point: Point, h: Any) -> list[tuple[Any, ...]]:
            """d(dA) = 0, computed from the candidate F by finite differences (a different quantity)."""
            d = []
            for a in range(n):
                plus = self.o._cand(
                    ev, "exterior_derivative", result, self._shifted(ev, point, a, h)
                )
                minus = self.o._cand(
                    ev, "exterior_derivative", result, self._shifted(ev, point, a, -h)
                )
                keys = set(plus) | set(minus)
                d.append({k: (plus.get(k, 0) - minus.get(k, 0)) / (2 * h) for k in keys})
            out: list[tuple[Any, ...]] = []
            for a in range(n):
                for b in range(a + 1, n):
                    for c in range(b + 1, n):
                        total = d[a].get((b, c), 0) + d[b].get((c, a), 0) + d[c].get((a, b), 0)
                        size = (
                            abs(d[a].get((b, c), 0))
                            + abs(d[b].get((c, a), 0))
                            + abs(d[c].get((a, b), 0))
                        )
                        out.append((total, 0, size))
            return out

        checks = [self.o.numeric("agrees with finite differences of the components", fn)]
        if t.rank == 1 and n >= 3:
            checks.append(self.o.numeric("dF = 0 for F = dA, by finite differences of F", closed))
            x = self.ws.geo.coords
            checks.append(
                self.o.symbolic(
                    "dF = 0 symbolically",
                    lambda: [
                        e
                        for e in (
                            sp.diff(result.get((b, c)), x[a])
                            + sp.diff(result.get((c, a)), x[b])
                            + sp.diff(result.get((a, b)), x[c])
                            for a in range(n)
                            for b in range(a + 1, n)
                            for c in range(b + 1, n)
                        )
                        if e != 0
                    ],
                )
            )
        return _combine(checks)

    def antisymmetric_divergence(self) -> GeometryVerification:
        result = antisymmetric_divergence(self.ws)
        n = self.n

        def connection_form(ev: Evaluator, point: Point, h: Any) -> list[tuple[Any, ...]]:
            x, p = point
            cand = self.o._cand(ev, "antisymmetric_divergence", result, point)
            gam = ev.christoffel(x, p, h)
            up = self._raised_num(ev, point)
            d_up = []
            for a in range(n):
                plus = self._raised_num(ev, self._shifted(ev, point, a, h))
                minus = self._raised_num(ev, self._shifted(ev, point, a, -h))
                d_up.append((plus - minus) / (2 * h))
            out: list[tuple[Any, ...]] = []
            for nu in range(n):
                total = mp.mpf(0)
                size = mp.mpf(0)
                for m in range(n):
                    term = d_up[m][m, nu]
                    total += term
                    size += abs(term)
                    for lam in range(n):
                        term = gam.data[(m, m, lam)] * up[lam, nu]
                        total += term
                        size += abs(term)
                out.append((cand.get((nu,), 0), total, size))
            return out

        def density_form(ev: Evaluator, point: Point, h: Any) -> list[tuple[Any, ...]]:
            """|g|^{-1/2} d_m (|g|^{1/2} F^{mn}): a different formula, no Christoffel symbols."""
            cand = self.o._cand(ev, "antisymmetric_divergence", result, point)

            def density(pt: Point) -> mp.mpf:
                return mp.sqrt(abs(mp.det(mp.matrix(ev.metric(pt[0], pt[1])))))

            root = density(point)
            out: list[tuple[Any, ...]] = []
            for nu in range(n):
                total = mp.mpf(0)
                size = mp.mpf(0)
                for m in range(n):
                    plus_pt, minus_pt = (
                        self._shifted(ev, point, m, h),
                        self._shifted(ev, point, m, -h),
                    )
                    plus = density(plus_pt) * self._raised_num(ev, plus_pt)[m, nu]
                    minus = density(minus_pt) * self._raised_num(ev, minus_pt)[m, nu]
                    term = (plus - minus) / (2 * h) / root
                    total += term
                    size += abs(term)
                out.append((cand.get((nu,), 0), total, size))
            return out

        checks = [
            self.o.numeric(
                "agrees with finite differences plus the numerical connection", connection_form
            ),
            self.o.numeric("agrees with |g|^(-1/2) d_m (|g|^(1/2) F^(mn))", density_form),
            Check(
                name="independent symbolic route",
                kind="symbolic",
                passed=None,
                note="not run: a divergence has no independent symbolic route here",
            ),
        ]
        return _combine(checks)

    def field_invariant(self) -> GeometryVerification:
        t = self.ws.geo.tensor
        assert t is not None  # noqa: S101 - guarded by the caller
        value = field_invariant(self.ws)
        n = self.n

        def fn(ev: Evaluator, point: Point, h: Any) -> list[tuple[Any, ...]]:
            up = self._raised_num(ev, point)
            tc = self._tensor_at(ev, point)
            ref = sum((tc.get((i, j), 0) * up[i, j] for i in range(n) for j in range(n)), mp.mpf(0))
            size = sum(
                (abs(tc.get((i, j), 0) * up[i, j]) for i in range(n) for j in range(n)),
                mp.mpf(0),
            )
            cand = ev.candidate("field_invariant", [((), value)], point)[()]
            return [(cand, ref, size)]

        def trace_form() -> list[sp.Expr]:
            ginv = self.ws.inverse()
            f = sp.Matrix(n, n, lambda i, j: t.get((i, j)))
            return [self.ws.tidy(value + (ginv * f * ginv * f).trace())]

        return _combine(
            [
                self.o.numeric("agrees with the numerical contraction", fn),
                self.o.symbolic("equals -tr(g^-1 F g^-1 F)", trace_form),
            ]
        )

    def hodge_dual(self) -> GeometryVerification:
        result = hodge_dual(self.ws)
        eps = _levi_civita(4)

        def dual(ev: Evaluator, point: Point, comps: dict[Any, Any]) -> dict[Any, Any]:
            x, p = point
            ginv = ev.inverse_metric(x, p)
            f = mp.matrix(4, 4)
            for (i, j), v in comps.items():
                f[i, j] = v
            up = ginv * f * ginv.T
            det = mp.det(mp.matrix(ev.metric(x, p)))
            root = mp.sqrt(abs(det))
            out = {}
            for m in range(4):
                for n in range(4):
                    total = mp.mpf(0)
                    for (a, b, c, d), sign in eps.items():
                        if (a, b) == (m, n):
                            total += sign * up[c, d]
                    out[(m, n)] = root * total / 2
            return out

        def fn(ev: Evaluator, point: Point, h: Any) -> list[tuple[Any, ...]]:
            cand = self.o._cand(ev, "hodge_dual", result, point)
            ref = dual(ev, point, self._tensor_at(ev, point))
            size = max((abs(v) for v in ref.values()), default=mp.mpf(0))
            return [(cand.get((m, n), 0), ref[(m, n)], size) for m in range(4) for n in range(4)]

        def involution(ev: Evaluator, point: Point, h: Any) -> list[tuple[Any, ...]]:
            """** = sign(det g) on 2-forms in four dimensions, applied to the candidate."""
            x, p = point
            sign = 1 if mp.det(mp.matrix(ev.metric(x, p))) > 0 else -1
            cand = self.o._cand(ev, "hodge_dual", result, point)
            twice = dual(ev, point, cand)
            original = self._tensor_at(ev, point)
            size = max((abs(v) for v in twice.values()), default=mp.mpf(0))
            return [
                (twice[(m, n)], sign * original.get((m, n), 0), size)
                for m in range(4)
                for n in range(4)
            ]

        return _combine(
            [
                self.o.numeric("agrees with the numerical Levi-Civita contraction", fn),
                self.o.numeric("** = sign(det g) on 2-forms", involution),
            ]
        )


def _combine(checks: list[Check]) -> GeometryVerification:
    from .verify import NUMERIC_METHOD
    from .verify import _combine as combine

    return combine(NUMERIC_METHOD, checks)
