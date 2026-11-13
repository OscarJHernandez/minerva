"""Verification of geometric results.

A quantity is shown only if every numeric leg passed. The numeric legs use `numeric.py` (finite differences
of the lambdified metric, no SymPy differentiation) and compare either with an independent numerical
recomputation or with an identity the result must satisfy (metric compatibility, the Riemann symmetries,
the first and second Bianchi identities, the divergence-free Einstein tensor). Symbolic legs are extra
evidence, reported separately: they can show that a result is proved, and a failure there means only that
no symbolic proof was found. Nothing here says "proved" on numerical agreement.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

import mpmath as mp
import sympy as sp

from ..errors import LimitError
from ..lagrangian import DerivationError, derive_expr
from ..verify import MIN_VALID_POINTS, euler_lagrange_check
from .compute import Workspace
from .models import Check, GeometryVerification
from .numeric import (
    DPS,
    Evaluator,
    Point,
    close,
    make_evaluators,
    sample_point,
    step,
)
from .tensors import Tensor

Pairs = list[tuple[Any, Any]]
PairFn = Callable[[Evaluator, Point, Any], Pairs]

STABLE_TOL = 1e-8  # the numerical value must not move when the step is doubled
AGREE_TOL = 1e-9
SYMBOLIC_MAX_OPS = 1500  # a symbolic zero test is only attempted below this size
MAX_ATTEMPTS = 60

NUMERIC_METHOD = (
    "finite differences of the lambdified metric (mpmath, 50 digits) at random points; "
    "no SymPy differentiation"
)


class Verifier:
    def __init__(self, ws: Workspace, seed: int = 20260101) -> None:
        self.ws = ws
        self.geo = ws.geo
        self.rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
        self.evaluators = make_evaluators(self.geo, self.rng, 3)
        self._alt: dict[tuple[int, int, int, int], sp.Expr] = {}

    # -- numeric runner ------------------------------------------------------------------
    def numeric(self, name: str, fn: PairFn, *, want: int = 8, tol: float = AGREE_TOL) -> Check:
        """Run `fn` at random points. `fn(evaluator, point, h)` returns (result, reference) pairs.

        A point counts only if the metric is well conditioned there and every value is unchanged
        (to `STABLE_TOL`) when the finite-difference step is doubled; unstable points are skipped,
        so a branch cut or pole near a point cannot make a correct result fail.
        """
        checked = agreed = 0
        count = len(self.evaluators)
        want = min(want, 2 * count) if self.geo.function_classes else want
        for attempt in range(MAX_ATTEMPTS):
            if checked >= want:
                break
            self.ws.check_deadline(name)
            ev = self.evaluators[attempt % count]
            with mp.workdps(DPS):
                point = sample_point(self.geo, self.rng)
                if not ev.is_regular(point):
                    continue
                try:
                    h = step()
                    first = fn(ev, point, h)
                    second = fn(ev, point, 2 * h)
                except (ZeroDivisionError, ValueError, ArithmeticError, TypeError, OverflowError):
                    continue
                if not _usable(first, second):
                    continue
                checked += 1
                if all(close(a, b, tol) for a, b in first):
                    agreed += 1
        passed = checked >= MIN_VALID_POINTS and agreed == checked
        if checked < MIN_VALID_POINTS:
            note = (
                "too few valid sample points (the metric was singular or unstable at most points)"
            )
        elif agreed != checked:
            note = "the numerical check found a disagreement"
        else:
            note = "agreed at every sampled point"
        return Check(
            name=name,
            kind="numeric",
            passed=passed,
            points_checked=checked,
            points_agreed=agreed,
            note=note,
        )

    # -- symbolic runner ---------------------------------------------------------------------
    def symbolic(self, name: str, exprs: Callable[[], list[sp.Expr]]) -> Check:
        if time.monotonic() > self.ws.soft_deadline:
            return Check(name=name, kind="symbolic", passed=None, note="not run: time budget")
        try:
            items = exprs()
        except LimitError:
            return Check(name=name, kind="symbolic", passed=None, note="not run: expression size")
        for e in items:
            if time.monotonic() > self.ws.soft_deadline:
                return Check(name=name, kind="symbolic", passed=None, note="not run: time budget")
            verdict = self._zero(e)
            if verdict is None:
                return Check(
                    name=name, kind="symbolic", passed=None, note="not run: expression too large"
                )
            if not verdict:
                return Check(
                    name=name,
                    kind="symbolic",
                    passed=False,
                    note="no symbolic proof found (this alone does not mean the result is wrong)",
                )
        return Check(name=name, kind="symbolic", passed=True, note="difference simplifies to 0")

    def _zero(self, e: sp.Expr) -> bool | None:
        if e == 0:
            return True
        if sp.count_ops(e) > SYMBOLIC_MAX_OPS:
            return None
        c = sp.cancel(e)
        if c == 0:
            return True
        if sp.count_ops(c) > 400:
            return False
        return bool(sp.simplify(c) == 0)

    # -- shared pieces -----------------------------------------------------------------------
    def alt_lowered_riemann(self, r: int, s: int, m: int, v: int) -> sp.Expr:
        """`R_{rsmv}` from second derivatives of the metric (a different route from the definition).

        R_{rsmv} = 1/2 (d_s d_m g_{rv} + d_r d_v g_{sm} - d_s d_v g_{rm} - d_r d_m g_{sv})
                   + g_{ab} (Gamma^a_{sm} Gamma^b_{rv} - Gamma^a_{sv} Gamma^b_{rm}).
        """
        key = (r, s, m, v)
        if key not in self._alt:
            geo, n = self.geo, self.geo.n
            g, x = geo.g, geo.coords
            gamma = self.ws.christoffel()
            second = (
                sp.diff(g[r, v], x[s], x[m])
                + sp.diff(g[s, m], x[r], x[v])
                - sp.diff(g[r, m], x[s], x[v])
                - sp.diff(g[s, v], x[r], x[m])
            ) / 2
            quad = sp.S.Zero
            for a in range(n):
                for b in range(n):
                    if g[a, b] == 0:
                        continue
                    quad += g[a, b] * (
                        gamma.get((a, s, m)) * gamma.get((b, r, v))
                        - gamma.get((a, s, v)) * gamma.get((b, r, m))
                    )
            self._alt[key] = second + quad
        return self._alt[key]

    def _cand(self, ev: Evaluator, key: str, tensor: Tensor, point: Point) -> dict[Any, Any]:
        items = list(tensor.data.items())
        return ev.candidate(key, items, point) if items else {}

    # -- quantities ----------------------------------------------------------------------------
    def inverse_metric(self) -> GeometryVerification:
        geo, n = self.geo, self.geo.n
        ginv = self.ws.inverse()
        items = [((i, j), ginv[i, j]) for i in range(n) for j in range(n) if ginv[i, j] != 0]

        def fn(ev: Evaluator, point: Point, h: Any) -> Pairs:
            x, p = point
            ref = ev.inverse_metric(x, p)
            cand = ev.candidate("inverse", items, point) if items else {}
            return [(cand.get((i, j), 0), ref[i, j]) for i in range(n) for j in range(n)]

        checks = [
            self.numeric("inverse matches the numerical inverse of g", fn),
            self.symbolic(
                "g g^-1 = identity",
                lambda: [e for e in (geo.g * ginv - sp.eye(n)) if e != 0],
            ),
        ]
        return _combine(NUMERIC_METHOD, checks)

    def determinant(self) -> GeometryVerification:
        geo = self.geo
        det = self.ws.determinant()

        def fn(ev: Evaluator, point: Point, h: Any) -> Pairs:
            x, p = point
            ref = mp.det(mp.matrix(ev.metric(x, p)))
            cand = ev.candidate("det", [((), det)], point)[()]
            return [(cand, ref)]

        checks = [
            self.numeric("determinant matches the numerical determinant", fn),
            self.symbolic(
                "determinant agrees with an independent expansion",
                lambda: [det - self.ws.tidy(geo.g.det(method="bareiss"))],
            ),
        ]
        return _combine(NUMERIC_METHOD, checks)

    def christoffel(self) -> GeometryVerification:
        geo, n = self.geo, self.geo.n
        gamma = self.ws.christoffel()

        def compatibility(ev: Evaluator, point: Point, h: Any) -> Pairs:
            x, p = point
            g = ev.metric(x, p)
            dg = ev.partial_metric(x, p, h)
            gam = self._cand(ev, "christoffel", gamma, point)
            out: Pairs = []
            for s in range(n):
                for m in range(n):
                    for v in range(n):
                        residual = dg[s][m][v]
                        for lam in range(n):
                            residual -= gam.get((lam, s, m), 0) * g[lam][v]
                            residual -= gam.get((lam, s, v), 0) * g[m][lam]
                        out.append((residual, mp.mpf(0)))
                        # torsion-free: symmetric in the lower indices
                        out.append((gam.get((s, m, v), 0), gam.get((s, v, m), 0)))
            return out

        def formula(ev: Evaluator, point: Point, h: Any) -> Pairs:
            x, p = point
            ref = ev.christoffel(x, p, h)
            gam = self._cand(ev, "christoffel", gamma, point)
            return [(gam.get(k, 0), v) for k, v in ref.data.items()]

        def compat_symbolic() -> list[sp.Expr]:
            return [
                sp.diff(geo.g[m, v], geo.coords[s])
                - sum(
                    gamma.get((lam, s, m)) * geo.g[lam, v] + gamma.get((lam, s, v)) * geo.g[m, lam]
                    for lam in range(n)
                )
                for s in range(n)
                for m in range(n)
                for v in range(m, n)
            ]

        checks = [
            self.numeric("metric compatibility and no torsion", compatibility),
            self.numeric("agrees with the numerical Christoffel formula", formula),
            self.symbolic("metric compatibility (symbolic)", compat_symbolic),
        ]
        return _combine(NUMERIC_METHOD, checks)

    def riemann(self) -> GeometryVerification:
        geo, n = self.geo, self.geo.n
        riem = self.ws.riemann()
        gamma = self.ws.christoffel()

        def lowered(ev: Evaluator, point: Point, at: list[Any] | None = None) -> dict[Any, Any]:
            x, p = point
            xx = at if at is not None else x
            g = ev.metric(xx, p)
            rc = self._cand(ev, "riemann", riem, (xx, p))
            out = {}
            for r in range(n):
                for s in range(n):
                    for m in range(n):
                        for v in range(n):
                            out[(r, s, m, v)] = sum(
                                (g[r][k] * rc.get((k, s, m, v), 0) for k in range(n)), mp.mpf(0)
                            )
            return out

        def curvature(ev: Evaluator, point: Point, h: Any) -> Pairs:
            x, p = point
            ref = ev.riemann(x, p, h)
            rc = self._cand(ev, "riemann", riem, point)
            return [(rc.get(k, 0), v) for k, v in ref.data.items()]

        def symmetries(ev: Evaluator, point: Point, h: Any) -> Pairs:
            low = lowered(ev, point)
            zero = mp.mpf(0)
            out: Pairs = []
            for r, s, m, v in low:
                out.append((low[(r, s, m, v)] + low[(s, r, m, v)], zero))
                out.append((low[(r, s, m, v)] - low[(m, v, r, s)], zero))
                out.append((low[(r, s, m, v)] + low[(r, m, v, s)] + low[(r, v, s, m)], zero))
            return out

        def second_bianchi(ev: Evaluator, point: Point, h: Any) -> Pairs:
            x, p = point
            gam = ev.christoffel(x, p, h)
            base = lowered(ev, point)
            d = []
            for lam in range(n):
                plus = lowered(ev, point, ev._shift(x, lam, h))
                minus = lowered(ev, point, ev._shift(x, lam, -h))
                d.append({k: (plus[k] - minus[k]) / (2 * h) for k in plus})

            def nabla(lam: int, r: int, s: int, m: int, v: int) -> Any:
                total = d[lam][(r, s, m, v)]
                for k in range(n):
                    total -= gam.data[(k, lam, r)] * base[(k, s, m, v)]
                    total -= gam.data[(k, lam, s)] * base[(r, k, m, v)]
                    total -= gam.data[(k, lam, m)] * base[(r, s, k, v)]
                    total -= gam.data[(k, lam, v)] * base[(r, s, m, k)]
                return total

            out: Pairs = []
            for r in range(n):
                for s in range(r + 1, n):
                    for lam in range(n):
                        for m in range(n):
                            for v in range(m + 1, n):
                                cyc = (
                                    nabla(lam, r, s, m, v)
                                    + nabla(m, r, s, v, lam)
                                    + nabla(v, r, s, lam, m)
                                )
                                out.append((cyc, mp.mpf(0)))
            return out

        def alt_route() -> list[sp.Expr]:
            g = geo.g
            out = []
            for r in range(n):
                for s in range(r + 1, n):
                    for m in range(n):
                        for v in range(m + 1, n):
                            cand = sum(
                                (g[r, k] * riem.get((k, s, m, v)) for k in range(n)), sp.S.Zero
                            )
                            out.append(cand - self.alt_lowered_riemann(r, s, m, v))
            return out

        checks = [
            self.numeric("agrees with the numerical curvature", curvature, want=6),
            self.numeric("antisymmetries, pair symmetry and first Bianchi identity", symmetries),
            self.numeric(
                "second Bianchi identity (covariant derivative by finite differences)",
                second_bianchi,
                want=4,
            ),
            self.symbolic("agrees with the second-derivative formula for R_{abcd}", alt_route),
        ]
        del gamma
        return _combine(NUMERIC_METHOD, checks)

    def _numeric_curvature_leg(
        self, name: str, cand_items: Callable[[Any, Any], Pairs], want: int = 6
    ) -> Check:
        """Compare with a contraction of the numerical Riemann tensor."""

        def fn(ev: Evaluator, point: Point, h: Any) -> Pairs:
            x, p = point
            riem = ev.riemann(x, p, h)
            return cand_items(ev, (point, riem, h))

        return self.numeric(name, fn, want=want)

    def ricci(self) -> GeometryVerification:
        geo, n = self.geo, self.geo.n
        ric = self.ws.ricci()
        ginv = self.ws.inverse()

        def compare(ev: Evaluator, ctx: Any) -> Pairs:
            point, riem, _ = ctx
            ref = ev.ricci(riem)
            cand = self._cand(ev, "ricci", ric, point)
            return [(cand.get(k, 0), v) for k, v in ref.data.items()]

        def alt_ricci() -> list[sp.Expr]:
            out = []
            for s in range(n):
                for v in range(s, n):
                    total = sum(
                        (
                            ginv[r, k] * self.alt_lowered_riemann(r, s, k, v)
                            for r in range(n)
                            for k in range(n)
                            if ginv[r, k] != 0 and r != s and k != v
                        ),
                        sp.S.Zero,
                    )
                    out.append(ric.get((s, v)) - total)
            return out

        checks = [
            self._numeric_curvature_leg(
                "agrees with the contraction of the numerical curvature", compare
            ),
            self.symbolic(
                "agrees with the contraction of the second-derivative formula", alt_ricci
            ),
        ]
        del geo
        return _combine(NUMERIC_METHOD, checks)

    def ricci_scalar(self) -> GeometryVerification:
        n = self.geo.n
        scalar = self.ws.ricci_scalar()
        ric = self.ws.ricci()
        ginv = self.ws.inverse()

        def compare(ev: Evaluator, ctx: Any) -> Pairs:
            point, riem, _ = ctx
            x, p = point
            ref = ev.scalar(ev.ricci(riem), ev.inverse_metric(x, p))
            cand = ev.candidate("scalar", [((), scalar)], point)[()]
            return [(cand, ref)]

        def alt_scalar() -> list[sp.Expr]:
            total = sum(
                (
                    ginv[m, v] * self.alt_lowered_riemann(r, m, r2, v) * ginv[r, r2]
                    for m in range(n)
                    for v in range(n)
                    for r in range(n)
                    for r2 in range(n)
                    if ginv[m, v] != 0 and ginv[r, r2] != 0 and r != m and r2 != v
                ),
                sp.S.Zero,
            )
            return [scalar - total]

        del ric
        checks = [
            self._numeric_curvature_leg("agrees with the numerical Ricci scalar", compare),
            self.symbolic(
                "agrees with the full contraction of the second-derivative formula", alt_scalar
            ),
        ]
        return _combine(NUMERIC_METHOD, checks)

    def einstein(self) -> GeometryVerification:
        geo, n = self.geo, self.geo.n
        ein = self.ws.einstein()
        scalar = self.ws.ricci_scalar()
        ginv = self.ws.inverse()

        def compare(ev: Evaluator, ctx: Any) -> Pairs:
            point, riem, _ = ctx
            x, p = point
            g = ev.metric(x, p)
            ric = ev.ricci(riem)
            ref = ev.einstein(ric, g, ev.scalar(ric, ev.inverse_metric(x, p)))
            cand = self._cand(ev, "einstein", ein, point)
            return [(cand.get(k, 0), v) for k, v in ref.data.items()]

        def divergence(ev: Evaluator, point: Point, h: Any) -> Pairs:
            x, p = point
            gam = ev.christoffel(x, p, h)

            def up(at: list[Any]) -> Any:
                g = mp.matrix(ev.metric(at, p))
                gi = mp.inverse(g)
                cand = self._cand(ev, "einstein", ein, (at, p))
                gm = mp.matrix(n, n)
                for i in range(n):
                    for j in range(n):
                        gm[i, j] = cand.get((i, j), 0)
                return gi * gm * gi

            base = up(x)
            d = []
            for m in range(n):
                plus, minus = up(ev._shift(x, m, h)), up(ev._shift(x, m, -h))
                d.append((plus - minus) / (2 * h))
            out: Pairs = []
            for v in range(n):
                total = mp.mpf(0)
                for m in range(n):
                    total += d[m][m, v]
                    for lam in range(n):
                        total += gam.data[(m, m, lam)] * base[lam, v]
                        total += gam.data[(v, m, lam)] * base[m, lam]
                out.append((total, mp.mpf(0)))
            return out

        def trace() -> list[sp.Expr]:
            total = sum(
                (ginv[m, v] * ein.get((m, v)) for m in range(n) for v in range(n)), sp.S.Zero
            )
            return [total - (1 - sp.Rational(n, 2)) * scalar]

        del geo
        checks = [
            self._numeric_curvature_leg("agrees with the numerical Einstein tensor", compare),
            self.numeric(
                "contracted Bianchi identity: the divergence vanishes", divergence, want=5
            ),
            self.symbolic("trace identity g^{ab}G_{ab} = (1 - n/2) R", trace),
        ]
        return _combine(NUMERIC_METHOD, checks)

    def kretschmann(self) -> GeometryVerification:
        scalar = self.ws.kretschmann()

        def compare(ev: Evaluator, ctx: Any) -> Pairs:
            point, riem, _ = ctx
            x, p = point
            g = ev.metric(x, p)
            ref = ev.kretschmann(riem, g, ev.inverse_metric(x, p))
            cand = ev.candidate("kretschmann", [((), scalar)], point)[()]
            return [(cand, ref)]

        checks = [
            self._numeric_curvature_leg("agrees with the numerical Kretschmann scalar", compare),
            Check(
                name="independent symbolic route",
                kind="symbolic",
                passed=None,
                note="not run: no independent symbolic route for this invariant",
            ),
        ]
        return _combine(NUMERIC_METHOD, checks)

    def ricci_squared(self) -> GeometryVerification:
        scalar = self.ws.ricci_squared()

        def compare(ev: Evaluator, ctx: Any) -> Pairs:
            point, riem, _ = ctx
            x, p = point
            ref = ev.ricci_squared(ev.ricci(riem), ev.inverse_metric(x, p))
            cand = ev.candidate("ricci_squared", [((), scalar)], point)[()]
            return [(cand, ref)]

        checks = [
            self._numeric_curvature_leg(
                "agrees with the numerical Ricci-squared invariant", compare
            ),
            Check(
                name="independent symbolic route",
                kind="symbolic",
                passed=None,
                note="not run: no independent symbolic route for this invariant",
            ),
        ]
        return _combine(NUMERIC_METHOD, checks)

    # -- geodesics -----------------------------------------------------------------------------
    def geodesics(self) -> GeometryVerification:
        geo, n = self.geo, self.geo.n
        acc = self.ws.accelerations()
        v = geo.velocities
        g = geo.g
        parameter = sp.Symbol("affine_parameter", real=True)
        qdd = [sp.Symbol(f"{c.name}_ddot", real=True) for c in geo.coords]
        lagrangian = sum((g[i, j] * v[i] * v[j] for i in range(n) for j in range(n)), sp.S.Zero) / 2

        # Euler-Lagrange equations in the form g_{im}(x''^m - a^m) = 0, checked against the nested
        # numerical derivatives of L (no SymPy differentiation, no derive()).
        euler = [sum((g[i, m] * (qdd[m] - acc[m]) for m in range(n)), sp.S.Zero) for i in range(n)]
        checked = agreed = 0
        for k, ev in enumerate(self.evaluators):
            self.ws.check_deadline("geodesics:verify")
            lag = ev.substitute(lagrangian)
            derived = [ev.substitute(e) for e in euler]
            results = euler_lagrange_check(
                lag, geo.coords, v, qdd, parameter, derived, seed=20260102 + k
            )
            checked += sum(r.numeric_points_checked for r in results)
            agreed += sum(r.numeric_points_agreed for r in results)
            if any(not r.passed for r in results):
                break
        passed = checked >= MIN_VALID_POINTS and agreed == checked
        el = Check(
            name="Euler-Lagrange equations of g(x', x')/2, by nested numerical derivatives",
            kind="numeric",
            passed=passed,
            points_checked=checked,
            points_agreed=agreed,
            note="agreed at every sampled point"
            if passed
            else "too few valid points or a disagreement was found",
        )

        def norm_conserved(ev: Evaluator, point: Point, h: Any) -> Pairs:
            x, p = point
            vel = [mp.mpf(self.rng.uniform(-2, 2)) for _ in range(n)]
            gm = ev.metric(x, p)
            dg = ev.partial_metric(x, p, h)
            items = [(i, a) for i, a in enumerate(acc)]
            values = ev.candidate("accelerations", items, point, v, vel)
            total = mp.mpf(0)
            for s in range(n):
                for a in range(n):
                    for b in range(n):
                        total += dg[s][a][b] * vel[s] * vel[a] * vel[b]
            for a in range(n):
                for b in range(n):
                    total += 2 * gm[a][b] * values[a] * vel[b]
            return [(total, mp.mpf(0))]

        def derive_route() -> list[sp.Expr]:
            d = derive_expr(
                lagrangian, [c.name for c in geo.coords], "s", q=geo.coords, qd=v, t=parameter
            )
            if not d.solution:
                raise LimitError("no solved accelerations")
            return [acc[i] - d.solution[d.qdd[i]] for i in range(n)]

        def conserved_symbolic() -> list[sp.Expr]:
            total = sp.S.Zero
            for s in range(n):
                for a in range(n):
                    for b in range(n):
                        total += sp.diff(g[a, b], geo.coords[s]) * v[s] * v[a] * v[b]
            for a in range(n):
                for b in range(n):
                    total += 2 * g[a, b] * acc[a] * v[b]
            return [self.ws.tidy(total)]

        checks = [
            el,
            self.numeric("g(x', x') is conserved along the equations", norm_conserved, want=6),
            self.symbolic(
                "agrees with the Euler-Lagrange accelerations from lagrangian.derive", derive_route
            ),
            self.symbolic("g(x', x') is conserved (symbolic)", conserved_symbolic),
        ]
        try:
            return _combine(
                "nested numerical differentiation of L along random paths (mpmath)", checks
            )
        except DerivationError:  # pragma: no cover - derive failures only lose the symbolic leg
            return _combine("nested numerical differentiation of L", checks[:2])

    # -- user tensors -----------------------------------------------------------------------------
    def covariant_derivative(self) -> GeometryVerification:
        geo, n = self.geo, self.geo.n
        assert geo.tensor is not None  # noqa: S101 - guarded by the caller
        t = geo.tensor
        result = self.ws.covariant_derivative()

        def fn(ev: Evaluator, point: Point, h: Any) -> Pairs:
            x, p = point
            gam = ev.christoffel(x, p, h)
            tc = self._cand(ev, "tensor", t, point)
            dt = []
            for a in range(n):
                plus = self._cand(ev, "tensor", t, (ev._shift(x, a, h), p))
                minus = self._cand(ev, "tensor", t, (ev._shift(x, a, -h), p))
                dt.append({k: (plus.get(k, 0) - minus.get(k, 0)) / (2 * h) for k in t.indices()})
            cand = self._cand(ev, "covariant_derivative", result, point)
            out: Pairs = []
            for a in range(n):
                for idx in t.indices():
                    total = dt[a][idx]
                    for k, var in enumerate(t.variance):
                        for lam in range(n):
                            probe = (*idx[:k], lam, *idx[k + 1 :])
                            comp = tc.get(probe, 0)
                            if var == "up":
                                total += gam.data[(idx[k], a, lam)] * comp
                            else:
                                total -= gam.data[(lam, a, idx[k])] * comp
                    out.append((cand.get((a, *idx), 0), total))
            return out

        checks = [
            self.numeric("agrees with finite differences plus the numerical connection", fn),
            Check(
                name="independent symbolic route",
                kind="symbolic",
                passed=None,
                note="not run: a derivative has no independent symbolic route",
            ),
        ]
        return _combine(NUMERIC_METHOD, checks)

    def index_operation(self, kind: str, axis: int) -> GeometryVerification:
        geo, n = self.geo, self.geo.n
        assert geo.tensor is not None  # noqa: S101 - guarded by the caller
        t = geo.tensor
        result = self.ws.index_operation(kind, axis)

        def fn(ev: Evaluator, point: Point, h: Any) -> Pairs:
            x, p = point
            matrix = (
                ev.inverse_metric(x, p) if kind == "raise_index" else mp.matrix(ev.metric(x, p))
            )
            tc = self._cand(ev, "tensor", t, point)
            cand = self._cand(ev, kind, result, point)
            out: Pairs = []
            for idx in t.indices():
                total = sum(
                    (
                        matrix[idx[axis], b] * tc.get((*idx[:axis], b, *idx[axis + 1 :]), 0)
                        for b in range(n)
                    ),
                    mp.mpf(0),
                )
                out.append((cand.get(idx, 0), total))
            return out

        def round_trip() -> list[sp.Expr]:
            from .tensors import apply_matrix

            back = (
                apply_matrix(result, axis, lambda a, b: geo.g[a, b], "down")
                if kind == "raise_index"
                else apply_matrix(result, axis, lambda a, b: self.ws.inverse()[a, b], "up")
            )
            return [back.get(i) - t.get(i) for i in t.indices() if back.get(i) != t.get(i)]

        checks = [
            self.numeric("agrees with the numerical contraction with the metric", fn),
            self.symbolic("the inverse operation restores the original tensor", round_trip),
        ]
        return _combine(NUMERIC_METHOD, checks)


def _usable(first: Pairs, second: Pairs) -> bool:
    if len(first) != len(second):
        return False
    for (a1, b1), (a2, b2) in zip(first, second, strict=True):
        if not all(mp.isfinite(v) for v in (a1, b1, a2, b2)):
            return False
        if not (close(a1, a2, STABLE_TOL) and close(b1, b2, STABLE_TOL)):
            return False
    return True


def _combine(method: str, checks: list[Check]) -> GeometryVerification:
    numeric = [c for c in checks if c.kind == "numeric"]
    symbolic = [c for c in checks if c.kind == "symbolic"]
    numeric_passed = bool(numeric) and all(c.passed is True for c in numeric)
    if any(c.passed is False for c in symbolic):
        symbolic_zero: bool | None = False
    elif symbolic and all(c.passed is True for c in symbolic):
        symbolic_zero = True
    else:
        symbolic_zero = None
    return GeometryVerification(
        method=method,
        numeric_passed=numeric_passed,
        symbolic_zero=symbolic_zero,
        checks=checks,
    )
