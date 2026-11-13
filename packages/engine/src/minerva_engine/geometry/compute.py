"""Symbolic stages of the geometry computation.

Each stage is memoised and timed by `Workspace.stage`; a stage records how long it took and how big its
result is (operation count), so a slow or exploding stage can be named. Before any simplification the
size of the expression is checked against a cap and the deadline is checked, because `sp.simplify`
cannot be interrupted. The conventions are fixed here and printed with every result.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from typing import Any

import sympy as sp

from ..errors import LimitError
from ..latexprint import to_latex
from ..models import Limits, Status
from .models import StageDiagnostic
from .setup import Geometry, GeometryError
from .tensors import Tensor, apply_matrix

CONVENTIONS = [
    r"Christoffel symbols: \Gamma^{\rho}{}_{\mu\nu} = \tfrac12 g^{\rho\sigma}"
    r"(\partial_\mu g_{\sigma\nu} + \partial_\nu g_{\sigma\mu} - \partial_\sigma g_{\mu\nu})",
    r"Riemann tensor: R^{\rho}{}_{\sigma\mu\nu} = \partial_\mu \Gamma^{\rho}{}_{\nu\sigma} "
    r"- \partial_\nu \Gamma^{\rho}{}_{\mu\sigma} + \Gamma^{\rho}{}_{\mu\lambda}\Gamma^{\lambda}{}_{\nu\sigma} "
    r"- \Gamma^{\rho}{}_{\nu\lambda}\Gamma^{\lambda}{}_{\mu\sigma}",
    r"Ricci tensor R_{\mu\nu} = R^{\rho}{}_{\mu\rho\nu}; Ricci scalar R = g^{\mu\nu}R_{\mu\nu}; "
    r"Einstein tensor G_{\mu\nu} = R_{\mu\nu} - \tfrac12 g_{\mu\nu} R (a sphere of radius a has R = 2/a^2)",
    r"Covariant derivative: \nabla_\mu V^{\nu} = \partial_\mu V^{\nu} + \Gamma^{\nu}{}_{\mu\rho}V^{\rho}, "
    r"\nabla_\mu \omega_{\nu} = \partial_\mu \omega_{\nu} - \Gamma^{\rho}{}_{\mu\nu}\omega_{\rho}; "
    "the derivative index comes first in the result",
    r"Geodesics: \ddot{x}^{\mu} = -\Gamma^{\mu}{}_{\alpha\beta}\dot{x}^{\alpha}\dot{x}^{\beta} "
    "with an affine parameter; dots are derivatives with respect to it",
    "Connection: Levi-Civita (metric-compatible and torsion-free); no other connection is supported",
    "Signature and definiteness are never assumed; the metric is only required to be symmetric and "
    "non-degenerate",
]

_TRIG = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc, sp.sinh, sp.cosh, sp.tanh, sp.coth)


def _score(e: sp.Expr) -> tuple[int, int]:
    """Smaller is better: operations, with multiple-angle arguments (sin(2t)) counted as extra."""
    multi = sum(
        1
        for f in e.atoms(sp.sin, sp.cos, sp.tan)
        if f.args[0].is_Mul and f.args[0].args[0].is_Number and f.args[0].args[0] != 1
    )
    return int(sp.count_ops(e)) + 2 * multi, len(str(e))


def _cotangent(e: sp.Expr) -> sp.Expr:
    """Write tan(x)**-n as cot(x)**n."""
    out = e.replace(
        lambda x: x.is_Pow and isinstance(x.base, sp.tan) and x.exp.is_Integer and x.exp < 0,
        lambda x: sp.cot(x.base.args[0]) ** (-x.exp),
    )
    return out


CANCEL_MAX_OPS = 4000  # beyond this a stage is stopped as expression growth (resource_limit)
SIMPLIFY_MAX_OPS = 400  # simplify() is not interruptible: only run it on expressions this small


class Workspace:
    def __init__(
        self, geo: Geometry, limits: Limits, deadline: float, soft_deadline: float | None = None
    ) -> None:
        self.geo = geo
        self.limits = limits
        self.deadline = deadline
        # symbolic verification legs (extra evidence) are skipped after this point
        self.soft_deadline = soft_deadline if soft_deadline is not None else deadline
        self.diagnostics: list[StageDiagnostic] = []
        self.coord_latex = [to_latex(c) for c in geo.coords]
        self._memo: dict[str, Any] = {}
        self._stage = "setup"

    # -- bookkeeping ---------------------------------------------------------------------
    def check_deadline(self, stage: str | None = None) -> None:
        if time.monotonic() > self.deadline:
            raise LimitError(f"The time budget ran out in the {stage or self._stage} stage.")

    def record(self, diag: StageDiagnostic) -> None:
        self.diagnostics.append(diag)

    def stage(self, name: str, produce: Callable[[], Any]) -> Any:
        if name in self._memo:
            return self._memo[name]
        self.check_deadline(name)
        previous, self._stage = self._stage, name
        start = time.perf_counter()
        try:
            value = produce()
        except LimitError:
            self._finish(name, start, None, "failed", "resource_limit")
            raise
        except RecursionError as exc:
            self._finish(name, start, None, "failed", "resource_limit")
            raise LimitError(
                f"The expression became too deeply nested in the {name} stage."
            ) from exc
        except GeometryError as exc:
            self._finish(name, start, None, "failed", exc.status.value)
            raise
        finally:
            self._stage = previous
        self._memo[name] = value
        self._finish(name, start, value, "ok", None)
        return value

    def _finish(
        self, name: str, start: float, value: Any, status: str, failure: str | None
    ) -> None:
        ops, components = _measure(value)
        self.diagnostics.append(
            StageDiagnostic(
                stage=name,
                duration_ms=round((time.perf_counter() - start) * 1000),
                ops=ops,
                components=components,
                status=status,
                failure=failure,
            )
        )

    # -- simplification --------------------------------------------------------------------
    def tidy(self, expr: Any) -> sp.Expr:
        e = sp.sympify(expr)
        if e == 0:
            return sp.S.Zero
        ops = int(sp.count_ops(e))
        if ops > CANCEL_MAX_OPS:
            raise LimitError(
                f"An expression grew past {CANCEL_MAX_OPS} operations in the {self._stage} stage."
            )
        self.check_deadline()
        with contextlib.suppress(sp.PolynomialError, NotImplementedError, TypeError, ValueError):
            e = sp.cancel(e)
        if sp.count_ops(e) <= SIMPLIFY_MAX_OPS:
            self.check_deadline()
            best = sp.simplify(e)
            if best.has(*_TRIG) and sp.count_ops(best) <= SIMPLIFY_MAX_OPS:
                # sin(2t)/2 and 1/tan(t) are correct but not the form a reader expects
                start = best
                for alt in (
                    sp.expand_trig(start),
                    sp.simplify(sp.expand_trig(start)),
                    sp.trigsimp(start),
                ):
                    if _score(alt) < _score(best):
                        best = alt
            e = best
        return _cotangent(e)

    # -- stages ----------------------------------------------------------------------------
    def determinant(self) -> sp.Expr:
        def produce() -> sp.Expr:
            det = self.geo.g.det(method="berkowitz")
            return self.tidy(det)

        det: sp.Expr = self.stage("determinant", produce)
        return det

    def inverse(self) -> sp.Matrix:
        def produce() -> sp.Matrix:
            det = self.determinant()
            if det == 0:
                raise GeometryError(
                    Status.UNSUPPORTED,
                    "The metric is degenerate (its determinant is identically zero), so it has "
                    "no inverse and no Levi-Civita connection.",
                )
            n = self.geo.n
            if all(self.geo.g[i, j] == 0 for i in range(n) for j in range(n) if i != j):
                return sp.diag(*[self.tidy(1 / self.geo.g[i, i]) for i in range(n)])
            adj = self.geo.g.adjugate(method="berkowitz")
            return sp.Matrix(n, n, lambda i, j: self.tidy(adj[i, j] / det))

        inv: sp.Matrix = self.stage("inverse_metric", produce)
        return inv

    def christoffel(self) -> Tensor:
        def produce() -> Tensor:
            geo, n = self.geo, self.geo.n
            ginv = self.inverse()
            dg = [[[sp.diff(geo.g[m, v], geo.coords[s]) for v in range(n)] for m in range(n)]
                  for s in range(n)]  # fmt: skip
            out = Tensor(n, ("up", "down", "down"))
            for rho in range(n):
                for mu in range(n):
                    for nu in range(mu, n):
                        total = sp.S.Zero
                        for sigma in range(n):
                            if ginv[rho, sigma] == 0:
                                continue
                            total += ginv[rho, sigma] * (
                                dg[mu][sigma][nu] + dg[nu][sigma][mu] - dg[sigma][mu][nu]
                            )
                        value = self.tidy(total / 2)
                        out.set((rho, mu, nu), value)
                        out.set((rho, nu, mu), value)
            return out

        gamma: Tensor = self.stage("christoffel", produce)
        return gamma

    def riemann(self) -> Tensor:
        def produce() -> Tensor:
            geo, n = self.geo, self.geo.n
            gamma = self.christoffel()
            out = Tensor(n, ("up", "down", "down", "down"))
            for rho in range(n):
                for sigma in range(n):
                    for mu in range(n):
                        for nu in range(mu + 1, n):
                            total = sp.diff(gamma.get((rho, nu, sigma)), geo.coords[mu]) - sp.diff(
                                gamma.get((rho, mu, sigma)), geo.coords[nu]
                            )
                            for lam in range(n):
                                total += gamma.get((rho, mu, lam)) * gamma.get((lam, nu, sigma))
                                total -= gamma.get((rho, nu, lam)) * gamma.get((lam, mu, sigma))
                            value = self.tidy(total)
                            out.set((rho, sigma, mu, nu), value)
                            out.set((rho, sigma, nu, mu), -value)
            return out

        riem: Tensor = self.stage("riemann", produce)
        return riem

    def ricci(self) -> Tensor:
        def produce() -> Tensor:
            n = self.geo.n
            riem = self.riemann()
            out = Tensor(n, ("down", "down"))
            for sigma in range(n):
                for nu in range(sigma, n):
                    total = sum((riem.get((rho, sigma, rho, nu)) for rho in range(n)), sp.S.Zero)
                    value = self.tidy(total)
                    out.set((sigma, nu), value)
                    out.set((nu, sigma), value)
            return out

        ric: Tensor = self.stage("ricci", produce)
        return ric

    def ricci_scalar(self) -> sp.Expr:
        def produce() -> sp.Expr:
            n, ginv, ric = self.geo.n, self.inverse(), self.ricci()
            total = sp.S.Zero
            for mu in range(n):
                for nu in range(n):
                    total += ginv[mu, nu] * ric.get((mu, nu))
            return self.tidy(total)

        scalar: sp.Expr = self.stage("ricci_scalar", produce)
        return scalar

    def einstein(self) -> Tensor:
        def produce() -> Tensor:
            n, g = self.geo.n, self.geo.g
            ric, scalar = self.ricci(), self.ricci_scalar()
            out = Tensor(n, ("down", "down"))
            for mu in range(n):
                for nu in range(mu, n):
                    value = self.tidy(ric.get((mu, nu)) - g[mu, nu] * scalar / 2)
                    out.set((mu, nu), value)
                    out.set((nu, mu), value)
            return out

        ein: Tensor = self.stage("einstein", produce)
        return ein

    def lowered_riemann(self) -> Tensor:
        def produce() -> Tensor:
            g = self.geo.g
            return apply_matrix(self.riemann(), 0, lambda a, b: g[a, b], "down", tidy=self.tidy)

        t: Tensor = self.stage("riemann_lowered", produce)
        return t

    def kretschmann(self) -> sp.Expr:
        def produce() -> sp.Expr:
            ginv = self.inverse()
            lowered = self.lowered_riemann()
            raised = lowered
            for axis in range(4):
                raised = apply_matrix(raised, axis, lambda a, b: ginv[a, b], "up", tidy=self.tidy)
            total = sp.S.Zero
            for idx, value in lowered.data.items():
                total += value * raised.get(idx)
            return self.tidy(total)

        k: sp.Expr = self.stage("kretschmann", produce)
        return k

    def ricci_squared(self) -> sp.Expr:
        def produce() -> sp.Expr:
            ginv, ric = self.inverse(), self.ricci()
            up = apply_matrix(ric, 0, lambda a, b: ginv[a, b], "up", tidy=self.tidy)
            up = apply_matrix(up, 1, lambda a, b: ginv[a, b], "up", tidy=self.tidy)
            total = sp.S.Zero
            for idx, value in ric.data.items():
                total += value * up.get(idx)
            return self.tidy(total)

        value: sp.Expr = self.stage("ricci_squared", produce)
        return value

    def accelerations(self) -> list[sp.Expr]:
        """`a^mu = -Gamma^mu_{ab} v^a v^b`: the geodesic equations read `x''^mu = a^mu`."""

        def produce() -> list[sp.Expr]:
            geo, n = self.geo, self.geo.n
            gamma = self.christoffel()
            v = geo.velocities
            out = []
            for mu in range(n):
                # built term by term from the simplified Christoffel symbols, so each coefficient
                # of a velocity monomial stays in its factored form
                total = sp.S.Zero
                for a in range(n):
                    for b in range(a, n):
                        coefficient = gamma.get((mu, a, b)) * (1 if a == b else 2)
                        total -= coefficient * v[a] * v[b]
                out.append(total)
            return out

        acc: list[sp.Expr] = self.stage("geodesics", produce)
        return acc

    def covariant_derivative(self) -> Tensor:
        def produce() -> Tensor:
            geo, n = self.geo, self.geo.n
            assert geo.tensor is not None  # noqa: S101 - guarded by the caller
            t = geo.tensor
            gamma = self.christoffel()
            out = Tensor(n, ("down", *t.variance))
            for a in range(n):
                for idx in t.indices():
                    total = sp.diff(t.get(idx), geo.coords[a])
                    for k, var in enumerate(t.variance):
                        for lam in range(n):
                            probe = (*idx[:k], lam, *idx[k + 1 :])
                            comp = t.get(probe)
                            if comp == 0:
                                continue
                            if var == "up":
                                total += gamma.get((idx[k], a, lam)) * comp
                            else:
                                total -= gamma.get((lam, a, idx[k])) * comp
                    out.set((a, *idx), self.tidy(total))
            return out

        result: Tensor = self.stage("covariant_derivative", produce)
        return result

    def index_operation(self, kind: str, axis: int) -> Tensor:
        def produce() -> Tensor:
            geo = self.geo
            assert geo.tensor is not None  # noqa: S101 - guarded by the caller
            t = geo.tensor
            if axis >= t.rank:
                raise GeometryError(
                    Status.PARSE_ERROR,
                    f"The tensor has rank {t.rank}; there is no index {axis + 1}.",
                )
            want = "down" if kind == "raise_index" else "up"
            if t.variance[axis] != want:
                verb = "raised" if kind == "raise_index" else "lowered"
                raise GeometryError(
                    Status.PARSE_ERROR,
                    f"Index {axis + 1} is already {t.variance[axis]}; it cannot be {verb} again "
                    f"(that would contract {t.variance[axis]} with {t.variance[axis]}).",
                )
            if kind == "raise_index":
                ginv = self.inverse()
                return apply_matrix(t, axis, lambda a, b: ginv[a, b], "up", tidy=self.tidy)
            return apply_matrix(t, axis, lambda a, b: geo.g[a, b], "down", tidy=self.tidy)

        result: Tensor = self.stage(f"{kind}", produce)
        return result


def _measure(value: Any) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    if isinstance(value, Tensor):
        items = list(value.data.values())
        return sum(int(sp.count_ops(v)) for v in items), len(items)
    if isinstance(value, sp.MatrixBase):
        items = [e for e in value if e != 0]
        return sum(int(sp.count_ops(v)) for v in items), len(items)
    if isinstance(value, list):
        return sum(int(sp.count_ops(v)) for v in value), len(value)
    if isinstance(value, sp.Basic):
        return int(sp.count_ops(value)), 1
    return None, None
