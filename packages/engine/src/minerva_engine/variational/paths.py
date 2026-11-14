"""Trial paths supplied by the user: analytic expressions and sampled data.

* `AnalyticField` parses one expression per function in the independent variable. Derivatives are
  symbolic and each one is checked against a numerical derivative of the previous one
  (`differentiate.check_derivative`), so a wrong derivative cannot reach the action.
* `SampledField` interpolates data: a not-a-knot cubic spline from values alone (derivatives up to
  order 2), or cubic Hermite interpolation when slopes are given. Higher derivatives are not
  available and a functional that needs them is `unsupported` for sampled paths.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import mpmath as mp
import sympy as sp

from ..differentiate import check_derivative
from ..errors import LimitError, ParseError, UnsupportedInput
from ..latexparse import parse_latex
from ..textparse import parse_text
from .models import SampledFunction
from .setup import Problem


def parse_in_variable(
    problem: Problem, source: str, fmt: str, extra: Mapping[str, Any] | None = None
) -> sp.Expr:
    """An expression in the independent variable and the problem's parameters (no jets)."""
    assumptions = {k: list(v) for k, v in problem.assumptions.items()}
    assumptions.setdefault(problem.variable.name, ["real"])
    kwargs: dict[str, Any] = {"assumptions": assumptions, "limits": problem.limits}
    expr = parse_text(source, **kwargs) if fmt == "text" else parse_latex(source, **kwargs)
    return sp.sympify(expr)


def numeric_value(problem: Problem, source: str, fmt: str) -> sp.Expr:
    expr = parse_in_variable(problem, source, fmt)
    if not expr.is_number or expr.has(sp.zoo, sp.oo, sp.nan, -sp.oo):
        raise ParseError(f"{source!r} is not a finite number")
    return expr


class AnalyticField:
    """One expression per function; every derivative up to `depth` is checked numerically."""

    def __init__(
        self,
        problem: Problem,
        expressions: Sequence[sp.Expr],
        values: Mapping[sp.Symbol, Any],
        *,
        depth: int,
        lower: Any,
        upper: Any,
    ) -> None:
        t = problem.jets.variable
        self.depth = depth
        self.derivatives: list[list[Any]] = []
        substitution = {s: to_sympy(v) for s, v in values.items()}
        for expr in expressions:
            expr = sp.sympify(expr).xreplace(substitution)
            unknown = expr.free_symbols - {t}
            if unknown:
                names = ", ".join(sorted(s.name for s in unknown))
                raise ParseError(f"the path uses symbols with no value: {names}")
            chain = [expr]
            for _ in range(depth):
                chain.append(sp.diff(chain[-1], t))
                if not check_derivative(chain[-2], t, chain[-1]).passed:
                    raise LimitError("a derivative of the path could not be confirmed numerically")
            functions = [sp.lambdify(t, e, modules="mpmath") for e in chain]
            self.derivatives.append(functions)
        for n in range(9):  # finite and real on the interval, or the path is refused
            s = lower + (upper - lower) * n / 8
            for row in self.at(s):
                for value in row:
                    if not mp.isfinite(value) or abs(mp.im(value)) > 1e-20 * (1 + abs(value)):
                        raise ParseError("the path must be finite and real on the whole interval")

    def at(self, s: Any) -> list[list[Any]]:
        return [[fn(s) for fn in fns] for fns in self.derivatives]


def to_sympy(value: Any) -> sp.Expr:
    if isinstance(value, mp.mpc):
        return sp.Float(str(value.real), 30) + sp.I * sp.Float(str(value.imag), 30)
    return sp.Float(str(value), 30)


# -- sampled paths -------------------------------------------------------------------------------
class SampledField:
    """Piecewise-cubic interpolation of sampled functions (values, optionally slopes)."""

    MAX_ORDER = 2

    def __init__(
        self,
        functions: Sequence[SampledFunction],
        depth: int,
        orders: Sequence[int],
        lower: float,
        upper: float,
    ) -> None:
        self.depth = depth
        self.pieces: list[_Piece] = []
        for fn, order in zip(functions, orders, strict=True):
            if order > self.MAX_ORDER:
                raise UnsupportedInput(
                    "sampled paths support derivatives up to order 2; use an analytic path"
                )
            if order == 2 and fn.slopes is None:
                raise UnsupportedInput(
                    "a functional with second derivatives needs slopes with the sampled values"
                )
            if fn.t[0] > lower or fn.t[-1] < upper:
                raise ParseError("the samples must cover the whole interval")
            self.pieces.append(_Piece(fn))

    def at(self, s: Any) -> list[list[Any]]:
        x = float(s)
        out = []
        for piece in self.pieces:
            row = piece.derivatives(x)
            out.append(
                [mp.mpf(row[k]) if k < len(row) else mp.mpf(0) for k in range(self.depth + 1)]
            )
        return out


class _Piece:
    def __init__(self, fn: SampledFunction) -> None:
        t, y = list(fn.t), list(fn.values)
        if len(t) != len(y):
            raise ParseError("each sampled function needs as many values as times")
        if any(b <= a for a, b in pairwise(t)):
            raise ParseError("the sample times must increase")
        self.t = t
        self.y = y
        if fn.slopes is not None:
            if len(fn.slopes) != len(t):
                raise ParseError("slopes need one entry per sample")
            self.slopes = list(fn.slopes)
            self.kind = "hermite"
        else:
            if len(t) < 4:
                raise ParseError("a spline needs at least four samples")
            self.moments = _not_a_knot_moments(t, y)
            self.kind = "spline"

    def derivatives(self, x: float) -> list[float]:
        t, y = self.t, self.y
        lo, hi = 0, len(t) - 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if t[mid] <= x:
                lo = mid
            else:
                hi = mid - 1
        i = lo
        h = t[i + 1] - t[i]
        if self.kind == "hermite":
            u = (x - t[i]) / h
            y0, y1, m0, m1 = y[i], y[i + 1], self.slopes[i] * h, self.slopes[i + 1] * h
            h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
            h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
            value = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
            d1 = (
                (6 * u**2 - 6 * u) * (y0 - y1)
                + (3 * u**2 - 4 * u + 1) * m0
                + (3 * u**2 - 2 * u) * m1
            ) / h
            d2 = ((12 * u - 6) * (y0 - y1) + (6 * u - 4) * m0 + (6 * u - 2) * m1) / h**2
            return [value, d1, d2]
        M = self.moments
        a, b = (t[i + 1] - x) / h, (x - t[i]) / h
        value = a * y[i] + b * y[i + 1] + ((a**3 - a) * M[i] + (b**3 - b) * M[i + 1]) * h * h / 6
        d1 = (
            (y[i + 1] - y[i]) / h
            - (3 * a * a - 1) / 6 * h * M[i]
            + (3 * b * b - 1) / 6 * h * M[i + 1]
        )
        d2 = a * M[i] + b * M[i + 1]
        return [value, d1, d2]


def _not_a_knot_moments(t: list[float], y: list[float]) -> list[float]:
    """Second derivatives of the not-a-knot cubic spline (tridiagonal after eliminating the ends)."""
    n = len(t) - 1
    h = [t[i + 1] - t[i] for i in range(n)]
    d = [(y[i + 1] - y[i]) / h[i] for i in range(n)]
    if n == 3:  # four points: a single cubic through all of them
        return _cubic_moments(t, y)
    size = n - 1  # unknowns M_1 .. M_{n-1}
    lower = [0.0] * size
    diag = [0.0] * size
    upper = [0.0] * size
    rhs = [0.0] * size
    for k in range(size):
        i = k + 1
        lower[k] = h[i - 1]
        diag[k] = 2 * (h[i - 1] + h[i])
        upper[k] = h[i]
        rhs[k] = 6 * (d[i] - d[i - 1])
    # M_0 = (1 + h0/h1) M_1 - (h0/h1) M_2   and   M_n = (1 + h_{n-1}/h_{n-2}) M_{n-1} - (h_{n-1}/h_{n-2}) M_{n-2}
    r0 = h[0] / h[1]
    diag[0] += lower[0] * (1 + r0)
    upper[0] -= lower[0] * r0
    rn = h[n - 1] / h[n - 2]
    diag[size - 1] += upper[size - 1] * (1 + rn)
    lower[size - 1] -= upper[size - 1] * rn
    inner = _thomas(lower, diag, upper, rhs)
    m0 = (1 + r0) * inner[0] - r0 * inner[1]
    mn = (1 + rn) * inner[-1] - rn * inner[-2]
    return [m0, *inner, mn]


def _cubic_moments(t: list[float], y: list[float]) -> list[float]:
    # Lagrange cubic through four points: second derivative at each node from divided differences.
    import itertools

    def second(x: float) -> float:
        total = 0.0
        for j in range(4):
            others = [m for m in range(4) if m != j]
            denom = 1.0
            for m in others:
                denom *= t[j] - t[m]
            # second derivative of prod (x - t_m) over the three others
            acc = 0.0
            for a, b in itertools.combinations(others, 2):
                acc += 2 * (x - t[next(c for c in others if c not in (a, b))])
            total += y[j] * acc / denom
        return total

    return [second(x) for x in t]


def _thomas(
    lower: list[float], diag: list[float], upper: list[float], rhs: list[float]
) -> list[float]:
    n = len(diag)
    c = [0.0] * n
    d = [0.0] * n
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / denom if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom
    x = [0.0] * n
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x
