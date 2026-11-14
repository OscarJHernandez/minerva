"""Two independent quadrature rules and a finite-difference table.

`gauss_legendre` is a composite rule with its nodes computed here; `tanh_sinh` is mpmath's own
adaptive rule. They share no code, and a value is trusted only where they agree.
`fd_table` is pure arithmetic on numbers it is given, so tests can feed it ill-conditioned or wrong
data and see the verdict.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

import mpmath as mp

Number = Any  # mp.mpf or mp.mpc


@cache
def _legendre_rule(n: int, dps: int) -> tuple[tuple[mp.mpf, ...], tuple[mp.mpf, ...]]:
    with mp.workdps(dps + 10):
        nodes: list[mp.mpf] = []
        weights: list[mp.mpf] = []
        for i in range(1, n + 1):
            x = mp.cos(mp.pi * (i - mp.mpf(1) / 4) / (n + mp.mpf(1) / 2))
            for _ in range(100):
                p0, p1 = mp.mpf(1), x
                for k in range(2, n + 1):
                    p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                dp = n * (x * p1 - p0) / (x * x - 1)
                step = p1 / dp
                x -= step
                if abs(step) < mp.mpf(10) ** (-(dps + 5)):
                    break
            p0, p1 = mp.mpf(1), x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            nodes.append(x)
            weights.append(2 / ((1 - x * x) * dp * dp))
        return tuple(nodes), tuple(weights)


def gauss_legendre(
    f: Callable[[mp.mpf], Number], a: Number, b: Number, panels: int = 3, order: int = 16
) -> Number:
    nodes, weights = _legendre_rule(order, mp.mp.dps)
    total: Number = mp.mpf(0)
    width = (b - a) / panels
    for p in range(panels):
        lo = a + p * width
        half = width / 2
        mid = lo + half
        total += half * mp.fsum(w * f(mid + half * x) for x, w in zip(nodes, weights, strict=True))
    return total


def tanh_sinh(f: Callable[[mp.mpf], Number], a: Number, b: Number) -> tuple[Number, mp.mpf]:
    value, error = mp.quad(f, [a, b], error=True)
    return value, mp.mpf(error)


@dataclass(frozen=True)
class Integral:
    value: Number
    error: mp.mpf  # disagreement of the two rules: an estimate, never a bound


def integrate(
    f: Callable[[mp.mpf], Number], a: Number, b: Number, *, fast: bool = False
) -> Integral:
    """Composite Gauss-Legendre, checked against a finer one (or tanh-sinh when not `fast`)."""
    coarse = gauss_legendre(f, a, b, panels=3, order=16)
    fine = gauss_legendre(f, a, b, panels=6, order=16)
    if fast:
        return Integral(fine, abs(fine - coarse))
    other, _ = tanh_sinh(f, a, b)
    return Integral(fine, max(abs(fine - coarse), abs(fine - other)))


# -- finite-difference verdict ------------------------------------------------------------------
Verdict = Literal["agree", "disagree", "inconclusive"]

EXACT_REL_TOL = mp.mpf("1e-14")  # every step agrees to this: the difference quotient is exact
RICHARDSON_REL_TOL = mp.mpf("1e-10")
PERSISTENT_REL = mp.mpf("1e-12")  # an error this large that stops shrinking is a disagreement


@dataclass(frozen=True)
class FDRow:
    step: mp.mpf
    fd: Number
    abs_error: mp.mpf
    rel_error: mp.mpf


@dataclass(frozen=True)
class FDTable:
    rows: tuple[FDRow, ...]
    richardson: tuple[Number, ...]
    richardson_rel_error: mp.mpf
    observed_order: float | None
    verdict: Verdict
    reason: str


def fd_table(
    action: Callable[[mp.mpf], Number],
    analytic: Number,
    scale: mp.mpf,
    steps: Sequence[mp.mpf],
    *,
    order: Literal[1, 2] = 1,
    noise: mp.mpf | None = None,
) -> FDTable:
    """Central differences of `action(h)`-type data against an analytic value.

    `action(e)` is S[x + e η]. For `order=1` the quotient is (S(h) - S(-h)) / 2h; for `order=2` it is
    (S(h) - 2 S(0) + S(-h)) / h². `scale` is the size of the terms that cancel (never |analytic|),
    `noise` the quadrature error of one S evaluation: a noise floor above the tolerance makes the
    verdict `inconclusive`. Agreement means the error is at rounding level at every step (the
    quotient is exact, as for a quadratic action) or falls like h² and the Richardson value agrees.
    """
    noise = mp.mpf(0) if noise is None else noise
    scale = max(abs(scale), mp.mpf("1e-30"))
    zero = action(mp.mpf(0)) if order == 2 else mp.mpf(0)
    rows: list[FDRow] = []
    for h in steps:
        plus, minus = action(h), action(-h)
        fd = (plus - minus) / (2 * h) if order == 1 else (plus - 2 * zero + minus) / (h * h)
        err = abs(fd - analytic)
        rows.append(FDRow(h, fd, err, err / scale))
    richardson = tuple((4 * rows[i + 1].fd - rows[i].fd) / 3 for i in range(len(rows) - 1))
    r_err = abs(richardson[-1] - analytic) / scale if richardson else rows[-1].rel_error
    ratio: float | None = None
    if len(rows) >= 2 and rows[-1].abs_error > 0 and rows[-2].abs_error > 0:
        ratio = float(rows[-2].abs_error / rows[-1].abs_error)
    h_ratio = float(steps[-2] / steps[-1]) if len(steps) >= 2 else 2.0
    observed = float(mp.log(ratio) / mp.log(h_ratio)) if ratio and h_ratio > 1 else None

    floor = noise / min(steps) / scale * (1 if order == 1 else 1 / min(steps))
    if floor > RICHARDSON_REL_TOL:
        return FDTable(
            tuple(rows), richardson, r_err, observed, "inconclusive",
            "the quadrature noise divided by the step exceeds the tolerance",
        )  # fmt: skip
    if all(r.rel_error <= EXACT_REL_TOL for r in rows):
        return FDTable(tuple(rows), richardson, r_err, observed, "agree", "exact at every step")
    second_order = observed is not None and 1.6 <= observed <= 2.4
    if second_order and r_err <= RICHARDSON_REL_TOL and rows[-1].rel_error <= rows[0].rel_error:
        return FDTable(
            tuple(rows), richardson, r_err, observed, "agree", "second-order convergence"
        )
    falling = ratio is not None and ratio >= 2.5
    if rows[-1].rel_error > PERSISTENT_REL and not falling:
        return FDTable(
            tuple(rows), richardson, r_err, observed, "disagree",
            "the error does not fall with the step",
        )  # fmt: skip
    return FDTable(tuple(rows), richardson, r_err, observed, "inconclusive", "no clear convergence")
