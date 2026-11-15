"""Composite Gauss-Legendre node sets for coefficient integrals, and the places a function jumps.

A set of nodes and weights is built once; every coefficient (cos(n w x), sin(n w x), P_n) is then a
weighted sum over the same function values. Two sets with different panel counts must give the same
coefficient: that agreement is the quadrature check.
"""

from __future__ import annotations

import contextlib
import itertools
import math
import signal
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import mpmath
import sympy as sp

from ..variational.quadrature import _legendre_rule

ORDER = 16
DPS = 40


@dataclass
class Nodes:
    xs: list[Any]
    ws: list[Any]
    ys: list[Any]  # the function at xs


def panels_for(cycles: float) -> int:
    """Panels of 16 nodes that resolve `cycles` oscillations on the piece, generously."""
    return max(3, math.ceil(2.5 * cycles) + 2)


def composite(points: list[Any], cycles_per_unit: float, f: Any, factor: int = 1) -> Nodes:
    """Gauss-Legendre panels on each piece between consecutive points, `factor` times as many."""
    nodes, weights = _legendre_rule(ORDER, DPS)
    xs: list[Any] = []
    ws: list[Any] = []
    for lo, hi in itertools.pairwise(points):
        count = panels_for(cycles_per_unit * float(hi - lo)) * factor
        width = (hi - lo) / count
        for k in range(count):
            left = lo + k * width
            for t, w in zip(nodes, weights, strict=True):
                xs.append(left + width * (t + 1) / 2)
                ws.append(w * width / 2)
    ys = []
    for x in xs:
        v = f(x)
        ys.append(None if v is None else mpmath.re(v))
    return Nodes(xs, ws, ys)


def usable(rule: Nodes) -> bool:
    return all(y is not None for y in rule.ys)


# --- where a function is not smooth -----------------------------------------------------------
_BREAKING = (sp.Abs, sp.sign, sp.Heaviside, sp.floor, sp.ceiling)


def breakpoints(expr: sp.Expr, var: sp.Symbol, a: Any, b: Any, deadline: float = 3.0) -> list[Any]:
    """Interior points where an Abs / sign / step / floor argument or a Piecewise condition vanishes."""
    conditions: list[sp.Expr] = []
    for atom in expr.atoms(*_BREAKING):
        conditions.append(atom.args[0])
    for pw in expr.atoms(sp.Piecewise):
        for _, cond in pw.args:
            if isinstance(cond, sp.core.relational.Relational):
                conditions.append(cond.lhs - cond.rhs)
    found: list[Any] = []
    with mpmath.workdps(DPS), time_limit(deadline):
        for g in conditions:
            if var not in g.free_symbols or g.free_symbols - {var}:
                continue
            try:
                roots = sp.solveset(
                    g, var, sp.Interval(sp.Float(str(a), DPS), sp.Float(str(b), DPS))
                )
            except (NotImplementedError, ValueError, TypeError):
                continue
            if isinstance(roots, sp.FiniteSet):
                for r in roots:
                    v = mpmath.mpf(str(sp.N(r, DPS)))
                    if a < v < b:
                        found.append(v)
    return sorted(set(found), key=float)


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Stop a SymPy call that runs too long (main thread only; elsewhere it is a no-op)."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(
        signal, "setitimer"
    ):
        yield
        return

    class _TimeUp(Exception):
        pass

    def handler(signum: int, frame: Any) -> None:
        raise _TimeUp

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    except _TimeUp:
        return
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
