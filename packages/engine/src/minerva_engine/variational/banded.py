"""A sparse-row Gaussian elimination with partial pivoting for banded systems (pure Python floats).

The Hessians of a discretised action are symmetric but may be indefinite (stationary paths that are
not minima), so no definiteness is assumed and the pivoting is by magnitude. A pivot that is tiny
compared with the row it came from is reported as `Singular`: the stationary path is then not
isolated, or the problem has no solution near the iterate.
"""

from __future__ import annotations

from collections.abc import Sequence

Row = dict[int, float]


class Singular(Exception):
    """The matrix is singular to working precision (relative pivot below the threshold)."""


def solve(
    rows: Sequence[Row], rhs: Sequence[float], bandwidth: int, *, tol: float = 1e-13
) -> list[float]:
    """Solve A x = rhs where `rows[i]` maps column -> value and |i - j| <= bandwidth for entries."""
    n = len(rows)
    a: list[Row] = [dict(r) for r in rows]
    b = list(rhs)
    scale = [max((abs(v) for v in r.values()), default=0.0) for r in a]
    for k in range(n):
        last = min(n - 1, k + bandwidth)
        pivot_row = max(range(k, last + 1), key=lambda r: abs(a[r].get(k, 0.0)))
        pivot = a[pivot_row].get(k, 0.0)
        reference = max(scale[pivot_row], max(scale[k : last + 1]), 1e-300)
        if abs(pivot) <= tol * reference:
            raise Singular
        if pivot_row != k:
            a[k], a[pivot_row] = a[pivot_row], a[k]
            b[k], b[pivot_row] = b[pivot_row], b[k]
            scale[k], scale[pivot_row] = scale[pivot_row], scale[k]
        row_k = a[k]
        for r in range(k + 1, last + 1):
            entry = a[r].get(k)
            if not entry:
                continue
            factor = entry / pivot
            row_r = a[r]
            for col, value in row_k.items():
                row_r[col] = row_r.get(col, 0.0) - factor * value
            row_r.pop(k, None)
            b[r] -= factor * b[k]
    x = [0.0] * n
    for k in range(n - 1, -1, -1):
        row = a[k]
        total = b[k] - sum(v * x[c] for c, v in row.items() if c > k)
        x[k] = total / row[k]
    return x
