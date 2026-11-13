"""A small sparse tensor of components with index variance.

Only the plumbing lives here (storage, applying a matrix to one index, labels); the geometric formulas
are written separately for the symbolic route (`compute.py`) and for the independent numerical
route (`numeric.py`). Elements are SymPy expressions or mpmath numbers; absent keys are zero.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal

Index = tuple[int, ...]
Var = Literal["up", "down"]


@dataclass
class Tensor:
    n: int
    variance: tuple[Var, ...]  # per index
    data: dict[Index, Any] = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return len(self.variance)

    def get(self, idx: Index) -> Any:
        return self.data.get(idx, 0)

    def set(self, idx: Index, value: Any) -> None:
        if _is_zero(value):
            self.data.pop(idx, None)
        else:
            self.data[idx] = value

    def indices(self) -> Iterator[Index]:
        return product(range(self.n), repeat=self.rank)

    def nonzero(self) -> list[tuple[Index, Any]]:
        return sorted(self.data.items())

    def map(self, fn: Callable[[Any], Any]) -> Tensor:
        out = Tensor(self.n, self.variance)
        for idx, value in self.data.items():
            out.set(idx, fn(value))
        return out


def _is_zero(value: Any) -> bool:
    try:
        return bool(value == 0)
    except Exception:
        return False


def apply_matrix(
    t: Tensor,
    axis: int,
    matrix: Callable[[int, int], Any],
    variance: Var,
    tidy: Callable[[Any], Any] | None = None,
) -> Tensor:
    """`out[..., a, ...] = sum_b matrix(a, b) * t[..., b, ...]` along `axis` (raising or lowering)."""
    out_var = (*t.variance[:axis], variance, *t.variance[axis + 1 :])
    acc: dict[Index, Any] = {}
    for idx, value in t.data.items():
        b = idx[axis]
        for a in range(t.n):
            m = matrix(a, b)
            if _is_zero(m):
                continue
            key = (*idx[:axis], a, *idx[axis + 1 :])
            acc[key] = acc.get(key, 0) + m * value
    out = Tensor(t.n, out_var)
    for idx, value in acc.items():
        out.set(idx, tidy(value) if tidy else value)
    return out


def index_label(name: str, variance: tuple[Var, ...], idx: Index, coord_latex: list[str]) -> str:
    """LaTeX label such as `\\Gamma^{r}{}_{\\theta\\theta}`: indices in order, `{}` where the position flips."""
    out = name
    previous: str | None = None
    for var, i in zip(variance, idx, strict=True):
        if previous is not None and var != previous:
            out += "{}"
        mark = "^" if var == "up" else "_"
        # same-position neighbours share one script: T_{ab}
        if previous == var:
            out = out[:-1] + " " + coord_latex[i] + "}"
        else:
            out += f"{mark}{{{coord_latex[i]}}}"
        previous = var
    return out
