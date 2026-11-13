"""Parse and validate a geometric problem: coordinates, declared functions, metric, optional tensor.

Everything user-written goes through the strict parsers (`ops.parse_expression`) with declared
functions and `strict_calls`, so `f(r)` is a function only if declared and an undeclared `f(r)` is an
error, never the product f*r. Coordinates are real symbols. Nothing here assumes a signature.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import sympy as sp
from sympy.core.function import AppliedUndef

from ..build import Builder
from ..errors import MinervaMathError
from ..models import ExpressionInput, Limits, Status
from .models import GeometryInput
from .tensors import Tensor

SYMMETRY_SIMPLIFY_MAX_OPS = 400


class GeometryError(Exception):
    """The problem cannot be analysed; carries the explicit status to report."""

    def __init__(self, status: Status, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Geometry:
    n: int
    coords: list[sp.Symbol]
    params: list[sp.Symbol]
    g: sp.Matrix
    velocities: list[sp.Symbol]
    functions: dict[str, tuple[str, ...]]
    function_classes: dict[str, type]  # name -> undefined function class used in the expressions
    assumptions: dict[str, list[str]]
    tensor: Tensor | None = None
    assumptions_used: list[str] = field(default_factory=list)

    @property
    def symbols(self) -> list[sp.Symbol]:
        """Arguments of every lambdified expression: coordinates first, then parameters."""
        return [*self.coords, *self.params]


def _clean_name(name: str) -> str:
    return name.strip().lstrip("\\").replace("{", "").replace("}", "")


def build_geometry(data: GeometryInput, limits: Limits, max_dimension: int) -> Geometry:
    from ..ops import parse_expression  # local import: ops imports this package lazily

    names = [_clean_name(c) for c in data.coordinates]
    n = len(names)
    if n > max_dimension:
        raise GeometryError(
            Status.UNSUPPORTED,
            f"Dimension {n} is above the supported maximum of {max_dimension} coordinates.",
        )
    if len(set(names)) != n:
        raise GeometryError(Status.PARSE_ERROR, "Coordinates must be distinct.")
    if len(data.metric) != n or any(len(row) != n for row in data.metric):
        raise GeometryError(
            Status.PARSE_ERROR,
            f"The metric must be a {n} by {n} grid (one row and one column per coordinate).",
        )

    # Coordinates are real; the user's other assumptions are kept as declared.
    assumptions = {k: list(v) for k, v in data.assumptions.items()}
    for name in names:
        items = assumptions.setdefault(name, [])
        if "real" not in items:
            items.append("real")

    functions: dict[str, tuple[str, ...]] = {}
    for decl in data.functions:
        name = _clean_name(decl.name)
        if name in functions:
            raise GeometryError(Status.PARSE_ERROR, f"Function {name!r} is declared twice.")
        if name in names:
            raise GeometryError(
                Status.PARSE_ERROR, f"{name!r} is a coordinate and cannot also be a function."
            )
        args = tuple(_clean_name(a) for a in decl.args)
        unknown = [a for a in args if a not in names]
        if unknown:
            raise GeometryError(
                Status.PARSE_ERROR,
                f"Function {name} can only depend on coordinates; {unknown[0]!r} is not one.",
            )
        functions[name] = args

    try:
        builder = Builder(assumptions, limits, functions=functions)
        coords = [builder.symbol(c) for c in names]
        ctx = _Parser(data, limits, assumptions, functions, parse_expression)
        g = sp.Matrix(n, n, lambda i, j: ctx.parse(data.metric[i][j]))
        tensor = _parse_tensor(data, n, ctx) if data.tensor is not None else None
    except GeometryError:
        raise
    except MinervaMathError as exc:
        raise GeometryError(Status.PARSE_ERROR, str(exc)) from exc

    _check_symmetric(g, names)

    exprs = list(g) + ([e for _, e in tensor.nonzero()] if tensor else [])
    used = set().union(*(e.free_symbols for e in exprs)) if exprs else set()
    clash = sorted(s.name for s in used if s.name.endswith(("_dot", "_ddot")))
    if clash:
        raise GeometryError(
            Status.PARSE_ERROR,
            f"The name {clash[0]!r} is reserved for velocities; rename that symbol.",
        )
    params = sorted(used - set(coords), key=lambda s: s.name)
    applied = set().union(*(e.atoms(AppliedUndef) for e in exprs)) if exprs else set()
    classes = {type(a).__name__: type(a) for a in applied}

    velocities = [sp.Symbol(f"{c.name}_dot", real=True) for c in coords]
    return Geometry(
        n=n,
        coords=coords,
        params=params,
        g=g,
        velocities=velocities,
        functions=functions,
        function_classes=classes,
        assumptions=assumptions,
        tensor=tensor,
        assumptions_used=_describe(assumptions, names, functions),
    )


class _Parser:
    def __init__(
        self,
        data: GeometryInput,
        limits: Limits,
        assumptions: dict[str, list[str]],
        functions: dict[str, tuple[str, ...]],
        parse_expression: Callable[..., sp.Expr],
    ) -> None:
        self.data = data
        self.limits = limits
        self.assumptions = assumptions
        self.functions = functions
        self._parse = parse_expression

    def parse(self, text: str) -> sp.Expr:
        return self._parse(
            ExpressionInput(input=text, format=self.data.format, assumptions=self.assumptions),
            self.limits,
            functions=self.functions,
            strict_calls=True,
        )


def _parse_tensor(data: GeometryInput, n: int, ctx: _Parser) -> Tensor:
    assert data.tensor is not None  # noqa: S101 - guarded by the caller
    spec = data.tensor
    rank = len(spec.variance)
    if len(spec.components) != n**rank:
        raise GeometryError(
            Status.PARSE_ERROR,
            f"A rank-{rank} tensor in {n} dimensions has {n**rank} components "
            f"({len(spec.components)} given).",
        )
    tensor = Tensor(n, tuple(spec.variance))
    for k, text in enumerate(spec.components):
        idx = []
        rest = k
        for _ in range(rank):
            idx.append(rest % n)
            rest //= n
        tensor.set(tuple(reversed(idx)), ctx.parse(text))
    return tensor


def _check_symmetric(g: sp.Matrix, names: list[str]) -> None:
    n = g.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            diff = sp.expand(g[i, j] - g[j, i])
            if diff == 0:
                continue
            if sp.count_ops(diff) <= SYMMETRY_SIMPLIFY_MAX_OPS and sp.simplify(diff) == 0:
                continue
            raise GeometryError(
                Status.PARSE_ERROR,
                f"The metric must be symmetric: g_({names[i]},{names[j]}) and "
                f"g_({names[j]},{names[i]}) differ (or could not be shown equal).",
            )


def _describe(
    assumptions: dict[str, list[str]], coords: list[str], functions: dict[str, tuple[str, ...]]
) -> list[str]:
    out = [f"coordinates are real: {', '.join(coords)}"]
    for name, items in sorted(assumptions.items()):
        extra = [a for a in items if not (name in coords and a == "real")]
        if extra:
            out.append(f"{name}: {', '.join(extra)}")
    for name, args in functions.items():
        out.append(f"{name}({', '.join(args)}) is an unspecified real function")
    out.append("symbols without a declared assumption are complex numbers")
    return out
