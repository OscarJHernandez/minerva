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
from ..errors import MinervaMathError, ParseError, UnsupportedInput, diagnostic_of
from ..latexparse import reject_text_syntax_in
from ..models import ExpressionInput, Limits, Status
from ..notation.models import InputCode, InputDiagnostic
from ..notation.syntax import Syntax, choose_syntax
from .models import GeometryInput
from .tensors import Tensor

SYMMETRY_SIMPLIFY_MAX_OPS = 400


class GeometryError(Exception):
    """The problem cannot be analysed; carries the explicit status to report."""

    def __init__(
        self, status: Status, message: str, diagnostic: InputDiagnostic | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.diagnostic = diagnostic


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
    cell_syntax: list[list[Syntax]] | None = None
    tensor_syntax: list[Syntax] | None = None

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
        cells = [c for row in data.metric for c in row]
        cells += list(data.tensor.components) if data.tensor is not None else []
        ctx = _Parser(data, limits, assumptions, functions, parse_expression)
        whole: Syntax = choose_syntax(" ".join(cells)) if data.format == "auto" else data.format
        try:
            _check_cells(cells, ctx, whole)
            ctx.syntax = {}
            g = sp.Matrix(n, n, lambda i, j: ctx.parse(data.metric[i][j]))
            tensor = _parse_tensor(data, n, ctx) if data.tensor is not None else None
        except MinervaMathError:
            if data.format != "auto":
                raise
            split = _split_syntax(cells, ctx)
            if split is None:
                raise
            ctx.syntax = split
            g = sp.Matrix(n, n, lambda i, j: ctx.parse(data.metric[i][j]))
            tensor = _parse_tensor(data, n, ctx) if data.tensor is not None else None
        builder = Builder(assumptions, limits, functions=functions)
        coords = [builder.symbol(c) for c in names]
    except GeometryError:
        raise
    except MinervaMathError as exc:
        status = Status.UNSUPPORTED if isinstance(exc, UnsupportedInput) else Status.PARSE_ERROR
        raise GeometryError(status, str(exc), diagnostic_of(exc, "metric")) from exc

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
        cell_syntax=(
            [[ctx.syntax[data.metric[i][j]] for j in range(n)] for i in range(n)]
            if ctx.syntax
            else None
        ),
        tensor_syntax=(
            [ctx.syntax[c] for c in data.tensor.components]
            if ctx.syntax and data.tensor is not None
            else None
        ),
    )


def _check_cells(cells: list[str], ctx: _Parser, syntax: Syntax) -> None:
    """Name every problem of every entry at once, for one syntax.

    LaTeX: the same refusal every cell gets from parse_latex, checked over all of them first, so a metric written
    in text syntax is refused with the typed reason instead of the generic error of whichever cell happens to be
    parsed first (`^(-1)`: "expected an exponent"). Text: every product written without `*`.
    """
    ctx.whole = syntax
    if syntax == "latex":
        reject_text_syntax_in(cells)
    else:
        _reject_text_products(cells, ctx)


def _split_syntax(cells: list[str], ctx: _Parser) -> dict[str, Syntax] | None:
    """Under `format: auto`, the parser for each entry when one syntax does not read them all, or None.

    An entry is assigned only if that is unambiguous: one parser reads it and the other refuses it, or both read
    it to the same expression. An entry both read *differently* (`r0^2` is 0 in LaTeX and r0**2 in text, `pi` is
    i*p and pi) or neither reads leaves the whole input to the single syntax of before, and its errors. Entries
    that read the same either way are labelled by the per-entry rule (`choose_syntax`).
    """
    split: dict[str, Syntax] = {}
    for cell in dict.fromkeys(cells):
        readings: dict[Syntax, sp.Expr] = {}
        for syntax in ("latex", "text"):
            try:
                readings[syntax] = ctx.parse(cell, syntax)
            except (MinervaMathError, RecursionError):
                continue
        if len(readings) == 1:
            split[cell] = next(iter(readings))
        elif len(readings) == 2 and readings["latex"] == readings["text"]:
            split[cell] = choose_syntax(cell)
        else:
            return None
    return split


def _reject_text_products(cells: list[str], ctx: _Parser) -> None:
    """In text syntax, name every product written without `*` in every cell in one error.

    Only the coded `implicit_multiplication` errors are merged; any other error is left to the parse of its
    cell, so the first cell's own message is unchanged.
    """
    symbols: list[str] = []
    suggestions: list[str] = []
    for cell in cells:
        try:
            ctx.parse(cell)
        except ParseError as exc:
            if exc.code is not InputCode.IMPLICIT_MULTIPLICATION:
                continue
            for symbol, suggestion in zip(exc.symbols, exc.suggestions, strict=True):
                if symbol not in symbols:
                    symbols.append(symbol)
                    suggestions.append(suggestion)
        except MinervaMathError:
            continue
    if not symbols:
        return
    fixes = "; ".join(
        "use ( ) for grouping instead of [ ]" if w == "[" else f"`{w}` → `{f}`"
        for w, f in zip(symbols, suggestions, strict=True)
    )
    raise ParseError(
        "could not parse the metric: text syntax needs `*` for every product and ( ) for grouping "
        f"({fixes})",
        code=InputCode.IMPLICIT_MULTIPLICATION,
        symbols=symbols,
        suggestions=suggestions,
        recoverable=True,
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

        self.whole: Syntax = "latex" if data.format == "auto" else data.format
        self.syntax: dict[str, Syntax] = {}  # per entry, when an `auto` input was split

    def parse(self, text: str, syntax: Syntax | None = None) -> sp.Expr:
        chosen = syntax or self.syntax.get(text) or self.whole
        return self._parse(
            ExpressionInput(input=text, format=chosen, assumptions=self.assumptions),
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
