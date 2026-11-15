"""The shared setup of a vector request: declarations, parsing and the evaluation of one expression.

`compute.py` (evaluate, Jacobian, convert) and `steps.py` (controlled identity steps) both start from a
`Context`. `Context.evaluate` is the single place where an expression is lowered, checked for the
smoothness it needs and verified by the independent Cartesian reading; a step that claims two expressions
are equal asks `Context.residual` for the evaluation of their difference, so a rewrite is never
trusted because the matcher produced it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import sympy as sp

from ..errors import MinervaMathError
from ..models import ALLOWED_ASSUMPTIONS, Limits, Status
from ..textparse import parse_text
from .algebra import exact_zero
from .coordinates import TIME, System, get_system
from .expressions import depth_of_operators, kind_of
from .lower import Declared, Lowering, Undeclared
from .models import Blocked, FieldDecl, Reason, VectorInput, VectorVerification
from .numeric import verify_result
from .parse import parse_vector_latex, parse_vector_text
from .printing import vector_latex

MAX_OPERATOR_DEPTH = 4
SIMPLIFY_MAX_OPS = 400
SEED = 20260514
TIME_BUDGET_SECONDS = 8.0


def time_budget(limits: Limits) -> float:
    """The request's own budget: 8 s, or 80% of a longer sandbox timeout the caller chose (so that a slow
    machine given a generous timeout is not cut off by a shorter internal constant)."""
    return max(TIME_BUDGET_SECONDS, 0.8 * limits.timeout_seconds)


RESERVED = frozenset({TIME, "pi", "oo"})


class Refused(Exception):
    def __init__(self, status: Status, message: str, reason: Reason) -> None:
        super().__init__(message)
        self.status, self.message, self.reason = status, message, reason


def tidy(expr: sp.Expr) -> sp.Expr:
    expr = sp.sympify(expr)
    if expr.count_ops() > SIMPLIFY_MAX_OPS:
        return sp.cancel(expr)
    return sp.simplify(expr)


def _flags(items: list[str]) -> dict[str, bool]:
    return {item: True for item in items if item in ALLOWED_ASSUMPTIONS}


@dataclass
class Evaluation:
    """One expression lowered, checked for smoothness and verified."""

    status: Literal["ok", "blocked", "not_verified"]
    kind: str
    comps: list[sp.Expr] = field(default_factory=list)
    verification: VectorVerification | None = None
    blocked: list[Blocked] = field(default_factory=list)

    @property
    def zero(self) -> bool:
        return self.status == "ok" and all(c == 0 for c in self.comps)


@dataclass
class Context:
    inp: VectorInput
    limits: Limits
    system: System
    low: Lowering
    assumptions: dict[str, list[str]]
    vector_names: list[str]
    deadline: float
    denominators: list[sp.Expr] = field(default_factory=list)

    # -- parsing -----------------------------------------------------------------------------
    def parse(self, text: str, *, denominators: list[sp.Expr] | None = None) -> sp.Expr:
        sink = self.denominators if denominators is None else denominators
        if self.inp.format == "latex":
            expr = parse_vector_latex(
                text,
                vectors=self.vector_names,
                directions=self.system.coords,
                assumptions=self.assumptions,
                limits=self.limits,
                denominators=sink,
            )
        else:
            expr = parse_vector_text(
                text,
                vectors=self.vector_names,
                names=[*self.low.declared, *self.low.constants],
                directions=self.system.coords,
                assumptions=self.assumptions,
                limits=self.limits,
                denominators=sink,
            )
        unknown = self.low.unknown_symbols(expr)
        if unknown:
            raise Undeclared(unknown)
        if depth_of_operators(expr) > MAX_OPERATOR_DEPTH:
            raise Refused(
                Status.RESOURCE_LIMIT,
                f"operators are nested {depth_of_operators(expr)} deep; the limit is "
                f"{MAX_OPERATOR_DEPTH}",
                "derivative_depth",
            )
        return expr

    def latex(self, expr: sp.Expr) -> str:
        return vector_latex(expr)

    # -- evaluation --------------------------------------------------------------------------
    def evaluate(self, expr: sp.Expr) -> Evaluation:
        kind = kind_of(expr)
        lowered = self.low.lower(expr)
        comps = [tidy(c) for c in lowered] if isinstance(lowered, tuple) else [tidy(lowered)]
        comps = [sp.Integer(0) if exact_zero(c) else c for c in comps]
        blocked = self.low.merge_blocks(comps)
        if blocked:
            return Evaluation("blocked", kind, comps, None, blocked)
        verification = verify_result(
            self.low,
            expr,
            comps,
            kind == "vector",
            seed=SEED,
            deadline=self.deadline,
            depth=depth_of_operators(expr),
        )
        status: Literal["ok", "not_verified"] = (
            "ok" if verification.numeric_passed else "not_verified"
        )
        return Evaluation(status, kind, comps, verification, [])

    def residual(self, before: sp.Expr, after: sp.Expr) -> Evaluation:
        """The evaluation of `before - after`: zero (with both legs agreeing) means the two are equal
        for every field of the declared smoothness. A literal 0 stands for a zero of either kind."""
        if after == 0:
            return self.evaluate(before)
        if before == 0:
            return self.evaluate(sp.Mul(-1, after))
        if kind_of(before) != kind_of(after):
            raise Refused(
                Status.PARSE_ERROR, "the two sides are not of the same kind", "type_error"
            )
        return self.evaluate(sp.Add(before, sp.Mul(-1, after)))

    def lacking_smoothness(self, expr: sp.Expr, needs: str = "c2") -> list[str]:
        """Declared abstract fields in `expr` whose declared smoothness is below `needs`."""
        order = {"c1": 1, "c2": 2, "smooth": 3}
        want = order.get(needs, 2)
        out = []
        for s in expr.free_symbols:
            d = self.low.declared.get(s.name)
            if d is None or d.explicit is not None:
                continue
            have = order.get(d.decl.smoothness or "", 0)
            if have < want and d.decl.name not in out:
                out.append(d.decl.name)
        return sorted(out)


def build_context(inp: VectorInput, limits: Limits, started: float) -> Context:
    system = get_system(inp.coordinates)
    taken: set[str] = set(system.coords) | RESERVED
    constants: dict[str, sp.Symbol] = {}
    assumptions: dict[str, list[str]] = dict(system.assumptions())
    assumptions[TIME] = ["real"]
    for name in dict.fromkeys([*inp.constants, *inp.assumptions]):
        if name in taken:
            raise Refused(
                Status.PARSE_ERROR,
                f"{name!r} is reserved or already a coordinate",
                "bad_declaration",
            )
        taken.add(name)
        items = list(inp.assumptions.get(name, []))
        constants[name] = sp.Symbol(name, **_flags(items))
        if items:
            assumptions[name] = items
    declared: dict[str, Declared] = {}
    allowed_args = [*system.coords, TIME]
    pending: list[tuple[Declared, FieldDecl]] = []
    vector_names: list[str] = []
    for decl in inp.fields:
        if decl.name in taken:
            raise Refused(
                Status.PARSE_ERROR,
                f"the name {decl.name!r} is declared twice or is reserved",
                "bad_declaration",
            )
        taken.add(decl.name)
        depends = tuple(decl.depends_on) if decl.depends_on is not None else system.coords
        if any(a not in allowed_args for a in depends) or len(set(depends)) != len(depends):
            raise Refused(
                Status.PARSE_ERROR,
                f"{decl.name}: depends_on may list {', '.join(allowed_args)}",
                "bad_declaration",
            )
        comps = (decl.name,) if decl.kind == "scalar" else system.component_names(decl.name)
        if decl.kind == "vector":
            vector_names.append(decl.name)
        d = Declared(decl, tuple(comps), tuple(depends))
        declared[decl.name] = d
        if decl.components is not None:
            pending.append((d, decl))
    for d in declared.values():
        for comp in d.comps:
            if comp != d.decl.name and comp in taken:
                raise Refused(
                    Status.PARSE_ERROR,
                    f"component name {comp!r} clashes with a declared name",
                    "bad_declaration",
                )
            taken.add(comp)
    low = Lowering(system, declared, constants)
    for d, decl in pending:
        parts: list[sp.Expr] = []
        for text in decl.components or []:
            try:
                parsed = parse_text(text, assumptions, limits)
            except MinervaMathError as exc:
                raise Refused(Status.PARSE_ERROR, f"{decl.name}: {exc}", "bad_declaration") from exc
            known_scalar = set(system.coords) | {TIME} | set(constants)
            bad = sorted({s.name for s in parsed.free_symbols} - known_scalar)
            if bad:
                raise Refused(
                    Status.PARSE_ERROR,
                    f"{decl.name}: components may use only the coordinates, t and declared "
                    f"constants (found {', '.join(bad)})",
                    "bad_declaration",
                )
            parts.append(parsed)
        d.explicit = tuple(low._scalar(p) for p in parts)
    return Context(
        inp, limits, system, low, assumptions, vector_names, started + time_budget(limits)
    )


def now() -> float:
    return time.monotonic()
