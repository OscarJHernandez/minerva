"""Dimensional analysis: dimensionless groups, missing quantities, and unit checks of equations.

Given quantities with units (or named kinds) and optionally one unknown ("target"), this finds the
dimensionless groups (Buckingham Pi) as the null space of the dimension matrix and says what the
dimensions alone can tell about the unknown:

* `determined`: one group contains the unknown, so it is fixed up to a dimensionless constant;
* `up_to_function`: further groups exist, so the unknown is that times a function of them;
* `missing`: the unknown's dimension is not a product of powers of the knowns' dimensions. The
  report gives the dimension a missing quantity must carry and which handbook constants (tested
  one at a time, then in pairs) would close the gap.

Every group is checked independently of the null-space computation: it is rebuilt as an expression
and its dimension evaluated by `dimensions.dimension_of`; a group that is not dimensionless is never
shown. Dimensional analysis does not decide whether the list of quantities is physically complete,
and it does not determine the function of the groups or any dimensionless constant.
"""

from __future__ import annotations

import itertools
import re
import time
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd, lcm
from typing import Literal

import sympy as sp
from pydantic import BaseModel, Field, field_validator, model_validator

from .claims import split_equation
from .dimensions import (
    BASE_UNITS,
    ZERO,
    Dim,
    DimensionError,
    UnsupportedDimensionConstruct,
    dim_latex,
    dim_text,
    dimension_of,
    read_kind,
    read_unit,
)
from .errors import MinervaMathError
from .latexparse import parse_latex
from .models import SYMBOL_NAME_MAX, Limits, Status

MAX_QUANTITIES = 16
MAX_CANDIDATES = 32
MAX_SUGGESTIONS = 6
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_PLACEHOLDER = re.compile(r"Qq[a-z]{2}")

Closure = Literal["determined", "up_to_function", "missing", "none"]


# --- models ---------------------------------------------------------------------------------------
class QuantitySpec(BaseModel):
    """A named quantity with either a unit (`kg m/s^2`, `J s`) or a kind (`dynamic viscosity`)."""

    name: str
    unit: str | None = Field(default=None, max_length=200)
    kind: str | None = Field(default=None, max_length=64)
    meaning: str = Field(default="", max_length=100)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if len(value) > SYMBOL_NAME_MAX or not _NAME.match(value):
            raise ValueError(
                f"invalid quantity name {value!r}: use letters, digits and underscores, "
                "starting with a letter (theta, rho, x_0)"
            )
        return value

    @model_validator(mode="after")
    def _one_of_unit_or_kind(self) -> QuantitySpec:
        if (self.unit is None) == (self.kind is None):
            raise ValueError(f"give exactly one of unit or kind for {self.name!r}")
        return self


class ConstantCandidate(BaseModel):
    """A physical constant offered as a possible missing quantity (from the handbook)."""

    id: str = Field(max_length=80)
    latex: str = Field(max_length=40)
    unit: str = Field(max_length=200)


class QuantityReading(BaseModel):
    """How a quantity was read: what was given and the base-unit dimension it became."""

    name: str
    latex: str
    meaning: str = ""
    given: str
    dimension_text: str
    dimension_latex: str


class PiGroup(BaseModel):
    label: str
    latex: str
    exponents: dict[str, str]
    contains_target: bool = False
    dimensionless_verified: bool = False


class ConstantSuggestion(BaseModel):
    ids: list[str]
    latex: list[str]
    closure: Closure
    relation_latex: str | None = None
    group_latex: str | None = None
    verified: bool = False


class MissingReport(BaseModel):
    target_dimension_text: str
    target_dimension_latex: str
    unbalanced: list[str]
    missing_dimension_text: str
    missing_dimension_latex: str
    verified: bool = False
    explanation: str
    suggestions: list[ConstantSuggestion] = Field(default_factory=list)


class DimensionalReport(BaseModel):
    status: Status
    quantities: list[QuantityReading] = Field(default_factory=list)
    rank: int = 0
    target: str | None = None
    repeating: list[str] = Field(default_factory=list)
    groups: list[PiGroup] = Field(default_factory=list)
    closure: Closure | None = None
    relation_latex: str | None = None
    cannot_enter: list[str] = Field(default_factory=list)
    missing: MissingReport | None = None
    tested: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0


class SideReading(BaseModel):
    latex: str
    dimension_text: str
    dimension_latex: str


class UnitsCheckReport(BaseModel):
    status: Status
    equation_latex: str | None = None
    consistent: bool | None = None
    sides: list[SideReading] = Field(default_factory=list)
    missing_factor_text: str | None = None
    missing_factor_latex: str | None = None
    problem: str | None = None
    problem_terms: list[SideReading] = Field(default_factory=list)
    quantities: list[QuantityReading] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0


def _fail(status: Status, message: str) -> DimensionalReport:
    return DimensionalReport(status=status, message=message)


# --- helpers ---------------------------------------------------------------------------------------
def _placeholder(i: int) -> str:
    return "Qq" + chr(97 + i // 26) + chr(97 + i % 26)


def _atom(display: str) -> str:
    atomic = re.fullmatch(r"(\\[A-Za-z]+|[A-Za-z0-9]|\\mathrm\{[A-Za-z]+\})", display)
    return display if atomic else "{" + display + "}"


def _tex(expr: sp.Expr, displays: list[str]) -> str:
    """LaTeX of an expression built from placeholder symbols, with their display forms put back."""
    text = sp.latex(expr)
    return _PLACEHOLDER.sub(lambda m: _atom(displays[_placeholder_index(m.group())]), text)


def _placeholder_index(name: str) -> int:
    return (ord(name[2]) - 97) * 26 + (ord(name[3]) - 97)


def _symbol_latex(name: str) -> str:
    text = sp.latex(sp.Symbol(name))
    return rf"\mathrm{{{text}}}" if re.fullmatch(r"[A-Za-z]{2,}", text) else text


def _rational(f: Fraction) -> sp.Rational:
    return sp.Rational(f.numerator, f.denominator)


def _matrix(dims: list[Dim], order: list[int]) -> sp.Matrix:
    return sp.Matrix(len(BASE_UNITS), len(order), lambda r, c: _rational(dims[order[c]][r]))


def _nullspace(matrix: sp.Matrix) -> list[sp.Matrix]:
    return list(matrix.nullspace())


def _read_quantities(specs: list[QuantitySpec]) -> tuple[list[QuantityReading], list[Dim]]:
    readings: list[QuantityReading] = []
    dims: list[Dim] = []
    for spec in specs:
        try:
            d = read_unit(spec.unit) if spec.unit is not None else read_kind(spec.kind or "")
        except MinervaMathError as exc:
            raise type(exc)(f"{spec.name}: {exc}") from exc
        dims.append(d)
        readings.append(
            QuantityReading(
                name=spec.name,
                latex=_symbol_latex(spec.name),
                meaning=spec.meaning,
                given=spec.unit if spec.unit is not None else f"kind: {spec.kind}",
                dimension_text=dim_text(d),
                dimension_latex=dim_latex(d),
            )
        )
    return readings, dims


def _integer_direction(d: Dim) -> Dim:
    """Smallest integer vector in the direction of `d`, first non-zero entry positive."""
    denominator = lcm(*(x.denominator for x in d))
    ints = [int(x * denominator) for x in d]
    divisor = gcd(*ints) or 1
    sign = -1 if next(x for x in ints if x) < 0 else 1
    return tuple(Fraction(sign * x, divisor) for x in ints)


# --- one analysis of a fixed list of quantities ---------------------------------------------------
@dataclass
class _Analysis:
    ok: bool
    problem: str = ""
    rank: int = 0
    groups: list[PiGroup] = field(default_factory=list)
    closure: Closure = "none"
    relation_latex: str | None = None
    repeating_used: list[str] = field(default_factory=list)
    dependent_requested: list[str] = field(default_factory=list)
    cannot_enter: list[str] = field(default_factory=list)


def _analyse(
    names: list[str],
    displays: list[str],
    dims: list[Dim],
    target: int | None,
    repeating: list[int],
) -> _Analysis:
    n = len(names)
    rest = [i for i in range(n) if i != target and i not in repeating]
    order = [*repeating, *rest, *([target] if target is not None else [])]
    matrix = _matrix(dims, order)
    rank = int(matrix.rank())
    pivots = set(matrix.rref()[1])
    null = _nullspace(matrix)
    if len(null) != n - rank:
        return _Analysis(False, "the number of groups does not equal quantities minus rank")

    raw: list[tuple[int, dict[int, Fraction]]] = []  # (free column, exponents by quantity index)
    for vector in null:
        free = [k for k in range(n) if k not in pivots and vector[k] != 0]
        if len(free) != 1 or vector[free[0]] != 1:
            return _Analysis(False, "the null-space basis was not in reduced form")
        exps = {
            order[k]: Fraction(int(vector[k].p), int(vector[k].q))
            for k in range(n)
            if vector[k] != 0
        }
        is_target = target is not None and order[free[0]] == target
        if not is_target:
            scale = lcm(*(e.denominator for e in exps.values()))
            exps = {i: e * scale for i, e in exps.items()}
        raw.append((free[0], exps))
    raw.sort(key=lambda item: (target is None or order[item[0]] != target, item[0]))

    # Independent verification: rebuild each group as an expression and take its dimension.
    symbols = [sp.Symbol(_placeholder(i)) for i in range(n)]
    units = {_placeholder(i): dims[i] for i in range(n)}
    groups: list[PiGroup] = []
    exponent_rows: list[list[sp.Rational]] = []
    for number, (_, exps) in enumerate(raw, start=1):
        expr = sp.Mul(*(symbols[i] ** _rational(e) for i, e in exps.items()))
        try:
            dimensionless = dimension_of(expr, units) == ZERO
        except MinervaMathError:
            dimensionless = False
        if not dimensionless:
            return _Analysis(False, f"group {number} is not dimensionless")
        groups.append(
            PiGroup(
                label=rf"\Pi_{{{number}}}",
                latex=_tex(expr, displays),
                exponents={names[i]: str(e) for i, e in sorted(exps.items())},
                contains_target=target is not None and target in exps,
                dimensionless_verified=True,
            )
        )
        exponent_rows.append([_rational(exps.get(i, Fraction(0))) for i in range(n)])
    if exponent_rows and sp.Matrix(exponent_rows).rank() != len(exponent_rows):
        return _Analysis(False, "the groups are not independent")
    if target is not None and any(
        g.contains_target and raw[k][1].get(target) != 1 for k, g in enumerate(groups)
    ):
        return _Analysis(False, "the unknown does not have exponent 1 in its group")

    analysis = _Analysis(
        True,
        rank=rank,
        groups=groups,
        repeating_used=[names[order[k]] for k in sorted(pivots) if order[k] != target],
        dependent_requested=[names[order[k]] for k in range(len(repeating)) if k not in pivots],
    )
    if target is None:
        return analysis
    if not groups or not groups[0].contains_target:
        analysis.closure = "missing"
        return analysis

    # Solve the group containing the unknown for the unknown: T = const * prod(q_i^-e_i).
    exps = raw[0][1]
    others = _product({i: -e for i, e in exps.items() if i != target}, symbols)
    others_tex = "" if others == 1 else _tex(others, displays)
    target_tex = _tex(symbols[target], displays)
    const = "C_{0}" if "C" in displays else "C"
    if len(groups) == 1:
        analysis.closure = "determined"
        analysis.relation_latex = (
            rf"{target_tex} = {const}\,{others_tex}" if others_tex else rf"{target_tex} = {const}"
        )
        analysis.cannot_enter = [
            names[order[k]] for k in sorted(pivots) if order[k] != target and order[k] not in exps
        ]
    else:
        analysis.closure = "up_to_function"
        args = ",".join(g.label for g in groups[1:])
        analysis.relation_latex = (
            rf"{target_tex} = {others_tex}\,f\left({args}\right)"
            if others_tex
            else rf"{target_tex} = f\left({args}\right)"
        )
    return analysis


def _product(exps: dict[int, Fraction], symbols: list[sp.Symbol]) -> sp.Expr:
    """Product of powers, with fractional powers of a ratio written together: sqrt(L/g)."""
    terms: list[sp.Expr] = []
    by_size: dict[Fraction, list[tuple[int, bool]]] = {}
    for i, e in exps.items():
        if e.denominator == 1:
            terms.append(symbols[i] ** _rational(e))
        else:
            by_size.setdefault(abs(e), []).append((i, e > 0))
    for size, items in by_size.items():
        up = [symbols[i] for i, positive in items if positive]
        down = [symbols[i] for i, positive in items if not positive]
        if up and down:
            ratio = sp.Mul(*up) / sp.Mul(*down)
            terms.append(sp.Pow(ratio, _rational(size), evaluate=False))
        else:
            terms.extend(symbols[i] ** _rational(size if p else -size) for i, p in items)
    return sp.Mul(*terms)


def _is_missing(dims: list[Dim], target: int) -> bool:
    """True if the target's dimension is not a product of powers of the other quantities'."""
    others = [i for i in range(len(dims)) if i != target]
    order = [*others, target]
    return len(dims) - 1 in set(_matrix(dims, order).rref()[1])


def _missing_dimension(dims: list[Dim], target: int) -> Dim:
    """What is left of the target's dimension after using up everything the knowns can supply."""
    others = [dims[i] for i in range(len(dims)) if i != target]
    if others:
        basis = sp.Matrix([[_rational(x) for x in d] for d in others]).rref()
        rows, pivots = basis
        residual = [_rational(x) for x in dims[target]]
        for row, pivot in zip(range(len(pivots)), pivots, strict=True):
            factor = residual[pivot]
            residual = [residual[c] - factor * rows[row, c] for c in range(len(BASE_UNITS))]
    else:
        residual = [_rational(x) for x in dims[target]]
    return _integer_direction(tuple(Fraction(int(x.p), int(x.q)) for x in residual))


def _suggest(
    names: list[str],
    displays: list[str],
    dims: list[Dim],
    target: int,
    candidates: list[ConstantCandidate],
    limits: Limits,
    tested: list[str],
) -> list[ConstantSuggestion]:
    usable: list[tuple[ConstantCandidate, Dim, str]] = []
    skipped: list[str] = []
    for c in candidates[:MAX_CANDIDATES]:
        try:
            d = read_unit(c.unit)
        except MinervaMathError:
            skipped.append(c.id)
            continue
        if d == ZERO:
            continue
        shown = (
            c.latex
            if c.latex not in {displays[i] for i in range(len(names))}
            else (c.latex + r"_{\mathrm{const}}")
        )
        usable.append((c, d, shown))
    if skipped:
        tested.append("Skipped constants whose unit could not be read: " + ", ".join(skipped))

    deadline = time.monotonic() + 0.6 * limits.timeout_seconds

    def closes(extra: tuple[int, ...]) -> _Analysis | None:
        all_dims = [d for i, d in enumerate(dims) if i != target]
        all_dims = [*all_dims, *(usable[k][1] for k in extra), dims[target]]
        if _is_missing(all_dims, len(all_dims) - 1):
            return None
        all_names = [n for i, n in enumerate(names) if i != target]
        all_names = [*all_names, *(usable[k][0].id for k in extra), names[target]]
        all_disp = [d for i, d in enumerate(displays) if i != target]
        all_disp = [*all_disp, *(usable[k][2] for k in extra), displays[target]]
        analysis = _analyse(all_names, all_disp, all_dims, len(all_dims) - 1, [])
        return analysis if analysis.ok and analysis.closure != "missing" else None

    found: list[ConstantSuggestion] = []

    def record(extra: tuple[int, ...]) -> None:
        analysis = closes(extra)
        if analysis is None or len(found) >= MAX_SUGGESTIONS:
            return
        found.append(
            ConstantSuggestion(
                ids=[usable[k][0].id for k in extra],
                latex=[usable[k][2] for k in extra],
                closure=analysis.closure,
                relation_latex=analysis.relation_latex,
                group_latex=analysis.groups[0].latex if analysis.groups else None,
                verified=True,
            )
        )

    for k in range(len(usable)):
        record((k,))
    tested.append(
        f"Tested each of {len(usable)} handbook constants alone: {len(found)} close the gap."
    )
    if not found:
        pairs = 0
        for pair in itertools.combinations(range(len(usable)), 2):
            if time.monotonic() > deadline:
                tested.append("Stopped searching pairs of constants: time budget used up.")
                break
            pairs += 1
            record(pair)
        tested.append(f"Tested {pairs} pairs of constants: {len(found)} close the gap.")
    tested.append("Combinations of three or more constants are not searched.")
    return found


# --- public operation: dimensional analysis -------------------------------------------------------
def analyze_dimensions(
    quantities: list[QuantitySpec],
    target: str | None,
    repeating: list[str],
    candidates: list[ConstantCandidate],
    limits: Limits,
) -> DimensionalReport:
    if not 1 <= len(quantities) <= MAX_QUANTITIES:
        return _fail(Status.UNSUPPORTED, f"Give between 1 and {MAX_QUANTITIES} quantities.")
    names = [q.name for q in quantities]
    if len(set(names)) != len(names):
        return _fail(Status.PARSE_ERROR, "Quantity names must be distinct.")
    if target is not None and target not in names:
        return _fail(Status.PARSE_ERROR, f"The unknown {target!r} is not among the quantities.")
    if len(set(repeating)) != len(repeating):
        return _fail(Status.PARSE_ERROR, "Repeating quantities must be distinct.")
    for r in repeating:
        if r not in names:
            return _fail(
                Status.PARSE_ERROR, f"Repeating quantity {r!r} is not among the quantities."
            )
        if r == target:
            return _fail(Status.PARSE_ERROR, "The unknown cannot be a repeating quantity.")
    try:
        readings, dims = _read_quantities(quantities)
    except MinervaMathError as exc:
        return _fail(Status.PARSE_ERROR, str(exc))

    displays = [r.latex for r in readings]
    t = names.index(target) if target is not None else None
    rep = [names.index(r) for r in repeating]
    analysis = _analyse(names, displays, dims, t, rep)
    if not analysis.ok:
        return DimensionalReport(
            status=Status.NOT_VERIFIED,
            quantities=readings,
            message="A set of groups was produced but failed its independent dimension check, "
            f"so it is not shown ({analysis.problem}).",
        )

    notes = [
        "Dimensional analysis fixes the groups, not the function of them or any dimensionless "
        "constant, and it cannot tell whether the list of quantities is complete.",
        "The choice of groups is not unique: products and powers of groups are also groups.",
    ]
    if analysis.dependent_requested:
        notes.append(
            "Not used as repeating quantities because their dimensions depend on earlier ones: "
            + ", ".join(analysis.dependent_requested)
            + "."
        )
    report = DimensionalReport(
        status=Status.OK,
        quantities=readings,
        rank=analysis.rank,
        target=target,
        repeating=analysis.repeating_used,
        groups=analysis.groups,
        closure=analysis.closure if target is not None else None,
        relation_latex=analysis.relation_latex,
        cannot_enter=analysis.cannot_enter,
        notes=notes,
        features=["dimensional:pi-groups"],
    )
    count = len(analysis.groups)

    if t is None:
        report.message = (
            f"{len(names)} quantities of rank {analysis.rank} give {count} independent "
            f"dimensionless group{'s' if count != 1 else ''}, each checked to be dimensionless."
            if count
            else f"The {len(names)} quantities have rank {analysis.rank}: no dimensionless group "
            "can be formed from them."
        )
        return report

    assert t is not None  # noqa: S101
    report.features.append("dimensional:target")
    if analysis.closure != "missing":
        report.message = (
            f"The dimensions fix {target} up to a dimensionless constant."
            if analysis.closure == "determined"
            else f"{target} is fixed up to a function of the other {count - 1} "
            f"group{'s' if count != 2 else ''}."
        )
        return report
    return _with_missing(report, names, displays, dims, t, candidates, limits)


def _with_missing(
    report: DimensionalReport,
    names: list[str],
    displays: list[str],
    dims: list[Dim],
    t: int,
    candidates: list[ConstantCandidate],
    limits: Limits,
) -> DimensionalReport:
    missing_dim = _missing_dimension(dims, t)
    # Re-solve with a hypothetical quantity of that dimension appended: the unknown must now be in a group.
    verified = not _is_missing([*dims[:t], *dims[t + 1 :], missing_dim, dims[t]], len(dims))
    if not verified:
        return DimensionalReport(
            status=Status.NOT_VERIFIED,
            quantities=report.quantities,
            message="The missing dimension could not be verified, so it is not shown.",
        )
    unbalanced = [u for u, e in zip(BASE_UNITS, missing_dim, strict=True) if e != 0]
    tested: list[str] = []
    suggestions = _suggest(names, displays, dims, t, candidates, limits, tested)
    report.features.append("dimensional:missing-quantity")
    report.tested = tested
    report.missing = MissingReport(
        target_dimension_text=dim_text(dims[t]),
        target_dimension_latex=dim_latex(dims[t]),
        unbalanced=unbalanced,
        missing_dimension_text=dim_text(missing_dim),
        missing_dimension_latex=dim_latex(missing_dim),
        verified=True,
        explanation=(
            f"{names[t]} has dimension {dim_text(dims[t])}, which is not a product of powers of "
            f"the dimensions of the other quantities. What they cannot supply is {dim_text(missing_dim)}: "
            "a missing quantity must carry that dimension, up to a power and a product of the knowns."
        ),
        suggestions=suggestions,
    )
    report.message = (
        f"{names[t]} cannot be built from the other quantities: a quantity with dimension "
        f"{dim_text(missing_dim)} is missing."
    )
    return report


# --- public operation: unit check of one equation -----------------------------------------------
def _side(expr: sp.Expr, d: Dim) -> SideReading:
    return SideReading(
        latex=sp.latex(expr), dimension_text=dim_text(d), dimension_latex=dim_latex(d)
    )


def check_units(
    equation_latex: str,
    quantities: list[QuantitySpec],
    limits: Limits,
) -> UnitsCheckReport:
    def fail(status: Status, message: str) -> UnitsCheckReport:
        return UnitsCheckReport(status=status, message=message)

    if not quantities or len(quantities) > MAX_QUANTITIES:
        return fail(Status.UNSUPPORTED, f"Give between 1 and {MAX_QUANTITIES} quantities.")
    names = [q.name for q in quantities]
    if len(set(names)) != len(names):
        return fail(Status.PARSE_ERROR, "Quantity names must be distinct.")
    sides = split_equation(equation_latex)
    if sides is None:
        return fail(Status.PARSE_ERROR, "Write an equation with an = sign.")
    try:
        readings, dims = _read_quantities(quantities)
        parsed = [parse_latex(s, {}, limits) for s in sides]
    except MinervaMathError as exc:
        return fail(Status.PARSE_ERROR, str(exc))
    units = {n: d for n, d in zip(names, dims, strict=True)}

    report = UnitsCheckReport(
        status=Status.OK,
        equation_latex=" = ".join(sp.latex(p) for p in parsed),
        quantities=readings,
        notes=[
            "A matching dimension does not prove an equation is correct; a mismatch proves it is "
            "wrong as written."
        ],
    )
    side_dims: list[Dim] = []
    try:
        for expr in parsed:
            side_dims.append(dimension_of(expr, units))
    except DimensionError as exc:
        report.consistent = False
        report.problem = str(exc)
        report.problem_terms = [_side(e, d) for e, d in exc.parts]
        report.message = f"Not dimensionally consistent: {exc}."
        return report
    except UnsupportedDimensionConstruct as exc:
        return fail(Status.UNSUPPORTED, str(exc))
    except MinervaMathError as exc:
        return fail(Status.PARSE_ERROR, str(exc))

    report.sides = [_side(e, d) for e, d in zip(parsed, side_dims, strict=True)]
    for k in range(1, len(side_dims)):
        if side_dims[k] != side_dims[0]:
            factor = tuple(a - b for a, b in zip(side_dims[0], side_dims[k], strict=True))
            report.consistent = False
            report.missing_factor_text = dim_text(factor)
            report.missing_factor_latex = dim_latex(factor)
            report.message = (
                f"The sides differ: {dim_text(side_dims[0])} on the left, {dim_text(side_dims[k])} "
                f"on the right. A factor of dimension {dim_text(factor)} would be needed on the right."
            )
            return report
    report.consistent = True
    report.message = f"Both sides have dimension {dim_text(side_dims[0])}."
    return report
