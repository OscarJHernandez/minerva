"""Vector calculus: evaluate, Jacobian and conversion, with the two routes kept apart.

`evaluate` reads a typed vector expression over declared fields and constants, builds the result in the
orthonormal frame of the chosen coordinate system (`lower.py`, exact) and shows it only if the
independent Cartesian finite-difference reading (`numeric.py`) agrees. A result that needs mixed
partial derivatives to commute is `conditions_not_satisfied` unless the field was declared `c2` or
`smooth`. Numerical agreement is evidence, not proof, and is worded that way.
"""

from __future__ import annotations

import time

import sympy as sp

from ..errors import LimitError, MinervaMathError, ParseError
from ..models import Limits, Status
from .algebra import exact_zero, regularise
from .context import MAX_OPERATOR_DEPTH, SEED, Refused, build_context, tidy, time_budget
from .coordinates import SYSTEMS, System, get_system, source_in_target
from .expressions import depth_of_operators, kind_of
from .lower import Lowered, Lowering, Undeclared, Vec
from .models import (
    STEP_ACTIONS,
    VectorDiagnostics,
    VectorInput,
    VectorResult,
    VectorStage,
)
from .numeric import verify_conversion, verify_jacobian, verify_result
from .parse import parse_vector_latex, parse_vector_text
from .printing import vector_latex

GENERAL_CONVENTIONS = (
    "Fields and constants are real-valued when sampled; coordinates are real.",
    "Square-root, logarithm and inverse-function branch domains are not analysed.",
)

_Refused = Refused
_tidy = tidy


def _build(
    inp: VectorInput, system: System, limits: Limits
) -> tuple[Lowering, dict[str, list[str]], list[str]]:
    ctx = build_context(inp, limits, time.monotonic())
    return ctx.low, ctx.assumptions, ctx.vector_names


def _factors(den: sp.Expr) -> list[sp.Expr]:
    return [f for f, _power in sp.factor_list(den)[1] if f.free_symbols]


def _denominators(exprs: list[sp.Expr]) -> list[sp.Expr]:
    """Irreducible factors of the denominators of `exprs` (which may contain coordinates, constants
    and field jets)."""
    out: list[sp.Expr] = []
    for e in exprs:
        _, den = sp.fraction(sp.together(regularise(e)))
        if den.free_symbols:
            out += [f for f in _factors(den) if f not in out]
    return out


def _restrictions(
    system: System, written: list[sp.Expr], results: list[sp.Expr], low: Lowering
) -> tuple[list[str], bool]:
    """Where the coordinates break down (the system's own axes) and every denominator of the input
    (`written`: denominators met while reading) and of the result. Each condition once."""
    conditions: list[sp.Expr] = list(system.singular_conditions())
    complete = True
    try:
        found: list[sp.Expr] = []
        for d in written:
            found += _factors(sp.together(regularise(d)))
        found += _denominators(results)
        for f in found:
            if f not in conditions:
                conditions.append(f)
    except (sp.PolynomialError, ValueError, NotImplementedError):
        complete = False
    return [rf"{vector_latex(low.display(c))} \neq 0" for c in conditions], complete


def _vector_latex(system: System, comps: list[sp.Expr], low: Lowering) -> str:
    terms: list[str] = []
    for i, c in enumerate(comps):
        if c == 0:
            continue
        text = vector_latex(low.display(c))
        if isinstance(c, sp.Add):
            text = rf"\left({text}\right)"
        terms.append(rf"{text}\,{system.unit_latex(i)}")
    return " + ".join(terms) if terms else r"\mathbf{0}"


def _stage(stages: list[VectorStage], name: str, started: float) -> None:
    stages.append(VectorStage(name=name, duration_ms=int((time.monotonic() - started) * 1000)))


def vector_calculus(inp: VectorInput, limits: Limits | None = None) -> VectorResult:
    limits = limits or Limits()
    started = time.monotonic()
    diag = VectorDiagnostics(action=inp.action, coordinates=inp.coordinates, trace_id=inp.trace_id)
    result = VectorResult(
        status=Status.INTERNAL_ERROR,
        action=inp.action,
        coordinates=inp.coordinates,
        to=inp.to,
        diagnostics=diag,
    )
    try:
        _run(inp, limits, result, diag, started)
    except _Refused as refusal:
        result.status, result.message, diag.reason = refusal.status, refusal.message, refusal.reason
    except Undeclared as exc:
        result.status, result.message = Status.PARSE_ERROR, str(exc)
        result.undeclared = exc.names
        diag.reason = "undeclared_symbol"
    except LimitError as exc:
        result.status, result.message, diag.reason = (
            Status.RESOURCE_LIMIT,
            str(exc),
            "resource_limit",
        )
    except ParseError as exc:
        result.status, result.message = Status.PARSE_ERROR, str(exc)
        diag.reason = (
            "type_error"
            if "vector" in str(exc) or "scalar" in str(exc)
            else "unsupported_expression"
        )
    except MinervaMathError as exc:
        result.status, result.message, diag.reason = (
            Status.UNSUPPORTED,
            str(exc),
            "unsupported_expression",
        )
    result.duration_ms = int((time.monotonic() - started) * 1000)
    return result


def _run(
    inp: VectorInput, limits: Limits, result: VectorResult, diag: VectorDiagnostics, started: float
) -> None:
    system = get_system(inp.coordinates)
    deadline = started + time_budget(limits)
    if inp.action not in ("jacobian", "em_derive", "em_compute") and not inp.expression.strip():
        raise _Refused(Status.PARSE_ERROR, "an expression is required", "unsupported_expression")
    result.conventions = [*system.conventions, *GENERAL_CONVENTIONS]
    if inp.action == "jacobian":
        _jacobian(system, result, diag, deadline)
        return

    if inp.action == "em_compute":
        from .em_numeric import run_em_compute

        ctx = build_context(inp, limits, started)
        result.conventions = [*system.conventions, *GENERAL_CONVENTIONS]
        run_em_compute(ctx, result, diag)
        return

    if inp.action == "integrate":
        from .integrals import run_integral

        ctx = build_context(inp, limits, started)
        diag.abstract_fields = sum(1 for d in ctx.low.declared.values() if d.explicit is None)
        diag.explicit_fields = sum(1 for d in ctx.low.declared.values() if d.explicit is not None)
        result.conventions = [*system.conventions, *GENERAL_CONVENTIONS]
        run_integral(ctx, result, diag)
        return

    if inp.action in STEP_ACTIONS:
        from .steps import run_step

        ctx = build_context(inp, limits, started)
        diag.abstract_fields = sum(1 for d in ctx.low.declared.values() if d.explicit is None)
        diag.explicit_fields = sum(1 for d in ctx.low.declared.values() if d.explicit is not None)
        result.conventions = [*system.conventions, *GENERAL_CONVENTIONS]
        run_step(ctx, result, diag)
        return

    low, assumptions, vector_names = _build(inp, system, limits)
    diag.abstract_fields = sum(1 for d in low.declared.values() if d.explicit is None)
    diag.explicit_fields = sum(1 for d in low.declared.values() if d.explicit is not None)

    t0 = time.monotonic()
    denominators: list[sp.Expr] = []
    if inp.format == "latex":
        expr = parse_vector_latex(
            inp.expression,
            vectors=vector_names,
            directions=system.coords,
            assumptions=assumptions,
            limits=limits,
            denominators=denominators,
        )
    else:
        expr = parse_vector_text(
            inp.expression,
            vectors=vector_names,
            names=[*low.declared, *low.constants],
            directions=system.coords,
            assumptions=assumptions,
            limits=limits,
            denominators=denominators,
        )
    kind = kind_of(expr)
    unknown = low.unknown_symbols(expr)
    if unknown:
        raise Undeclared(unknown)
    diag.operator_depth = depth_of_operators(expr)
    diag.result_kind = kind
    _stage(diag.stages, "parse", t0)
    result.input_latex = vector_latex(expr)
    if diag.operator_depth > MAX_OPERATOR_DEPTH:
        raise _Refused(
            Status.RESOURCE_LIMIT,
            f"operators are nested {diag.operator_depth} deep; the limit is {MAX_OPERATOR_DEPTH}",
            "derivative_depth",
        )

    if inp.action == "convert":
        _convert(inp, low, system, expr, kind, result, diag, deadline)
        return

    t0 = time.monotonic()
    lowered = low.lower(expr)
    _stage(diag.stages, "lower", t0)
    t0 = time.monotonic()
    comps = [_tidy(c) for c in lowered] if isinstance(lowered, tuple) else [_tidy(lowered)]
    comps = [sp.Integer(0) if exact_zero(c) else c for c in comps]
    comps = [sp.Integer(0) if exact_zero(c) else c for c in comps]
    _stage(diag.stages, "tidy", t0)

    blocked = low.merge_blocks(comps)
    if blocked:
        result.blocked = blocked
        fields = ", ".join(sorted({f for b in blocked for f in b.fields}))
        result.status = Status.CONDITIONS_NOT_SATISFIED
        result.message = (
            f"not shown: the result needs mixed partial derivatives of {fields} to commute, and the "
            "smoothness of that field was not declared (use smoothness c2, or smooth for higher orders)"
        )
        diag.reason = "needs_smoothness"
        return

    t0 = time.monotonic()
    verification = verify_result(
        low, expr, comps, kind == "vector", seed=SEED, deadline=deadline, depth=diag.operator_depth
    )
    _stage(diag.stages, "verify", t0)
    diag.points = verification.points_checked
    result.verification = verification
    if not verification.numeric_passed:
        result.status = Status.NOT_VERIFIED
        result.message = (
            "a result was computed but the independent numerical check did not confirm it, so it is "
            "not shown: " + verification.note
        )
        diag.reason = (
            "too_few_points"
            if verification.points_checked == verification.points_agreed
            else "verification_failed"
        )
        return

    _present(inp, low, system, kind, comps, expr, denominators, result, diag)


def _present(
    inp: VectorInput,
    low: Lowering,
    system: System,
    kind: str,
    comps: list[sp.Expr],
    expr: sp.Expr,
    denominators: list[sp.Expr],
    result: VectorResult,
    diag: VectorDiagnostics,
) -> None:
    t0 = time.monotonic()
    result.result_kind = "vector" if kind == "vector" else "scalar"
    shown = [low.display(c) for c in comps]
    if kind == "vector":
        result.components_latex = [vector_latex(c) for c in shown]
        result.components_text = [str(c) for c in shown]
        result.basis_latex = [system.unit_latex(i) for i in range(3)]
        result.result_latex = _vector_latex(system, comps, low)
        result.result_text = "(" + ", ".join(result.components_text) + ")"
    else:
        result.result_latex = vector_latex(shown[0])
        result.result_text = str(shown[0])
    result.restrictions_latex, result.restrictions_complete = _restrictions(
        system, denominators, comps, low
    )
    used = ["every field is differentiable as often as the expression differentiates it"]
    sorted_fields = sorted(
        {
            d.decl.name
            for d in low.declared.values()
            if d.decl.smoothness in ("c2", "smooth") and d.explicit is None
        }
    )
    for name in sorted_fields:
        smooth = next(d.decl.smoothness for d in low.declared.values() if d.decl.name == name)
        used.append(
            f"{name} declared {smooth}: its mixed partial derivatives commute (as declared)"
        )
    result.assumptions_used = used
    result.status = Status.OK
    result.message = (
        "numerically checked against an independent Cartesian finite-difference evaluation "
        f"at {result.verification.points_agreed if result.verification else 0} points; "
        "this is evidence, not a proof"
    )
    _stage(diag.stages, "render", t0)


# -- conversion between coordinate systems ----------------------------------------------------
def _convert(
    inp: VectorInput,
    low: Lowering,
    system: System,
    expr: sp.Expr,
    kind: str,
    result: VectorResult,
    diag: VectorDiagnostics,
    deadline: float,
) -> None:
    if inp.to is None or inp.to == inp.coordinates:
        raise _Refused(
            Status.PARSE_ERROR,
            "convert needs `to`, a different coordinate system",
            "bad_declaration",
        )
    if any(
        d.explicit is None
        for d in low.declared.values()
        if d.decl.name in {s.name for s in expr.free_symbols}
    ):
        raise _Refused(
            Status.UNSUPPORTED,
            "convert needs explicit fields (give `components`); an abstract field has no formula "
            "to re-express",
            "unsupported_expression",
        )
    target = get_system(inp.to)
    t0 = time.monotonic()
    lowered = low.lower(expr)
    values: list[Lowered] = list(lowered) if isinstance(lowered, tuple) else [lowered]
    mapping = source_in_target(system, target)
    substitution = {
        sym: mapping[name] for sym, name in zip(system.symbols(), system.coords, strict=True)
    }
    new_values: list[sp.Expr] = [sp.sympify(v).xreplace(substitution) for v in values]
    if kind == "vector":
        cart = system.frame().xreplace(substitution) * sp.Matrix(new_values)
        new_values = list(target.frame().T * cart)
    comps = [_tidy(v) for v in new_values]
    _stage(diag.stages, "lower", t0)
    t0 = time.monotonic()
    target_low = Lowering(target, {}, low.constants)
    verification = verify_conversion(
        system,
        target,
        [sp.sympify(v) for v in values],
        comps,
        kind == "vector",
        (target.symbols()[0], target_low.t),
        list(low.constants.values()),
        seed=SEED,
        deadline=deadline,
    )
    _stage(diag.stages, "verify", t0)
    diag.points = verification.points_checked
    result.verification = verification
    result.conventions = [*target.conventions, *GENERAL_CONVENTIONS]
    result.coordinates = target.name
    if not verification.numeric_passed:
        result.status = Status.NOT_VERIFIED
        result.message = (
            "the conversion was computed but the independent check did not confirm it: "
            + verification.note
        )
        diag.reason = "verification_failed"
        return
    result.result_kind = "vector" if kind == "vector" else "scalar"
    if kind == "vector":
        result.components_latex = [vector_latex(c) for c in comps]
        result.components_text = [str(c) for c in comps]
        result.basis_latex = [target.unit_latex(i) for i in range(3)]
        result.result_latex = _vector_latex(target, comps, target_low)
        result.result_text = "(" + ", ".join(result.components_text) + ")"
    else:
        result.result_latex = vector_latex(comps[0])
        result.result_text = str(comps[0])
    result.restrictions_latex, result.restrictions_complete = _restrictions(
        target, [], comps, target_low
    )
    result.assumptions_used = ["the change of coordinates holds where both systems are regular"]
    result.status = Status.OK
    result.message = (
        f"converted from {system.name} to {target.name} and checked numerically by an independent "
        "route (through Cartesian coordinates); this is evidence, not a proof"
    )


# -- Jacobian ---------------------------------------------------------------------------------
def _jacobian(
    system: System, result: VectorResult, diag: VectorDiagnostics, deadline: float
) -> None:
    q = system.symbols()
    x = system.to_cartesian()
    matrix = sp.Matrix(3, 3, lambda i, j: sp.diff(x[i], q[j]))
    determinant = sp.simplify(matrix.det())
    verification = verify_jacobian(system, matrix, determinant, seed=SEED, deadline=deadline)
    result.verification = verification
    diag.points = verification.points_checked
    diag.result_kind = "matrix"
    if not verification.numeric_passed:
        result.status = Status.NOT_VERIFIED
        result.message = (
            "the Jacobian was not confirmed by the independent check: " + verification.note
        )
        diag.reason = "verification_failed"
        return
    result.result_kind = "matrix"
    result.result_latex = vector_latex(matrix)
    result.determinant_latex = vector_latex(determinant)
    result.scale_factors_latex = [vector_latex(h) for h in system.scale_factors()]
    result.basis_latex = [system.unit_latex(i) for i in range(3)]
    result.restrictions_latex = list(system.restriction_latex)
    result.status = Status.OK
    result.message = (
        "Jacobian of the map to Cartesian coordinates, checked against central differences of the map; "
        "this is evidence, not a proof"
    )


__all__ = ["SYSTEMS", "Vec", "vector_calculus"]
