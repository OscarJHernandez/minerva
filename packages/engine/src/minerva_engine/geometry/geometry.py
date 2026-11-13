"""Entry point: coordinates and a metric in, verified geometric quantities out.

Stages run in dependency order and are timed individually (`diagnostics`). A quantity is returned only if
its numeric verification passed; otherwise it is `not_verified` and carries no components. The geometry
(coordinates, functions, metric, assumptions) is part of every request, so any later operation on the same
problem sends the same data and gets the same conventions.
"""

from __future__ import annotations

import resource
import sys
import time
from collections.abc import Callable
from typing import Any

import sympy as sp

from ..errors import LimitError, MinervaMathError, UnsupportedInput
from ..latexprint import to_latex
from ..models import Limits, Status
from .compute import CONVENTIONS, Workspace
from .domain import restrictions, signature
from .models import (
    MAX_DIMENSION,
    TENSOR_QUANTITIES,
    Component,
    GeometryInput,
    GeometryResult,
    GeometryVerification,
    QuantityResult,
    StageDiagnostic,
)
from .setup import Geometry, GeometryError, build_geometry
from .tensors import Tensor, index_label
from .verify import Verifier

DEADLINE_FRACTION = 0.8
SOFT_DEADLINE_FRACTION = 0.5

TITLES = {
    "inverse_metric": "Inverse metric",
    "determinant": "Determinant of the metric",
    "christoffel": "Christoffel symbols (symmetric in the lower indices; one of each pair shown)",
    "riemann": "Riemann tensor R^rho_{sigma mu nu} (antisymmetric in the last two indices; mu < nu shown)",
    "ricci": "Ricci tensor (symmetric; one of each pair shown)",
    "ricci_scalar": "Ricci scalar R",
    "einstein": "Einstein tensor (symmetric; one of each pair shown)",
    "kretschmann": "Kretschmann scalar (full contraction of the Riemann tensor with itself)",
    "ricci_squared": "Ricci-squared invariant (Ricci tensor contracted with itself)",
    "geodesics": "Geodesic equations (affine parameter, dots are derivatives)",
    "covariant_derivative": "Covariant derivative of the tensor (derivative index first)",
    "raise_index": "Tensor with the chosen index raised",
    "lower_index": "Tensor with the chosen index lowered",
}


def geometry(
    data: GeometryInput, limits: Limits, *, max_dimension: int = MAX_DIMENSION
) -> GeometryResult:
    started = time.monotonic()
    deadline = started + DEADLINE_FRACTION * limits.timeout_seconds
    soft = started + SOFT_DEADLINE_FRACTION * limits.timeout_seconds
    diagnostics: list[StageDiagnostic] = []
    try:
        geo = _timed(diagnostics, "setup", lambda: build_geometry(data, limits, max_dimension))
    except GeometryError as exc:
        return _fail(diagnostics, exc.status, exc.message)
    except MinervaMathError as exc:
        if isinstance(exc, LimitError):
            status = Status.RESOURCE_LIMIT
        elif isinstance(exc, UnsupportedInput):
            status = Status.UNSUPPORTED
        else:
            status = Status.PARSE_ERROR
        return _fail(diagnostics, status, str(exc))
    except RecursionError:
        return _fail(diagnostics, Status.RESOURCE_LIMIT, "The expression is nested too deeply.")

    ws = Workspace(geo, limits, deadline, soft)
    ws.diagnostics = diagnostics
    verifier = Verifier(ws)
    result = GeometryResult(
        status=Status.OK,
        dimension=geo.n,
        coordinates_latex=ws.coord_latex,
        metric_latex=to_latex(geo.g),
        conventions=CONVENTIONS,
        assumptions_used=geo.assumptions_used,
    )
    try:
        det = ws.determinant()
        ginv = ws.inverse()
        result.signature, result.signature_note = signature(geo)
        result.restrictions_latex, result.restrictions_complete = restrictions(geo, det, ginv)
    except GeometryError as exc:
        return _fail(ws.diagnostics, exc.status, exc.message, result)
    except (LimitError, RecursionError) as exc:
        message = (
            str(exc) if isinstance(exc, LimitError) else "The expression is nested too deeply."
        )
        return _fail(ws.diagnostics, Status.RESOURCE_LIMIT, message, result)

    seen: set[str] = set()
    for kind in data.quantities:
        if kind in seen:
            continue
        seen.add(kind)
        result.quantities.append(_quantity(kind, data, ws, verifier))

    result.diagnostics = ws.diagnostics
    result.status = _overall(result.quantities)
    result.message = _summary(result)
    result.peak_memory_mb = _peak_memory_mb()
    return result


def _quantity(kind: str, data: GeometryInput, ws: Workspace, verifier: Verifier) -> QuantityResult:
    geo = ws.geo
    title = TITLES[kind]
    try:
        if kind in TENSOR_QUANTITIES and geo.tensor is None:
            raise GeometryError(
                Status.PARSE_ERROR,
                f"{kind.replace('_', ' ')} needs a tensor field; none was given.",
            )
        value = _compute(kind, data, ws)
        verification = _verify(kind, data, ws, verifier)
    except GeometryError as exc:
        _fail_stage(ws, kind, exc.status)
        return QuantityResult(kind=kind, status=exc.status, title=title, message=exc.message)
    except LimitError as exc:
        _fail_stage(ws, kind, Status.RESOURCE_LIMIT)
        return QuantityResult(
            kind=kind,
            status=Status.RESOURCE_LIMIT,
            title=title,
            message=str(exc),
        )
    except RecursionError:
        _fail_stage(ws, kind, Status.RESOURCE_LIMIT)
        return QuantityResult(
            kind=kind,
            status=Status.RESOURCE_LIMIT,
            title=title,
            message="The expression is nested too deeply.",
        )
    except Exception:
        _fail_stage(ws, kind, Status.INTERNAL_ERROR)
        return QuantityResult(
            kind=kind,
            status=Status.INTERNAL_ERROR,
            title=title,
            message="This quantity could not be computed.",
        )

    if not verification.numeric_passed:
        return QuantityResult(
            kind=kind,
            status=Status.NOT_VERIFIED,
            title=title,
            verification=verification,
            message="A computed candidate failed its independent numerical check, so it is not shown.",
        )
    return _present(kind, title, value, verification, data, ws)


def _compute(kind: str, data: GeometryInput, ws: Workspace) -> Any:
    table: dict[str, Callable[[], Any]] = {
        "inverse_metric": ws.inverse,
        "determinant": ws.determinant,
        "christoffel": ws.christoffel,
        "riemann": ws.riemann,
        "ricci": ws.ricci,
        "ricci_scalar": ws.ricci_scalar,
        "einstein": ws.einstein,
        "kretschmann": ws.kretschmann,
        "ricci_squared": ws.ricci_squared,
        "geodesics": ws.accelerations,
        "covariant_derivative": ws.covariant_derivative,
        "raise_index": lambda: ws.index_operation("raise_index", data.tensor_index),
        "lower_index": lambda: ws.index_operation("lower_index", data.tensor_index),
    }
    return table[kind]()


def _verify(
    kind: str, data: GeometryInput, ws: Workspace, verifier: Verifier
) -> GeometryVerification:
    table: dict[str, Callable[[], GeometryVerification]] = {
        "inverse_metric": verifier.inverse_metric,
        "determinant": verifier.determinant,
        "christoffel": verifier.christoffel,
        "riemann": verifier.riemann,
        "ricci": verifier.ricci,
        "ricci_scalar": verifier.ricci_scalar,
        "einstein": verifier.einstein,
        "kretschmann": verifier.kretschmann,
        "ricci_squared": verifier.ricci_squared,
        "geodesics": verifier.geodesics,
        "covariant_derivative": verifier.covariant_derivative,
        "raise_index": lambda: verifier.index_operation("raise_index", data.tensor_index),
        "lower_index": lambda: verifier.index_operation("lower_index", data.tensor_index),
    }
    start = time.perf_counter()
    try:
        verification = table[kind]()
    except (LimitError, RecursionError):
        ws.record(
            StageDiagnostic(
                stage=f"{kind}:verify",
                duration_ms=round((time.perf_counter() - start) * 1000),
                status="failed",
                failure="resource_limit",
            )
        )
        raise
    ws.record(
        StageDiagnostic(
            stage=f"{kind}:verify",
            duration_ms=round((time.perf_counter() - start) * 1000),
            status="ok" if verification.numeric_passed else "failed",
            failure=None if verification.numeric_passed else "not_verified",
        )
    )
    return verification


def _fail_stage(ws: Workspace, kind: str, status: Status) -> None:
    if not any(d.stage == kind and d.status == "failed" for d in ws.diagnostics):
        ws.record(StageDiagnostic(stage=kind, status="failed", failure=status.value))


# -- presentation ------------------------------------------------------------------------------
def _present(
    kind: str,
    title: str,
    value: Any,
    verification: GeometryVerification,
    data: GeometryInput,
    ws: Workspace,
) -> QuantityResult:
    geo = ws.geo
    names = {v: rf"\dot{{{ws.coord_latex[i]}}}" for i, v in enumerate(geo.velocities)}
    out = QuantityResult(
        kind=kind,
        status=Status.OK,
        title=title,
        verification=verification,
    )

    def latex(e: sp.Expr) -> str:
        return str(to_latex(e, symbol_names=names))

    if kind in {"determinant", "ricci_scalar", "kretschmann", "ricci_squared"}:
        out.scalar_latex = latex(value)
        out.scalar_text = str(value)
        out.all_zero = value == 0
        return out

    if kind == "geodesics":
        for mu, rhs in enumerate(value):
            lhs = rf"\ddot{{{ws.coord_latex[mu]}}}"
            out.components.append(
                Component(
                    indices=[mu],
                    label_latex=lhs,
                    latex=f"{lhs} = {latex(rhs)}",
                    text=f"{geo.coords[mu].name}_ddot = {rhs}",
                )
            )
        out.all_zero = all(r == 0 for r in value)
        out.symmetry_note = (
            "One equation per coordinate; the right-hand sides are quadratic in the velocities."
        )
        return out

    tensor: Tensor
    if kind == "inverse_metric":
        tensor = Tensor(geo.n, ("up", "up"))
        for i in range(geo.n):
            for j in range(geo.n):
                tensor.set((i, j), value[i, j])
        name = "g"
    else:
        tensor = value
        name = {
            "christoffel": r"\Gamma",
            "riemann": "R",
            "ricci": "R",
            "einstein": "G",
            "covariant_derivative": r"\nabla T",
            "raise_index": "T",
            "lower_index": "T",
        }[kind]
    out.variance = list(tensor.variance)
    shown = _shown(kind, tensor)
    for idx, expr in shown:
        out.components.append(
            Component(
                indices=list(idx),
                label_latex=index_label(name, tensor.variance, idx, ws.coord_latex),
                latex=latex(expr),
                text=str(expr),
            )
        )
    out.all_zero = not tensor.data
    out.symmetry_note = _symmetry_note(kind)
    if kind == "covariant_derivative":
        out.notes.append("The first index is the derivative index.")
    return out


def _shown(kind: str, tensor: Tensor) -> list[tuple[tuple[int, ...], Any]]:
    items = tensor.nonzero()
    if kind in {"christoffel", "ricci", "einstein", "inverse_metric"}:
        lo = 1 if kind == "christoffel" else 0
        return [(i, e) for i, e in items if i[lo] <= i[lo + 1]]
    if kind == "riemann":
        return [(i, e) for i, e in items if i[2] < i[3]]
    return items


def _symmetry_note(kind: str) -> str:
    return {
        "christoffel": "Symmetric in the lower indices; components not listed are zero or follow by symmetry.",
        "riemann": "Antisymmetric in the last two indices; components not listed are zero or follow by symmetry.",
        "ricci": "Symmetric; components not listed are zero or follow by symmetry.",
        "einstein": "Symmetric; components not listed are zero or follow by symmetry.",
        "inverse_metric": "Symmetric; components not listed are zero or follow by symmetry.",
    }.get(kind, "Components not listed are zero.")


def _overall(quantities: list[QuantityResult]) -> Status:
    statuses = [q.status for q in quantities]
    if all(s is Status.OK for s in statuses):
        return Status.OK
    for wanted in (
        Status.NOT_VERIFIED,
        Status.RESOURCE_LIMIT,
        Status.UNSUPPORTED,
        Status.PARSE_ERROR,
        Status.INTERNAL_ERROR,
    ):
        if wanted in statuses:
            return wanted
    return Status.INTERNAL_ERROR


def _summary(result: GeometryResult) -> str:
    done = sum(1 for q in result.quantities if q.status is Status.OK)
    total = len(result.quantities)
    if done == total:
        return f"{done} of {total} requested quantities computed and numerically verified."
    failed = [
        f"{q.kind} ({q.status.value})" for q in result.quantities if q.status is not Status.OK
    ]
    return f"{done} of {total} quantities computed; not shown: {', '.join(failed)}."


def _timed(
    diagnostics: list[StageDiagnostic],
    name: str,
    fn: Callable[[], Geometry],
) -> Geometry:
    start = time.perf_counter()
    try:
        value = fn()
    except Exception as exc:
        diagnostics.append(
            StageDiagnostic(
                stage=name,
                duration_ms=round((time.perf_counter() - start) * 1000),
                status="failed",
                failure=getattr(getattr(exc, "status", None), "value", "parse_error"),
            )
        )
        raise
    diagnostics.append(
        StageDiagnostic(stage=name, duration_ms=round((time.perf_counter() - start) * 1000))
    )
    return value


def _fail(
    diagnostics: list[StageDiagnostic],
    status: Status,
    message: str,
    base: GeometryResult | None = None,
) -> GeometryResult:
    result = base or GeometryResult(status=status)
    result.status = status
    result.message = message
    result.diagnostics = diagnostics
    result.peak_memory_mb = _peak_memory_mb()
    return result


def _peak_memory_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
