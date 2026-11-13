"""Reproducible benchmark of the geometry engine.

Measures, on a fixed corpus of metrics with independently known answers:

* correctness against closed-form reference values (and against `sympy.diffgeom`, an independent
  implementation, for the explicit metrics);
* coverage: how many corpus problems are fully computed and verified, and that invalid or unsupported
  problems are refused with the expected status instead of a plausible result;
* the verification evidence per quantity (how many independent numeric checks, whether a symbolic proof
  was found);
* latency (median and p95 over repeats), expression size per stage, peak memory, and how all of that
  grows with the dimension (the n-sphere series), including where limits are hit.

`python -m minerva_engine.geometry.benchmark` writes `docs/benchmarks/geometry.json` and
`docs/geometry-benchmark.md`. The JSON has a `deterministic` section (statuses, correctness, check counts,
expression sizes: the same on every machine) and a `measurements` section (timings and memory: machine
dependent). `--check` recomputes only the deterministic section and compares it with the stored one.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sympy as sp

from ..models import Limits, Status
from .compute import Workspace
from .geometry import geometry
from .models import GeometryInput, GeometryResult
from .setup import GeometryError, build_geometry
from .tensors import Index, Tensor

FORMAT = 1
CASE_LIMITS = Limits(timeout_seconds=300)
CURVATURE = ["christoffel", "riemann", "ricci", "ricci_scalar", "einstein", "kretschmann"]


# --- the corpus -------------------------------------------------------------------------------------
Expected = dict[str, Any]  # quantity -> sympy expression (scalar) or {index: expression} (tensor)


@dataclass
class Case:
    id: str
    title: str
    coordinates: list[str]
    metric: list[list[str]]
    quantities: list[str]
    expect: Callable[[Workspace], Expected]
    reference: str  # where the expected values come from
    assumptions: dict[str, list[str]] = field(default_factory=dict)
    functions: list[dict[str, Any]] = field(default_factory=list)
    format: str = "latex"
    use_diffgeom: bool = False  # compare with sympy.diffgeom (explicit metrics only)
    signature: str | None = None


def _sym(ws: Workspace, name: str) -> sp.Symbol:
    return {s.name: s for s in [*ws.geo.coords, *ws.geo.params]}[name]


def _lower_symmetric(entries: dict[tuple[int, int, int], sp.Expr]) -> dict[Index, sp.Expr]:
    """`Gamma^a_{bc}` given for b <= c: add the mirrored components."""
    out: dict[Index, sp.Expr] = {}
    for (a, b, c), value in entries.items():
        out[(a, b, c)] = value
        out[(a, c, b)] = value
    return out


def _diag(ws: Workspace, scale: sp.Expr) -> dict[Index, sp.Expr]:
    return {(i, i): scale * ws.geo.g[i, i] for i in range(ws.geo.n)}


def _flat(ws: Workspace) -> Expected:
    return {
        "christoffel": {},
        "riemann": {},
        "ricci": {},
        "ricci_scalar": sp.S.Zero,
        "einstein": {},
        "kretschmann": sp.S.Zero,
    }


def _polar(ws: Workspace) -> Expected:
    r = _sym(ws, "r")
    return {
        **_flat(ws),
        "christoffel": _lower_symmetric({(0, 1, 1): -r, (1, 0, 1): 1 / r}),
    }


def _sphere2(ws: Workspace) -> Expected:
    th, a = _sym(ws, "theta"), _sym(ws, "a")
    return {
        "christoffel": _lower_symmetric(
            {(0, 1, 1): -sp.sin(th) * sp.cos(th), (1, 0, 1): sp.cos(th) / sp.sin(th)}
        ),
        "riemann": {
            (0, 1, 0, 1): sp.sin(th) ** 2,
            (0, 1, 1, 0): -(sp.sin(th) ** 2),
            (1, 0, 0, 1): -1,
            (1, 0, 1, 0): 1,
        },
        "ricci": {(0, 0): sp.S.One, (1, 1): sp.sin(th) ** 2},
        "ricci_scalar": 2 / a**2,
        "einstein": {},  # identically zero in two dimensions
        "kretschmann": 4 / a**4,
    }


def _sphere_sheared(ws: Workspace) -> Expected:
    a = _sym(ws, "a")
    return {"ricci_scalar": 2 / a**2, "kretschmann": 4 / a**4, "einstein": {}}


def _sphere3(ws: Workspace) -> Expected:
    a = _sym(ws, "a")
    return {"ricci_scalar": 6 / a**2, "kretschmann": 12 / a**4}


def _schwarzschild(ws: Workspace) -> Expected:
    r, th, M = _sym(ws, "r"), _sym(ws, "theta"), _sym(ws, "M")
    gamma = {
        (0, 0, 1): M / (r * (r - 2 * M)),
        (1, 0, 0): M * (r - 2 * M) / r**3,
        (1, 1, 1): -M / (r * (r - 2 * M)),
        (1, 2, 2): -(r - 2 * M),
        (1, 3, 3): -(r - 2 * M) * sp.sin(th) ** 2,
        (2, 1, 2): 1 / r,
        (2, 3, 3): -sp.sin(th) * sp.cos(th),
        (3, 1, 3): 1 / r,
        (3, 2, 3): sp.cos(th) / sp.sin(th),
    }
    return {
        "christoffel": _lower_symmetric(gamma),
        "ricci": {},
        "ricci_scalar": sp.S.Zero,
        "einstein": {},
        "kretschmann": 48 * M**2 / r**6,
    }


def _de_sitter(ws: Workspace) -> Expected:
    L = _sym(ws, "L")
    return {
        "ricci": _diag(ws, 3 / L**2),
        "ricci_scalar": 12 / L**2,
        "einstein": _diag(ws, -3 / L**2),
        "kretschmann": 24 / L**4,
    }


def _reissner_nordstrom(ws: Workspace) -> Expected:
    r, M, Q = _sym(ws, "r"), _sym(ws, "M"), _sym(ws, "Q")
    return {
        "ricci_scalar": sp.S.Zero,
        "kretschmann": (48 * M**2 * r**2 - 96 * M * Q**2 * r + 56 * Q**4) / r**8,
    }


def _flrw(ws: Workspace) -> Expected:
    t = _sym(ws, "t")
    a = sp.Function("a", real=True)(t)
    da, dda = a.diff(t), a.diff(t, 2)
    return {
        "ricci_scalar": 6 * (a * dda + da**2) / a**2,
        "einstein": {
            (0, 0): 3 * (da / a) ** 2,
            (1, 1): -(2 * a * dda + da**2),
            (2, 2): -(2 * a * dda + da**2) * _sym(ws, "r") ** 2,
            (3, 3): -(2 * a * dda + da**2) * _sym(ws, "r") ** 2 * sp.sin(_sym(ws, "theta")) ** 2,
        },
    }


def _static_f(ws: Workspace) -> Expected:
    r = _sym(ws, "r")
    f = sp.Function("f", real=True)(r)
    return {"ricci_scalar": -f.diff(r, 2) - 4 * f.diff(r) / r - 2 * (f - 1) / r**2}


def _skew_flat(ws: Workspace) -> Expected:
    return _flat(ws)


SPHERE2 = [["a^2", "0"], ["0", r"a^2 \sin^2\theta"]]
BLACK_HOLE_ANGLES = ["r^2", r"r^2 \sin^2\theta"]


def _static(f: str) -> list[list[str]]:
    return [
        [f"-({f})", "0", "0", "0"],
        ["0", f"({f})^{{-1}}", "0", "0"],
        ["0", "0", BLACK_HOLE_ANGLES[0], "0"],
        ["0", "0", "0", BLACK_HOLE_ANGLES[1]],
    ]


EYE3 = [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]]
ETA = [
    ["-1", "0", "0", "0"],
    ["0", "1", "0", "0"],
    ["0", "0", "1", "0"],
    ["0", "0", "0", "1"],
]
POSITIVE_A = {"a": ["positive"]}

CASES: list[Case] = [
    Case(
        "euclid3",
        "Euclidean 3-space, Cartesian",
        ["x", "y", "z"],
        EYE3,
        CURVATURE,
        _flat,
        "flat space: every Christoffel symbol and curvature vanishes",
        use_diffgeom=True,
        signature="(+, +, +)",
    ),
    Case(
        "polar",
        "Euclidean plane, polar coordinates",
        ["r", r"\theta"],
        [["1", "0"], ["0", "r^2"]],
        CURVATURE,
        _polar,
        "standard result: flat, Gamma^r_tt = -r",
        assumptions={"r": ["positive"]},
        use_diffgeom=True,
        signature="(+, +)",
    ),
    Case(
        "sphere2",
        "2-sphere of radius a",
        [r"\theta", r"\phi"],
        SPHERE2,
        [*CURVATURE, "geodesics"],
        _sphere2,
        "standard result: R = 2/a^2, K = 4/a^4, G = 0 in 2D",
        assumptions=POSITIVE_A,
        use_diffgeom=True,
    ),
    Case(
        "sphere_sheared",
        "2-sphere in sheared coordinates (non-diagonal metric)",
        [r"\theta", r"\psi"],
        [
            ["a^2(1+\\sin^2\\theta)", "-a^2\\sin^2\\theta"],
            ["-a^2\\sin^2\\theta", "a^2\\sin^2\\theta"],
        ],
        ["ricci_scalar", "kretschmann", "einstein"],
        _sphere_sheared,
        "coordinate invariance: the scalar invariants of the 2-sphere do not depend on coordinates",
        assumptions=POSITIVE_A,
    ),
    Case(
        "sphere3",
        "3-sphere of radius a",
        [r"\chi", r"\theta", r"\phi"],
        [
            ["a^2", "0", "0"],
            ["0", r"a^2 \sin^2\chi", "0"],
            ["0", "0", r"a^2 \sin^2\chi \sin^2\theta"],
        ],
        ["ricci_scalar", "kretschmann"],
        _sphere3,
        "standard result: R = n(n-1)/a^2, K = 2n(n-1)/a^4",
        assumptions=POSITIVE_A,
    ),
    Case(
        "minkowski",
        "Minkowski space",
        ["t", "x", "y", "z"],
        ETA,
        [*CURVATURE, "geodesics"],
        _flat,
        "flat space; signature from the entries",
        signature="(-, +, +, +)",
    ),
    Case(
        "skew_flat",
        "Plane in skew coordinates (constant non-diagonal metric)",
        ["u", "v"],
        [["1", "c"], ["c", "1"]],
        CURVATURE,
        _skew_flat,
        "a constant metric is flat",
        format="text",
    ),
    Case(
        "schwarzschild",
        "Schwarzschild",
        ["t", "r", r"\theta", r"\phi"],
        _static("1-2M/r"),
        [*CURVATURE, "geodesics"],
        _schwarzschild,
        "standard result: Ricci-flat, K = 48 M^2 / r^6, Gamma^r_tt = M (r - 2M) / r^3",
        assumptions={"M": ["positive"], "r": ["positive"]},
        use_diffgeom=True,
    ),
    Case(
        "de_sitter",
        "de Sitter, static patch",
        ["t", "r", r"\theta", r"\phi"],
        _static("1-r^2/L^2"),
        ["ricci", "ricci_scalar", "einstein", "kretschmann"],
        _de_sitter,
        "standard result: R_mn = 3 g_mn / L^2, R = 12 / L^2, K = 24 / L^4",
        assumptions={"L": ["positive"], "r": ["positive"]},
    ),
    Case(
        "reissner_nordstrom",
        "Reissner-Nordstrom",
        ["t", "r", r"\theta", r"\phi"],
        _static("1-2M/r+Q^2/r^2"),
        ["ricci_scalar", "kretschmann"],
        _reissner_nordstrom,
        "standard result: R = 0, K = (48 M^2 r^2 - 96 M Q^2 r + 56 Q^4) / r^8",
        assumptions={"M": ["positive"], "Q": ["positive"], "r": ["positive"]},
    ),
    Case(
        "static_f",
        "Static spherical metric with an unspecified f(r)",
        ["t", "r", r"\theta", r"\phi"],
        _static("f(r)"),
        ["christoffel", "ricci_scalar", "einstein"],
        _static_f,
        "R = -f'' - 4 f'/r - 2 (f - 1)/r^2",
        assumptions={"r": ["positive"]},
        functions=[{"name": "f", "args": ["r"]}],
    ),
    Case(
        "flrw",
        "Flat FLRW with an unspecified a(t)",
        ["t", "r", r"\theta", r"\phi"],
        [
            ["-1", "0", "0", "0"],
            ["0", "a(t)^2", "0", "0"],
            ["0", "0", "a(t)^2 r^2", "0"],
            ["0", "0", "0", r"a(t)^2 r^2 \sin^2\theta"],
        ],
        ["ricci_scalar", "einstein"],
        _flrw,
        "standard result: R = 6 (a a'' + a'^2)/a^2, G_tt = 3 (a'/a)^2, G_rr = -(2 a a'' + a'^2)",
        assumptions={"r": ["positive"]},
        functions=[{"name": "a", "args": ["t"]}],
    ),
]


@dataclass
class Refusal:
    id: str
    title: str
    request: dict[str, Any]
    status: Status
    needle: str  # must appear in the message


REFUSALS: list[Refusal] = [
    Refusal(
        "degenerate",
        "metric with determinant 0",
        {"coordinates": ["x", "y"], "metric": [["1", "0"], ["0", "0"]], "format": "text"},
        Status.UNSUPPORTED,
        "degenerate",
    ),
    Refusal(
        "hidden_degenerate",
        "metric whose determinant cancels only after simplification",
        {"coordinates": ["x", "y"], "metric": [["1", "x"], ["x", "x^2"]], "format": "text"},
        Status.UNSUPPORTED,
        "degenerate",
    ),
    Refusal(
        "asymmetric",
        "metric that is not symmetric",
        {"coordinates": ["x", "y"], "metric": [["1", "x"], ["0", "1"]], "format": "text"},
        Status.PARSE_ERROR,
        "symmetric",
    ),
    Refusal(
        "not_square",
        "metric with the wrong shape",
        {"coordinates": ["x", "y"], "metric": [["1", "0"]], "format": "text"},
        Status.PARSE_ERROR,
        "by 2 grid",
    ),
    Refusal(
        "undeclared_function",
        "f(r) used without declaring f",
        {"coordinates": ["r"], "metric": [["f(r)"]]},
        Status.PARSE_ERROR,
        "declare",
    ),
    Refusal(
        "too_many_dimensions",
        "more coordinates than supported",
        {
            "coordinates": ["a1", "a2", "a3", "a4", "a5"],
            "metric": [["1" if i == j else "0" for j in range(5)] for i in range(5)],
        },
        Status.UNSUPPORTED,
        "Dimension 5",
    ),
    Refusal(
        "reserved_velocity_name",
        "symbol named like a velocity",
        {"coordinates": ["x"], "metric": [["x_dot"]], "format": "text"},
        Status.PARSE_ERROR,
        "reserved",
    ),
    Refusal(
        "tensor_missing",
        "index operation without a tensor",
        {"coordinates": ["x"], "metric": [["1"]], "quantities": ["lower_index"]},
        Status.PARSE_ERROR,
        "needs a tensor",
    ),
]


# --- running ----------------------------------------------------------------------------------------
def _request(case: Case) -> GeometryInput:
    return GeometryInput(
        coordinates=case.coordinates,
        metric=case.metric,
        format=case.format,
        assumptions=case.assumptions,
        functions=case.functions,
        quantities=case.quantities,
    )


def _tensor_matches(tensor: Tensor, expected: dict[Index, Any]) -> bool:
    for key in set(tensor.data) | set(expected):
        if sp.simplify(tensor.get(key) - expected.get(key, 0)) != 0:
            return False
    return True


def _scalar_matches(value: Any, expected: Any) -> bool:
    return bool(sp.simplify(value - expected) == 0)


def _correct(case: Case) -> dict[str, bool]:
    """Compare every expected quantity with the symbolic result, in the workspace the engine uses."""
    import time as _time

    geo = build_geometry(_request(case), CASE_LIMITS, 6)
    ws = Workspace(geo, CASE_LIMITS, _time.monotonic() + 1e9)
    getters: dict[str, Callable[[], Any]] = {
        "christoffel": ws.christoffel,
        "riemann": ws.riemann,
        "ricci": ws.ricci,
        "ricci_scalar": ws.ricci_scalar,
        "einstein": ws.einstein,
        "kretschmann": ws.kretschmann,
    }
    out: dict[str, bool] = {}
    for kind, expected in case.expect(ws).items():
        value = getters[kind]()
        out[kind] = (
            _tensor_matches(value, expected)
            if isinstance(value, Tensor)
            else _scalar_matches(value, expected)
        )
    if "geodesics" in case.quantities:
        # the geodesic accelerations must equal -Gamma^mu_{ab} v^a v^b built from the reference Christoffels
        ref = case.expect(ws).get("christoffel")
        if ref is not None:
            v = ws.geo.velocities
            n = ws.geo.n
            acc = ws.accelerations()
            out["geodesics"] = all(
                sp.simplify(
                    acc[m]
                    + sum(ref.get((m, a, b), 0) * v[a] * v[b] for a in range(n) for b in range(n))
                )
                == 0
                for m in range(n)
            )
    return out


def _diffgeom_reference(case: Case) -> tuple[bool, float]:
    """Does `sympy.diffgeom` give the same Christoffel, Riemann and Ricci components? And how long does it take?"""
    from sympy.diffgeom import (
        CoordSystem,
        Manifold,
        Patch,
        TensorProduct,
        metric_to_Christoffel_2nd,
        metric_to_Ricci_components,
        metric_to_Riemann_components,
    )

    geo = build_geometry(_request(case), CASE_LIMITS, 6)
    n = geo.n
    # plain symbols for the chart: diffgeom does not cope with coordinates that carry assumptions
    plain = [sp.Symbol(c.name) for c in geo.coords]
    chart = CoordSystem("benchmark", Patch("P", Manifold("M", n)), plain)
    forms, funcs = chart.base_oneforms(), chart.coord_functions()
    back = {
        **dict(zip(funcs, geo.coords, strict=True)),
        **dict(zip(plain, geo.coords, strict=True)),
    }
    to_funcs = dict(zip(geo.coords, funcs, strict=True))
    start = time.perf_counter()
    metric = sum(
        (
            geo.g[i, j].subs(to_funcs) * TensorProduct(forms[i], forms[j])
            for i in range(n)
            for j in range(n)
            if geo.g[i, j] != 0
        ),
        sp.S.Zero,
    )
    gamma = metric_to_Christoffel_2nd(metric)
    riemann = metric_to_Riemann_components(metric)
    ricci = metric_to_Ricci_components(metric)
    elapsed = (time.perf_counter() - start) * 1000

    ours = Workspace(geo, CASE_LIMITS, time.monotonic() + 1e9)
    ok = True
    gm, rm, cm = ours.christoffel(), ours.riemann(), ours.ricci()
    for r in range(n):
        for a in range(n):
            for b in range(n):
                ok &= sp.simplify(gm.get((r, a, b)) - sp.sympify(gamma[r][a][b]).subs(back)) == 0
                for c in range(n):
                    ok &= (
                        sp.simplify(
                            rm.get((r, a, b, c)) - sp.sympify(riemann[r][a][b][c]).subs(back)
                        )
                        == 0
                    )
    for a in range(n):
        for b in range(n):
            ok &= sp.simplify(cm.get((a, b)) - sp.sympify(ricci[a][b]).subs(back)) == 0
    return bool(ok), elapsed


def _same_three(case: Case, repeats: int) -> dict[str, float]:
    """The engine on exactly what `sympy.diffgeom` computes (Christoffel, Riemann, Ricci): with its
    independent verification, and the symbolic stages alone."""
    data = _request(case).model_copy(update={"quantities": ["christoffel", "riemann", "ricci"]})
    verified: list[float] = []
    bare: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = geometry(data, CASE_LIMITS, max_dimension=6)
        verified.append((time.perf_counter() - start) * 1000)
        bare.append(
            sum(d.duration_ms for d in result.diagnostics if not d.stage.endswith(":verify"))
        )
    return {
        "ours_verified_ms": round(statistics.median(verified), 1),
        "ours_symbolic_only_ms": round(statistics.median(bare), 1),
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _run_case(case: Case, repeats: int) -> tuple[GeometryResult, list[float]]:
    times: list[float] = []
    result: GeometryResult | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = geometry(_request(case), CASE_LIMITS, max_dimension=6)
        times.append((time.perf_counter() - start) * 1000)
    assert result is not None  # noqa: S101 - repeats >= 1
    return result, times


def _traced_peak_mb(case: Case) -> float:
    tracemalloc.start()
    try:
        geometry(_request(case), CASE_LIMITS, max_dimension=6)
        return round(tracemalloc.get_traced_memory()[1] / 1e6, 1)
    finally:
        tracemalloc.stop()


def _quantity_summary(result: GeometryResult) -> dict[str, Any]:
    out: dict[str, Any] = {}
    stage_ops = {d.stage: d.ops for d in result.diagnostics}
    for q in result.quantities:
        v = q.verification
        numeric = [c for c in v.checks if c.kind == "numeric"] if v else []
        symbolic = [c for c in v.checks if c.kind == "symbolic"] if v else []
        out[q.kind] = {
            "status": q.status.value,
            "numeric_checks": len(numeric),
            "numeric_passed": bool(v and v.numeric_passed),
            "symbolic_checks": len(symbolic),
            "symbolic_proved": None if v is None else v.symbolic_zero,
            "ops": stage_ops.get(q.kind),
        }
    return out


def run(
    repeats: int, max_dimension: int, scaling_timeout: float, check_only: bool
) -> dict[str, Any]:
    deterministic: dict[str, Any] = {"cases": {}, "refusals": {}}
    measurements: dict[str, Any] = {"latency_ms": {}, "memory_mb": {}, "stages": {}}
    reference_rows: list[dict[str, Any]] = []

    supported = 0
    for case in CASES:
        result, times = _run_case(case, 1 if check_only else repeats)
        correct = _correct(case)
        entry: dict[str, Any] = {
            "status": result.status.value,
            "dimension": result.dimension,
            "correct": all(correct.values()) and bool(correct),
            "correct_by_quantity": dict(sorted(correct.items())),
            "signature_ok": case.signature is None or result.signature == case.signature,
            "restrictions": len(result.restrictions_latex),
            "quantities": _quantity_summary(result),
        }
        if case.use_diffgeom:
            agrees, ref_ms = _diffgeom_reference(case)
            entry["diffgeom_agrees"] = agrees
            if not check_only:
                reference_rows.append(
                    {"id": case.id, "diffgeom_ms": round(ref_ms, 1), **_same_three(case, repeats)}
                )
        deterministic["cases"][case.id] = entry
        if result.status is Status.OK and entry["correct"]:
            supported += 1
        if not check_only:
            measurements["latency_ms"][case.id] = {
                "median": round(statistics.median(times), 1),
                "p95": round(_percentile(times, 0.95), 1),
                "runs": [round(t, 1) for t in times],
            }
            measurements["memory_mb"][case.id] = _traced_peak_mb(case)
            measurements["stages"][case.id] = [
                {"stage": d.stage, "ms": d.duration_ms, "ops": d.ops} for d in result.diagnostics
            ]
    for refusal in REFUSALS:
        data = GeometryInput(**{"quantities": ["christoffel"], **refusal.request})
        got = geometry(data, CASE_LIMITS)
        quantity_status = got.quantities[0].status if got.quantities else got.status
        message = " ".join([got.message, *(q.message for q in got.quantities)])
        deterministic["refusals"][refusal.id] = {
            "status": quantity_status.value,
            "as_expected": quantity_status is refusal.status
            and refusal.needle in message
            and not any(q.components or q.scalar_latex for q in got.quantities),
        }
    deterministic["coverage"] = {
        "computed_and_correct": supported,
        "corpus": len(CASES),
        "refusals_as_expected": sum(r["as_expected"] for r in deterministic["refusals"].values()),
        "refusal_cases": len(REFUSALS),
    }
    if not check_only:
        measurements["vs_diffgeom"] = reference_rows
        measurements["scaling"] = _scaling(max_dimension, scaling_timeout, repeats=min(repeats, 3))
    return {
        "format": FORMAT,
        "environment": {
            "python": platform.python_version(),
            "sympy": sp.__version__,
            "platform": f"{platform.system()} {platform.machine()}",
            "repeats": repeats,
        },
        "deterministic": deterministic,
        "measurements": measurements,
    }


def _nsphere(n: int) -> tuple[list[str], list[list[str]]]:
    names = [f"t{i}" for i in range(1, n + 1)]
    diag = ["a^2"]
    for k in range(1, n):
        diag.append("a^2*" + "*".join(f"sin(t{j})^2" for j in range(1, k + 1)))
    return names, [[diag[i] if i == j else "0" for j in range(n)] for i in range(n)]


def _scaling(max_dimension: int, timeout: float, repeats: int) -> list[dict[str, Any]]:
    """The n-sphere, n = 2 ... max: time, expression size and memory against the dimension."""
    rows: list[dict[str, Any]] = []
    for n in range(2, max_dimension + 1):
        names, grid = _nsphere(n)
        data = GeometryInput(
            coordinates=names,
            metric=grid,
            format="text",
            assumptions={"a": ["positive"]},
            quantities=["christoffel", "riemann", "ricci_scalar"],
        )
        times: list[float] = []
        result: GeometryResult | None = None
        for _ in range(repeats):
            start = time.perf_counter()
            result = geometry(data, Limits(timeout_seconds=timeout), max_dimension=max_dimension)
            times.append((time.perf_counter() - start) * 1000)
            if result.status is not Status.OK:
                break
        assert result is not None  # noqa: S101
        tracemalloc.start()
        geometry(data, Limits(timeout_seconds=timeout), max_dimension=max_dimension)
        peak = round(tracemalloc.get_traced_memory()[1] / 1e6, 1)
        tracemalloc.stop()
        scalar = next((q.scalar_text for q in result.quantities if q.kind == "ricci_scalar"), None)
        rows.append(
            {
                "n": n,
                "status": result.status.value,
                "median_ms": round(statistics.median(times), 1),
                "p95_ms": round(_percentile(times, 0.95), 1),
                "ricci_scalar": scalar,
                "expected_ricci_scalar": f"{n * (n - 1)}/a**2",
                "stage_ops": {d.stage: d.ops for d in result.diagnostics if d.ops is not None},
                "failed_stage": next(
                    (d.stage for d in result.diagnostics if d.status == "failed"), None
                ),
                "peak_traced_mb": peak,
            }
        )
    return rows


# --- the report -------------------------------------------------------------------------------------
def _in_order(mapping: dict[str, Any], order: list[str]) -> list[tuple[str, Any]]:
    """The items in corpus order, whatever order the keys were stored in (the JSON is key-sorted)."""
    return [(k, mapping[k]) for k in order if k in mapping]


def render_report(data: dict[str, Any]) -> str:
    det, meas, env = data["deterministic"], data["measurements"], data["environment"]
    case_ids, refusal_ids = [c.id for c in CASES], [r.id for r in REFUSALS]
    cov = det["coverage"]
    lines = [
        "# Geometry benchmark",
        "",
        "Generated by `python -m minerva_engine.geometry.benchmark`; do not edit. The corpus and the expected",
        "values are in `packages/engine/src/minerva_engine/geometry/benchmark.py`. Machine-readable results:",
        "`docs/benchmarks/geometry.json` (the `deterministic` section is checked in CI, the `measurements`",
        "section is machine dependent).",
        "",
        f"Environment of the stored measurements: Python {env['python']}, SymPy {env['sympy']}, "
        f"{env['platform']}, {env['repeats']} runs per case.",
        "",
        "## Correctness and coverage",
        "",
        f"- {cov['computed_and_correct']} of {cov['corpus']} corpus problems computed, verified, and equal "
        "to their closed-form reference values.",
        f"- {cov['refusals_as_expected']} of {cov['refusal_cases']} invalid or unsupported problems were "
        "refused with the expected status and no result.",
        "",
        "| Case | Status | Reference values | Signature | `sympy.diffgeom` | Quantities | "
        "Numeric checks passed | Symbolic proofs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cid, e in _in_order(det["cases"], case_ids):
        qs = e["quantities"]
        proved = sum(1 for q in qs.values() if q["symbolic_proved"] is True)
        passed = sum(1 for q in qs.values() if q["numeric_passed"])
        diff = e.get("diffgeom_agrees")
        lines.append(
            f"| {cid} | {e['status']} | {'match' if e['correct'] else 'MISMATCH'} | "
            f"{'ok' if e['signature_ok'] else 'WRONG'} | "
            f"{'agrees' if diff else ('DISAGREES' if diff is False else 'n/a')} | {len(qs)} | "
            f"{passed}/{len(qs)} | {proved}/{len(qs)} |"
        )
    lines += [
        "",
        "Refusals: "
        + ", ".join(f"{rid} ({r['status']})" for rid, r in _in_order(det["refusals"], refusal_ids))
        + ".",
        "",
        "## Latency, expression size and memory",
        "",
        "| Case | Median ms | p95 ms | Peak traced MB | Largest stage (ms) | Largest result (ops) |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for cid, lat in _in_order(meas["latency_ms"], case_ids):
        stages = meas["stages"][cid]
        slow = max(stages, key=lambda s: s["ms"])
        big = max((s for s in stages if s["ops"] is not None), key=lambda s: s["ops"], default=None)
        lines.append(
            f"| {cid} | {lat['median']} | {lat['p95']} | {meas['memory_mb'][cid]} | "
            f"{slow['stage']} ({slow['ms']}) | {big['ops'] if big else ''} |"
        )
    lines += [
        "",
        "## Against `sympy.diffgeom`",
        "",
        "Christoffel symbols, Riemann and Ricci components for the explicit metrics, computed by SymPy's own",
        "differential geometry module (an independent implementation), compared component by component with",
        "the engine's results. Times are median milliseconds for the same three tensors. The engine's",
        "first column includes the independent verification of every quantity it returns, which",
        "`sympy.diffgeom` does not do; the second column is its symbolic stages alone.",
        "",
        "| Case | Engine with verification | Engine, symbolic stages only | `sympy.diffgeom` |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {r['id']} | {r['ours_verified_ms']} | {r['ours_symbolic_only_ms']} | {r['diffgeom_ms']} |"
        for r in sorted(meas["vs_diffgeom"], key=lambda r: case_ids.index(r["id"]))
    ]
    lines += [
        "",
        "## Growth with the dimension (n-sphere)",
        "",
        "Christoffel symbols, Riemann tensor and Ricci scalar of the round n-sphere, with the Ricci scalar",
        "compared with the known `n(n-1)/a^2`. A status other than `ok` is where a limit was reached.",
        "",
        "| n | Status | Median ms | p95 ms | Ricci scalar | Riemann result (ops) | Peak traced MB |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in meas["scaling"]:
        match = (
            ""
            if row["ricci_scalar"] is None
            else (
                " (matches)"
                if row["ricci_scalar"].replace(" ", "") == row["expected_ricci_scalar"]
                else " (DIFFERS)"
            )
        )
        failed = f", stopped in {row['failed_stage']}" if row["failed_stage"] else ""
        lines.append(
            f"| {row['n']} | {row['status']}{failed} | {row['median_ms']} | {row['p95_ms']} | "
            f"{row['ricci_scalar'] or ''}{match} | {row['stage_ops'].get('riemann', '')} | "
            f"{row['peak_traced_mb']} |"
        )
    lines += [
        "",
        "## Limits of this evidence",
        "",
        "- The corpus is small and the expected values are classical closed forms; agreement with them and with",
        "  `sympy.diffgeom` shows the common cases are right, not that every metric is.",
        "- Numerical agreement is evidence, not proof; the symbolic column counts quantities for which a symbolic",
        "  zero test also succeeded. A missing proof means SymPy's simplifier did not finish the job.",
        "- Timings are one machine and a few runs; p95 of a handful of runs is close to the maximum. Peak memory",
        "  is the peak of Python allocations (`tracemalloc`) in one process, not the resident size.",
        "- Where the dimension series reports a limit, that is the engine's budget on that machine, not a",
        "  property of the mathematics.",
        "",
    ]
    return "\n".join(lines)


# --- command line -----------------------------------------------------------------------------------
def _digest_diff(old: Any, new: Any, path: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        out: list[str] = []
        for key in sorted(set(old) | set(new)):
            out += _digest_diff(
                old.get(key, "<missing>"), new.get(key, "<missing>"), f"{path}/{key}"
            )
        return out
    return [] if old == new else [f"{path}: stored {old!r}, now {new!r}"]


def compare(stored: dict[str, Any], fresh: dict[str, Any], report_text: str) -> list[str]:
    """Differences between the stored results and a fresh run, in the deterministic section only."""
    problems = _digest_diff(stored["deterministic"], fresh["deterministic"])
    if report_text != render_report(stored):
        problems.append("the report is not the report of the stored JSON; regenerate it")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the geometry engine.")
    parser.add_argument("--repeats", type=int, default=5, help="timed runs per case (default 5)")
    parser.add_argument("--max-dimension", type=int, default=5, help="largest n-sphere (default 5)")
    parser.add_argument("--scaling-timeout", type=float, default=120.0)
    parser.add_argument("--json", type=Path, default=Path("docs/benchmarks/geometry.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/geometry-benchmark.md"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute only the deterministic section and compare with the stored JSON and report",
    )
    args = parser.parse_args(argv)

    if args.check:
        stored = json.loads(args.json.read_text())
        problems = compare(stored, run(1, 2, 1.0, check_only=True), args.report.read_text())
        if problems:
            print("geometry benchmark is out of date:", *problems, sep="\n  ", file=sys.stderr)
            return 1
        print("geometry benchmark is up to date")
        return 0

    data = run(args.repeats, args.max_dimension, args.scaling_timeout, check_only=False)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    args.report.write_text(render_report(data))
    cov = data["deterministic"]["coverage"]
    print(
        f"{cov['computed_and_correct']}/{cov['corpus']} corpus problems correct; "
        f"{cov['refusals_as_expected']}/{cov['refusal_cases']} refusals as expected; wrote "
        f"{args.json} and {args.report}"
    )
    failures = cov["computed_and_correct"] != cov["corpus"] or (
        cov["refusals_as_expected"] != cov["refusal_cases"]
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

_ = GeometryError  # re-exported for tests that build workspaces directly
