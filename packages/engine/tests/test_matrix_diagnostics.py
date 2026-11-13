"""Diagnostics of a matrix computation: what was chosen and checked, never the mathematics."""

from __future__ import annotations

import json
from typing import Any

import pytest
from minerva_engine import Engine, ExpressionInput, Limits, Status
from minerva_engine.matrix import MatrixCalculusInput, MatrixCalculusResult, MatrixDecl
from pydantic import ValidationError

SECRET_NAME = "Qmarker"
SECRET_ENTRY = "7919"
SECRET_BOUND = "9876"


def run(
    engine: Engine, source: str, matrices: list[MatrixDecl], fmt: str = "latex", **kw: Any
) -> MatrixCalculusResult:
    return engine.matrix_calculus(
        MatrixCalculusInput(
            expression=ExpressionInput(input=source, format=fmt), matrices=matrices, **kw
        )
    )


def abstract(name: str, rows: int = 2, **kw: Any) -> MatrixDecl:
    return MatrixDecl(name=name, rows=rows, **kw)


def test_a_verified_result_reports_how_it_was_reached(engine: Engine) -> None:
    r = run(
        engine,
        r"\exp(t^2 A + t B)",
        [abstract("A"), abstract("B", 2)],
        action="differentiate",
        variable="t",
        trace_id="req-1_a",
    )
    d = r.diagnostics
    assert d is not None and r.status is Status.OK
    assert d.trace_id == "req-1_a" and d.method == "exp_frechet_derivative"
    assert [(m.id, m.outcome) for m in d.methods_tried] == [
        ("exp_derivative", "unsupported"),
        ("exp_frechet_derivative", "selected"),
    ]
    assert d.dimensions == [(2, 2)] and d.matrices == 2 and d.explicit == 0
    assert d.commutation == "not_used" and d.failure_reason is None
    names = [s.name for s in d.stages]
    assert names == ["parse", "transform", "verify_numeric", "verify_exact", "total"]
    assert all(s.ms >= 0 for s in d.stages)
    parts = sum(s.ms for s in d.stages if s.name != "total")
    total = next(s.ms for s in d.stages if s.name == "total")
    assert parts <= total + 5
    assert d.digits is not None and d.digits >= 16
    assert d.tolerance == pytest.approx(1e-9)
    assert d.max_residual is not None and d.max_residual < 1e-9
    assert d.points_checked == r.verification.points_checked  # type: ignore[union-attr]
    assert d.verification == "central_difference+quadrature+series"


def test_commutation_sources_are_reported(engine: Engine) -> None:
    declared = run(
        engine,
        r"\exp(t^2 A + t B)",
        [abstract("A"), abstract("B")],
        action="differentiate",
        variable="t",
        commuting=[("A", "B")],
    )
    assert declared.diagnostics is not None and declared.diagnostics.commutation == "declared"
    from_entries = run(
        engine,
        r"\exp(t^2 A + t B)",
        [
            MatrixDecl(name="A", entries=[["1", "0"], ["0", "2"]]),
            MatrixDecl(name="B", entries=[["3", "0"], ["0", "5"]]),
        ],
        action="differentiate",
        variable="t",
    )
    assert from_entries.diagnostics is not None
    assert from_entries.diagnostics.commutation == "from_entries"
    assert from_entries.diagnostics.explicit == 2
    invertible = run(
        engine,
        r"\exp(tA)",
        [abstract("A")],
        action="integrate",
        variable="t",
        lower="0",
        upper="1",
        invertible=["A"],
    )
    assert invertible.diagnostics is not None
    assert invertible.diagnostics.invertibility == "declared"


def test_failures_have_codes_not_messages(engine: Engine) -> None:
    cases: list[tuple[MatrixCalculusResult, str]] = [
        (run(engine, "AB", [abstract("A", 2), abstract("B", 3)]), "parse_error"),
        (run(engine, r"\sin(A)", [abstract("A")]), "unsupported"),
        (
            run(engine, r"\exp(A)\exp(B)", [abstract("A"), abstract("B")]),
            "conditions_not_satisfied",
        ),
        (run(engine, "(A+B)^{12}", [abstract("A"), abstract("B")]), "size_limit"),
    ]
    for result, code in cases:
        assert result.diagnostics is not None
        assert result.diagnostics.failure_reason == code, result.message
        assert result.diagnostics.points_checked == 0


def test_a_time_budget_is_a_coded_failure() -> None:
    quick = Engine(Limits(timeout_seconds=0.05, max_concurrency=1))
    r = run(
        quick,
        r"\exp(t^2 A + t B)",
        [abstract("A", 8), abstract("B", 8)],
        action="differentiate",
        variable="t",
    )
    assert r.status is Status.RESOURCE_LIMIT
    assert r.diagnostics is not None
    assert r.diagnostics.failure_reason in ("time_budget", "sandbox_timeout")


def test_a_closed_form_that_was_too_large_is_a_recorded_fallback(engine: Engine) -> None:
    a = [[2, -1, 2], [0, 1, -2], [2, -1, 2]]
    b = [[-1, 0, 1], [-1, 0, 1], [-2, 0, -1]]

    def decl(name: str, m: list[list[int]]) -> MatrixDecl:
        return MatrixDecl(name=name, entries=[[str(x) for x in row] for row in m])

    r = run(
        engine,
        r"\exp((t-s)A) B \exp(sA)",
        [decl("A", a), decl("B", b)],
        action="integrate",
        variable="s",
        lower="0",
        upper="1",
    )
    assert r.status is Status.OK and r.result_kind == "representation"
    assert r.diagnostics is not None
    assert "closed_form_too_large" in r.diagnostics.fallbacks or (
        "symbolic_timeout" in r.diagnostics.fallbacks
    )


def test_diagnostics_never_contain_the_mathematics(engine: Engine) -> None:
    evaluated = run(
        engine,
        f"{SECRET_NAME}*Rsecret",
        [
            MatrixDecl(name=SECRET_NAME, entries=[[SECRET_ENTRY, "0"], ["0", "1"]]),
            MatrixDecl(name="Rsecret", entries=[["2", "1"], ["0", "1"]]),
        ],
        fmt="text",
        action="evaluate",
        trace_id="trace-privacy",
    )
    integrated = run(
        engine,
        "3*t**2*Rsecret",
        [MatrixDecl(name="Rsecret", rows=2)],
        fmt="text",
        action="integrate",
        variable="t",
        lower="0",
        upper=SECRET_BOUND,
    )
    for r in (evaluated, integrated):
        assert r.status is Status.OK and r.diagnostics is not None
        blob = json.dumps(r.diagnostics.model_dump())
        for secret in (SECRET_NAME, SECRET_ENTRY, SECRET_BOUND, "Rsecret"):
            assert secret not in blob, secret
    assert evaluated.diagnostics is not None and "trace-privacy" in json.dumps(
        evaluated.diagnostics.model_dump()
    )


def test_trace_ids_are_opaque_tokens() -> None:
    base: dict[str, Any] = {
        "expression": ExpressionInput(input="A", format="latex"),
        "matrices": [abstract("A")],
    }
    MatrixCalculusInput(**base, trace_id="abc-DEF_123")
    for bad in ("has space", "x" * 65, "semi;colon", "$(rm)", ""):
        with pytest.raises(ValidationError):
            MatrixCalculusInput(**base, trace_id=bad)


def test_numeric_evolution_records_double_precision_and_the_worst_residual(
    engine: Engine,
) -> None:
    r = run(
        engine,
        "A(t)",
        [MatrixDecl(name="A", entries=[["0", "t"], ["-1", "0"]])],
        action="evolve",
        variable="t",
        upper="1",
    )
    d = r.diagnostics
    assert d is not None and d.digits == 15
    assert d.verification == "rk4+backward+liouville"
    assert d.max_residual is not None and d.max_residual < 1e-7
    assert d.time_dependent == 1
