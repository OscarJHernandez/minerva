"""The statistics benchmark: cases are well formed, pending cases are honest, a baseline guards."""

import json
from pathlib import Path

import pytest
from minerva_engine import Engine
from minerva_knowledge.stat_benchmark import (
    GROUPS,
    MINIMUM_OUTCOME,
    NOT_EXPRESSIBLE,
    OPERATIONS,
    CaseResult,
    StatCase,
    StatReport,
    compare,
    load_cases,
    run_case,
    run_stat_benchmark,
)

BASELINE = Path(__file__).resolve().parents[3] / "docs" / "benchmarks" / "statistics.json"
CASES = load_cases()


@pytest.fixture(scope="module")
def report(engine: Engine) -> StatReport:
    return run_stat_benchmark(engine)


def test_cases_are_well_formed() -> None:
    assert {c.group for c in CASES} == set(GROUPS)
    assert {c.operation for c in CASES} <= set(OPERATIONS)
    assert {c.kind for c in CASES} == {"supported", "unsupported", "invalid_candidate"}
    assert all(c.source.strip() for c in CASES)
    ids = {c.id: c for c in CASES}
    for case_id in MINIMUM_OUTCOME:
        assert ids[case_id].kind == "supported", case_id


def test_the_benchmark_has_real_breadth() -> None:
    kinds = [c.kind for c in CASES]
    assert kinds.count("supported") >= 45
    assert kinds.count("unsupported") >= 25
    assert kinds.count("invalid_candidate") >= 25


def test_every_adversarial_theme_of_the_brief_is_covered() -> None:
    refused = {(c.expect.status, c.expect.reason) for c in CASES if c.kind == "unsupported"}
    reasons = {reason for _, reason in refused}
    assert {
        "parameter_invalid",  # invalid distribution parameters
        "unsupported_family",  # unsupported distributions
        "independence_not_declared",  # incorrect independence assumptions
        "candidate_inadmissible",  # nonexistent or boundary MLEs
        "regularity_not_satisfied",  # differentiation under the integral is not allowed
        "convergence_not_established",  # divergent or undefined moments
        "kernel_not_recognised",  # no conjugate form
        "invertibility_not_declared",  # singular / undeclared inverses
        "non_scalar_gradient",  # matrix by matrix
        "range_not_provable",
    } <= reasons
    statuses = {status for status, _ in refused}
    assert {
        "resource_limit",
        "parse_error",
        "unchanged",
    } <= statuses  # limits, shapes, no invention
    ids = {c.id for c in CASES}
    assert "matrix-singular-covariance-declared-invertible" in ids
    assert "matrix-nonsymmetric-covariance-declared-symmetric" in ids
    assert "matrix-check-unconstrained-gradient-for-symmetric-variable" in ids


def test_every_wrong_candidate_comes_with_a_correct_control() -> None:
    for case in CASES:
        if case.kind == "invalid_candidate":
            assert case.candidate and case.control, case.id
            assert set(case.candidate) <= set(case.control), case.id


@pytest.mark.parametrize("case_id", [c.id for c in CASES])
def test_each_case(report: StatReport, case_id: str) -> None:
    result = next(r for r in report.results if r.id == case_id)
    if result.pending:
        assert not result.passed, f"{case_id} now passes: remove `pending`"
    else:
        assert result.passed, result.problems


def test_a_request_the_engine_cannot_express_is_not_a_pass(engine: Engine) -> None:
    case = StatCase(
        id="synthetic",
        group="matrix",
        kind="supported",
        operation="matrix_calculus",
        source="synthetic",
        input="tr(A)",
        matrices=[{"name": "A", "rows": 2, "cols": 2}],
        request={"action": "no_such_action", "variable": "A"},
        expect={"outputs": {"result": "A"}},
    )
    result = run_case(case, engine)
    assert result.status == NOT_EXPRESSIBLE
    assert not result.passed


def test_summary_counts(report: StatReport) -> None:
    s = report.summary()
    assert s["cases"] == len(CASES)
    assert s["minimum_outcome_total"] == 5
    assert report.failures == []
    assert report.stale_pending == []


def _doc(passed: bool, status: str = "ok") -> dict:
    case = CaseResult(
        id="c", group="sums", kind="supported", pending=False, passed=passed, status=status
    )
    return {"format": 1, "summary": {"cases": 1}, "cases": [case.model_dump(mode="json")]}


def test_compare_reports_a_regression_but_not_an_improvement() -> None:
    _, regressed = compare(_doc(True), _doc(False))
    assert regressed
    text, regressed = compare(_doc(False), _doc(True))
    assert not regressed and "now passes c" in text
    assert compare({"format": 0, "cases": []}, _doc(True))[1]


def test_the_committed_baseline_has_no_regression(report: StatReport) -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    text, regressed = compare(baseline, report.to_json())
    assert not regressed, text
