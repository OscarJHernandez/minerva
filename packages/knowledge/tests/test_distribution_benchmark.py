"""The distribution benchmark: every case runs, floors guard the summary, comparisons are honest."""

import json
from pathlib import Path

import pytest
from minerva_engine import Engine
from minerva_knowledge import KnowledgeBase
from minerva_knowledge.distribution_benchmark import (
    DistReport,
    compare,
    load_distribution_cases,
    run_distribution_benchmark,
)

BASELINE = Path(__file__).resolve().parents[3] / "docs" / "benchmarks" / "distributions.json"


@pytest.fixture(scope="module")
def report(kb: KnowledgeBase, engine: Engine) -> DistReport:
    return run_distribution_benchmark(kb, engine)


def test_cases_are_well_formed(kb: KnowledgeBase) -> None:
    cases = load_distribution_cases()
    kinds = {c.kind for c in cases}
    assert kinds == {"supported", "unsupported", "invalid_transform"}
    for case in cases:
        for entry_id in [*case.entries, *case.retrieval]:
            assert kb.get(entry_id) is not None, (case.id, entry_id)
        if case.kind == "supported":
            assert case.result is not None, case.id
        if case.kind == "unsupported":
            assert case.status and case.reason, case.id
        if case.kind == "invalid_transform":
            assert case.candidate is not None, case.id


@pytest.mark.parametrize("case_id", [c.id for c in load_distribution_cases()])
def test_each_case(report: DistReport, case_id: str) -> None:
    result = next(r for r in report.results if r.id == case_id)
    assert result.passed, result.problems


def test_summary_floors(report: DistReport) -> None:
    s = report.summary()
    assert s["supported_correct"] == s["supported_cases"] >= 25
    assert s["unsupported_refused_correctly"] == s["unsupported_cases"] >= 10
    assert s["invalid_rejected"] == s["invalid_cases"] >= 8  # every wrong rewriting is caught
    assert s["verification_success"] == 1.0
    assert s["retrieval_recall_at_5"] >= 0.9
    assert s["retrieval_recall_at_3"] >= 0.8
    assert set(s["refusal_reasons"]) <= {
        "unsupported_distribution",
        "singular_root",
        "assumptions_insufficient",
        "integration_domain_ambiguous",
    }
    assert s["failed_transformations"] == []


def test_json_is_machine_readable(report: DistReport) -> None:
    document = json.loads(json.dumps(report.to_json(label="test")))
    assert document["format"] == 1 and document["label"] == "test"
    assert len(document["cases"]) == len(report.results)
    assert {"latency_median_ms", "latency_p95_ms", "coverage"} <= set(document["summary"])


def test_the_committed_baseline_has_no_regression(report: DistReport) -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    text, regressed = compare(baseline, report.to_json())
    assert not regressed, text


def test_compare_reports_regressions_fixes_and_changes() -> None:
    def doc(*cases: dict) -> dict:  # type: ignore[type-arg]
        return {
            "format": 1,
            "summary": {"cases_passed": sum(c["passed"] for c in cases)},
            "cases": list(cases),
        }

    def case(case_id: str, passed: bool, result: str = "1") -> dict:  # type: ignore[type-arg]
        return {
            "id": case_id,
            "passed": passed,
            "status": "ok",
            "result_text": result,
            "problems": [] if passed else ["wrong"],
        }

    before = doc(case("a", True), case("b", False), case("c", True, "1"))
    after = doc(case("a", False), case("b", True), case("c", True, "2"), case("d", True))
    text, regressed = compare(before, after)
    assert regressed
    assert "REGRESSION a" in text and "fixed b" in text and "changed c" in text
    assert "new case d" in text
    clean, regressed_again = compare(before, before)
    assert not regressed_again and "no per-case differences" in clean
    assert compare({"format": 0, "cases": []}, after)[1]
