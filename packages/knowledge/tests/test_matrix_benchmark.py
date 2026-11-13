"""The matrix calculus benchmark: asserted cases hold, and progress on pending cases is locked in."""

import pytest
from minerva_engine import Engine
from minerva_knowledge.matrix_benchmark import (
    GROUPS,
    CaseResult,
    format_markdown,
    format_report,
    load_cases,
    percentile,
    run_case,
    run_matrix_benchmark,
    write_section,
)

CASES = load_cases()


def test_case_ids_are_unique_and_groups_known() -> None:
    assert len({c.id for c in CASES}) == len(CASES)
    assert {c.group for c in CASES} <= set(GROUPS)
    assert all(c.input.strip() for c in CASES)


def test_every_group_has_cases_and_a_known_unsupported_case_is_asserted() -> None:
    assert {c.group for c in CASES} == set(GROUPS)
    assert any(c.group == "unsupported" and not c.pending for c in CASES)


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_case(engine: Engine, case) -> None:  # type: ignore[no-untyped-def]
    result = run_case(engine, case)
    if case.pending:
        assert not result.passed, f"{case.id} now passes: remove `pending`"
    else:
        assert result.passed, (
            f"{case.id}: status={result.status} method={result.method} kind={result.kind} "
            f"evaluated={result.evaluated}"
        )


def test_report_and_markdown_render(engine: Engine, tmp_path) -> None:  # type: ignore[no-untyped-def]
    cases = [c for c in CASES if c.id in {"exp-diagonal", "dimension-mismatch"}]
    report = run_matrix_benchmark(engine, cases, repeat=2)
    assert not report.failures and not report.stale_pending
    summary = report.summary()
    assert summary["cases"] == 2 and summary["latency_p95_ms"] >= summary["latency_median_ms"]
    assert "exp-diagonal" in format_report(report)
    path = tmp_path / "bench.md"
    write_section(path, "Before", format_markdown(report, "Before", 2))
    write_section(path, "After", format_markdown(report, "After", 2))
    write_section(path, "Before", format_markdown(report, "Before", 2))  # replaces, keeps order
    text = path.read_text()
    assert text.count("## Run: Before") == 1 and text.count("## Run: After") == 1
    assert text.index("## Run: Before") < text.index("## Run: After")


def test_percentile_is_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 95) == 95.0 and percentile(values, 50) == 50.0
    assert percentile([], 95) == 0.0 and percentile([7.0], 95) == 7.0


def test_unknown_request_fields_are_reported_not_hidden() -> None:
    case = next(c for c in CASES if c.id == "evolution-constant-abstract")
    # a request the engine cannot express never counts as a pass
    result = CaseResult(
        case=case,
        status="not_expressible",
        method=None,
        kind=None,
        evaluated=None,
        verified=None,
        exact_leg=None,
        residual=None,
        milliseconds=[],
    )
    assert not result.passed
