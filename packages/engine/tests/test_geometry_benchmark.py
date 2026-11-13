"""The geometry benchmark: the corpus is correct, refusals hold, and the stored results are current."""

import json
from pathlib import Path

import pytest
from minerva_engine.geometry import benchmark as B

ROOT = Path(__file__).resolve().parents[3]
STORED = ROOT / "docs" / "benchmarks" / "geometry.json"
REPORT = ROOT / "docs" / "geometry-benchmark.md"


@pytest.fixture(scope="module")
def fresh() -> dict:  # type: ignore[type-arg]
    return B.run(1, 2, 1.0, check_only=True)


def test_every_corpus_problem_is_computed_verified_and_equal_to_its_reference(fresh) -> None:  # type: ignore[no-untyped-def]
    cases = fresh["deterministic"]["cases"]
    assert set(cases) == {c.id for c in B.CASES}
    for cid, e in cases.items():
        assert e["status"] == "ok", cid
        assert e["correct"], (cid, e["correct_by_quantity"])
        assert e["signature_ok"], cid
        assert all(q["numeric_passed"] and q["status"] == "ok" for q in e["quantities"].values()), (
            cid
        )
        assert e.get("diffgeom_agrees", True), cid  # sympy.diffgeom gives the same components
    cov = fresh["deterministic"]["coverage"]
    assert cov["computed_and_correct"] == cov["corpus"] == len(B.CASES)


def test_the_corpus_covers_the_required_reference_cases() -> None:
    ids = {c.id for c in B.CASES}
    assert {"euclid3", "polar", "sphere2", "minkowski", "schwarzschild"} <= ids
    assert {c.id for c in B.CASES if c.use_diffgeom} >= {
        "euclid3",
        "polar",
        "sphere2",
        "schwarzschild",
    }
    # declared functions and a non-diagonal metric are covered too
    assert any(c.functions for c in B.CASES) and any(c.id == "sphere_sheared" for c in B.CASES)


def test_invalid_and_unsupported_problems_are_refused_with_the_expected_status(fresh) -> None:  # type: ignore[no-untyped-def]
    refusals = fresh["deterministic"]["refusals"]
    assert set(refusals) == {r.id for r in B.REFUSALS}
    assert all(r["as_expected"] for r in refusals.values()), refusals
    cov = fresh["deterministic"]["coverage"]
    assert cov["refusals_as_expected"] == cov["refusal_cases"] == len(B.REFUSALS)


def test_the_stored_results_match_a_fresh_run(fresh) -> None:  # type: ignore[no-untyped-def]
    """If this fails, rerun `python -m minerva_engine.geometry.benchmark` and commit both outputs."""
    stored = json.loads(STORED.read_text())
    assert stored["format"] == B.FORMAT
    problems = B.compare(stored, fresh, REPORT.read_text())
    assert problems == []


def test_the_stored_measurements_are_complete() -> None:
    stored = json.loads(STORED.read_text())["measurements"]
    for case in B.CASES:
        lat = stored["latency_ms"][case.id]
        assert lat["median"] > 0 and lat["p95"] >= lat["median"]
        assert stored["memory_mb"][case.id] > 0 and stored["stages"][case.id]
    assert [row["n"] for row in stored["scaling"]][:3] == [2, 3, 4]
    for row in stored["scaling"]:
        if row["status"] == "ok":
            assert row["ricci_scalar"].replace(" ", "") == row["expected_ricci_scalar"]
    assert {r["id"] for r in stored["vs_diffgeom"]} == {c.id for c in B.CASES if c.use_diffgeom}
    for row in stored["vs_diffgeom"]:
        assert row["ours_verified_ms"] >= row["ours_symbolic_only_ms"] >= 0  # flat space is ~0 ms
        assert row["diffgeom_ms"] > 0


def test_percentiles_use_the_nearest_rank() -> None:
    assert B._percentile([5, 1, 3, 2, 4], 0.95) == 5
    assert B._percentile([5, 1, 3, 2, 4], 0.5) == 3
    assert B._percentile([7.0], 0.95) == 7.0


def test_the_n_sphere_series_builds_the_round_metric() -> None:
    names, grid = B._nsphere(3)
    assert names == ["t1", "t2", "t3"]
    assert grid[0][0] == "a^2" and grid[1][1] == "a^2*sin(t1)^2"
    assert grid[2][2] == "a^2*sin(t1)^2*sin(t2)^2" and grid[0][1] == "0"


def test_a_digest_difference_is_reported_with_its_path() -> None:
    stored = {"cases": {"x": {"status": "ok", "ops": 10}}}
    changed = {"cases": {"x": {"status": "ok", "ops": 11}}}
    assert B._digest_diff(stored, stored) == []
    (problem,) = B._digest_diff(stored, changed)
    assert "/cases/x/ops" in problem and "stored 10" in problem and "now 11" in problem


def test_the_report_names_what_it_cannot_show() -> None:
    stored = json.loads(STORED.read_text())
    text = B.render_report(stored)
    assert "Limits of this evidence" in text and "not proof" in text
    assert text == B.render_report(stored)  # rendering is deterministic
