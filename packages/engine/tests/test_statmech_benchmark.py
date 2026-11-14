"""The statistical-mechanics corpus stays as expected, and its judge cannot pass a wrong answer."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from minerva_engine import Engine, Limits
from minerva_engine.statmech.benchmark import (
    CASES,
    agrees,
    deterministic,
    judge_case,
    markdown,
    run_cases,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(Limits(timeout_seconds=120, max_concurrency=2))


@pytest.fixture(scope="module")
def rows(engine: Engine) -> list[dict[str, object]]:
    return run_cases(engine)


def test_every_problem_comes_out_as_expected(rows: list[dict[str, object]]) -> None:
    bad = [
        (r["id"], r["outcome"], r["detail"])
        for r in rows
        if r["outcome"] not in ("correct", "refused_as_expected")
    ]
    assert not bad


def test_outcomes_match_the_committed_report(rows: list[dict[str, object]]) -> None:
    committed = json.loads((ROOT / "docs/benchmarks/statmech.json").read_text())["problems"]
    assert deterministic(rows) == committed, "run: python -m minerva_engine.statmech.benchmark"
    assert (ROOT / "docs/statmech-benchmark.md").read_text().split("\n", 5)[4] == markdown(
        rows
    ).split("\n", 5)[4]


def test_the_corpus_covers_the_requested_systems() -> None:
    categories = {c.category for c in CASES}
    assert categories >= {
        "two-level system",
        "spin-1/2 in a field",
        "classical harmonic oscillator",
        "quantum harmonic oscillator",
        "classical ideal gas",
        "finite-state canonical ensembles",
        "density-of-states integrals",
        "thermodynamic derivative identities",
        "fluctuation-response relations",
        "invalid and divergent partition functions",
        "quantum systems",
        "noncommuting quantum operators",
        "Bose and Fermi occupation numbers",
        "microcanonical counting",
        "Ising chains",
        "thermodynamic limits",
        "high- and low-temperature expansions",
        "Stirling approximation",
    }
    assert len({c.id for c in CASES}) == len(CASES) and all(c.source for c in CASES)
    refusals = [c for c in CASES if c.expected == "refused"]
    assert len(refusals) >= 8 and all(c.reason for c in refusals)
    reasons = {c.reason for c in refusals}
    assert reasons >= {
        "divergent",
        "convergence_not_provable",
        "invalid_degeneracy",
        "not_separable",
        "convention_unconfirmed",
        "ensemble_mismatch",
        "coupling_not_linear",
        "operators_do_not_commute",
        "not_hermitian",
        "energy_not_attainable",
        "singular_expansion_point",
        "ordering_not_decided",
        "nothing_to_approximate",
    }


def test_the_judge_cannot_pass_a_wrong_answer() -> None:
    assert agrees([("x", 1.0, 1.0)])[0]
    assert not agrees([("x", 1.000001, 1.0)])[0]  # a 1e-6 relative error is wrong
    assert not agrees([("x", 0.0, 1e-3)])[0]


def _case(case_id: str) -> object:
    return next(c for c in CASES if c.id == case_id)


def test_a_wrong_reference_a_wrong_reason_and_a_lost_answer_are_each_caught(
    engine: Engine,
) -> None:
    good = _case("two-level-thermodynamics")
    wrong_reference = replace(good, judge=lambda r: (False, "reference disagrees"))  # type: ignore[type-var]
    assert judge_case(wrong_reference, engine)[0] == "wrong"  # type: ignore[arg-type]

    refusal = _case("divergent-partition-function")
    wrong_reason = replace(refusal, reason="not_separable")  # type: ignore[type-var]
    assert judge_case(wrong_reason, engine)[0] == "wrong"  # type: ignore[arg-type]

    answer_where_refusal_due = replace(good, expected="refused", reason="divergent")  # type: ignore[type-var]
    assert judge_case(answer_where_refusal_due, engine)[0] == "wrong"  # type: ignore[arg-type]

    lost = replace(good, request={**good.request, "action": "analyze"})  # type: ignore[attr-defined]
    outcome = judge_case(replace(lost, judge=None), engine)  # type: ignore[type-var,arg-type]
    assert outcome[0] == "unsupported_regression"
