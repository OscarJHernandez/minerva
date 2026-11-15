"""The approximation reference corpus: every case agrees with an answer from outside Minerva,
every refusal keeps its stated reason, and the stored report is current."""

from __future__ import annotations

from minerva_engine.approx import benchmark


def test_every_case_agrees_with_its_reference_and_the_report_is_current() -> None:
    assert benchmark.main(["--check"]) == 0


def test_the_corpus_covers_every_family_and_the_refusals() -> None:
    ids = {case_id for case_id, _, _ in benchmark.CASES}
    assert {
        "taylor",
        "relativistic",
        "laurent",
        "asymptotic",
        "multivariable",
        "limits",
        "pade",
        "chebyshev",
        "chebyshev-tolerance",
        "fourier",
        "legendre",
        "piecewise",
        "continued-fractions",
        "euler-maclaurin",
        "acceleration",
        "integral-expansion",
        "root-and-inverse",
        "laplace",
        "strategy",
        "refusals",
    } <= ids


def test_a_wrong_reference_is_reported_as_a_failure() -> None:
    check = benchmark.close("deliberately wrong", 1.0000001, 1.0)
    assert not check.ok and benchmark.close("right", 1.0, 1.0).ok
