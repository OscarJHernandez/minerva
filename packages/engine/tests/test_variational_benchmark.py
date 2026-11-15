"""The reference corpus: every case agrees with its closed form, and the stored report is current."""

from __future__ import annotations

from minerva_engine.variational import benchmark


def test_every_case_agrees_with_its_closed_form_and_the_report_is_current() -> None:
    assert benchmark.main(["--check"]) == 0


def test_the_corpus_covers_the_required_families() -> None:
    ids = {case_id for case_id, _, _ in benchmark.CASES}
    assert {
        "free-particle",
        "harmonic",
        "coupled",
        "time-dependent",
        "higher-derivative",
        "endpoint-term",
        "free-end",
        "cantilever",
        "catenary",  # no unique stationary path
        "harmonic-long",  # stationary, not a minimum
        "singular",
        "null-lagrangians",
        "mixed-orders",
    } <= ids
