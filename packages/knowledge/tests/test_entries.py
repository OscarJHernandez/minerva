"""Every knowledge-base entry is verified at the level it declares."""

import pytest
from minerva_knowledge import KnowledgeBase, SearchQuery, load_entries
from minerva_knowledge.checks import run_check

ENTRIES = load_entries()


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.id)
def test_entry_passes_its_declared_check(entry) -> None:  # type: ignore[no-untyped-def]
    report = run_check(entry)
    assert report.passed, f"{entry.id}: {report.detail}"
    if entry.verification == "symbolic":
        assert report.symbolic_proof, f"{entry.id} is labelled symbolic but no symbolic zero found"


def test_stated_entries_are_never_applicable(kb: KnowledgeBase) -> None:
    assert all(e.rule is None for e in kb.entries if e.verification == "stated")


def test_ids_are_unique_and_chapters_exist(kb: KnowledgeBase) -> None:
    ids = [e.id for e in kb.entries]
    assert len(ids) == len(set(ids))
    chapters = {c.id for c in kb.chapters}
    assert {e.chapter for e in kb.entries} <= chapters


def test_toc_hides_planned_chapters(kb: KnowledgeBase) -> None:
    shown = {c["id"] for c in kb.toc()}
    planned = {c.id for c in kb.chapters if c.planned}
    assert planned and not (shown & planned)
    assert "math.trigonometry" in shown
    assert "physics.constants" in shown


def test_every_entry_has_a_statement(kb: KnowledgeBase) -> None:
    for e in kb.entries:
        assert kb.statement_latex(e).strip()


def test_references_are_citations_only(kb: KnowledgeBase) -> None:
    for e in kb.entries:
        if e.reference:
            assert e.reference.source
            assert len(e.reference.locator or "") < 80  # a locator, never quoted text


def test_constant_candidates_cover_every_constant_with_a_readable_unit() -> None:
    from minerva_engine.dimensions import read_unit
    from minerva_knowledge.identities import constant_candidates

    kb = KnowledgeBase.load()
    candidates = constant_candidates(kb)
    constants = [e for e in kb.entries if e.kind == "constant"]
    assert {c.id for c in candidates} == {e.id for e in constants}
    by_id = {c.id: c for c in candidates}
    assert by_id["const.c"].latex == "c" and by_id["const.k_b"].latex == "k_B"
    assert by_id["const.hbar"].latex == r"\hbar"
    for c in candidates:
        read_unit(c.unit)  # every handbook unit is in the vocabulary


def test_dimensional_methods_are_retrieved_by_feature() -> None:
    kb = KnowledgeBase.load()
    for feature, expected in [
        (
            "dimensional:pi-groups",
            {"method.phys.buckingham_pi", "method.phys.dimensional_homogeneity"},
        ),
        ("dimensional:missing-quantity", {"method.phys.dimensional_homogeneity"}),
    ]:
        found = {e.id for e in kb.search(SearchQuery(features=[feature], limit=10))}
        assert expected <= found
