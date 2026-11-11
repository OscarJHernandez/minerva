"""Every knowledge-base entry is verified at the level it declares."""

import pytest
from minerva_knowledge import KnowledgeBase, load_entries
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
