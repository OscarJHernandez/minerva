"""The three-dimensional point-source entries: their level is honest and a wrong statement fails its check."""

import pytest
from minerva_knowledge import KnowledgeBase
from minerva_knowledge.checks import run_check
from minerva_knowledge.models import Entry, PointSourceCheck

KB = KnowledgeBase.load()
POINT = [e for e in KB.entries if isinstance(e.check, PointSourceCheck)]
IDS = {
    "distributions.coulomb_divergence",
    "distributions.newtonian_laplacian",
    "distributions.point_source_translation",
    "distributions.point_source_scaling",
    "em.static.gauss_point_charge",
}


def test_the_point_source_entries_are_the_five_the_handbook_promises() -> None:
    assert {e.id for e in POINT} == IDS


@pytest.mark.parametrize("entry", POINT, ids=lambda e: e.id)
def test_each_entry_is_distributional_has_no_rule_and_cites_a_locator(entry: Entry) -> None:
    assert entry.verification == "distributional" and entry.rule is None
    assert entry.reference is not None and entry.reference.locator
    assert entry.provenance in ("original", "public-domain-fact")
    assert r"\delta^{(3)}" in entry.statement_latex  # the engine printer's spelling, not a 1D delta
    assert "DiracDelta" not in entry.statement_latex


@pytest.mark.parametrize("entry", POINT, ids=lambda e: e.id)
def test_each_entry_is_confirmed_by_the_engine(entry: Entry) -> None:
    report = run_check(entry)
    assert report.passed and report.level == "distributional", report.detail
    assert report.symbolic_proof is False  # nothing here is called a symbolic proof


@pytest.mark.parametrize(
    ("entry_id", "wrong_target"),
    [
        ("distributions.coulomb_divergence", r"\delta^{(3)}(\mathbf r)"),  # normalisation
        ("distributions.coulomb_divergence", r"-4\pi\delta^{(3)}(\mathbf r)"),  # sign
        ("distributions.coulomb_divergence", "0"),  # the classical statement
        ("distributions.newtonian_laplacian", r"4\pi\delta^{(3)}(\mathbf r)"),  # the +4 pi mistake
        ("distributions.point_source_scaling", r"4\pi\delta^{(3)}(\mathbf r)"),  # dropped constant
        ("em.static.gauss_point_charge", r"\frac{q}{4\pi\epsilon_0}\delta^{(3)}(\mathbf r)"),
    ],
)
def test_a_wrong_right_hand_side_fails_the_check(entry_id: str, wrong_target: str) -> None:
    entry = KB.get(entry_id)
    wrong = entry.model_copy(
        update={"check": entry.check.model_copy(update={"target": wrong_target})}
    )
    report = run_check(wrong)
    assert not report.passed, report.detail
