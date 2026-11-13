"""The distribution entries: their level is honest and the engine only cites entries that exist."""

import pytest
from minerva_engine.distributions.models import CITED_ENTRIES
from minerva_knowledge import KnowledgeBase
from minerva_knowledge.checks import run_check
from minerva_knowledge.models import Entry
from pydantic import ValidationError


def entry(**over):  # type: ignore[no-untyped-def]
    data = {
        "id": "distributions.test_entry",
        "title": "t",
        "kind": "identity",
        "chapter": "math.distributions",
        "verification": "distributional",
        "statement_latex": r"x\operatorname{DiracDelta}(x) = 0",
        "check": {
            "kind": "distribution",
            "action": "simplify",
            "input": "x*DiracDelta(x)",
            "expected": "0",
        },
    }
    data.update(over)
    return Entry(**data)


def test_every_entry_the_engine_cites_is_a_checked_distributional_entry(
    kb: KnowledgeBase,
) -> None:
    for entry_id in CITED_ENTRIES:
        found = kb.get(entry_id)
        assert found is not None, entry_id
        assert found.verification == "distributional", entry_id
        assert found.rule is None  # the pointwise rule search must never apply it


def test_distribution_entries_pass_their_own_check(kb: KnowledgeBase) -> None:
    entries = [e for e in kb.entries if e.verification == "distributional"]
    assert len(entries) >= 8
    for e in entries:
        report = run_check(e)
        assert report.passed, (e.id, report.detail)
        assert report.level == "distributional"
        assert report.symbolic_proof is False  # never presented as a symbolic proof


def test_a_wrong_expected_result_fails_the_check() -> None:
    wrong = entry(
        check={
            "kind": "distribution",
            "action": "simplify",
            "input": "DiracDelta(2*x)",
            "expected": "DiracDelta(x)",  # the factor 1/2 is missing
        }
    )
    assert run_check(wrong).passed is False


def test_an_unsupported_instance_fails_the_check() -> None:
    bad = entry(
        check={
            "kind": "distribution",
            "action": "simplify",
            "input": "DiracDelta(x**2)",  # repeated root
            "expected": "DiracDelta(x)",
        }
    )
    report = run_check(bad)
    assert report.passed is False and "unsupported" in report.detail


def test_the_distributional_level_only_takes_a_distribution_check() -> None:
    with pytest.raises(ValidationError):
        entry(check={"kind": "derivative", "expression": "x", "variable": "x", "derivative": "1"})
    with pytest.raises(ValidationError):
        entry(
            check=None,
            rule={"id": "distributions.test_entry", "lhs": "a", "rhs": "a", "variables": ["a"]},
        )


def test_a_pointwise_rule_with_a_delta_cannot_pass() -> None:
    # The old route: declaring a delta identity as an ordinary rule. Sampling must not accept it.
    bad = Entry(
        id="distributions.bad_rule",
        title="t",
        kind="identity",
        chapter="math.distributions",
        verification="numeric",
        rule={
            "id": "distributions.bad_rule",
            "lhs": "DiracDelta(a*x)",
            "rhs": "5*DiracDelta(x)",
            "variables": ["a"],
            "conditions": [{"variable": "a", "assumption": "positive"}],
        },
    )
    assert run_check(bad).passed is False
