import pytest
from minerva_knowledge.models import Entry
from pydantic import ValidationError


def base(**over):  # type: ignore[no-untyped-def]
    data = {
        "id": "x.y",
        "title": "t",
        "kind": "identity",
        "chapter": "math.algebra",
        "verification": "symbolic",
        "rule": {"id": "x.y", "lhs": "a", "rhs": "a", "variables": ["a"]},
    }
    data.update(over)
    return data


def test_valid_entry() -> None:
    Entry(**base())


def test_rejects_bad_id() -> None:
    with pytest.raises(ValidationError):
        Entry(**base(id="NoDots"))


def test_stated_cannot_carry_a_rule() -> None:
    with pytest.raises(ValidationError):
        Entry(**base(verification="stated", statement_latex="x"))


def test_stated_needs_a_statement() -> None:
    with pytest.raises(ValidationError):
        Entry(id="a.b", title="t", kind="theorem", chapter="c", verification="stated")


def test_verification_must_match_content() -> None:
    with pytest.raises(ValidationError):
        Entry(**base(verification="dimensional"))


def test_unknown_assumption_rejected() -> None:
    rule = {
        "id": "x.y",
        "lhs": "a",
        "rhs": "a",
        "variables": ["a"],
        "conditions": [{"variable": "a", "assumption": "magical"}],
    }
    with pytest.raises(ValidationError):
        Entry(**base(rule=rule))
