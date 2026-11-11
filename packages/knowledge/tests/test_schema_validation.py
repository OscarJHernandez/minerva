import pytest
from minerva_engine import Rule
from minerva_knowledge.models import Entry, Reference
from pydantic import ValidationError


def entry(**over):  # type: ignore[no-untyped-def]
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


def test_rule_conditions_must_use_declared_variables() -> None:
    with pytest.raises(ValidationError, match="undeclared"):
        Rule(
            id="r",
            lhs="a",
            rhs="a",
            variables=["a"],
            conditions=[{"variable": "c", "assumption": "positive"}],  # type: ignore[list-item]
        )


def test_rule_id_must_match_entry_id() -> None:
    bad = entry(rule={"id": "other.id", "lhs": "a", "rhs": "a", "variables": ["a"]})
    with pytest.raises(ValidationError, match=r"rule\.id"):
        Entry(**bad)


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "data:text/html,x", "//evil.example", "ftp://x"]
)
def test_reference_urls_must_be_http(url: str) -> None:
    with pytest.raises(ValidationError):
        Reference(source="s", url=url)


def test_reference_accepts_https_and_none() -> None:
    assert Reference(source="s", url="https://dlmf.nist.gov/4.21").url
    assert Reference(source="s").url is None
