"""Structural features index rules and expressions; an expression with none retrieves nothing."""

import pytest
from minerva_engine import parse_text
from minerva_engine.features import expression_features


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("sin(x)", {"f:sin"}),
        ("x**2 - y**2", {"op:square"}),
        ("x**3 + y**3", {"op:cube"}),
        ("sqrt(x)", {"op:sqrt"}),
        ("x**n", {"op:power"}),
        ("x/y", {"op:quotient"}),
        ("sin(x)**2 + cos(x)**2", {"f:sin", "f:cos", "op:square"}),
    ],
)
def test_existing_features(source: str, expected: set[str]) -> None:
    assert expression_features(parse_text(source)) == expected


@pytest.mark.parametrize("source", ["x**4 - y**4", "(a + b)**4", "x**5", "x**10 + 1"])
def test_integer_powers_of_four_and_above_are_indexed(source: str) -> None:
    """Without a feature, rules like a**4 - b**4 could never be retrieved by structure."""
    assert "op:power" in expression_features(parse_text(source))


@pytest.mark.parametrize("source", ["x", "x + 1", "2*x", "x**1"])
def test_plain_expressions_have_no_power_feature(source: str) -> None:
    assert "op:power" not in expression_features(parse_text(source))
