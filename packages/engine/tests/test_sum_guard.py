"""Sums, indexed data and expectations never reach a pointwise check.

One sample per free symbol gives every x[i] the same value, so sum x^2 = (sum x)^2 / n would "agree".
These tests pin the refusals: the parsers refuse the syntax unless an operation opts in, and every
checker refuses a hand-built sum, indexed element or moment.
"""

import pytest
import sympy as sp
from minerva_engine import (
    DiffVariable,
    Engine,
    ExpressionInput,
    Limits,
    ParseError,
    Status,
    UnsupportedInput,
    parse_latex,
    parse_text,
)
from minerva_engine.differentiate import check_derivative
from minerva_engine.indexed import Expectation, Variance, contains_indexed
from minerva_engine.integrate import check_antiderivative, check_definite
from minerva_engine.models import Rule
from minerva_engine.ncverify import verify_equal_nc
from minerva_engine.ops import parse_expression, rule_rewrites
from minerva_engine.restrictions import domain_restrictions
from minerva_engine.verify import numeric_detail, verify_equal

n = sp.Symbol("n", integer=True, positive=True)
i = sp.Symbol("i", integer=True)
mu = sp.Symbol("mu", real=True)
x = sp.IndexedBase("x", real=True)
X = sp.Symbol("X")

SUM_OF_SQUARES = sp.Sum(x[i] ** 2, (i, 1, n))
SQUARE_OF_SUM = sp.Sum(x[i], (i, 1, n)) ** 2 / n  # equal to the above only for n = 1

REFUSED_INPUTS = [
    ("latex", r"\sum_{i=1}^{n} x_i"),
    ("latex", r"\prod_{i=1}^{n} x_i"),
    ("text", "Sum(x[i], (i, 1, n))"),
    ("text", "Product(x[i], (i, 1, n))"),
    ("text", "x[1]"),
    ("text", "E[X]"),
    ("text", "Var(X)"),
]


@pytest.mark.parametrize(("fmt", "source"), REFUSED_INPUTS)
def test_the_parsers_refuse_sums_unless_an_operation_opts_in(fmt: str, source: str) -> None:
    parse = parse_latex if fmt == "latex" else parse_text
    with pytest.raises((ParseError, UnsupportedInput)):
        parse(source, {"n": ["integer", "positive"]})


@pytest.mark.parametrize(("fmt", "source"), REFUSED_INPUTS[:5])
def test_parse_expression_refuses_even_with_a_hand_made_gate(fmt: str, source: str) -> None:
    with pytest.raises((ParseError, UnsupportedInput)):
        parse_expression(
            ExpressionInput(input=source, format=fmt, assumptions={"n": ["integer", "positive"]}),  # type: ignore[arg-type]
            Limits(),
        )


@pytest.mark.parametrize("source", ["Sum(x[i], (i, 1, n))", "x[1] + x[2]"])
def test_every_operation_that_takes_an_expression_refuses(engine: Engine, source: str) -> None:
    assumptions = {"n": ["integer", "positive"], "x": ["real"]}
    expression = ExpressionInput(input=source, format="text", assumptions=assumptions)
    equation = ExpressionInput(input=f"y = {source}", format="text", assumptions=assumptions)
    results = [
        engine.simplify(expression),
        engine.differentiate(expression, [DiffVariable(name="n", order=1)]),
        engine.integrate(expression, "n"),
        engine.series(expression, "n"),
        engine.substitute(expression, ["n = 2"]),
        engine.isolate(equation, "y"),
    ]
    for result in results:
        assert result.status in {Status.PARSE_ERROR, Status.UNSUPPORTED}, result


def test_a_claim_with_a_sum_is_never_marked_false_or_checked(engine: Engine) -> None:
    claims = engine.check_claims(
        [r"\sum_{i=1}^{n} x_i^2 = \frac{1}{n}\left(\sum_{i=1}^{n} x_i\right)^2"]
    )
    assert [c.status for c in claims] == ["unparsed"] or all(
        c.status in {"unverified", "unparsed"} for c in claims
    )


def test_pointwise_verification_refuses_the_classic_false_identity() -> None:
    # Sampling would see x[i] equal for every i and "verify" this.
    result = verify_equal(SUM_OF_SQUARES, SQUARE_OF_SUM)
    assert result.passed is False
    assert "sampling" in result.note
    assert numeric_detail(SUM_OF_SQUARES, SQUARE_OF_SUM) == (0, 0, 0)


def test_moments_are_refused_by_pointwise_verification() -> None:
    # E[XY] = E[X]E[Y] "agrees" when a random variable is sampled as a number.
    y = sp.Symbol("Y")
    assert verify_equal(Expectation(X * y), Expectation(X) * Expectation(y)).passed is False
    assert verify_equal(Variance(X), Expectation(X**2) - Expectation(X) ** 2).passed is False


def test_the_derivative_and_integral_checkers_refuse_sums() -> None:
    score = sp.Sum(2 * mu - 2 * x[i], (i, 1, n))
    f = sp.Sum((x[i] - mu) ** 2, (i, 1, n))
    assert check_derivative(f, mu, score).passed is False
    assert check_derivative(f, mu, sp.Integer(0)).passed is False
    assert check_antiderivative(score, mu, f).passed is False
    t = sp.Symbol("t", real=True)
    assert (
        check_definite(
            sp.Sum(x[i] * t, (i, 1, n)),
            t,
            sp.Integer(0),
            sp.Integer(1),
            sp.Integer(1),
            deadline=1e12,
        ).passed
        is False
    )


def test_other_checkers_and_matchers_refuse_sums() -> None:
    with pytest.raises(UnsupportedInput):
        verify_equal_nc(SUM_OF_SQUARES, SUM_OF_SQUARES)
    assert domain_restrictions(1 / SUM_OF_SQUARES, 1 / SUM_OF_SQUARES) == ([], False)
    rule = Rule(id="commute", lhs="a*b", rhs="b*a", variables=["a", "b"])
    assert list(rule_rewrites(SUM_OF_SQUARES * mu, rule)) == []  # a plain product would match


def test_contains_indexed_sees_every_kind() -> None:
    assert contains_indexed(SUM_OF_SQUARES)
    assert contains_indexed(sp.Product(x[i], (i, 1, n)))
    assert contains_indexed(x[1] + 1)
    assert contains_indexed(Expectation(X))
    assert not contains_indexed(sp.sin(mu) + n)
