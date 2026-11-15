"""Expectation, variance and covariance: independence is declared, never assumed."""

import pytest
import sympy as sp
from minerva_engine import Engine, ExpectationInput, ExpressionInput, Status, parse_text
from minerva_engine.indexed.randoms import Covariance, Expectation, Variance
from minerva_engine.stats.expectation import Moments
from minerva_engine.stats.pmf_check import joint_models, verify_moments

A = {"n": ["integer", "positive"], "a": ["real"], "b": ["real"], "c": ["real"]}
X, Y = sp.symbols("X Y")
a, b, c = sp.symbols("a b c", real=True)


def P(source: str, random=("X", "Y")) -> sp.Expr:
    return parse_text(source, A, allow_sums=True, random_variables=list(random))


def expand(source: str, independent=(), random=("X", "Y")):
    m = Moments(frozenset(random), [list(g) for g in independent])
    return m.expand(P(source, random)), m.premises


def test_linearity_needs_no_independence() -> None:
    result, premises = expand("E[a*X + b*Y + c]")
    assert result == a * Expectation(X) + b * Expectation(Y) + c and premises == []


def test_the_variance_of_a_sum_keeps_the_covariance_unless_independence_is_declared() -> None:
    result, premises = expand("Var(X + Y)")
    assert result == Variance(X) + Variance(Y) + 2 * Covariance(X, Y) and premises == []
    result, premises = expand("Var(X + Y)", [["X", "Y"]])
    assert result == Variance(X) + Variance(Y) and premises == ["X and Y are independent"]


def test_the_expectation_of_a_product_factors_only_for_declared_independence() -> None:
    assert expand("E[X*Y]")[0] == Expectation(X * Y)
    assert expand("E[X*Y]", [["X", "Y"]])[0] == Expectation(X) * Expectation(Y)


def test_variance_scaling_and_the_definition_of_covariance() -> None:
    assert expand("Var(a*X + b)", random=("X",))[0] == a**2 * Variance(X)
    m = Moments(frozenset({"X", "Y"}), [])
    assert m.to_moments(P("Cov(X, Y)")) == Expectation(X * Y) - Expectation(X) * Expectation(Y)
    assert m.to_moments(P("Var(X)")) == Expectation(X**2) - Expectation(X) ** 2


def test_indexed_variables_and_component_independence() -> None:
    n = sp.Symbol("n", integer=True, positive=True)
    i = sp.Symbol("i", integer=True, positive=True)
    j = sp.Symbol("j", integer=True)
    x = sp.IndexedBase("X")
    general, _ = expand("Var(Sum(X[i], (i, 1, n)))", random=("X",))
    assert general.has(Covariance(x[i], x[j])) or general.has(
        Covariance(x[i], x[sp.Symbol("j", integer=True)])
    )
    independent, premises = expand("Var(Sum(X[i], (i, 1, n)))", [["X"]], ("X",))
    assert independent == sp.Sum(Variance(x[i]), (i, 1, n))
    assert premises == ["the components of X are mutually independent"]


def test_joint_models_are_dependent_unless_independence_is_declared() -> None:
    x, y = sp.symbols("X__1 Y__1")
    for independent, expect_product in [([], False), ([["X", "Y"]], True)]:
        outcomes = joint_models([x, y], independent, seed=3)
        px = sum(w for o, w in outcomes if o[x] == outcomes[0][0][x])
        py = sum(w for o, w in outcomes if o[y] == outcomes[0][0][y])
        pxy = sum(w for o, w in outcomes if o[x] == outcomes[0][0][x] and o[y] == outcomes[0][0][y])
        assert sum(w for _, w in outcomes) == 1
        assert (pxy == px * py) is expect_product


@pytest.mark.parametrize(
    ("original", "wrong"),
    [
        ("Var(X + Y)", "Var(X) + Var(Y)"),  # the covariance term dropped
        ("E[X*Y]", "E[X]*E[Y]"),  # independence assumed
        ("Var(a*X)", "a*Var(X)"),  # a, not a squared
        ("Cov(X, Y)", "E[X*Y]"),
        ("Var(X)", "E[X**2]"),
        ("Var(X + Y)", "Var(X) + Var(Y) + Cov(X, Y)"),  # factor 2 missing
        ("E[X**2]", "E[X]**2"),
    ],
)
def test_wrong_moment_identities_are_rejected(original: str, wrong: str) -> None:
    assert not verify_moments(P(original), P(wrong), frozenset({"X", "Y"}), []).passed


def test_a_result_that_needs_independence_passes_only_when_it_is_declared() -> None:
    v_dep = verify_moments(P("Var(X + Y)"), P("Var(X) + Var(Y)"), frozenset({"X", "Y"}), [])
    v_ind = verify_moments(
        P("Var(X + Y)"), P("Var(X) + Var(Y)"), frozenset({"X", "Y"}), [["X", "Y"]]
    )
    assert not v_dep.passed and v_ind.passed


@pytest.fixture(scope="module")
def run(engine: Engine):
    def go(source, action="expand", random=("X", "Y"), independent=(), **kw):
        return engine.expectation(
            ExpectationInput(
                expression=ExpressionInput(input=source, format="text", assumptions=A),
                random=list(random),
                independent=[list(g) for g in independent],
                action=action,  # type: ignore[arg-type]
                **kw,
            )
        )

    return go


def test_the_operation_marks_results_that_used_independence_as_conditional(run) -> None:
    free = run("Var(X + Y)")
    assert free.status is Status.OK and free.evidence == "identity" and free.given == []
    given = run("Var(X + Y)", independent=[["X", "Y"]])
    assert given.status is Status.OK and given.evidence == "conditional"
    assert given.given == ["X and Y are independent"] and "holds given" in given.message


def test_without_a_derivable_result_the_operation_says_unchanged(run) -> None:
    assert run("E[X*Y]").status is Status.UNCHANGED


def test_check_accepts_only_what_the_declarations_allow(run) -> None:
    assert run("E[X*Y]", "check", candidate={"result": "E[X]*E[Y]"}).status is Status.NOT_VERIFIED
    declared = run("E[X*Y]", "check", candidate={"result": "E[X]*E[Y]"}, independent=[["X", "Y"]])
    assert declared.status is Status.OK and declared.evidence == "conditional"


@pytest.mark.parametrize(
    ("source", "random", "independent", "status"),
    [
        ("E[Z]", ("X",), (), Status.PARSE_ERROR),  # Z was not declared random
        (
            "Var(X)",
            ("X",),
            [["X", "Q"]],
            Status.PARSE_ERROR,
        ),  # a group names an undeclared variable
        (
            "Var(X + Y + W)",
            ("X", "Y", "W"),
            [["X", "Y"], ["Y", "W"]],
            Status.UNSUPPORTED,
        ),  # overlapping
    ],
)
def test_bad_declarations_are_refused(run, source, random, independent, status) -> None:
    assert run(source, random=random, independent=independent).status is status
