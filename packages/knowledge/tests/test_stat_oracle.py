"""The benchmark oracle must tell right from wrong answers, and must never execute its input."""

import mpmath as mp
import pytest
from minerva_knowledge.stat_oracle import (
    OracleError,
    environments,
    equal_texts,
    evaluate,
)

SUMS = {"n": "size", "mu": "real", "x": "vector_real", "y": "vector_real"}


def same(a: str, b: str, spec: dict[str, str] | None = None, **kw) -> bool:
    return equal_texts(a, b, environments(spec or SUMS, **kw))[0]


def test_a_correct_expansion_is_accepted() -> None:
    assert same(
        "Sum((x[i] - mu)**2, (i, 1, n))",
        "Sum(x[i]**2, (i, 1, n)) - 2*mu*Sum(x[i], (i, 1, n)) + n*mu**2",
    )


@pytest.mark.parametrize(
    "wrong",
    [
        "Sum(x[i]**2, (i, 1, n)) - n*mu**2",  # dropped term
        "Sum(x[i]**2, (i, 1, n)) - 2*mu*Sum(x[i], (i, 1, n)) + n*mu**2 + 1/1000000",  # 1e-6
        "Sum(x[i]**2, (i, 1, n)) - 2*mu*Sum(x[i], (i, 1, n-1)) + n*mu**2",  # off by one
    ],
)
def test_wrong_sum_identities_are_rejected(wrong: str) -> None:
    right = "Sum(x[i]**2, (i, 1, n)) - 2*mu*Sum(x[i], (i, 1, n)) + n*mu**2"
    assert not same(wrong, right)


def test_the_classic_false_identity_is_rejected_although_it_holds_for_n_equal_1() -> None:
    assert not same("Sum(x[i]**2, (i, 1, n))", "Sum(x[i], (i, 1, n))**2/n")
    one = {"n": "size", "x": "vector_real"}
    envs = environments(one)
    for env, _ in envs:
        env["n"] = mp.mpf(1)
        env["x"] = env["x"][:1]
    assert equal_texts("Sum(x[i]**2, (i, 1, n))", "Sum(x[i], (i, 1, n))**2/n", envs)[0]


def test_independence_is_never_assumed() -> None:
    rv = {"X": "random", "Y": "random"}
    assert same("Var(X + Y)", "Var(X) + Var(Y) + 2*Cov(X, Y)", rv)
    assert not same("Var(X + Y)", "Var(X) + Var(Y)", rv)
    assert not same("E[X*Y]", "E[X]*E[Y]", rv)
    assert same("Var(X + Y)", "Var(X) + Var(Y)", rv, independent=[["X", "Y"]])
    assert same("E[X*Y]", "E[X]*E[Y]", rv, independent=[["X", "Y"]])


def test_indexed_random_variables_and_component_independence() -> None:
    rv = {"n": "size", "X": "random_vector"}
    covariances = "Sum(Sum(Cov(X[i], X[j]), (j, 1, n)), (i, 1, n))"
    assert same("Var(Sum(X[i], (i, 1, n)))", covariances, rv)
    assert not same("Var(Sum(X[i], (i, 1, n)))", "Sum(Var(X[i]), (i, 1, n))", rv)
    assert same("Var(Sum(X[i], (i, 1, n)))", "Sum(Var(X[i]), (i, 1, n))", rv, independent=[["X"]])


def test_matrix_products_and_scalar_blocks() -> None:
    decl = [{"name": "A", "rows": 3, "cols": 3}, {"name": "x", "rows": 3, "cols": 1}]
    envs = environments({}, decl)
    assert equal_texts("tr(A) + transpose(x)*A*x", "tr(transpose(A)) + transpose(x)*A*x", envs)[0]
    assert not equal_texts("transpose(x)*A*x", "transpose(x)*transpose(A)*A*x", envs)[0]
    pd = environments({}, [{"name": "S", "rows": 3, "cols": 3, "positive_definite": True}])
    assert equal_texts("inv(S)*S", "S*inv(S)", pd)[0]
    assert equal_texts("transpose(inv(S))", "inv(S)", pd)[0]  # symmetric by declaration


@pytest.mark.parametrize(
    "hostile",
    [
        "__import__('os').system('true')",
        "(1).__class__",
        "[i for i in range(3)]",
        "lambda: 1",
        "open('/etc/passwd')",
        "x.real",
        "Sum(1, (i, 1, 10**9))",
        "2**(10**7)",
        "1/0",
        "'text'",
    ],
)
def test_hostile_or_unreadable_text_raises_oracle_error(hostile: str) -> None:
    with pytest.raises(OracleError):
        evaluate(hostile, {"x": mp.mpf(1)})


def test_evaluation_is_exact_enough_to_see_1e_minus_6() -> None:
    env = {"a": mp.mpf(3)}
    assert evaluate("a/7", env) != evaluate("a/7 + 1/1000000", env)
