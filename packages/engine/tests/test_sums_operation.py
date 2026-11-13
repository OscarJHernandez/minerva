"""The instance checker rejects wrong identities; the sums operation shows only verified rewrites."""

import pytest
import sympy as sp
from minerva_engine import Engine, ExpressionInput, Status, SumsInput, parse_text
from minerva_engine.indexed.instantiate import (
    check_derivative_indexed,
    explicit,
    size_assignments,
    verify_equal_indexed,
)

A = {"n": ["integer", "positive"], "x": ["real"], "y": ["real"], "mu": ["real"], "c": ["real"]}


def P(source: str, **assumptions: list[str]) -> sp.Expr:
    return parse_text(source, {**A, **assumptions}, allow_sums=True)


def test_sizes_always_include_a_size_of_at_least_two() -> None:
    sizes = [
        a for assignment in size_assignments(P("Sum(x[i], (i, 1, n))")) for a in assignment.values()
    ]
    assert 1 in sizes and any(s >= 2 for s in sizes)


def test_explicit_expansion_uses_independent_symbols_and_no_closed_forms() -> None:
    n = sp.Symbol("n", integer=True, positive=True)
    expanded = explicit(P("Sum(x[i]**2, (i, 1, n))").subs(n, 3))
    names = sorted(s.name for s in expanded.free_symbols)
    assert names == ["x__1", "x__2", "x__3"]
    assert explicit(P("Product(x[i], (i, 1, n))").subs(n, 2)).is_Mul


@pytest.mark.parametrize(
    "wrong",
    [
        "Sum(x[i], (i, 1, n))**2/n",  # true only for n = 1
        "Sum(x[i]**2, (i, 1, n - 1))",  # range off by one
        "Sum(x[i]**2, (i, 2, n))",  # starts at 2
        "Sum(x[i]**2, (i, 1, n)) + 1/1000000",  # 1e-6
        "n*x[1]**2",  # all elements equal the first
        "Sum(x[i]**2, (i, 1, n)) + Sum(x[i], (i, 1, n))",  # an extra term
    ],
)
def test_wrong_identities_are_rejected(wrong: str) -> None:
    assert verify_equal_indexed(P("Sum(x[i]**2, (i, 1, n))"), P(wrong)).passed is False


def test_the_classic_false_identity_holds_at_n_equal_1_so_one_size_proves_nothing() -> None:
    v = verify_equal_indexed(P("Sum(x[i]**2, (i, 1, n))"), P("Sum(x[i], (i, 1, n))**2/n"))
    assert v.passed is False and 0 < v.instances_agreed < v.instances_checked
    assert {"n": 1} in v.sizes and any(s["n"] >= 2 for s in v.sizes)


@pytest.mark.parametrize(
    ("original", "right"),
    [
        (
            "Sum((x[i] - mu)**2, (i, 1, n))",
            "Sum(x[i]**2, (i, 1, n)) - 2*mu*Sum(x[i], (i, 1, n)) + n*mu**2",
        ),
        ("Sum(x[i] + y[i], (i, 1, n))", "Sum(x[i], (i, 1, n)) + Sum(y[i], (i, 1, n))"),
        ("Sum(c*x[i], (i, 1, n))", "c*Sum(x[i], (i, 1, n))"),
        ("Sum(Sum(x[i]*y[j], (j, 1, n)), (i, 1, n))", "Sum(x[i], (i, 1, n))*Sum(y[j], (j, 1, n))"),
        ("Sum(x[i], (i, 1, n))", "Sum(x[i], (i, 1, n - 1)) + x[n]"),
    ],
)
def test_right_identities_are_accepted(original: str, right: str) -> None:
    v = verify_equal_indexed(P(original), P(right))
    assert v.passed and v.exact_zero_instances == v.instances_checked


def test_a_wrong_derivative_is_rejected_on_every_instance() -> None:
    f = P("Sum((x[i] - mu)**2, (i, 1, n))")
    mu = next(s for s in f.free_symbols if isinstance(s, sp.Symbol) and s.name == "mu")
    right = P("-2*Sum(x[i] - mu, (i, 1, n))")
    assert check_derivative_indexed(f, mu, right)[0].passed is True
    for wrong in ["-Sum(x[i] - mu, (i, 1, n))", "2*Sum(x[i] - mu, (i, 1, n))", "-2*(x[1] - mu)"]:
        assert check_derivative_indexed(f, mu, P(wrong))[0].passed is False


@pytest.fixture(scope="module")
def run(engine: Engine):
    def go(source: str, action: str = "expand", assumptions=None, **kw):
        return engine.sums(
            SumsInput(
                expression=ExpressionInput(
                    input=source, format="text", assumptions=assumptions or A
                ),
                action=action,  # type: ignore[arg-type]
                **kw,
            )
        )

    return go


def test_expand_the_sample_variance_identity(run) -> None:
    r = run("Sum((x[i] - Sum(x[j], (j, 1, n))/n)**2, (i, 1, n))")
    assert r.status is Status.OK and r.verification and r.verification.passed
    assert verify_equal_indexed(
        P(r.result_text or ""), P("Sum(x[i]**2, (i, 1, n)) - Sum(x[i], (i, 1, n))**2/n")
    ).passed
    assert "not proved for every size" in r.message


def test_nothing_to_do_is_unchanged_not_a_made_up_result(run) -> None:
    assert run("Sum(x[i], (i, 1, n))").status is Status.UNCHANGED


def test_log_of_a_product_needs_positive_factors(run) -> None:
    refused = run("log(Product(x[i], (i, 1, n)))", "log_of_product")
    assert refused.status is Status.CONDITIONS_NOT_SATISFIED and refused.result_text is None
    ok = run("log(Product(x[i], (i, 1, n)))", "log_of_product", {**A, "x": ["positive"]})
    assert ok.status is Status.OK and ok.conditions and ok.conditions[0].established


def test_differentiate_through_a_sum(run) -> None:
    r = run("Sum((x[i] - mu)**2, (i, 1, n))", "differentiate", variable="mu")
    assert r.status is Status.OK and r.derivative_verification and r.derivative_verification.passed


@pytest.mark.parametrize(
    ("source", "assumptions", "status", "reason"),
    [
        (
            "Sum(x[i], (i, 1, m))",
            {"x": ["real"]},
            Status.CONDITIONS_NOT_SATISFIED,
            "range_not_provable",
        ),
        ("Sum(x[i], (i, 1, oo))", A, Status.UNSUPPORTED, "unsupported_form"),
        ("Sum(x[i], (i, 1, n)) + i", A, Status.PARSE_ERROR, "parse_error"),
        ("x[1]" + "+x[1]" * 700, A, Status.RESOURCE_LIMIT, "resource_limit"),
    ],
)
def test_refusals_carry_a_status_and_a_reason(run, source, assumptions, status, reason) -> None:
    r = run(source, assumptions=assumptions)
    assert (r.status, r.reason) == (status, reason)
    assert r.result_text is None


def test_check_rejects_a_wrong_answer_and_accepts_a_right_one(run) -> None:
    wrong = run(
        "Sum(x[i]**2, (i, 1, n))", "check", candidate={"result": "Sum(x[i], (i, 1, n))**2/n"}
    )
    assert wrong.status is Status.NOT_VERIFIED and wrong.result_text is None
    right = run(
        "Sum(x[i]**2, (i, 1, n))",
        "check",
        candidate={"result": "Sum(x[i]**2, (i, 1, n-1)) + x[n]**2"},
    )
    assert right.status is Status.OK
    assert run("Sum(x[i], (i, 1, n))", "check").status is Status.PARSE_ERROR  # nothing to check
