"""The grammar of finite sums, products and indexed data, and its round trips."""

import pytest
import sympy as sp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from minerva_engine import ParseError, UnsupportedInput, parse_latex, parse_text
from minerva_engine.errors import RangeNotProvable
from minerva_engine.latexprint import to_latex, to_text

A = {"n": ["integer", "positive"], "x": ["real"], "y": ["real"], "mu": ["real"], "b": ["real"]}
PROFILE = settings(
    max_examples=25, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


def tex(source: str, **kw) -> sp.Expr:
    return parse_latex(source, A, allow_sums=True, **kw)


def txt(source: str, **kw) -> sp.Expr:
    return parse_text(source, A, allow_sums=True, **kw)


def test_both_syntaxes_read_the_same_sum() -> None:
    assert tex(r"\sum_{i=1}^{n}(x_i-\mu)^2") == txt("Sum((x[i] - mu)**2, (i, 1, n))")
    assert tex(r"\prod_{i=1}^{n} x_i") == txt("Product(x[i], (i, 1, n))")


def test_the_body_is_one_product_up_to_the_next_plus() -> None:
    assert tex(r"\sum_{i=1}^{n} x_i + b") == txt("Sum(x[i], (i, 1, n)) + b")
    assert tex(r"\sum_{i=1}^{n} x_i y_i") == txt("Sum(x[i]*y[i], (i, 1, n))")
    assert tex(r"\frac{1}{n}\sum_{i=1}^{n} x_i") == txt("Sum(x[i], (i, 1, n))/n")
    assert tex(r"\sum_{i=1}^{n}\left(x_i+y_i\right)") == txt("Sum(x[i] + y[i], (i, 1, n))")


def test_nested_sums_and_dependent_bounds() -> None:
    nested = tex(r"\sum_{i=1}^{n}\sum_{j=1}^{i} x_j")
    assert nested == txt("Sum(x[j], (j, 1, i), (i, 1, n))")
    assert tex(r"\sum_{i=1}^{n}\sum_{j=1}^{n} x_i y_j") == txt(
        "Sum(Sum(x[i]*y[j], (j, 1, n)), (i, 1, n))"
    )


def test_subscripts_outside_a_sum_stay_plain_symbols() -> None:
    assert parse_latex("x_1 + y_a", A) == sp.Symbol("x_1") + sp.Symbol("y_a")
    assert tex("x_1 + y_a") == sp.Symbol("x_1") + sp.Symbol("y_a")


def test_a_base_subscripted_by_an_index_is_indexed_everywhere() -> None:
    e = tex(r"x_1 + \sum_{i=1}^{n} x_{i+1}")
    base = sp.IndexedBase("x", real=True)
    assert e.has(base[1]) and e.has(base[sp.Symbol("i", integer=True, positive=True) + 1])


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (r"\sum_{i=1}^{n} x_{i,j}", ParseError),  # two indices
        (r"\sum_{i=1}^{n}\sum_{j=1}^{n} x_{ij}", ParseError),  # ambiguous
        (r"\sum_i x_i", ParseError),  # no bounds
        (r"\sum_{i=1} x_i", ParseError),  # no upper bound
        (r"\sum_{i=1}^{\infty} x_i", ParseError),  # \infty is not supported anywhere
        (r"\sum_{i=1}^{n} x_i + i", ParseError),  # the index escaped its sum
        (r"\sum_{i=1}^{n}\sum_{i=1}^{n} x_i", ParseError),  # shadowing
        (r"\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{k=1}^{n} x_i", ParseError),  # depth 3
        (r"\sum_{x=1}^{n} x", ParseError),  # the index is already a variable
        (r"\sum_{i=1}^{n} x_i + x", ParseError),  # x is data and a scalar
        (r"\sum_{i=3}^{1} x_i", ParseError),  # empty numeric range
        (r"\sum_{i=1}^{500} x_i", Exception),  # beyond the explicit-range limit
    ],
)
def test_ambiguous_or_unsound_forms_are_refused(source: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        tex(source)


def test_text_refusals() -> None:
    for source in [
        "Sum(x[i], (i, 1, oo))",  # infinite
        "x[i]",  # i is not bound
        "Sum(x[i], (i, 3, 1))",
        "Sum(x[i], i)",
        "x[1:2]",
        "x[1, 2]",
    ]:
        with pytest.raises((ParseError, UnsupportedInput)):
            txt(source)


def test_a_range_that_cannot_be_shown_non_empty_is_a_distinct_error() -> None:
    with pytest.raises(RangeNotProvable):
        parse_text("Sum(x[i], (i, 1, m))", {"x": ["real"]}, allow_sums=True)
    with pytest.raises(RangeNotProvable):
        parse_text("Sum(x[i], (i, 1, k))", {"k": ["integer"], "x": ["real"]}, allow_sums=True)
    # n - 1 with n a positive integer is a non-empty (possibly zero-term) range
    assert txt("Sum(x[i], (i, 1, n - 1))").has(sp.Sum)


def test_the_assumptions_of_a_base_apply_to_its_elements() -> None:
    e = parse_text("log(Product(x[i], (i, 1, n)))", {**A, "x": ["positive"]}, allow_sums=True)
    assert e.args[0].function.is_positive is True
    assert txt("Product(x[i], (i, 1, n))").function.is_positive is None


def test_moments_need_declared_random_variables() -> None:
    assert tex(r"\mathbb{E}[X+Y]", random_variables=["X", "Y"]) == txt(
        "E[X + Y]", random_variables=["X", "Y"]
    )
    for source in [r"\mathbb{E}[Z]", r"\operatorname{Var}(2)"]:
        with pytest.raises(ParseError):
            tex(source, random_variables=["X"])
    with pytest.raises(ParseError):
        tex(r"\mathbb{E}[X]")  # no random variable declared: not read as an expectation


@pytest.mark.parametrize(
    "source",
    [
        r"\sum_{i=1}^{n}(x_i-\mu)^2",
        r"b\sum_{i=1}^{n} x_i",
        r"\frac{\left(\sum_{i=1}^{n} x_i\right)^{2}}{n}",
        r"\sum_{i=1}^{n}\sum_{j=1}^{i} x_j y_i",
        r"-\sum_{i=1}^{n} x_i",
        r"\sum_{i=1}^{n} -x_i",
        r"\log\left(\prod_{i=1}^{n} x_i\right)",
        r"x_1+\sum_{i=1}^{n} x_{i+1}",
    ],
)
def test_printed_sums_read_back_in_both_syntaxes(source: str) -> None:
    e = tex(source)
    assert tex(to_latex(e)).dummy_eq(e)
    assert txt(to_text(e)).dummy_eq(e)


def test_printed_moments_read_back() -> None:
    for source in [r"\mathbb{E}\left[a X+b Y\right]", r"\operatorname{Cov}\left(X, Y\right)"]:
        e = parse_latex(
            source, {"a": ["real"], "b": ["real"]}, allow_sums=True, random_variables=["X", "Y"]
        )
        back = parse_text(
            to_text(e), {"a": ["real"], "b": ["real"]}, allow_sums=True, random_variables=["X", "Y"]
        )
        assert back.dummy_eq(e)


_atoms = st.sampled_from(["x[i]", "y[i]", "mu", "2", "i", "b"])
_bodies = st.recursive(
    _atoms,
    lambda inner: st.one_of(
        st.tuples(inner, st.sampled_from(["+", "-", "*"]), inner).map(
            lambda t: f"({t[0]} {t[1]} {t[2]})"
        ),
        inner.map(lambda s: f"({s})**2"),
    ),
    max_leaves=5,
)


@PROFILE
@given(body=_bodies, kind=st.sampled_from(["Sum", "Product"]))
def test_property_text_and_latex_round_trip(body: str, kind: str) -> None:
    e = txt(f"{kind}({body}, (i, 1, n))")
    assert txt(to_text(e)).dummy_eq(e)
    assert tex(to_latex(e)).dummy_eq(e)


def test_a_lowercase_sum_says_what_to_write() -> None:
    for source in ["sum(x[i], (i, 1, n))", "prod(x[i], (i, 1, n))"]:
        with pytest.raises(ParseError, match="capital letter"):
            txt(source)
    # the same call in an operation that does not allow sums is refused the same way, not misread
    with pytest.raises(ParseError, match="capital letter"):
        parse_text("sum(1, 2)", {})


def test_the_beta_function_prints_its_arguments_as_symbols() -> None:
    a, b = sp.symbols("alpha beta", positive=True)
    assert to_latex(sp.beta(a, b)) == r"\operatorname{B}\left(\alpha, \beta\right)"
