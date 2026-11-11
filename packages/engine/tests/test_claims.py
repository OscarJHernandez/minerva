import pytest
from minerva_engine import Engine, Limits, check_claim, split_equation


@pytest.mark.parametrize(
    ("latex", "sides"),
    [
        (r"a = b", ["a", "b"]),
        (r"\frac{x^2-1}{x-1} = x+1 \quad \text{for } x \neq 1", [r"\frac{x^2-1}{x-1}", "x+1"]),
        (r"a = b = c", ["a", "b", "c"]),
        (r"f_{x=1}(y) = 2", ["f_{x=1}(y)", "2"]),
        (r"x = 2, \text{ if } y > 0", ["x", "2"]),
    ],
)
def test_split_equation(latex: str, sides: list[str]) -> None:
    assert split_equation(latex) == sides


@pytest.mark.parametrize("latex", ["x+1", "=", "x =", "= y"])
def test_no_equation(latex: str) -> None:
    assert split_equation(latex) is None


L = Limits()


def test_true_equation_holds_and_false_one_fails() -> None:
    assert check_claim(r"\frac{x^2-1}{x-1} = x+1", {}, L).status == "holds"
    assert check_claim(r"\sin^2 x + \cos^2 x = 1", {}, L).status == "holds"
    assert check_claim(r"(x+1)^2 = x^2 + 1", {}, L).status == "fails"


def test_claims_depend_on_assumptions() -> None:
    claim = r"\sqrt{x^2} = x"
    assert (
        check_claim(claim, {}, L).status == "conditional"
    )  # false for complex x, true for positive x
    assert check_claim(claim, {"x": ["positive"]}, L).status == "holds"
    assert check_claim(claim, {"x": ["negative"]}, L).status == "fails"


def test_chains_and_trailing_conditions() -> None:
    assert check_claim(r"2x = x + x = 2x \quad (x \ne 0)", {}, L).status in ("holds", "unparsed")
    assert check_claim(r"x + x = 2x \text{ for all } x", {}, L).status == "holds"


def test_undecidable_inputs_are_reported_not_guessed() -> None:
    assert check_claim("x + 1", {}, L).status == "not_an_equation"
    assert check_claim(r"\foo = 1", {}, L).status == "unparsed"
    assert check_claim(r"\int x\,dx = \frac{x^2}{2}", {}, L).status == "unparsed"


def test_engine_batch_runs_in_the_sandbox() -> None:
    engine = Engine(Limits(timeout_seconds=30))
    results = engine.check_claims([r"x+x = 2x", r"x+x = 3x", "y"], {})
    assert [r.status for r in results] == ["holds", "fails", "not_an_equation"]
    assert engine.check_claims([]) == []


def test_claims_true_only_for_real_or_positive_symbols_are_conditional_not_false() -> None:
    real = check_claim(r"\sqrt{x^2} = |x|", {}, L)
    assert real.status == "conditional" and "real" in real.detail
    positive = check_claim(r"\sqrt{x^2} = x", {}, L)
    assert positive.status == "conditional" and "positive" in positive.detail
    assert check_claim(r"(x+1)^2 = x^2 + 1", {}, L).status == "fails"  # false under any assumption
    assert check_claim(r"\sqrt{x^2} = x", {"x": ["positive"]}, L).status == "holds"
    assert check_claim(r"\sqrt{x^2} = -x", {"x": ["positive"]}, L).status == "fails"
