"""`\\gamma` is the Greek letter in LaTeX (a damping rate, a Lorentz factor); the Gamma function is `\\Gamma`.

It used to be refused as "a reserved function or constant name", because the text syntax has a function named
`gamma`: a model could not even write the decay rate of a damped oscillator. In text syntax `gamma` stays the function."""

import pytest
import sympy as sp
from minerva_engine import ParseError, parse_latex, parse_text
from minerva_engine.latexprint import to_latex
from minerva_engine.models import Limits, Status
from minerva_engine.testing import budget_limits

L = Limits()


def test_gamma_in_latex_is_a_symbol_with_the_declared_assumptions() -> None:
    g = parse_latex(r"\gamma", {"gamma": ["positive"]}, L)
    assert isinstance(g, sp.Symbol) and g.name == "gamma" and g.is_positive is True
    plain = parse_latex(r"\gamma", {}, L)
    assert isinstance(plain, sp.Symbol) and plain.is_positive is None  # nothing is assumed


def test_a_subscripted_gamma_and_products_are_ordinary() -> None:
    assert parse_latex(r"\gamma_1", {}, L) == sp.Symbol("gamma_1")
    t = sp.Symbol("t")
    assert parse_latex(r"\exp(-\gamma t)", {}, L) == sp.exp(-sp.Symbol("gamma") * t)


def test_the_gamma_function_is_still_big_gamma_and_text_gamma_is_still_the_function() -> None:
    x = sp.Symbol("x")
    assert parse_latex(r"\Gamma(x)", {}, L) == sp.gamma(x)
    assert parse_text("gamma(x)", {}, L) == sp.gamma(x)
    with pytest.raises(ParseError, match="reserved"):
        parse_text("gamma*x", {}, L)  # in text syntax a bare gamma is not a symbol


def test_the_letter_prints_back_as_the_letter() -> None:
    expr = parse_latex(r"-\gamma t", {"gamma": ["positive"], "t": ["real"]}, L)
    again = parse_latex(to_latex(expr), {"gamma": ["positive"], "t": ["real"]}, L)
    assert to_latex(expr).count(r"\gamma") == 1 and again == expr


def test_the_causal_decay_differentiates_to_a_delta_minus_the_decay_for_a_positive_rate() -> None:
    """The motivating example: d/dt [exp(-gamma t) H(t)] = delta(t) - gamma exp(-gamma t) H(t)."""
    from minerva_engine import DistributionInput, ExpressionInput
    from minerva_engine.distributions.operation import distribution

    r = distribution(
        DistributionInput(
            expression=ExpressionInput(
                input=r"\exp(-\gamma t)\operatorname{Heaviside}(t)",
                format="latex",
                assumptions={"gamma": ["positive"], "t": ["real"]},
            ),
            action="differentiate",
            variable="t",
        ),
        budget_limits(),
    )
    assert r.status is Status.OK and r.verification is not None and r.verification.passed
    assert r.equality == "distributional"
    latex = r.result_latex or ""
    # the two terms of the textbook result, each exactly once and with the right signs
    assert latex.count(r"\operatorname{DiracDelta}\left(t\right)") == 1
    assert r"- \gamma \mathrm{e}^{- \gamma t} \operatorname{Heaviside}\left(t\right)" in latex
    assert latex.count("+") == 1
