"""Second variation: the quadratic form, forged forms, and what may be concluded."""

from __future__ import annotations

from typing import Any

import mpmath as mp
import pytest
from minerva_engine.models import Limits, Status
from minerva_engine.variational import VariationInput, variation
from minerva_engine.variational.classify import ALL_FACTS, SecondEvidence, classify
from minerva_engine.variational.secondvar import (
    QuadraticForm,
    check_second_variation,
    second_form,
)
from minerva_engine.variational.setup import read_problem

FIXED = [
    {"function": "x", "order": 0, "end": "lower", "kind": "fixed", "value": "0"},
    {"function": "x", "order": 0, "end": "upper", "kind": "fixed", "value": "1"},
]


def run(integrand: str, **kw: Any) -> Any:
    kw.setdefault("functions", ["x"])
    kw.setdefault("lower", "0")
    kw.setdefault("upper", "1")
    data = VariationInput(action="second", integrand=integrand, **kw)
    return variation(data, Limits(timeout_seconds=120))


def problem(integrand: str, **kw: Any):  # type: ignore[no-untyped-def]
    kw.setdefault("functions", ["x"])
    kw.setdefault("lower", "0")
    kw.setdefault("upper", "1")
    return read_problem(VariationInput(action="second", integrand=integrand, **kw), Limits())


# -- the symbolic form --------------------------------------------------------------------------
def test_quadratic_form_of_the_harmonic_oscillator() -> None:
    r = run(r"\frac12\dot{x}^2-\frac12 x^2")
    assert r.status is Status.OK
    s = r.second
    assert s.checks.deciding == "passed"
    assert (
        "\\dot{\\eta}_{x}^{2}" in s.quadratic_form_latex
        and "\\eta_{x}^{2}" in s.quadratic_form_latex
    )
    assert s.legendre_matrix_latex == r"\begin{pmatrix}1\end{pmatrix}"


def test_cross_terms_and_endpoint_terms_are_part_of_the_form() -> None:
    r = run(r"\frac12\dot{x}^2+\frac12 x^2+x\dot{x}", upper_term=r"\frac{k}{2}x^2")
    assert r.status is Status.OK
    assert "\\dot{\\eta}_{x}" in r.second.quadratic_form_latex
    assert r.second.endpoint_form_latex  # the end term contributes k eta^2


def test_a_missing_cross_term_is_rejected() -> None:
    p = problem(r"\frac12\dot{x}^2+\frac12 x^2+x\dot{x}")
    form = second_form(p)
    assert check_second_variation(p, form).verdict == "passed"
    one_sided = QuadraticForm(
        {pq: h for pq, h in form.interior.items() if pq != ((0, 1), (0, 0))}, form.endpoint
    )
    assert check_second_variation(p, one_sided).verdict == "failed"


def test_a_wrong_factor_of_two_is_rejected() -> None:
    p = problem(r"\frac12\dot{x}^2-\frac12 x^2")
    form = second_form(p)
    doubled = QuadraticForm({k: 2 * v for k, v in form.interior.items()}, form.endpoint)
    halved = QuadraticForm({k: v / 2 for k, v in form.interior.items()}, form.endpoint)
    assert check_second_variation(p, doubled).verdict == "failed"
    assert check_second_variation(p, halved).verdict == "failed"


def test_higher_order_nonlinear_form_is_confirmed() -> None:
    r = run(r"\frac12\ddot{x}^2+\dot{x}^3+\sin(x)\dot{x}")
    assert r.status is Status.OK and r.second.checks.deciding == "passed"


# -- diagnostics along a path -------------------------------------------------------------------------
def test_long_interval_oscillator_has_a_certified_negative_direction() -> None:
    r = run(
        r"\frac12\dot{x}^2-\frac12 x^2",
        upper="4",
        conditions=[*FIXED[:1], {**FIXED[1], "value": "1"}],
        paths=[{"analytic": {"x": r"\sin(t)/\sin(4)"}}],
    )
    n = r.second.numeric
    assert n.stationary == "consistent"
    assert n.ritz.negative >= 1
    assert "not_a_weak_minimum" in n.facts
    assert any(d.sign == "negative" and d.certified for d in n.directions)
    # the closed form for eta = sin(pi t / 4) is 2 (pi^2/16 - 1); the Ritz space contains better
    assert float(n.ritz.eigenvalues[0]) < 0
    assert any("conjugate" in m for m in n.not_established)


def test_short_interval_oscillator_makes_no_claim_of_a_minimum() -> None:
    r = run(
        r"\frac12\dot{x}^2-\frac12 x^2",
        conditions=FIXED,
        paths=[{"analytic": {"x": r"\sin(t)/\sin(1)"}}],
    )
    n = r.second.numeric
    assert n.stationary == "consistent"
    assert n.ritz.negative == 0 and n.ritz.positive == n.ritz.dimension
    # no negative direction among those tried: said as exactly that, never as "a minimum"
    assert "no_negative_direction_found" in n.facts
    assert "not_a_weak_minimum" not in n.facts
    assert "not_a_weak_maximum" in n.facts  # a positive direction is a genuine fact too
    assert n.not_established, "sufficient conditions must be listed as not established"


def test_cubic_functional_with_a_negative_endpoint_is_not_a_minimum_and_violates_legendre() -> None:
    r = run(
        r"\dot{x}^3",
        conditions=[FIXED[0], {**FIXED[1], "value": "-1"}],
        paths=[{"analytic": {"x": "-t"}}],
    )
    n = r.second.numeric
    assert n.stationary == "consistent"
    assert "not_a_weak_minimum" in n.facts and "legendre_violated_for_minimum" in n.facts
    assert n.legendre.largest_eigenvalue < 0 and abs(n.legendre.smallest_eigenvalue + 6) < 1e-9


def test_a_path_that_is_not_stationary_is_not_classified() -> None:
    r = run(
        r"\frac12\dot{x}^2",
        conditions=FIXED,
        paths=[{"analytic": {"x": "t^2"}}],
    )
    n = r.second.numeric
    assert n.stationary == "violated" and n.facts == [] and n.directions == []
    assert n.stationary_residual > 1e-4


def test_free_end_is_respected_in_stationarity() -> None:
    # ½ẋ² - x with the upper end free: stationary path x = t - t²/2 (natural condition x'(1) = 0)
    r = run(
        r"\frac12\dot{x}^2-x",
        conditions=[
            {"function": "x", "order": 0, "end": "lower", "kind": "fixed", "value": "0"},
            {"function": "x", "order": 0, "end": "upper", "kind": "free"},
        ],
        paths=[{"analytic": {"x": "t-t^2/2"}}],
    )
    assert r.second.numeric.stationary == "consistent"
    wrong = run(
        r"\frac12\dot{x}^2-x",
        conditions=[
            {"function": "x", "order": 0, "end": "lower", "kind": "fixed", "value": "0"},
            {"function": "x", "order": 0, "end": "upper", "kind": "free"},
        ],
        paths=[{"analytic": {"x": "t-t^2/4"}}],  # satisfies the equation? no: x'' = -1/2
    )
    assert wrong.second.numeric.stationary == "violated"


def test_complex_parameter_values_are_not_classified() -> None:
    r = run(
        "m/2*x_dot**2",
        format="text",
        conditions=FIXED,
        values={"m": "1+I"},
        paths=[{"analytic": {"x": "t"}}],
    )
    assert r.status is Status.OK
    n = r.second.numeric
    assert n.facts == [] and n.ritz is None and n.directions == [] and n.legendre is None


# -- what the classification can say -----------------------------------------------------------------
def test_no_fact_asserts_a_minimum() -> None:
    assert ALL_FACTS
    for fact in ALL_FACTS:
        assert fact.startswith(("not_a_weak", "legendre_violated", "no_"))
        assert "is_a_minimum" not in fact and fact != "minimum"


@pytest.mark.parametrize("negative", [False, True])
def test_positive_evidence_never_yields_a_minimum(negative: bool) -> None:
    facts, missing = classify(SecondEvidence(True, "consistent", negative, False, False, False))
    assert ("not_a_weak_minimum" in facts) == negative
    assert "no_negative_direction_found" in facts or negative
    assert missing and not any(f == "minimum" for f in facts)


def test_violated_or_untested_stationarity_gives_no_facts() -> None:
    for stationarity in ("violated", "not_tested"):
        assert classify(SecondEvidence(True, stationarity, True, True, True, True))[0] == []  # type: ignore[arg-type]


def test_second_variation_is_non_negative_for_a_convex_functional_by_values() -> None:
    p = problem(r"\frac12\dot{x}^2")
    form = second_form(p)
    from minerva_engine.variational.numerics import Numerics
    from minerva_engine.variational.secondvar import quadratic_value
    from minerva_engine.variational.verify import perturbation

    num = Numerics(p, 3)
    path = num.field(num.random_path(3))
    for _ in range(3):
        value, _, _ = quadratic_value(num, form, path, num.field(perturbation(num, "generic")))
        assert value >= 0 and isinstance(value, mp.mpf)
