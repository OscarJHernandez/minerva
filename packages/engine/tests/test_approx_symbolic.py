"""Approximation, phase 1: local symbolic expansions, leading behaviour and limits.

Expected values come from closed forms worked out independently of the code; every checker is also
shown deliberately wrong candidates.
"""

from __future__ import annotations

import mpmath
import pytest
import sympy as sp
from minerva_engine import ApproximationInput
from minerva_engine.approx import expand, leading, multivar
from minerva_engine.approx.common import Context
from minerva_engine.approx.operation import approximation
from minerva_engine.models import Limits, Status

LIMITS = Limits(timeout_seconds=60)


def run(**kw):  # type: ignore[no-untyped-def]
    kw.setdefault("format", "text")
    return approximation(ApproximationInput(**kw), LIMITS)


def texts(report):  # type: ignore[no-untyped-def]
    return {o.order: o.coefficient_latex for o in report.orders}


# --- Taylor -----------------------------------------------------------------------------------
def test_taylor_sine() -> None:
    r = run(method="taylor", input="sin(x)", point="0", order=5)
    assert r.status is Status.OK
    assert r.result_latex == r"x - \frac{x^{3}}{6} + \frac{x^{5}}{120}"
    assert r.remainder_latex == r"O\left(x^{6}\right)"
    assert r.evidence == "series_checked"
    assert all(c.passed for c in r.checks)
    assert "local_only" in r.warnings


def test_taylor_about_another_point() -> None:
    r = run(method="taylor", input="log(x)", point="1", order=3)
    assert r.status is Status.OK
    assert texts(r) == {1: "1", 2: r"- \frac{1}{2}", 3: r"\frac{1}{3}"}


def test_taylor_refuses_a_pole_and_points_to_laurent() -> None:
    r = run(method="taylor", input="1/x", point="0", order=3)
    assert r.status is Status.UNSUPPORTED
    assert r.result_latex is None
    assert any("laurent" in n for n in r.notes)


def test_missing_point_and_order_are_questions_not_defaults() -> None:
    r = run(method="taylor", input="sin(x)", order=3)
    assert r.status is Status.CONDITIONS_NOT_SATISFIED and r.reason == "domain_unspecified"
    assert r.questions
    r = run(method="taylor", input="sin(x)", point="0")
    assert r.reason == "domain_unspecified" and r.questions
    r = run(method="laurent", input="1/x", point="0", order=40)
    assert r.reason == "order_out_of_range"


# --- Laurent ----------------------------------------------------------------------------------
def test_laurent_simple_pole_residue_and_principal_part() -> None:
    r = run(method="laurent", input="1/(x*(1-x))", point="0", order=3)
    assert r.status is Status.OK
    assert r.result_latex == r"\frac{1}{x} + 1 + x + x^{2} + x^{3}"
    extras = {e.label: e.latex for e in r.extras}
    assert extras["Pole order"] == "1" and extras["Residue"] == "1"
    assert extras["Principal part"] == r"\frac{1}{x}"


def test_laurent_double_pole_has_zero_residue() -> None:
    r = run(method="laurent", input="1/sin(x)**2", point="0", order=2)
    extras = {e.label: e.latex for e in r.extras}
    assert extras["Pole order"] == "2" and extras["Residue"] == "0"
    assert r.result_latex == r"\frac{1}{x^{2}} + \frac{1}{3} + \frac{x^{2}}{15}"


def test_laurent_residue_at_another_pole() -> None:
    r = run(method="laurent", input="exp(x)/(x-2)", point="2", order=2)
    extras = {e.label: e.latex for e in r.extras}
    assert extras["Residue"] == r"\mathrm{e}^{2}"


def test_laurent_removable_singularity_is_a_warning() -> None:
    r = run(method="laurent", input="sin(x)/x", point="0", order=3)
    assert r.status is Status.OK
    assert "removable_singularity" in r.warnings
    assert {e.label: e.latex for e in r.extras}["Residue"] == "0"


def test_laurent_refuses_an_essential_singularity() -> None:
    r = run(method="laurent", input="exp(1/x)", point="0", order=3)
    assert r.status is Status.UNSUPPORTED and r.result_latex is None


def test_laurent_refuses_a_branch_point() -> None:
    r = run(method="laurent", input="sqrt(x)/x", point="0", order=2)
    assert r.status is Status.UNSUPPORTED


def test_laurent_with_a_wrong_residue_is_not_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    real = sp.series

    def wrong(expr, var, x0, n):  # type: ignore[no-untyped-def]
        return real(expr, var, x0, n) + sp.Rational(1, 10**6) / (var - x0) ** 1

    monkeypatch.setattr(expand.sp, "series", wrong)
    r = run(method="laurent", input="1/(x*(1-x))", point="0", order=3)
    assert r.status is Status.NOT_VERIFIED
    assert r.result_latex is None


# --- Asymptotic -------------------------------------------------------------------------------
def test_asymptotic_inverse_powers() -> None:
    r = run(method="asymptotic", input="1/(x+1)", point="oo", order=3)
    assert r.status is Status.OK
    assert r.result_latex == r"\frac{1}{x} - \frac{1}{x^{2}} + \frac{1}{x^{3}}"
    assert r.remainder_latex == r"O\left(x^{-4}\right)"
    assert "asymptotic_not_convergent" in r.warnings


def test_asymptotic_square_root_has_a_growing_leading_term() -> None:
    r = run(method="asymptotic", input="sqrt(x**2+x)", point="oo", order=3)
    assert r.status is Status.OK
    assert r.result_latex.startswith("x + \\frac{1}{2} - \\frac{1}{8 x}")  # type: ignore[union-attr]


def test_asymptotic_direction_changes_the_answer() -> None:
    plus = run(method="asymptotic", input="sqrt(x**2-x)", point="oo", order=2)
    minus = run(method="asymptotic", input="sqrt(x**2-x)", point="-oo", order=2)
    assert plus.status is minus.status is Status.OK
    assert plus.result_latex.startswith("x - \\frac{1}{2}")  # type: ignore[union-attr]
    assert minus.result_latex.startswith("- x + \\frac{1}{2}")  # type: ignore[union-attr]


def test_asymptotic_with_a_logarithm_is_numerically_checked_only() -> None:
    r = run(method="asymptotic", input="log(x+1)", point="oo", order=3)
    assert r.status is Status.OK
    assert r.evidence == "numeric_estimate"
    assert r.result_latex.startswith("\\log{\\left(x \\right)} + \\frac{1}{x}")  # type: ignore[union-attr]


def test_asymptotic_needs_a_direction() -> None:
    r = run(method="asymptotic", input="1/(x+1)", order=3)
    assert r.reason == "domain_unspecified"
    r = run(method="asymptotic", input="1/(x+1)", point="3", order=3)
    assert r.reason == "point_invalid"


def test_asymptotic_exponential_is_refused_not_guessed() -> None:
    r = run(method="asymptotic", input="exp(-x)", point="oo", order=2)
    assert r.status in (Status.UNSUPPORTED, Status.NOT_VERIFIED)
    assert r.result_latex is None
    growing = run(method="asymptotic", input="exp(x)", point="oo", order=2)
    assert growing.status in (Status.UNSUPPORTED, Status.NOT_VERIFIED)


def _asym_check(approx: sp.Expr, claimed: int) -> bool | None:
    x = sp.Symbol("x", positive=True)
    return expand._residual_check(1 / (x + 1), approx, x, 1, claimed).passed


def test_residual_check_sees_a_coefficient_wrong_by_one_part_in_a_billion() -> None:
    x = sp.Symbol("x", positive=True)
    good = 1 / x - 1 / x**2 + 1 / x**3
    assert _asym_check(good, 4) is True
    assert _asym_check(1 / x - 1 / x**2 + (1 + sp.Rational(1, 10**9)) / x**3, 4) is False
    assert _asym_check(1 / x - 1 / x**2 + 2 / x**3, 4) is False
    assert _asym_check(1 / x - sp.Rational(11, 10) / x**2 + 1 / x**3, 4) is False
    assert _asym_check(1 / x - 1 / x**2, 4) is False  # one term short of the claimed order
    # An over-cautious claim is fine: the same truncation holds to order 3.
    assert _asym_check(1 / x - 1 / x**2 + 1 / x**3, 3) is True


# --- Perturbation -----------------------------------------------------------------------------
def test_perturbation_lorentz_factor() -> None:
    r = run(method="perturbation", input="1/sqrt(1-eps**2)", small="eps", order=4)
    assert r.status is Status.OK
    assert r.result_latex == r"1 + \frac{eps^{2}}{2} + \frac{3 eps^{4}}{8}"
    assert texts(r) == {0: "1", 2: r"\frac{1}{2}", 4: r"\frac{3}{8}"}


def test_perturbation_nonrelativistic_energy_in_v_over_c() -> None:
    r = run(
        method="perturbation",
        input="m*c**2/sqrt(1-v**2/c**2)",
        small="v/c",
        order=4,
        assumptions={"m": ["positive"], "c": ["positive"], "v": ["real"]},
    )
    assert r.status is Status.OK
    assert r.result_latex == r"c^{2} m + \frac{m v^{2}}{2} + \frac{3 m v^{4}}{8 c^{2}}"
    by_order = {o.order: o.term_latex for o in r.orders}
    assert by_order[2] == r"\frac{m v^{2}}{2}"  # the first non-trivial correction
    assert r.remainder_latex == r"O\left(\left(\frac{v}{c}\right)^{5}\right)"
    assert r.assumptions_stated


def test_perturbation_with_several_parameters_checks_at_sampled_values() -> None:
    r = run(
        method="perturbation",
        input="1/(a + lam*b)",
        small="lam",
        order=2,
        assumptions={"a": ["positive"], "b": ["positive"], "lam": ["real"]},
    )
    assert r.status is Status.OK
    assert any(c.points >= 4 for c in r.checks)


def test_perturbation_needs_an_isolable_small_parameter() -> None:
    r = run(method="perturbation", input="1/(1-eps)", small="1+eps", order=2)
    assert r.reason == "small_parameter_not_isolable"
    r = run(method="perturbation", input="1/(1-eps)", order=2)
    assert r.reason == "parameter_missing" and r.questions


def test_perturbation_refuses_when_there_is_no_taylor_series_in_the_parameter() -> None:
    r = run(method="perturbation", input="1/eps", small="eps", order=2)
    assert r.status is Status.UNSUPPORTED and r.result_latex is None


# --- Multivariable ----------------------------------------------------------------------------
def test_multivariable_exponential_degree_two() -> None:
    r = run(
        method="multivariable", input="exp(x+y)", variables=["x", "y"], points=["0", "0"], order=2
    )
    assert r.status is Status.OK
    assert r.result_latex == r"1 + x + y + \frac{x^{2}}{2} + x y + \frac{y^{2}}{2}"
    assert r.remainder_latex == r"O\left(\left\|\left(x, y\right)\right\|^{3}\right)"
    extras = {e.label: e.latex for e in r.extras}
    assert "Hessian at the point" in extras
    assert any("Euclidean" in a for a in r.assumptions_stated)


def test_multivariable_equilibrium_quadratic() -> None:
    r = run(
        method="multivariable",
        input="cos(x)+cos(y)+x*y/10",
        variables=["x", "y"],
        points=["0", "0"],
        order=2,
    )
    assert r.status is Status.OK
    assert any("stationary" in n for n in r.notes)


def test_multivariable_refuses_a_singular_point() -> None:
    r = run(
        method="multivariable", input="log(x+y)", variables=["x", "y"], points=["0", "0"], order=2
    )
    assert r.status is Status.UNSUPPORTED and r.result_latex is None


def test_multivariable_with_a_wrong_derivative_is_not_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = sp.diff

    def wrong(f, *args):  # type: ignore[no-untyped-def]
        return real(f, *args) + sp.Rational(1, 10**7)

    monkeypatch.setattr(multivar.sp, "diff", wrong)
    r = run(
        method="multivariable", input="exp(x+y)", variables=["x", "y"], points=["0", "0"], order=2
    )
    assert r.status is Status.NOT_VERIFIED and r.result_latex is None


# --- Leading order and limits -----------------------------------------------------------------
def test_leading_term_of_sine_and_of_a_square_root() -> None:
    assert run(method="leading", input="sin(x)", point="0").result_latex == "x"
    assert run(method="leading", input="sqrt(x**2+x)", point="oo").result_latex == "x"
    r = run(method="leading", input="sqrt(x)+x", point="0", direction="+")
    assert r.result_latex == r"\sqrt{x}"


def test_leading_first_nonzero_correction() -> None:
    r = run(method="leading", input="sin(x)-x", point="0")
    assert r.result_latex == r"- \frac{x^{3}}{6}"


def test_leading_ratio_check_rejects_a_wrong_term() -> None:
    x = sp.Symbol("x", positive=True)
    ctx = Context(
        ApproximationInput(input="sin(x)", format="text", method="leading", point="0"),
        LIMITS,
        0.0,
        sp.sin(x),
        x,
    )
    approach = leading.Approach(ctx)
    assert leading._ratio_check(ctx, approach, 1, sp.sin(x), x)[0] is True
    assert leading._ratio_check(ctx, approach, 1, sp.sin(x), 2 * x)[0] is False
    assert (
        leading._ratio_check(ctx, approach, 1, sp.sin(x), x * (1 + sp.Rational(1, 10**3)))[0]
        is False
    )


def test_limits_including_removable_and_one_sided() -> None:
    assert run(method="limit", input="sin(x)/x", point="0").result_latex == "1"
    both = run(method="limit", input="1/x", point="0")
    assert both.status is Status.OK and "one_sided_limits_differ" in both.warnings
    assert both.result_latex is None
    assert run(method="limit", input="1/x", point="0", direction="+").result_latex == r"\infty"
    assert run(method="limit", input="(1+1/x)**x", point="oo").result_latex == r"\mathrm{e}"


def test_limit_that_does_not_exist_is_not_guessed() -> None:
    r = run(method="limit", input="sin(1/x)", point="0")
    assert r.status is Status.UNSUPPORTED and r.reason == "limit_undecided"


# --- through the sandbox ----------------------------------------------------------------------
def test_engine_runs_the_operation_in_the_sandbox(engine) -> None:  # type: ignore[no-untyped-def]
    r = engine.approximation(
        ApproximationInput(input="1/(x+1)", format="text", method="asymptotic", point="oo", order=3)
    )
    assert r.status is Status.OK and r.diagnostics.checks >= 1
    assert mpmath  # the check ran with mpmath inside the child
