"""Approximation, phase 5: the deterministic strategy and the physics regression problems.

The strategy reads the shape of the request, never an LLM. The physics cases are integration tests
of general methods against results worked out independently.
"""

from __future__ import annotations

import mpmath
import sympy as sp
from minerva_engine import ApproximationInput
from minerva_engine.approx.operation import approximation
from minerva_engine.models import Limits, Status

LIMITS = Limits(timeout_seconds=120)


def run(**kw):  # type: ignore[no-untyped-def]
    kw.setdefault("format", "text")
    return approximation(ApproximationInput(**kw), LIMITS)


def auto(**kw):  # type: ignore[no-untyped-def]
    return run(method="auto", **kw)


# --- the strategy -----------------------------------------------------------------------------
def test_several_variables_without_their_points_are_a_question() -> None:
    r = auto(input="cos(a - b)", variables=["a", "b"], order=2)
    assert r.status is Status.CONDITIONS_NOT_SATISFIED and r.reason == "domain_unspecified"
    assert r.questions and r.result_latex is None


def test_each_shape_of_request_gets_its_method() -> None:
    assert auto(input="sin(x)", point="0", order=5).method == "taylor"
    assert auto(input="1/(x+1)", point="oo", order=3).method == "asymptotic"
    assert auto(input="1/sqrt(1-eps**2)", small="eps", order=4).method == "perturbation"
    assert auto(input="x", lower="-pi", upper="pi", period="2*pi", modes=3).method == "fourier"
    assert auto(input="pi", order=5).method == "continued_fraction"
    assert (
        auto(input="exp(x+y)", variables=["x", "y"], points=["0", "0"], order=2).method
        == "multivariable"
    )
    assert auto(input="log(1+x)", lower="0", upper="2", degree=5).method == "compare"
    assert (
        auto(
            input="exp(x)", point="0", lower="-1", upper="1", degree=1, denominator_degree=1
        ).method
        == "pade"
    )


def test_a_pole_makes_the_strategy_move_from_taylor_to_laurent() -> None:
    r = auto(input="1/(x*(1-x))", point="0", order=3)
    assert r.method == "laurent" and r.status is Status.OK
    tried = {s.method: s.tried for s in r.suggestions}
    assert tried["taylor"] and tried["laurent"]  # Taylor was tried and refused, then Laurent


def test_every_rule_is_reported_with_its_reason() -> None:
    r = auto(input="sin(x)", point="0", order=5)
    assert {s.method for s in r.suggestions} >= {
        "taylor",
        "laurent",
        "asymptotic",
        "fourier",
        "compare",
    }
    assert all(s.why for s in r.suggestions)
    assert any(s.method == "taylor" and s.applies and s.tried for s in r.suggestions)
    assert any(s.method == "fourier" and not s.applies and not s.tried for s in r.suggestions)


def test_a_request_without_a_regime_gets_a_question_not_a_guess() -> None:
    r = auto(input="sin(x)")
    assert r.status is Status.CONDITIONS_NOT_SATISFIED and r.questions and r.result_latex is None


def test_the_strategy_does_not_run_more_than_the_bound() -> None:
    from minerva_engine.approx.strategy import MAX_TRIED

    r = auto(input="1/x", point="0", order=3)  # Taylor refuses, Laurent answers
    assert sum(s.tried for s in r.suggestions) <= MAX_TRIED


def test_the_strategy_never_reports_what_a_method_refused() -> None:
    r = auto(input="abs(x)", point="0", order=3)
    assert r.status is not Status.OK and r.result_latex is None


# --- physics regression ------------------------------------------------------------------------
def test_nonrelativistic_expansion_identifies_the_small_parameter_v_over_c() -> None:
    r = run(
        method="perturbation",
        input="m*c**2/sqrt(1-v**2/c**2)",
        small="v/c",
        order=4,
        assumptions={"m": ["positive"], "c": ["positive"], "v": ["real"]},
    )
    m, c, v = sp.symbols("m c v", positive=True)
    assert (
        sp.simplify(
            sp.sympify(r.result_text, locals={"m": m, "c": c, "v": v})
            - (m * c**2 + m * v**2 / 2 + 3 * m * v**4 / (8 * c**2))
        )
        == 0
    )


def test_small_oscillations_about_the_equilibrium_of_a_lennard_jones_potential() -> None:
    r = run(method="taylor", input="4*(r**-12 - r**-6)", variable="r", point="2**(1/6)", order=3)
    assert r.status is Status.OK
    by = {o.order: o.coefficient_latex for o in r.orders}
    assert 1 not in by  # V'(r0) = 0: no linear term
    rr = sp.Symbol("rr", positive=True)
    V = 4 * (rr**-12 - rr**-6)
    r0 = sp.Integer(2) ** sp.Rational(1, 6)
    assert sp.simplify(sp.diff(V, rr).subs(rr, r0)) == 0
    assert sp.simplify(V.subs(rr, r0)) == -1  # the well depth
    expected = 36 * sp.Integer(2) ** sp.Rational(-1, 3)  # V''(r0)/2, using r0^6 = 2
    assert abs(sp.N(sp.diff(V, rr, 2).subs(rr, r0) / 2 - expected, 30)) < 1e-25
    shown = sp.sympify(r.result_text, locals={"r": rr})
    assert (
        abs(sp.N(sp.diff(shown, rr, 2).subs(rr, r0) / 2 - expected, 25)) < 1e-15
    )  # what the engine returned


def test_multivariable_small_oscillations_give_the_hessian_form() -> None:
    r = run(
        method="multivariable",
        input="x**2*y + cos(x) + cos(y) - 2",
        variables=["x", "y"],
        points=["0", "0"],
        order=2,
    )
    assert r.status is Status.OK
    x, y = sp.symbols("x y")
    assert sp.expand(sp.sympify(r.result_text) - (-(x**2) / 2 - y**2 / 2)) == 0


def test_far_field_expansion_of_a_potential_in_the_displacement() -> None:
    r = run(
        method="perturbation",
        input="1/(r+h)",
        small="h",
        order=3,
        assumptions={"r": ["positive"], "h": ["real"]},
    )
    rr, h = sp.symbols("r h", positive=True)
    assert (
        sp.simplify(
            sp.sympify(r.result_text, locals={"r": rr, "h": h})
            - (1 / rr - h / rr**2 + h**2 / rr**3 - h**3 / rr**4)
        )
        == 0
    )
    far = run(method="asymptotic", input="1/(r+1)", variable="r", point="oo", order=3)
    assert far.status is Status.OK


def test_boltzmann_factor_in_beta_times_energy() -> None:
    r = run(
        method="perturbation",
        input="exp(-beta*E1)",
        small="beta*E1",
        order=3,
        assumptions={"beta": ["positive"], "E1": ["positive"]},
    )
    assert r.status is Status.OK
    b, e = sp.symbols("beta E1", positive=True)
    assert (
        sp.simplify(
            sp.sympify(r.result_text, locals={"beta": b, "E1": e})
            - (1 - b * e + b**2 * e**2 / 2 - b**3 * e**3 / 6)
        )
        == 0
    )


def test_two_level_partition_function_logarithm() -> None:
    r = run(
        method="perturbation",
        input="log(1 + exp(-beta*E1))",
        small="beta",
        order=2,
        assumptions={"E1": ["positive"], "beta": ["real"]},
    )
    b, e = sp.symbols("beta E1", positive=True)
    want = sp.log(2) - b * e / 2 + b**2 * e**2 / 8
    assert (
        r.status is Status.OK
        and sp.simplify(sp.sympify(r.result_text, locals={"beta": b, "E1": e}) - want) == 0
    )


def test_periodic_potential_reconstructs_and_a_step_shows_gibbs() -> None:
    smooth = run(method="fourier", input="exp(cos(x))", period="2*pi", modes=12)
    assert float(smooth.error.max_abs) < 1e-8 and "gibbs" not in smooth.warnings  # type: ignore[union-attr]
    step = run(method="fourier", input="abs(x)/x", lower="-1", upper="1.7", modes=15)
    assert "gibbs" in step.warnings and float(step.error.max_abs) > 0.9  # type: ignore[union-attr]


def test_angular_dependence_is_expanded_in_legendre_polynomials() -> None:
    r = run(
        method="legendre", input="x**2", lower="-1", upper="1", degree=2
    )  # cos^2(theta) with x = cos(theta)
    by = {o.order: mpmath.mpf(o.coefficient_latex) for o in r.orders}
    assert abs(by[0] - mpmath.mpf(1) / 3) < 1e-11 and abs(by[2] - mpmath.mpf(2) / 3) < 1e-11
    mono = run(
        method="legendre", input="1/sqrt(1 - 2*x*0.3 + 0.09)", lower="-1", upper="1", degree=5
    )
    assert all(
        abs(mpmath.mpf(o.coefficient_latex) - mpmath.mpf("0.3") ** o.order) < 1e-10
        for o in mono.orders
    )


def test_residue_and_leading_singular_term_of_a_propagator() -> None:
    r = run(method="laurent", input="1/(x*(x-3))", point="3", order=2)
    extras = {e.label: e.latex for e in r.extras}
    assert extras["Residue"] == r"\frac{1}{3}" and extras["Pole order"] == "1"
    lead = run(method="leading", input="1/(x*(x-3))", point="3", direction="+")
    assert lead.status is Status.OK and "frac{1}{3" in lead.result_latex  # type: ignore[operator]


def test_a_large_sum_is_approximated_with_its_error_reported() -> None:
    r = run(
        method="euler_maclaurin", input="1/k**2", variable="k", lower="10", upper="100000", order=2
    )
    assert r.status is Status.OK and r.candidates[0].error and r.candidates[-1].error
    first, last = float(r.candidates[0].error.max_abs), float(r.candidates[-1].error.max_abs)  # type: ignore[union-attr]
    assert last < first


def test_ignoring_everything_smaller_than_second_order_returns_the_truncated_expression() -> None:
    r = run(method="perturbation", input="exp(eps)*cos(eps)", small="eps", order=2)
    eps = sp.Symbol("eps")
    assert (
        sp.expand(sp.sympify(r.result_text, locals={"eps": eps}) - (1 + eps)) == 0
    )  # e^eps cos eps = 1 + eps + 0 eps^2 + ...


def test_limit_of_a_removable_singularity_and_one_sided_behaviour() -> None:
    assert run(method="limit", input="(exp(x)-1)/x", point="0").result_latex == "1"
    r = run(method="limit", input="sqrt(x**2+x) - x", point="oo")
    assert r.result_latex == r"\frac{1}{2}"
    left = run(method="limit", input="1/x", point="0", direction="-")
    assert left.result_latex == r"-\infty"
