"""Approximation, phase 4: continued fractions, Euler-Maclaurin, sequence acceleration, composed
expansions (integral, root, inverse) and Laplace's method.

Expected values are classical: the partial quotients of pi and e, Faulhaber power sums, the Stirling
and Bessel-I0 expansions, arcsin and log(1 + y) as inverses. Each checker is shown tampered results.
"""

from __future__ import annotations

import re
from fractions import Fraction

import mpmath
import pytest
import sympy as sp
from minerva_engine import ApproximationInput
from minerva_engine.approx import cfrac, composed, laplace, sums
from minerva_engine.approx.operation import approximation
from minerva_engine.models import Limits, Status

LIMITS = Limits(timeout_seconds=120)


def run(**kw):  # type: ignore[no-untyped-def]
    kw.setdefault("format", "text")
    return approximation(ApproximationInput(**kw), LIMITS)


def quotients(report) -> list[int]:  # type: ignore[no-untyped-def]
    inside = report.statement_latex.split("[")[1].split("]")[0]
    parts = [
        p.strip() for p in inside.replace(";", ",").split(",") if p.strip().lstrip("-").isdigit()
    ]
    return [int(p) for p in parts]


def number(latex: str) -> sp.Expr:
    """A rational printed by the engine as LaTeX, e.g. `- \\frac{1}{64}`."""
    text = re.sub(r"\\frac\{(-?\d+)\}\{(\d+)\}", r"(\1)/(\2)", latex).replace(" ", "")
    return sp.sympify(text)


def orders(report) -> dict[int, sp.Expr]:  # type: ignore[no-untyped-def]
    return {o.order: number(o.coefficient_latex) for o in report.orders}


# --- continued fractions ----------------------------------------------------------------------
def test_continued_fraction_of_pi_and_e() -> None:
    r = run(method="continued_fraction", input="pi", order=13)
    assert quotients(r) == [3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14]
    got = [o.coefficient_latex for o in r.orders]
    assert got[:4] == ["3/1", "22/7", "333/106", "355/113"] and got[4] == "103993/33102"
    r = run(method="continued_fraction", input="E", order=12)
    assert quotients(r) == [2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8]


def test_continued_fraction_of_quadratic_irrationals_is_periodic() -> None:
    assert quotients(run(method="continued_fraction", input="sqrt(2)", order=10)) == [1] + [2] * 9
    r = run(method="continued_fraction", input="(1+sqrt(5))/2", order=10)
    assert quotients(r) == [1] * 10
    assert [Fraction(o.coefficient_latex) for o in r.orders][:6] == [
        Fraction(a, b) for a, b in [(1, 1), (2, 1), (3, 2), (5, 3), (8, 5), (13, 8)]
    ]


def test_continued_fraction_of_a_rational_is_exact() -> None:
    r = run(method="continued_fraction", input="22/7", order=10)
    assert quotients(r) == [3, 7] and r.evidence == "bound_established"
    assert r.error.max_abs == "0 (exact)"  # type: ignore[union-attr]
    assert quotients(run(method="continued_fraction", input="-355/113", order=10))[0] == -4


def test_continued_fraction_convergents_are_compared_with_decimals() -> None:
    r = run(method="continued_fraction", input="pi", order=5)
    assert "3.14159292035398" in r.orders[3].term_latex and "2.67e-7" in r.orders[3].term_latex


def test_continued_fraction_tolerance_picks_the_first_convergent_that_meets_it() -> None:
    r = run(method="continued_fraction", input="pi", tolerance="1e-6", tolerance_kind="absolute")
    assert r.result_latex == r"\frac{355}{113}" and r.tolerance_met is True
    miss = run(
        method="continued_fraction",
        input="pi",
        order=3,
        tolerance="1e-12",
        tolerance_kind="absolute",
    )
    assert miss.tolerance_met is False


def test_continued_fraction_refuses_non_numbers_and_asks_for_a_budget() -> None:
    assert run(method="continued_fraction", input="x+1", order=5).status is Status.UNSUPPORTED
    assert run(method="continued_fraction", input="pi").questions
    assert (
        run(
            method="continued_fraction",
            input="pi",
            order=5,
            tolerance="1e-3",
            tolerance_kind="relative",
        ).status
        is Status.UNSUPPORTED
    )


def test_continued_fraction_with_quotients_that_disagree_between_precisions_is_not_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = cfrac._quotients

    def wrong(value, terms, dps):  # type: ignore[no-untyped-def]
        out = real(value, terms, dps)
        if dps > 136 and len(out) > 4:
            out[2] += 1
        return out

    monkeypatch.setattr(cfrac, "_quotients", wrong)
    r = run(method="continued_fraction", input="pi", order=8)
    assert r.status is Status.NOT_VERIFIED and r.result_latex is None


# --- Euler-Maclaurin --------------------------------------------------------------------------
def direct(f, a: int, b: int) -> mpmath.mpf:  # type: ignore[no-untyped-def]
    with mpmath.workdps(40):
        return sum((f(mpmath.mpf(k)) for k in range(a, b + 1)), mpmath.mpf(0))


def test_euler_maclaurin_for_inverse_squares_matches_direct_summation() -> None:
    r = run(
        method="euler_maclaurin", input="1/k**2", variable="k", lower="10", upper="1000", order=3
    )
    assert r.status is Status.OK
    exact = direct(lambda k: 1 / k**2, 10, 1000)
    assert abs(mpmath.mpf(r.result_latex) - exact) < 1e-9
    errors = [float(c.error.max_abs) for c in r.candidates]  # type: ignore[union-attr]
    assert errors[0] > errors[1] > errors[2] > errors[3]  # each Bernoulli correction helps here
    assert r.candidates[-1].chosen and any("neglected" in e.label for e in r.extras)
    assert all(c.passed for c in r.checks)


def test_euler_maclaurin_symbolic_limit_for_power_sums_is_exact() -> None:
    r = run(method="euler_maclaurin", input="k**3", variable="k", lower="1", upper="N", order=2)
    N = sp.Symbol("N")
    assert sp.simplify(sp.sympify(r.result_text, locals={"N": N}) - N**2 * (N + 1) ** 2 / 4) == 0
    assert any(
        "N = " in n for n in r.notes
    )  # worded as checked for instances, not proved for every N


def test_euler_maclaurin_harmonic_sum_gets_a_closed_form_in_n() -> None:
    r = run(
        method="euler_maclaurin",
        input="1/k",
        variable="k",
        lower="10",
        upper="N",
        order=2,
        assumptions={"N": ["positive"], "k": ["positive"]},
    )
    assert r.status is Status.OK and r.result_latex.startswith(r"\log{\left(N \right)}")  # type: ignore[union-attr]
    N = sp.Symbol("N", positive=True)
    value = sp.sympify(r.result_text, locals={"N": N}).subs(N, 5000)
    assert abs(mpmath.mpf(str(sp.N(value, 30))) - direct(lambda k: 1 / k, 10, 5000)) < 1e-7


def test_euler_maclaurin_huge_upper_limit_is_checked_on_a_feasible_instance() -> None:
    r = run(
        method="euler_maclaurin",
        input="sqrt(k)",
        variable="k",
        lower="1",
        upper="100000000",
        order=3,
    )
    assert r.status is Status.OK and any("direct summation allows" in n for n in r.notes)


def test_euler_maclaurin_refuses_when_the_summand_is_not_smooth_or_oscillates() -> None:
    assert (
        run(
            method="euler_maclaurin", input="abs(k-5)", variable="k", lower="1", upper="20", order=2
        ).status
        is Status.UNSUPPORTED
    )
    r = run(
        method="euler_maclaurin", input="cos(1000*k)", variable="k", lower="1", upper="200", order=2
    )
    assert r.status is Status.NOT_VERIFIED and r.result_latex is None
    assert (
        run(
            method="euler_maclaurin", input="1/(k-5)", variable="k", lower="1", upper="20", order=2
        ).reason
        == "domain_invalid"
    )


def test_euler_maclaurin_questions_and_conventions() -> None:
    assert run(method="euler_maclaurin", input="1/k", variable="k", order=2).questions
    assert (
        run(
            method="euler_maclaurin", input="a/k", variable="k", lower="1", upper="9", order=2
        ).reason
        == "parameter_missing"
    )
    assert (
        run(
            method="euler_maclaurin", input="1/k", variable="k", lower="1.5", upper="9", order=2
        ).reason
        == "point_invalid"
    )


def test_euler_maclaurin_with_a_dropped_bernoulli_term_is_not_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = sums.correction

    def wrong(f, k, a, b, j):  # type: ignore[no-untyped-def]
        return sp.Integer(0) if j == 2 else real(f, k, a, b, j)

    monkeypatch.setattr(sums, "correction", wrong)
    r = run(method="euler_maclaurin", input="k**4", variable="k", lower="1", upper="30", order=2)
    assert r.status is Status.NOT_VERIFIED and r.result_latex is None


def test_euler_maclaurin_with_a_wrong_bernoulli_coefficient_fails_the_polynomial_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sums, "_bernoulli_coefficient", lambda j: sp.bernoulli(2 * j) / sp.factorial(2 * j + 1)
    )
    r = run(
        method="euler_maclaurin", input="1/k**2", variable="k", lower="10", upper="500", order=2
    )
    assert r.status is Status.NOT_VERIFIED


# --- sequence acceleration --------------------------------------------------------------------
def test_acceleration_of_the_basel_series_and_an_alternating_series() -> None:
    r = run(method="accelerate", input="1/k**2", variable="k", lower="1", max_terms=40)
    assert r.status is Status.OK and abs(mpmath.mpf(r.result_latex) - mpmath.pi**2 / 6) < 1e-12
    plain = [c for c in r.candidates if "partial sum" in c.label][0]
    assert (
        abs(mpmath.mpf(plain.result_latex) - mpmath.pi**2 / 6) > 1e-2
    )  # the acceleration is what helped
    r = run(method="accelerate", input="(-1)**k/k", variable="k", lower="1", max_terms=40)
    assert abs(mpmath.mpf(r.result_latex) + mpmath.log(2)) < 1e-12


def test_acceleration_is_an_estimate_and_is_refused_when_the_terms_do_not_shrink() -> None:
    r = run(method="accelerate", input="1/k**2", variable="k", lower="1", max_terms=30)
    assert "not an identity" in r.message and r.evidence == "numeric_estimate"
    assert (
        run(method="accelerate", input="1 + 1/k", variable="k", lower="1", max_terms=30).status
        is Status.UNSUPPORTED
    )
    assert run(method="accelerate", input="1/k**2", variable="k", lower="1").questions


# --- integral expansion -----------------------------------------------------------------------
POSITIVE = {"eps": ["positive"], "x": ["positive"]}


def test_integral_expansion_of_an_exponential() -> None:
    r = run(
        method="integral_expansion",
        input="exp(-eps*x)",
        variable="x",
        small="eps",
        lower="0",
        upper="1",
        order=4,
        assumptions=POSITIVE,
    )
    assert r.status is Status.OK
    eps = sp.Symbol("eps", positive=True)
    want = sum((-1) ** k * eps**k / sp.factorial(k + 1) for k in range(5))
    assert sp.simplify(sp.sympify(r.result_text, locals={"eps": eps}) - want) == 0
    assert any(c.name.startswith("against quadrature") and c.passed for c in r.checks)


def test_integral_expansion_with_a_square_root_integrand() -> None:
    r = run(
        method="integral_expansion",
        input="sqrt(1+eps*x**2)",
        variable="x",
        small="eps",
        lower="0",
        upper="1",
        order=3,
        assumptions=POSITIVE,
    )
    eps = sp.Symbol("eps", positive=True)
    assert (
        sp.simplify(
            sp.sympify(r.result_text, locals={"eps": eps})
            - (1 + eps / 6 - eps**2 / 40 + eps**3 / 112)
        )
        == 0
    )


def test_integral_expansion_refuses_a_non_uniform_expansion() -> None:
    r = run(
        method="integral_expansion",
        input="1/(x+eps)",
        variable="x",
        small="eps",
        lower="0",
        upper="1",
        order=2,
        assumptions=POSITIVE,
    )
    assert r.status is Status.UNSUPPORTED and r.result_latex is None


def test_integral_expansion_with_a_wrong_term_is_not_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    real = composed.integrate
    calls = {"n": 0}

    def wrong(*args, **kwargs):  # type: ignore[no-untyped-def]
        out = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 3 and out.result_text:
            out.result_text = f"({out.result_text}) + 1/10000000"
        return out

    monkeypatch.setattr(composed, "integrate", wrong)
    r = run(
        method="integral_expansion",
        input="exp(-eps*x)",
        variable="x",
        small="eps",
        lower="0",
        upper="1",
        order=4,
        assumptions=POSITIVE,
    )
    assert r.status is Status.NOT_VERIFIED and r.result_latex is None


# --- roots and inverses -----------------------------------------------------------------------
def test_root_expansion_matches_the_closed_form() -> None:
    r = run(
        method="root_expansion",
        input="x**2 - 1 - eps*x",
        variable="x",
        small="eps",
        point="1",
        order=5,
    )
    eps = sp.Symbol("eps")
    exact = (eps + sp.sqrt(eps**2 + 4)) / 2
    got = sp.sympify(r.result_text, locals={"eps": eps})
    assert sp.simplify(sp.series(exact, eps, 0, 6).removeO() - got) == 0
    assert all(c.passed for c in r.checks)


def test_root_expansion_refuses_a_double_root_and_a_non_root() -> None:
    assert (
        run(
            method="root_expansion",
            input="x**2 - eps",
            variable="x",
            small="eps",
            point="0",
            order=2,
        ).reason
        == "degenerate_system"
    )
    assert (
        run(
            method="root_expansion",
            input="x**2 - 1 - eps*x",
            variable="x",
            small="eps",
            point="2",
            order=2,
        ).reason
        == "point_invalid"
    )
    assert run(
        method="root_expansion", input="x**2 - 1 - eps*x", variable="x", small="eps", order=2
    ).questions


def test_root_expansion_with_a_wrong_coefficient_is_not_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = composed.solve_implicit

    def wrong(F, x, eps, x0, order):  # type: ignore[no-untyped-def]
        out = real(F, x, eps, x0, order)
        out[2] = out[2] + sp.Rational(1, 10**9)
        return out

    monkeypatch.setattr(composed, "solve_implicit", wrong)
    r = run(
        method="root_expansion",
        input="x**2 - 1 - eps*x",
        variable="x",
        small="eps",
        point="1",
        order=4,
    )
    assert r.status is Status.NOT_VERIFIED and r.result_latex is None


def test_reversion_gives_the_classical_inverses() -> None:
    y = sp.Symbol("y")

    def series_of(report) -> sp.Expr:  # type: ignore[no-untyped-def]
        return sp.sympify(report.result_text.replace("eps_y", "y"), locals={"y": y})

    assert (
        sp.expand(
            series_of(
                run(
                    method="reversion",
                    input="x + x**3",
                    variable="x",
                    point="0",
                    small="y",
                    order=5,
                )
            )
            - (y - y**3 + 3 * y**5)
        )
        == 0
    )
    assert (
        sp.expand(
            series_of(
                run(method="reversion", input="sin(x)", variable="x", point="0", small="y", order=5)
            )
            - sp.series(sp.asin(y), y, 0, 6).removeO()
        )
        == 0
    )
    assert (
        sp.expand(
            series_of(
                run(
                    method="reversion",
                    input="exp(x) - 1",
                    variable="x",
                    point="0",
                    small="y",
                    order=4,
                )
            )
            - sp.series(sp.log(1 + y), y, 0, 5).removeO()
        )
        == 0
    )


def test_reversion_about_a_nonzero_point_and_where_it_does_not_exist() -> None:
    r = run(method="reversion", input="x**2", variable="x", point="2", small="y", order=2)
    assert orders(r)[1] == sp.Rational(1, 4) and orders(r)[2] == -sp.Rational(1, 64)
    assert any(e.label.startswith("Value of y") and e.latex == "4" for e in r.extras)
    assert (
        run(method="reversion", input="x**2", variable="x", point="0", small="y", order=3).reason
        == "degenerate_system"
    )
    assert run(method="reversion", input="x**2", variable="x", point="2", order=2).questions


# --- Laplace's method -------------------------------------------------------------------------
def test_laplace_reproduces_the_bessel_i0_expansion() -> None:
    r = run(
        method="laplace",
        input="exp(M*cos(x))",
        variable="x",
        small="M",
        lower="-3",
        upper="3",
        order=3,
    )
    assert r.status is Status.OK
    a = [number(o.coefficient_latex) for o in r.orders]
    assert a[:4] == [
        1,
        sp.Rational(1, 8),
        sp.Rational(9, 128),
        sp.Rational(75, 1024),
    ]  # I0(M) ~ e^M / sqrt(2 pi M) (1 + 1/8M + ...)
    assert "asymptotic_not_convergent" in r.warnings


def test_laplace_reproduces_stirlings_series() -> None:
    r = run(
        method="laplace",
        input="exp(M*(log(x)-x))",
        variable="x",
        small="M",
        lower="1/10",
        upper="3",
        order=2,
    )
    assert r.status is Status.OK
    a = [number(o.coefficient_latex) for o in r.orders]
    assert a == [1, sp.Rational(1, 12), sp.Rational(1, 288)]


def test_laplace_with_an_amplitude() -> None:
    r = run(
        method="laplace",
        input="(1+x**2)*exp(-M*x**2)",
        variable="x",
        small="M",
        lower="-2",
        upper="2",
        order=1,
    )
    a = [number(o.coefficient_latex) for o in r.orders]
    assert a == [1, sp.Rational(1, 2)]  # integral of x^2 e^{-M x^2} is sqrt(pi/M)/(2M)


def test_laplace_refuses_what_it_does_not_cover() -> None:
    assert (
        run(
            method="laplace",
            input="exp(-M*x)",
            variable="x",
            small="M",
            lower="0",
            upper="1",
            order=1,
        ).reason
        == "unsupported_input"
    )
    assert (
        run(
            method="laplace",
            input="exp(M*cos(2*x))",
            variable="x",
            small="M",
            lower="-4",
            upper="4",
            order=1,
        ).status
        is Status.UNSUPPORTED
    )
    assert (
        run(
            method="laplace",
            input="x*exp(-M*x**2)",
            variable="x",
            small="M",
            lower="-1",
            upper="1",
            order=1,
        ).status
        is Status.UNSUPPORTED
    )
    assert run(
        method="laplace", input="exp(M*cos(x))", variable="x", small="M", lower="-3", upper="3"
    ).questions
    assert (
        run(
            method="laplace",
            input="a*exp(M*cos(x))",
            variable="x",
            small="M",
            lower="-3",
            upper="3",
            order=1,
        ).reason
        == "parameter_missing"
    )


def test_laplace_with_a_wrong_correction_is_not_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    real = laplace.corrections

    def wrong(g, f, x, star, order):  # type: ignore[no-untyped-def]
        a, A = real(g, f, x, star, order)
        a[1] = a[1] * (1 + sp.Rational(1, 10**3))
        return a, A

    monkeypatch.setattr(laplace, "corrections", wrong)
    r = run(
        method="laplace",
        input="exp(M*cos(x))",
        variable="x",
        small="M",
        lower="-3",
        upper="3",
        order=1,
    )
    assert r.status is Status.NOT_VERIFIED and r.result_latex is None
