"""Approximation, phase 2: Padé, Chebyshev, comparison and tolerance-driven search.

Expected values are worked out independently (known Padé tables, Bessel-function Chebyshev
coefficients, the interpolation error bound); every checker is also given wrong candidates.
"""

from __future__ import annotations

import mpmath
import pytest
import sympy as sp
from minerva_engine import ApproximationInput
from minerva_engine.approx import chebyshev as cheb
from minerva_engine.approx import interval, pade
from minerva_engine.approx.operation import approximation
from minerva_engine.models import Limits, Status

LIMITS = Limits(timeout_seconds=120)
x = sp.Symbol("x")


def run(**kw):  # type: ignore[no-untyped-def]
    kw.setdefault("format", "text")
    return approximation(ApproximationInput(**kw), LIMITS)


def rational(report) -> sp.Expr:  # type: ignore[no-untyped-def]
    return sp.sympify(report.result_text, locals={"x": x})


# --- Padé -------------------------------------------------------------------------------------
def test_pade_one_one_of_the_exponential_and_its_spurious_pole() -> None:
    r = run(method="pade", input="exp(x)", point="0", degree=1, denominator_degree=1)
    assert r.status is Status.OK
    assert r.result_latex == r"\frac{1 + \frac{x}{2}}{1 - \frac{x}{2}}"
    assert r.remainder_latex == r"O\left(\left(x\right)^{3}\right)" or "^{3}" in (
        r.remainder_latex or ""
    )
    assert "spurious_pole" in r.warnings
    assert any("x = 2" in e.latex for e in r.extras)
    assert any("although the original function does not" in n for n in r.notes)


def test_pade_two_two_matches_the_known_table_entry() -> None:
    r = run(method="pade", input="exp(x)", point="0", degree=2, denominator_degree=2)
    want = (1 + x / 2 + x**2 / 12) / (1 - x / 2 + x**2 / 12)
    assert sp.simplify(rational(r) - want) == 0
    r = run(method="pade", input="log(1+x)", point="0", degree=2, denominator_degree=2)
    assert sp.simplify(rational(r) - (6 * x + 3 * x**2) / (6 + 6 * x + x**2)) == 0


def test_pade_about_another_point() -> None:
    r = run(method="pade", input="1/x", point="1", degree=0, denominator_degree=1)
    assert r.status is Status.OK
    assert sp.simplify(rational(r) - 1 / x) == 0  # 1/x is its own [0/1] approximant about 1


def test_pade_with_symbolic_parameters_is_checked_at_sampled_values() -> None:
    r = run(
        method="pade",
        input="exp(a*x)",
        point="0",
        degree=1,
        denominator_degree=1,
        assumptions={"a": ["positive"]},
    )
    assert r.status is Status.OK
    assert any(
        c.name.startswith("coefficients against mpmath") and c.passed is None for c in r.checks
    )


def test_pade_degenerate_system_is_refused() -> None:
    r = run(method="pade", input="cos(x)", point="0", degree=1, denominator_degree=1)
    assert r.reason == "degenerate_system" and r.result_latex is None


def test_pade_needs_a_taylor_series() -> None:
    r = run(method="pade", input="sqrt(x)", point="0", degree=1, denominator_degree=1)
    assert r.status is Status.UNSUPPORTED and r.result_latex is None
    r = run(method="pade", input="exp(x)", point="0", degree=1)
    assert r.reason == "domain_unspecified" and r.questions


def test_pade_pole_in_the_interval_and_not_better_than_taylor() -> None:
    r = run(
        method="pade",
        input="exp(x)",
        point="0",
        degree=1,
        denominator_degree=1,
        lower="-1",
        upper="3",
    )
    assert r.status is Status.OK
    assert "pole_in_interval" in r.warnings and r.warnings.count("pole_in_interval") == 1
    assert "unbounded" in (r.error.max_abs or "")
    assert any("not more accurate" in n for n in r.notes)


def test_pade_beats_taylor_on_an_interval_where_it_should() -> None:
    r = run(
        method="pade",
        input="log(1+x)",
        point="0",
        degree=2,
        denominator_degree=2,
        lower="0",
        upper="2",
    )
    errs = {c.label: float(c.error.max_abs) for c in r.candidates}  # type: ignore[union-attr]
    assert errs["Padé"] < errs["Taylor (same order)"]
    assert any("more accurate" in n for n in r.notes)


def test_pade_with_a_perturbed_denominator_is_not_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    real = pade.solve_pade

    def wrong(c, m, n):  # type: ignore[no-untyped-def]
        p, q = real(c, m, n)
        return p, [q[0], q[1] + sp.Rational(1, 10**9), *q[2:]]

    monkeypatch.setattr(pade, "solve_pade", wrong)
    r = run(method="pade", input="exp(x)", point="0", degree=1, denominator_degree=1)
    assert r.status is Status.NOT_VERIFIED and r.result_latex is None


# --- Chebyshev --------------------------------------------------------------------------------
def test_chebyshev_exponential_coefficients_and_error() -> None:
    r = run(method="chebyshev", input="exp(x)", lower="-1", upper="1", degree=6)
    assert r.status is Status.OK
    by = {o.order: mpmath.mpf(o.coefficient_latex) for o in r.orders}
    for k in range(5):  # exact Chebyshev coefficients of e^t are 2 I_k(1) (halved for k = 0)
        exact = mpmath.besseli(k, 1) * (1 if k == 0 else 2)
        assert abs(by[k] - exact) < 1e-9
    worst = float(r.error.max_abs)  # type: ignore[union-attr]
    assert 1e-7 < worst < 8.4e-6  # below the classical interpolation bound e / (7! 2^6)
    assert all(c.passed for c in r.checks)
    assert r.evidence == "numeric_estimate"
    assert "not a bound" in r.message


def test_chebyshev_on_another_interval_is_in_the_chebyshev_basis() -> None:
    r = run(method="chebyshev", input="sin(x)", lower="0", upper="3", degree=10)
    assert r.status is Status.OK
    assert r.statement_latex.startswith("t = ")  # type: ignore[union-attr]
    assert "T_{" in r.result_latex  # type: ignore[operator]
    assert float(r.error.max_abs) < 1e-8  # type: ignore[union-attr]


def test_chebyshev_tolerance_finds_the_lowest_degree_tried() -> None:
    r = run(
        method="chebyshev",
        input="sin(x)",
        lower="0",
        upper="3",
        tolerance="1e-6",
        tolerance_kind="absolute",
    )
    assert r.status is Status.OK and r.tolerance_met is True
    chosen = [c for c in r.candidates if c.chosen]
    assert len(chosen) == 1 and chosen[0].label == "degree 8"
    assert all(c.tolerance_met is False for c in r.candidates if not c.chosen)
    assert "sampled" in r.message


def test_chebyshev_unreachable_tolerance_is_reported_not_hidden() -> None:
    r = run(
        method="chebyshev",
        input="exp(x)",
        lower="0",
        upper="1",
        tolerance="1e-30",
        tolerance_kind="absolute",
        max_terms=6,
    )
    assert r.status is Status.OK and r.tolerance_met is False
    assert "NOT met" in r.message


def test_chebyshev_nonsmooth_function_is_flagged() -> None:
    r = run(method="chebyshev", input="abs(x)", lower="-1", upper="1", degree=8)
    assert r.status is Status.OK and "slow_coefficient_decay" in r.warnings
    assert float(r.error.max_abs) > 0.01  # type: ignore[union-attr]


def test_chebyshev_rapid_oscillation_does_not_meet_a_modest_tolerance() -> None:
    r = run(
        method="chebyshev",
        input="sin(50*x)",
        lower="0",
        upper="1",
        degree=8,
        tolerance="1e-2",
        tolerance_kind="absolute",
    )
    assert r.tolerance_met is False and "slow_coefficient_decay" in r.warnings


def test_chebyshev_zero_crossing_does_not_blow_up_the_relative_error() -> None:
    r = run(method="chebyshev", input="sin(x)", lower="-1", upper="1", degree=5)
    assert "zero" in r.error.relative_note  # type: ignore[union-attr]
    assert float(r.error.max_rel) < 1  # type: ignore[union-attr]


def test_chebyshev_refuses_poles_undefined_points_and_complex_values() -> None:
    assert (
        run(method="chebyshev", input="1/(x-1)", lower="0", upper="2", degree=8).reason
        == "domain_invalid"
    )
    assert (
        run(method="chebyshev", input="1/(x-1.0001)", lower="0", upper="2", degree=8).reason
        == "domain_invalid"
    )
    assert (
        run(method="chebyshev", input="log(x)", lower="0", upper="1", degree=8).reason
        == "domain_invalid"
    )
    assert (
        run(method="chebyshev", input="sqrt(x)", lower="-1", upper="1", degree=8).reason
        == "domain_invalid"
    )


def test_chebyshev_discontinuity_is_a_warning_with_a_large_measured_error() -> None:
    r = run(method="chebyshev", input="abs(x)/x", lower="-1", upper="2", degree=10)
    assert r.status is Status.OK
    assert "jump_discontinuity" in r.warnings
    assert float(r.error.max_abs) > 0.1  # type: ignore[union-attr]


def test_chebyshev_huge_values_do_not_overflow() -> None:
    r = run(method="chebyshev", input="exp(x)", lower="0", upper="800", degree=5)
    assert r.status is Status.OK
    assert any(c.passed is None for c in r.checks)  # numpy cannot take part in double precision


def test_chebyshev_questions_instead_of_defaults() -> None:
    assert run(method="chebyshev", input="exp(x)", degree=4).reason == "domain_unspecified"
    assert run(method="chebyshev", input="exp(x)", lower="0", upper="1").questions
    assert run(
        method="chebyshev", input="exp(x)", lower="0", upper="1", degree=3, tolerance="1e-3"
    ).questions
    assert (
        run(method="chebyshev", input="exp(a*x)", lower="0", upper="1", degree=3).reason
        == "parameter_missing"
    )
    assert (
        run(method="chebyshev", input="exp(x)", lower="1", upper="0", degree=3).reason
        == "point_invalid"
    )


def test_chebyshev_with_a_tampered_coefficient_is_not_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = cheb.coefficients

    def wrong(ref, degree):  # type: ignore[no-untyped-def]
        out = real(ref, degree)
        out[3] = out[3] + mpmath.mpf("1e-6")
        return out

    monkeypatch.setattr(cheb, "coefficients", wrong)
    r = run(method="chebyshev", input="exp(x)", lower="-1", upper="1", degree=6)
    assert r.status is Status.NOT_VERIFIED and r.result_latex is None


def test_error_estimate_that_differs_between_grids_is_flagged() -> None:
    ctx_input = ApproximationInput(
        input="sin(x)", format="text", method="chebyshev", lower="0", upper="1"
    )
    from minerva_engine.approx.common import build_context

    ctx = build_context(ctx_input, LIMITS, 0.0)
    ref = interval.build_reference(ctx)
    spikes = {id(v): v for v in ref.xs2}
    values = {mpmath.nstr(v, 30) for v in spikes.values()}

    def approx(t):  # type: ignore[no-untyped-def]
        bump = mpmath.mpf("1e-3") if mpmath.nstr(t, 30) in values else 0
        return mpmath.sin(t) + bump

    assert interval.measure(ref, approx).unstable is True
    assert interval.measure(ref, lambda t: mpmath.sin(t)).unstable is False


# --- Comparison -------------------------------------------------------------------------------
def test_compare_lists_every_candidate_with_its_cost_and_picks_one() -> None:
    r = run(method="compare", input="log(1+x)", lower="0", upper="2", degree=5)
    assert r.status is Status.OK
    assert {c.method for c in r.candidates} == {"taylor", "pade", "chebyshev"}
    assert sum(c.chosen for c in r.candidates) == 1
    best = min(r.candidates, key=lambda c: float(c.error.max_abs))  # type: ignore[union-attr]
    assert best.chosen
    assert all(c.complexity.coefficients == 6 for c in r.candidates)
    assert "not bounds" in r.message
    assert any("midpoint" in n for n in r.notes)


def test_compare_prefers_the_simplest_candidate_that_meets_the_tolerance() -> None:
    r = run(
        method="compare",
        input="log(1+x)",
        lower="0",
        upper="2",
        tolerance="1e-3",
        tolerance_kind="absolute",
    )
    assert r.tolerance_met is True
    meeting = [c for c in r.candidates if c.tolerance_met]
    chosen = next(c for c in r.candidates if c.chosen)
    assert chosen.tolerance_met
    assert chosen.complexity.coefficients == min(c.complexity.coefficients for c in meeting)  # type: ignore[type-var]
    simpler = [
        c
        for c in r.candidates
        if (c.complexity.coefficients or 0) < (chosen.complexity.coefficients or 0)
    ]
    assert all(not c.tolerance_met for c in simpler)


def test_compare_with_an_unreachable_tolerance_says_so() -> None:
    r = run(
        method="compare",
        input="exp(x)",
        lower="0",
        upper="1",
        tolerance="1e-30",
        tolerance_kind="absolute",
        degree=4,
    )
    assert r.status is Status.OK and r.tolerance_met is False
    assert "does NOT satisfy" in r.message


def test_compare_never_chooses_a_pade_with_a_pole_in_the_interval() -> None:
    r = run(
        method="compare",
        input="exp(x)",
        point="0",
        lower="-1",
        upper="3",
        degree=2,
        methods=["pade", "taylor"],
    )
    pade_candidate = next(c for c in r.candidates if c.method == "pade")
    assert "pole_in_interval" in pade_candidate.warnings and not pade_candidate.chosen


def test_compare_relative_tolerance_is_formed_away_from_zeros() -> None:
    r = run(
        method="compare",
        input="sin(x)",
        lower="0.5",
        upper="3",
        tolerance="1e-3",
        tolerance_kind="relative",
    )
    assert r.status is Status.OK
    assert r.error.relative_note  # type: ignore[union-attr]


def test_compare_needs_a_budget_or_a_target() -> None:
    r = run(method="compare", input="exp(x)", lower="0", upper="1")
    assert r.reason == "domain_unspecified" and r.questions


def test_engine_runs_compare_in_the_sandbox(engine) -> None:  # type: ignore[no-untyped-def]
    r = engine.approximation(
        ApproximationInput(
            input="exp(x)", format="text", method="chebyshev", lower="0", upper="1", degree=4
        )
    )
    assert r.status is Status.OK and r.diagnostics.checks == 2
