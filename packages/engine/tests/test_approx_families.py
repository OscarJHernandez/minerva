"""Approximation, phase 3: Fourier series, Legendre expansions and piecewise polynomials.

Expected values are classical closed forms (sawtooth and |x| Fourier coefficients, the Legendre
generating function, an exact projection); each checker is also given tampered coefficients.
"""

from __future__ import annotations

import mpmath
import numpy as np
import pytest
from minerva_engine import ApproximationInput
from minerva_engine.approx import fourier, legendre, piecewise
from minerva_engine.approx.operation import approximation
from minerva_engine.models import Limits, Status

LIMITS = Limits(timeout_seconds=120)


def run(**kw):  # type: ignore[no-untyped-def]
    kw.setdefault("format", "text")
    return approximation(ApproximationInput(**kw), LIMITS)


def coeff(report, order: int, prefix: str = "") -> mpmath.mpf:  # type: ignore[no-untyped-def]
    for o in report.orders:
        if o.order == order and o.term_latex.startswith(prefix):
            return mpmath.mpf(o.coefficient_latex)
    return mpmath.mpf(0)


# --- Fourier ----------------------------------------------------------------------------------
def test_fourier_sawtooth_has_the_classical_coefficients() -> None:
    r = run(method="fourier", input="x", lower="-pi", upper="pi", modes=6)
    assert r.status is Status.OK
    for n in range(1, 7):
        assert abs(coeff(r, n, "b") - mpmath.mpf(2) * (-1) ** (n + 1) / n) < 1e-10
        assert coeff(r, n, "a") == 0
    assert any("sine-only" in n for n in r.notes)
    closed = {e.label.split(" ")[0]: e.latex for e in r.extras}
    assert closed["b_n"] == r"- \frac{2 \left(-1\right)^{n}}{n}"


def test_fourier_of_the_absolute_value_is_cosine_only() -> None:
    r = run(method="fourier", input="abs(x)", lower="-pi", upper="pi", modes=7)
    assert r.status is Status.OK
    for n in range(1, 8):
        want = 0 if n % 2 == 0 else -4 / (mpmath.pi * n**2)
        assert abs(coeff(r, n, "a") - want) < 1e-10
    assert any("cosine-only" in n for n in r.notes)
    assert "gibbs" not in r.warnings  # |x| is continuous across the period


def test_fourier_period_given_alone_uses_a_stated_symmetric_interval() -> None:
    r = run(method="fourier", input="sin(x)", period="2*pi", modes=3)
    assert r.status is Status.OK and any("centred at 0" in n for n in r.notes)
    assert abs(coeff(r, 1, "b") - 1) < 1e-10 and float(r.error.max_abs) < 1e-10  # type: ignore[union-attr]


def test_fourier_period_must_match_the_interval() -> None:
    assert (
        run(method="fourier", input="x", lower="0", upper="1", period="3", modes=3).reason
        == "point_invalid"
    )
    r = run(method="fourier", input="x", modes=3)
    assert r.reason == "domain_unspecified" and r.questions


def test_fourier_discontinuity_reports_gibbs_and_does_not_converge_uniformly() -> None:
    few = run(method="fourier", input="abs(x)/x", lower="-2", upper="3.5", modes=9)
    many = run(method="fourier", input="abs(x)/x", lower="-2", upper="3.5", modes=33)
    for r in (few, many):
        assert r.status is Status.OK
        assert "gibbs" in r.warnings and "jump_discontinuity" in r.warnings
        assert any("midpoint" in n for n in r.notes)
        assert float(r.error.max_abs) > 0.9  # type: ignore[union-attr]  # no uniform convergence at the jump


def test_fourier_lanczos_sigma_factors_change_the_series_not_the_coefficients() -> None:
    plain = run(method="fourier", input="abs(x)/x", lower="-2", upper="3.5", modes=9)
    smooth = run(method="fourier", input="abs(x)/x", lower="-2", upper="3.5", modes=9, kind="sigma")
    assert smooth.status is Status.OK
    assert any("Lanczos" in n for n in smooth.notes)
    assert float(smooth.error.rms) != float(plain.error.rms)  # type: ignore[union-attr]
    assert plain.orders[0].coefficient_latex == smooth.orders[0].coefficient_latex


def test_fourier_tolerance_finds_the_fewest_modes_tried() -> None:
    r = run(
        method="fourier",
        input="exp(cos(x))",
        period="2*pi",
        tolerance="1e-9",
        tolerance_kind="absolute",
    )
    assert r.tolerance_met is True and r.complexity.modes == 16
    assert [c.label for c in r.candidates if c.chosen] == ["16 modes"]
    # and for the jump it cannot be met
    j = run(
        method="fourier",
        input="abs(x)/x",
        lower="-2",
        upper="3.5",
        tolerance="1e-3",
        tolerance_kind="absolute",
        max_terms=16,
    )
    assert j.tolerance_met is False and "NOT met" in j.message


def test_fourier_high_frequency_is_missed_by_few_modes_and_found_by_enough() -> None:
    low = run(method="fourier", input="sin(40*x)", lower="0", upper="2*pi", modes=8)
    high = run(method="fourier", input="sin(40*x)", lower="0", upper="2*pi", modes=48)
    assert float(low.error.max_abs) > 0.9 and float(high.error.max_abs) < 1e-9  # type: ignore[union-attr]
    assert abs(coeff(high, 40, "b") - 1) < 1e-9


def test_fourier_with_coefficients_that_the_second_rule_contradicts_is_not_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = fourier._coefficients
    calls = {"n": 0}

    def wrong(rule, a, omega, period, modes):  # type: ignore[no-untyped-def]
        out = real(rule, a, omega, period, modes)
        calls["n"] += 1
        if calls["n"] == 2:  # the finer rule
            out.an[1] += mpmath.mpf("1e-7")
        return out

    monkeypatch.setattr(fourier, "_coefficients", wrong)
    r = run(method="fourier", input="exp(cos(x))", period="2*pi", modes=4)
    assert r.status is Status.NOT_VERIFIED and r.result_latex is None


def test_fourier_energy_check_rejects_coefficients_that_are_too_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = fourier._tidy

    def wrong(coeffs):  # type: ignore[no-untyped-def]
        out, parity = real(coeffs)
        out.bn[0] = out.bn[0] * mpmath.mpf("1.01")
        return out, parity

    monkeypatch.setattr(fourier, "_tidy", wrong)
    r = run(method="fourier", input="sin(x)", period="2*pi", modes=3)
    assert r.status is Status.NOT_VERIFIED


# --- Legendre ---------------------------------------------------------------------------------
def test_legendre_generating_function_coefficients() -> None:
    # 1/sqrt(1 - 2 x h + h^2) = sum h^n P_n(x), here h = 1/2
    r = run(method="legendre", input="1/sqrt(5/4 - x)", lower="-1", upper="1", degree=7)
    assert r.status is Status.OK
    for n in range(8):
        assert abs(coeff(r, n) - mpmath.mpf(2) ** -n) < 1e-10
    assert all(c.passed for c in r.checks)


def test_legendre_projection_of_a_cubic_is_exact() -> None:
    r = run(method="legendre", input="x**3", lower="-1", upper="1", degree=3)
    assert (
        abs(coeff(r, 3) - mpmath.mpf(2) / 5) < 1e-11
        and abs(coeff(r, 1) - mpmath.mpf(3) / 5) < 1e-11
    )
    assert coeff(r, 0) == 0 and coeff(r, 2) == 0
    assert float(r.error.max_abs) < 1e-10  # type: ignore[union-attr]


def test_legendre_affine_map_to_another_interval() -> None:
    r = run(method="legendre", input="x", lower="0", upper="2", degree=1)  # x = 1 + t
    assert abs(coeff(r, 0) - 1) < 1e-11 and abs(coeff(r, 1) - 1) < 1e-11
    assert r.statement_latex.startswith("t = ")  # type: ignore[union-attr]


def test_legendre_tolerance_search() -> None:
    r = run(
        method="legendre",
        input="exp(x)",
        lower="0",
        upper="2",
        tolerance="1e-8",
        tolerance_kind="absolute",
    )
    assert r.tolerance_met is True and r.complexity.degree == 10
    chosen = [c for c in r.candidates if c.chosen]
    assert len(chosen) == 1 and all(not c.tolerance_met for c in r.candidates if not c.chosen)


def test_legendre_discontinuous_function_warns_and_stays_inaccurate() -> None:
    r = run(method="legendre", input="abs(x)/x", lower="-1", upper="2", degree=8)
    assert r.status is Status.OK and "jump_discontinuity" in r.warnings
    assert float(r.error.max_abs) > 0.1  # type: ignore[union-attr]


def test_legendre_with_a_tampered_numpy_comparison_is_not_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = legendre.numpy_coefficients
    monkeypatch.setattr(
        legendre,
        "numpy_coefficients",
        lambda ref, d, breaks=None: [c + 1e-3 for c in real(ref, d, breaks) or []],
    )
    r = run(method="legendre", input="exp(x)", lower="-1", upper="1", degree=4)
    assert r.status is Status.NOT_VERIFIED and r.result_latex is None


def test_legendre_refuses_a_pole_and_asks_for_the_degree() -> None:
    assert (
        run(method="legendre", input="1/x", lower="-1", upper="1", degree=3).reason
        == "domain_invalid"
    )
    assert run(method="legendre", input="exp(x)", lower="-1", upper="1").questions
    assert (
        run(method="legendre", input="exp(x)", lower="-1", upper="1", degree=50).reason
        == "order_out_of_range"
    )


# --- Piecewise --------------------------------------------------------------------------------
def test_piecewise_linear_is_exact_when_the_knots_hit_the_kink() -> None:
    r = run(method="piecewise", input="abs(x)", lower="-1", upper="1", segments=4, kind="linear")
    assert r.status is Status.OK and float(r.error.max_abs) < 1e-12  # type: ignore[union-attr]
    r = run(method="piecewise", input="abs(x)", lower="-1", upper="1", segments=3, kind="linear")
    assert float(r.error.max_abs) > 0.05  # type: ignore[union-attr]  # the kink falls inside a segment


def test_piecewise_cubic_spline_converges_with_segments() -> None:
    errors = [
        float(
            run(
                method="piecewise", input="sin(x)", lower="0", upper="3", segments=n, kind="cubic"
            ).error.max_abs
        )  # type: ignore[union-attr]
        for n in (4, 8, 16)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] / errors[2] > 8  # fourth order: halving the step gains about 16


def test_piecewise_subdivides_where_the_function_varies_quickly() -> None:
    r = run(
        method="piecewise",
        input="exp(-200*x**2)",
        lower="-1",
        upper="1",
        tolerance="1e-3",
        tolerance_kind="absolute",
        kind="pchip",
    )
    assert r.status is Status.OK and r.tolerance_met is True
    xs = [float(o.term_latex.split(",")[0].split("=")[1]) for o in r.orders]
    near = sum(1 for x in xs if abs(x) <= 0.25)
    far = sum(1 for x in xs if abs(x) > 0.25)
    assert near / 0.5 > 3 * far / 1.5  # many more knots per unit length near the spike


def test_piecewise_subdivision_is_bounded() -> None:
    r = run(
        method="piecewise",
        input="sin(100*x)",
        lower="0",
        upper="1",
        tolerance="1e-12",
        tolerance_kind="absolute",
        max_terms=20,
        kind="cubic",
    )
    assert r.status is Status.OK and r.tolerance_met is False
    assert r.complexity.segments is not None and r.complexity.segments <= 20


def test_piecewise_never_extrapolates() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    for kind in piecewise.KINDS:
        fn = piecewise._wrap(piecewise._interpolant(kind, xs, xs**2))
        assert fn(mpmath.mpf("1.5")) is not None
        assert fn(mpmath.mpf("-0.1")) is None and fn(mpmath.mpf("2.1")) is None
    r = run(method="piecewise", input="sin(x)", lower="0", upper="3", segments=6)
    assert "no_extrapolation" in r.warnings


def test_piecewise_jump_is_flagged_and_questions_are_asked() -> None:
    r = run(method="piecewise", input="abs(x)/x", lower="-1", upper="2", segments=10, kind="linear")
    assert "jump_discontinuity" in r.warnings and float(r.error.max_abs) > 0.3  # type: ignore[union-attr]
    assert run(method="piecewise", input="sin(x)", lower="0", upper="3").questions
    assert (
        run(
            method="piecewise", input="sin(x)", lower="0", upper="3", segments=3, kind="quintic"
        ).status
        is Status.UNSUPPORTED
    )
    rel = run(
        method="piecewise",
        input="sin(x)",
        lower="0",
        upper="3",
        tolerance="1e-3",
        tolerance_kind="relative",
    )
    assert rel.reason == "domain_unspecified"


def test_compare_adds_a_piecewise_candidate_for_a_function_that_jumps() -> None:
    r = run(method="compare", input="abs(x)/x", lower="-1", upper="2", degree=7)
    assert {c.method for c in r.candidates} >= {"chebyshev", "piecewise"}
    assert "jump_discontinuity" in r.warnings
