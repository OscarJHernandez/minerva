"""Radial effective potential of geodesics (issue #207).

Expected answers are written by hand from the textbook (Schwarzschild, `V_eff = (1 - 2M/r)(1 + L^2/r^2)` for a
timelike geodesic, `(1 - 2M/r) L^2/r^2` for a null one), never generated from the code. Wrong candidates are put
between the construction and the checks, so that only the verification stands between them and the card: each
must come back `not_verified`."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import sympy as sp
from minerva_engine import Engine, Status
from minerva_engine import effective_potential as ep
from minerva_engine.effective_potential import (
    EffectivePotentialInput,
    EffectivePotentialReport,
    effective_potential,
)
from minerva_engine.latexparse import parse_latex
from minerva_engine.notation.models import InputCode
from minerva_engine.testing import budget_limits

LIMITS = budget_limits()
F = r"\left(1-\frac{2M}{r}\right)"
SCHWARZSCHILD = [
    [f"-{F}", "0", "0", "0"],
    ["0", f"{F}^{{-1}}", "0", "0"],
    ["0", "0", "r^2", "0"],
    ["0", "0", "0", r"r^2\sin^2\theta"],
]
ASSUME = {"M": ["positive"], "r": ["positive"]}

M, r, pt, pphi = sp.symbols("M r p_t p_phi")
f = 1 - 2 * M / r


def request(**changes: object) -> EffectivePotentialInput:
    base: dict[str, object] = {
        "coordinates": ["t", "r", "theta", "phi"],
        "metric": SCHWARZSCHILD,
        "assumptions": ASSUME,
        "cyclic": ["t", "phi"],
        "radial": "r",
        "fixed": {"theta": r"\frac{\pi}{2}"},
        "normalisation": "-1",
    }
    base.update(changes)
    return EffectivePotentialInput(**base)  # type: ignore[arg-type]


def run(**changes: object) -> EffectivePotentialReport:
    return effective_potential(request(**changes), LIMITS)


def plain(latex: str) -> sp.Expr:
    value = parse_latex(latex)
    return value.xreplace({s: sp.Symbol(s.name) for s in value.free_symbols})


def veff_of(report: EffectivePotentialReport) -> sp.Expr:
    assert report.veff_latex is not None
    return plain(report.veff_latex.split("=", 1)[1])


def test_schwarzschild_timelike_is_the_textbook_effective_potential() -> None:
    report = run()
    assert report.status == Status.OK, report.message
    # Hand-written: g(v, v) = -1 gives (1 - 2M/r)^-1 (r_dot^2 - p_t^2) + p_phi^2/r^2 = -1.
    assert sp.simplify(veff_of(report) - f * (1 + pphi**2 / r**2)) == 0
    assert report.result_latex is not None
    lhs, rhs = (plain(side) for side in report.result_latex.split("=", 1))
    assert sp.simplify(rhs - pt**2) == 0
    assert sp.simplify(lhs - (sp.Symbol("r_dot") ** 2 + f * (1 + pphi**2 / r**2))) == 0
    # the momenta keep their verified names; E and L are the user's own step
    assert [h.symbol_latex for h in report.held_fixed] == ["p_{t}", "p_{\\phi}"]
    assert "E" not in (report.result_latex or "").replace("\\", "")


def test_the_checks_are_reported_separately_and_proved_only_with_symbolic_zeros() -> None:
    report = run()
    checks = report.checks
    assert checks.normalisation is not None and checks.normalisation.passed
    assert checks.radial_acceleration is not None and checks.radial_acceleration.passed
    assert checks.routhian is not None and all(c.passed for c in checks.routhian.euler_lagrange)
    assert checks.symbolic is True
    assert all(c.symbolic is True for c in report.consistency)
    assert report.verification is not None and report.verification.symbolic_zero is True
    assert report.verification.numeric_points_agreed == report.verification.numeric_points_checked


def test_null_geodesics_drop_the_one() -> None:
    report = run(normalisation="0")
    assert report.status == Status.OK, report.message
    assert sp.simplify(veff_of(report) - f * pphi**2 / r**2) == 0


def test_the_normalisation_is_the_users_number() -> None:
    report = run(normalisation="-1/4")
    assert report.status == Status.OK, report.message
    assert sp.simplify(veff_of(report) - f * (sp.Rational(1, 4) + pphi**2 / r**2)) == 0


@pytest.mark.parametrize("text", ["", "m", "-1.2.3", "one"])
def test_an_unreadable_normalisation_is_a_typed_refusal(text: str) -> None:
    report = run(normalisation=text if text else " ")
    assert report.status == Status.PARSE_ERROR
    assert report.reason == "normalisation_unreadable"
    assert report.diagnostic is not None
    assert report.diagnostic.code == InputCode.NORMALISATION_UNREADABLE


def test_the_normalisation_has_no_default() -> None:
    with pytest.raises(ValueError):
        EffectivePotentialInput(  # type: ignore[call-arg]
            coordinates=["t", "r"],
            metric=[["-1", "0"], ["0", "1"]],
            cyclic=["t"],
            radial="r",
        )


def test_a_plane_that_is_not_consistent_is_refused() -> None:
    report = run(fixed={"theta": r"\frac{\pi}{3}"})
    assert report.status == Status.UNSUPPORTED
    assert report.reason == "equatorial_not_consistent"
    assert report.diagnostic is not None
    assert report.diagnostic.code == InputCode.RESTRICTION_NOT_CONSISTENT
    assert report.result_latex is None


def test_a_metric_whose_equator_is_not_a_plane_of_symmetry_is_refused() -> None:
    # g_phi_phi = r^2 sin^2(theta) (1 + cos(theta)): the equator theta = pi/2 is not a stationary plane
    tilted = [row[:] for row in SCHWARZSCHILD]
    tilted[3][3] = r"r^2\sin^2\theta\cdot\left(1+\cos(\theta)\right)"
    report = run(metric=tilted)
    assert report.status in (Status.UNSUPPORTED, Status.NOT_VERIFIED)
    assert report.reason in ("equatorial_not_consistent", "consistency_undecided")
    assert report.result_latex is None


def test_every_coordinate_needs_a_role() -> None:
    report = run(fixed={})
    assert report.status == Status.PARSE_ERROR
    assert report.reason == "unassigned_coordinates"
    report = run(cyclic=["t", "phi", "r"])
    assert report.reason == "unassigned_coordinates"


def test_the_energy_like_momentum_must_come_first() -> None:
    # r_dot^2 = p_t^2 - V: with p_phi first the coefficient of p_phi^2 depends on r, so the shape is refused
    report = run(cyclic=["phi", "t"])
    assert report.status == Status.UNSUPPORTED
    assert report.reason == "not_separable"


def test_text_syntax_gives_the_same_result() -> None:
    text = [
        ["-(1-2*M/r)", "0", "0", "0"],
        ["0", "1/(1-2*M/r)", "0", "0"],
        ["0", "0", "r^2", "0"],
        ["0", "0", "0", "r^2*sin(theta)^2"],
    ]
    report = run(metric=text, format="text", fixed={"theta": "pi/2"})
    assert report.status == Status.OK, report.message
    assert sp.simplify(veff_of(report) - f * (1 + pphi**2 / r**2)) == 0


def test_a_general_static_spherical_metric() -> None:
    # Reissner-Nordstrom: g_tt = -(1 - 2M/r + Q^2/r^2) = -h, g_rr = 1/h; V_eff = h (1 + p_phi^2/r^2)
    h = r"\left(1-\frac{2M}{r}+\frac{Q^2}{r^2}\right)"
    metric = [row[:] for row in SCHWARZSCHILD]
    metric[0][0] = f"-{h}"
    metric[1][1] = f"{h}^{{-1}}"
    report = run(metric=metric, assumptions={**ASSUME, "Q": ["positive"]})
    assert report.status == Status.OK, report.message
    q = sp.Symbol("Q")
    hh = 1 - 2 * M / r + q**2 / r**2
    assert sp.simplify(veff_of(report) - hh * (1 + pphi**2 / r**2)) == 0


def test_restrictions_are_about_the_coordinates() -> None:
    report = run()
    assert report.restrictions_latex
    assert any("M" in text for text in report.restrictions_latex)
    assert "singular" not in " ".join(report.restrictions_latex).lower()


def test_a_declared_function_is_not_supported() -> None:
    report = run(
        metric=[["-f(r)", "0", "0", "0"], ["0", "1/f(r)", "0", "0"], *SCHWARZSCHILD[2:]],
    )
    assert report.status in (Status.PARSE_ERROR, Status.UNSUPPORTED)
    assert report.result_latex is None


def _inject(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[sp.Expr, sp.Symbol], sp.Expr]
) -> None:
    original = ep.split_radial

    def wrong(
        routhian: sp.Expr, momenta: list[sp.Symbol], rd: sp.Symbol, kappa: sp.Rational
    ) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
        norm, alpha, veff, lead = original(routhian, momenta, rd, kappa)
        return norm, alpha, change(veff, momenta[1]), lead

    monkeypatch.setattr(ep, "_split", wrong)


def _named(expr: sp.Expr, name: str) -> sp.Symbol:
    return next(s for s in expr.free_symbols if s.name == name)


WRONG_TWINS: dict[str, Callable[[sp.Expr, sp.Symbol], sp.Expr]] = {
    # a dropped factor 1 - 2M/r
    "dropped_f": lambda v, p: sp.cancel(v / (1 - 2 * _named(v, "M") / _named(v, "r"))),
    # the sign of the p_phi^2 / r^2 term
    "centrifugal_sign": lambda v, p: sp.expand(v).subs(p**2, -(p**2)),
    # a relative error of 1e-6
    "factor_1e6": lambda v, p: v * sp.Rational(1000001, 1000000),
}


@pytest.mark.parametrize("twin", sorted(WRONG_TWINS))
def test_wrong_twins_are_not_verified(monkeypatch: pytest.MonkeyPatch, twin: str) -> None:
    _inject(monkeypatch, WRONG_TWINS[twin])
    report = run()
    assert report.status == Status.NOT_VERIFIED, report.veff_latex
    assert report.reason == "not_verified"
    assert report.result_latex is None
    assert report.veff_latex is None


def test_the_twins_differ_from_the_correct_potential() -> None:
    correct = sp.Symbol("x")
    del correct
    r_pos = sp.Symbol("r", positive=True)
    v = (1 - 2 * M / r_pos) * (1 + pphi**2 / r_pos**2)
    for change in WRONG_TWINS.values():
        assert sp.simplify(change(v, pphi) - v) != 0


def test_engine_entry_point_runs_in_the_sandbox() -> None:
    engine = Engine(limits=LIMITS)
    report = engine.effective_potential(request())
    assert report.status == Status.OK, report.message
    assert report.result_latex is not None
