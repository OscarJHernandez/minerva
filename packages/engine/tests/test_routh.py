"""Routh reduction of cyclic coordinates (issue #153).

Expected Routhians are written by hand from the textbook definition R = L - sum p_c c_dot (never generated
from the code). Wrong candidates are injected at the solve so that only the verification stands between
them and the card: each must come back `not_verified`."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import sympy as sp
from minerva_engine import Engine, Status
from minerva_engine import routh as routh_module
from minerva_engine.latexparse import parse_latex
from minerva_engine.notation.models import InputCode
from minerva_engine.routh import Reduction, RouthReport, routh_reduce
from minerva_engine.testing import budget_limits

LIMITS = budget_limits()

SCHWARZSCHILD = (
    r"\frac{1}{2}\left(-\left(1-\frac{2M}{r}\right)\dot{t}^2+\frac{\dot{r}^2}{1-\frac{2M}{r}}"
    r"+r^2\dot{\theta}^2+r^2\sin^2(\theta)\dot{\phi}^2\right)"
)
SCHWARZSCHILD_COORDS = ["t", "r", "theta", "phi"]
SCHWARZSCHILD_ASSUME = {"M": ["positive"], "r": ["positive"]}
KEPLER = r"\frac{1}{2} m \left(\dot{r}^2 + r^2\dot{\theta}^2\right) + \frac{k}{r}"
KEPLER_ASSUME = {"m": ["positive"], "k": ["positive"], "r": ["positive"]}

M, r, theta, m, k, a = sp.symbols("M r theta m k a")
rd, thd, phd = sp.symbols("r_dot theta_dot phi_dot")
p_t, p_phi, p_theta = sp.symbols("p_t p_phi p_theta")
f = 1 - 2 * M / r


def schwarzschild(cyclic: list[str]) -> RouthReport:
    return routh_reduce(
        SCHWARZSCHILD, SCHWARZSCHILD_COORDS, "s", SCHWARZSCHILD_ASSUME, cyclic, LIMITS
    )


def read(latex: str | None) -> sp.Expr:
    """The printed result, read with plain symbols so it can be compared with the hand-written form."""
    assert latex is not None
    expr = parse_latex(latex)
    return expr.xreplace({s: sp.Symbol(s.name) for s in expr.free_symbols})


def rhs(latex: str) -> sp.Expr:
    return read(latex.split("=", 1)[1])


def test_schwarzschild_reduced_by_time_is_the_textbook_routhian() -> None:
    report = schwarzschild(["t"])
    assert report.status == Status.OK, report.message
    # Hand-written: p_t = -f t_dot, so t_dot = -p_t / f and R = L - p_t t_dot (issue #153 with E = -p_t).
    expected = sp.Rational(1, 2) * (
        rd**2 / f + r**2 * thd**2 + r**2 * sp.sin(theta) ** 2 * phd**2 + p_t**2 / f
    )
    assert sp.simplify(read(report.result_latex) - expected) == 0
    assert report.remaining == ["r", "theta", "phi"]
    [held] = report.held_fixed
    assert held.symbol_latex == "p_{t}"
    assert sp.simplify(rhs(held.latex) - (-f * sp.Symbol("t_dot"))) == 0
    assert "cyclic" in held.reason and "verified" in held.reason
    assert held.verification is not None and held.verification.passed
    [rebuilt] = report.reconstruction
    assert rebuilt.latex.startswith(r"\dot{t} =")
    assert sp.simplify(rhs(rebuilt.latex) - (-p_t / f)) == 0
    # The block determinant -f vanishes at the horizon: reported, never dropped.
    assert report.restrictions_latex and all(
        "M" in x and "r" in x for x in report.restrictions_latex
    )
    assert report.verification is not None and report.verification.passed
    assert report.checks.symbolic is True and report.verification.symbolic_zero
    assert report.checks.definition is not None and report.checks.definition.passed
    assert all(v.passed and v.numeric_points_checked >= 4 for v in report.checks.euler_lagrange)
    assert all(v.passed for v in report.checks.reconstruction)
    assert "Proved" in report.message


def test_kepler_reduced_by_angle_has_the_centrifugal_term_with_the_textbook_sign() -> None:
    """Goldstein §8.3 / Landau-Lifshitz §41: the centrifugal potential p^2/(2 m r^2) is added to the
    potential energy, so it enters R (which acts as a Lagrangian, T - V_eff) with a minus sign."""
    report = routh_reduce(KEPLER, ["r", "theta"], "t", KEPLER_ASSUME, ["theta"], LIMITS)
    assert report.status == Status.OK, report.message
    expected = m * rd**2 / 2 + k / r - p_theta**2 / (2 * m * r**2)
    assert sp.simplify(read(report.result_latex) - expected) == 0
    assert report.held_fixed[0].symbol_latex == r"p_{\theta}"
    assert sp.simplify(rhs(report.reconstruction[0].latex) - p_theta / (m * r**2)) == 0
    assert "Landau-Lifshitz §41" in report.convention


def test_two_cyclic_coordinates_at_once() -> None:
    report = schwarzschild(["t", "phi"])
    assert report.status == Status.OK, report.message
    expected = sp.Rational(1, 2) * (
        rd**2 / f + r**2 * thd**2 + p_t**2 / f - p_phi**2 / (r**2 * sp.sin(theta) ** 2)
    )
    assert sp.simplify(read(report.result_latex) - expected) == 0
    assert [h.symbol_latex for h in report.held_fixed] == ["p_{t}", r"p_{\phi}"]
    assert report.remaining == ["r", "theta"]
    assert any(r"\sin" in x for x in report.restrictions_latex)


def test_velocity_coupled_cross_term() -> None:
    """L = (-f t_dot^2 + 2 a t_dot phi_dot + r_dot^2 + r^2 phi_dot^2) / 2: p_t = -f t_dot + a phi_dot, so
    t_dot = (a phi_dot - p_t) / f and R keeps a term linear in phi_dot (hand-derived)."""
    source = (
        r"\frac{1}{2}\left(-\left(1-\frac{2M}{r}\right)\dot{t}^2 + 2 a \dot{t}\dot{\phi} + \dot{r}^2"
        r" + r^2\dot{\phi}^2\right)"
    )
    assume = {"M": ["positive"], "r": ["positive"], "a": ["real"]}
    report = routh_reduce(source, ["t", "r", "phi"], "s", assume, ["t"], LIMITS)
    assert report.status == Status.OK, report.message
    t_dot = (a * phd - p_t) / f
    lagrangian = (-f * t_dot**2 + 2 * a * t_dot * phd + rd**2 + r**2 * phd**2) / 2
    expected = lagrangian - p_t * t_dot
    assert sp.simplify(read(report.result_latex) - expected) == 0
    assert sp.simplify(rhs(report.reconstruction[0].latex) - t_dot) == 0


def test_non_cyclic_coordinate_is_unsupported() -> None:
    report = schwarzschild(["r"])
    assert report.status == Status.UNSUPPORTED
    assert report.reason == "not_cyclic"
    assert report.result_latex is None and not report.held_fixed


def test_degenerate_block_is_unsupported() -> None:
    # p_t = r_dot does not contain t_dot: the 1x1 block is 0 although the full mass matrix is not singular.
    report = routh_reduce(
        r"\frac{1}{2}\dot{r}^2 + \dot{r}\dot{t} - \frac{k}{r}", ["r", "t"], "s", {}, ["t"], LIMITS
    )
    assert report.status == Status.UNSUPPORTED
    assert report.reason == "degenerate_block"


def test_momentum_not_linear_in_the_velocity_is_unsupported() -> None:
    report = routh_reduce(
        r"\frac{1}{2}\dot{r}^2 + \frac{1}{4}\dot{t}^4", ["r", "t"], "s", {}, ["t"], LIMITS
    )
    assert report.status == Status.UNSUPPORTED
    assert report.reason == "nonlinear_momentum"


def test_momentum_symbol_already_in_the_lagrangian_is_a_typed_refusal() -> None:
    report = routh_reduce(
        r"\frac{1}{2}\dot{x}^2 + p_{x}\dot{y}^2 + \frac{1}{2}\dot{y}^2",
        ["x", "y"],
        "t",
        {},
        ["x"],
        LIMITS,
    )
    assert report.status == Status.PARSE_ERROR
    assert report.diagnostic is not None
    assert report.diagnostic.code == InputCode.MOMENTUM_SYMBOL_CLASH
    assert report.diagnostic.symbols == ["p_x"]
    assert report.result_latex is None


@pytest.mark.parametrize(
    ("cyclic", "reason"),
    [(["q"], "not_a_coordinate"), ([], "not_a_coordinate"), (["r", "r"], "not_a_coordinate")],
)
def test_requests_that_name_no_declared_coordinate(cyclic: list[str], reason: str) -> None:
    report = routh_reduce(KEPLER, ["r", "theta"], "t", KEPLER_ASSUME, cyclic, LIMITS)
    assert report.status == Status.PARSE_ERROR
    assert report.reason == reason


def test_removing_every_coordinate_is_refused() -> None:
    report = routh_reduce(r"\frac{1}{2}\dot{x}^2", ["x"], "t", {}, ["x"], LIMITS)
    assert report.status == Status.UNSUPPORTED
    assert report.reason == "no_coordinate_left"


def test_coordinate_named_t_needs_another_time_symbol() -> None:
    report = routh_reduce(
        SCHWARZSCHILD, SCHWARZSCHILD_COORDS, "t", SCHWARZSCHILD_ASSUME, ["t"], LIMITS
    )
    assert report.status == Status.PARSE_ERROR
    assert report.diagnostic is not None
    assert report.diagnostic.code == InputCode.TIME_SYMBOL_IS_COORDINATE


def test_text_syntax_gives_the_same_routhian() -> None:
    report = routh_reduce(
        "m/2*(r_dot^2 + r^2*theta_dot^2) + k/r",
        ["r", "theta"],
        "t",
        KEPLER_ASSUME,
        ["theta"],
        LIMITS,
        format="text",
    )
    assert report.status == Status.OK, report.message
    expected = m * rd**2 / 2 + k / r - p_theta**2 / (2 * m * r**2)
    assert sp.simplify(read(report.result_latex) - expected) == 0


# --- wrong twins: injected at the solve, caught by the verification --------------------------------


def _inject(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[sp.Expr, dict[str, sp.Symbol]], sp.Expr]
) -> None:
    original = routh_module.solve_reduction

    def wrong(*args: object, **kwargs: object) -> Reduction:
        good = original(*args, **kwargs)  # type: ignore[arg-type]
        names = {s.name: s for s in good.routhian.free_symbols}
        return Reduction(
            routhian=change(good.routhian, names),
            velocities=good.velocities,
            determinant=good.determinant,
        )

    monkeypatch.setattr(routh_module, "solve_reduction", wrong)


def _p_term(expr: sp.Expr, names: dict[str, sp.Symbol]) -> sp.Expr:
    p = names["p_t"]
    return sp.expand(expr).coeff(p, 2) * p**2


WRONG_TWINS: dict[str, Callable[[sp.Expr, dict[str, sp.Symbol]], sp.Expr]] = {
    # +p_t^2/(2f) written as -p_t^2/(2f)
    "sign_of_the_momentum_term": lambda R, n: R - 2 * _p_term(R, n),
    # p_t^2/(2f) written as p_t^2/2
    "dropped_one_over_f": lambda R, n: (
        R - _p_term(R, n) + _p_term(R, n) * (1 - 2 * n["M"] / n["r"])
    ),
    # the momentum term off by a relative 1e-6
    "momentum_term_off_by_1e-6": lambda R, n: R + sp.Rational(1, 10**6) * _p_term(R, n),
    # the whole Routhian scaled by 1 + 1e-6: same Euler-Lagrange equations, wrong reconstruction
    "whole_routhian_scaled_by_1e-6": lambda R, n: R * (1 + sp.Rational(1, 10**6)),
    # the same equations of motion and the same -dR/dp: only the value of R against L - p t_dot sees these
    "plus_a_constant": lambda R, n: R + sp.Rational(1, 1000),
    "plus_a_function_of_the_time": lambda R, n: (
        R + sp.sin(n["s"]) if "s" in n else R + sp.sin(sp.Symbol("s"))
    ),
    "plus_a_total_derivative": lambda R, n: R + n["r_dot"],
}


@pytest.mark.parametrize("twin", sorted(WRONG_TWINS))
def test_wrong_twins_are_not_verified(monkeypatch: pytest.MonkeyPatch, twin: str) -> None:
    _inject(monkeypatch, WRONG_TWINS[twin])
    report = schwarzschild(["t"])
    assert report.status == Status.NOT_VERIFIED, (twin, report.result_latex)
    assert report.reason == "reduction_not_confirmed"
    assert report.result_latex is None and not report.held_fixed
    definition = report.checks.definition
    assert definition is not None
    legs = [*report.checks.euler_lagrange, *report.checks.reconstruction, definition]
    assert legs and not all(v.passed for v in legs)  # caught by the numerical legs, not the reader


def test_the_twins_differ_from_the_correct_routhian(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard on the test itself: each twin really changes R."""
    good = schwarzschild(["t"])
    assert good.result_latex is not None
    correct = read(good.result_latex)
    names = {s.name: s for s in correct.free_symbols}
    names.setdefault("s", sp.Symbol("s"))
    for twin, change in WRONG_TWINS.items():
        assert sp.simplify(change(correct, names) - correct) != 0, twin


def test_engine_entry_point_runs_in_the_sandbox() -> None:
    report = Engine(LIMITS).routh_reduce(
        KEPLER, ["r", "theta"], ["theta"], assumptions=KEPLER_ASSUME
    )
    assert report.status == Status.OK, report.message
    assert report.duration_ms > 0


def test_restrictions_follow_the_declared_assumptions() -> None:
    """The block determinant -(1 - 2M/r) = (2M - r)/r: its factor r is reported only when nothing says r > 0."""
    declared = schwarzschild(["t"])
    bare = routh_reduce(SCHWARZSCHILD, SCHWARZSCHILD_COORDS, "s", {}, ["t"], LIMITS)
    assert declared.status == bare.status == Status.OK
    assert not any(x.replace(" ", "") == r"r\neq0" for x in declared.restrictions_latex)
    assert any(x.replace(" ", "") == r"r\neq0" for x in bare.restrictions_latex)
    assert sp.simplify(read(declared.result_latex) - read(bare.result_latex)) == 0


def test_a_removed_velocity_left_in_r_fails_every_leg_and_does_not_crash() -> None:
    """Called directly (the reader would already refuse the printed form): no NameError from the evaluation."""
    from minerva_engine.lagrangian import derive

    d = derive(SCHWARZSCHILD, SCHWARZSCHILD_COORDS, "s", SCHWARZSCHILD_ASSUME, LIMITS)
    momenta = [routh_module.momentum_symbol(d.q[0])]
    good = routh_module.solve_reduction(d, [0], momenta)
    wrong = good.routhian + d.qd[0] / 10**9
    el, rec, definition = routh_module.check_reduction(d, [0], momenta, wrong, good.velocities)
    assert not any(v.passed for v in [*el, *rec, definition])
