"""Numerical action evaluation and user-facing finite-difference verification."""

from __future__ import annotations

from typing import Any

import mpmath as mp
import pytest
from minerva_engine import Engine
from minerva_engine.models import Limits, Status
from minerva_engine.variational import VariationInput, variation
from minerva_engine.variational.action import check_action
from minerva_engine.variational.quadrature import fd_table


def run(action: str, integrand: str, **kw: Any) -> Any:
    kw.setdefault("functions", ["x"])
    kw.setdefault("lower", "0")
    kw.setdefault("upper", "1")
    data = VariationInput(action=action, integrand=integrand, **kw)
    return variation(data, Limits(timeout_seconds=120))


def number(text: str) -> mp.mpf:
    return mp.mpf(text)


# -- evaluate -----------------------------------------------------------------------------------
def test_harmonic_action_on_the_stationary_path_matches_the_closed_form() -> None:
    r = run(
        "evaluate",
        r"\frac12\dot{x}^2-\frac12 x^2",
        paths=[{"analytic": {"x": r"\sin(t)/\sin(1)"}}],
    )
    assert r.status is Status.OK
    assert abs(number(r.evaluation.trials[0].action) - mp.cot(1) / 2) < 1e-12
    assert r.evaluation.trials[0].quadrature_error < 1e-12


def test_free_particle_with_parameters_and_a_definition() -> None:
    r = run(
        "evaluate",
        "m/2*x_dot**2 - V(x)",
        given_functions=[{"name": "V", "args": ["x"]}],
        values={"m": "3"},
        definitions={"V": "x**2"},
        format="text",
        paths=[{"analytic": {"x": "t"}}],
    )
    assert r.status is Status.OK
    assert abs(number(r.evaluation.trials[0].action) - (mp.mpf(3) / 2 - mp.mpf(1) / 3)) < 1e-12


def test_a_trial_family_ritz_example_with_exact_values() -> None:
    r = run(
        "evaluate",
        r"\frac12\dot{x}^2-x",
        paths=[{"analytic": {"x": "c*t*(1-t)"}}],
        sweep={"name": "c", "values": ["0", "1/4", "1/2", "1"]},
    )
    assert r.status is Status.OK
    got = [number(t.action) for t in r.evaluation.trials]
    want = [mp.mpf(0), -mp.mpf(1) / 32, -mp.mpf(1) / 24, mp.mpf(0)]
    assert all(abs(g - w) < 1e-13 for g, w in zip(got, want, strict=True))
    assert r.evaluation.ordering is not None and r.evaluation.ordering[0].endswith("1/2")
    # the smallest value among a few trials is not called a minimizer
    assert "not claimed to be a minimum" in r.evaluation.statement


def test_paths_that_break_the_declared_conditions_are_flagged() -> None:
    r = run(
        "evaluate",
        r"\frac12\dot{x}^2",
        conditions=[
            {"function": "x", "order": 0, "end": "lower", "kind": "fixed", "value": "0"},
            {"function": "x", "order": 0, "end": "upper", "kind": "fixed", "value": "1"},
        ],
        paths=[
            {"label": "line", "analytic": {"x": "t"}},
            {"label": "wrong end", "analytic": {"x": "2*t"}},
        ],
    )
    assert r.status is Status.OK
    met = {t.label: [c.met for c in t.conditions] for t in r.evaluation.trials}
    assert met == {"line": [True, True], "wrong end": [True, False]}
    assert any("like with like" in n for n in r.evaluation.notes)


def test_missing_parameter_values_are_asked_for_not_guessed() -> None:
    r = run("evaluate", r"\frac{m}{2}\dot{x}^2", paths=[{"analytic": {"x": "t"}}])
    assert r.status is Status.CONDITIONS_NOT_SATISFIED and "m" in r.message


def test_a_symbolic_interval_needs_values_too() -> None:
    r = run("evaluate", r"\frac12\dot{x}^2", lower="a", upper="b", paths=[{"analytic": {"x": "t"}}])
    assert r.status is Status.CONDITIONS_NOT_SATISFIED
    ok = run(
        "evaluate",
        r"\frac12\dot{x}^2",
        lower="a",
        upper="b",
        values={"a": "0", "b": "2"},
        paths=[{"analytic": {"x": "t"}}],
    )
    assert ok.status is Status.OK and abs(number(ok.evaluation.trials[0].action) - 1) < 1e-12


def test_higher_derivatives_and_several_functions() -> None:
    r = run(
        "evaluate",
        r"\frac12\ddot{x}^2+\frac12\dot{y}^2+x y",
        functions=["x", "y"],
        paths=[{"analytic": {"x": "t^3", "y": "t"}}],
    )
    # ∫ (18 t)^2 / 2 ... = ∫ 18 t^2 ... : x'' = 6t -> 18 t^2 -> 6 ; y'^2/2 -> 1/2 ; x y = t^4 -> 1/5
    assert r.status is Status.OK
    assert abs(number(r.evaluation.trials[0].action) - (6 + mp.mpf(1) / 2 + mp.mpf(1) / 5)) < 1e-12


def test_sampled_path_with_a_spline_is_exact_for_a_cubic() -> None:
    ts = [i / 20 for i in range(21)]
    r = run(
        "evaluate",
        r"\dot{x}^2",
        paths=[{"sampled": {"x": {"t": ts, "values": [t * t for t in ts]}}}],
    )
    assert r.status is Status.OK
    trial = r.evaluation.trials[0]
    # a natural spline would force x'' = 0 at the ends and miss this by O(h^2); not-a-knot is exact
    assert abs(number(trial.action) - mp.mpf(4) / 3) < 1e-9
    assert trial.route == "sampled" and trial.half_data_difference is not None
    assert trial.half_data_difference < 1e-9


def test_second_derivatives_need_slopes_for_sampled_data() -> None:
    ts = [i / 10 for i in range(11)]
    values = [t**3 for t in ts]
    refused = run(
        "evaluate", r"\ddot{x}^2", paths=[{"sampled": {"x": {"t": ts, "values": values}}}]
    )
    assert refused.status is Status.UNSUPPORTED
    ok = run(
        "evaluate",
        r"\ddot{x}^2",
        paths=[{"sampled": {"x": {"t": ts, "values": values, "slopes": [3 * t * t for t in ts]}}}],
    )
    assert ok.status is Status.OK and abs(number(ok.evaluation.trials[0].action) - 12) < 1e-9


def test_a_singular_integrand_is_not_shown() -> None:
    r = run("evaluate", r"\frac{1}{x}", paths=[{"analytic": {"x": "t"}}])
    assert r.status in {Status.RESOURCE_LIMIT, Status.NOT_VERIFIED}
    assert r.evaluation is None


def test_check_action_rejects_a_value_that_is_off_by_one_part_in_ten_million() -> None:
    f = lambda s: s * s  # noqa: E731
    assert check_action(f, mp.mpf(0), mp.mpf(1), mp.mpf(1) / 3)
    assert not check_action(f, mp.mpf(0), mp.mpf(1), mp.mpf(1) / 3 * (1 + mp.mpf("1e-7")))


# -- the finite-difference table itself ---------------------------------------------------------------
STEPS = (mp.mpf("1e-3"), mp.mpf("5e-4"), mp.mpf("2.5e-4"))


def test_fd_table_accepts_exact_and_second_order_data_and_rejects_the_rest() -> None:
    cubic = lambda e: 2 * e + 3 * e**3  # noqa: E731 - S(e) with S'(0) = 2
    assert fd_table(cubic, mp.mpf(2), mp.mpf(2), STEPS).verdict == "agree"
    assert fd_table(lambda e: 2 * e, mp.mpf(2), mp.mpf(2), STEPS).verdict == "agree"
    wrong = fd_table(cubic, mp.mpf(2) * (1 + mp.mpf("1e-7")), mp.mpf(2), STEPS)
    assert wrong.verdict == "disagree"


def test_fd_table_reports_noise_instead_of_a_verdict() -> None:
    noisy = fd_table(lambda e: 2 * e, mp.mpf(2), mp.mpf(2), STEPS, noise=mp.mpf("1e-5"))
    assert noisy.verdict == "inconclusive"


def test_fd_table_second_order_quotient() -> None:
    # S(e) = 5 + 2e + 4e^2 + e^4: second derivative at 0 is 8
    s = lambda e: 5 + 2 * e + 4 * e**2 + e**4  # noqa: E731
    assert fd_table(s, mp.mpf(8), mp.mpf(8), STEPS, order=2).verdict == "agree"
    assert fd_table(s, mp.mpf(9), mp.mpf(8), STEPS, order=2).verdict == "disagree"


# -- verify -------------------------------------------------------------------------------------------
def test_verification_on_a_non_stationary_path_has_interior_and_boundary_parts() -> None:
    r = run(
        "verify",
        r"\frac12\dot{x}^2-\frac12 x^2",
        paths=[{"analytic": {"x": "t^2"}}],
    )
    assert r.status is Status.OK and r.verification is not None
    v = r.verification
    assert v.verdict == "passed" and v.route == "integrated"
    assert len(v.perturbations) >= 3
    free = [p for p in v.perturbations if p.kind == "free_end"]
    assert free and all(abs(number(p.boundary_part)) > 1e-3 for p in free)
    assert all(p.comparison.verdict == "agree" for p in v.perturbations)


def test_fixed_ends_keep_the_perturbations_admissible_and_free_ends_keep_the_boundary_term() -> (
    None
):
    r = run(
        "verify",
        r"\frac12\dot{x}^2",
        conditions=[
            {"function": "x", "order": 0, "end": "lower", "kind": "fixed", "value": "0"},
            {"function": "x", "order": 0, "end": "upper", "kind": "free"},
        ],
        paths=[{"analytic": {"x": "t^2"}}],
    )
    assert r.verification is not None and r.verification.verdict == "passed"
    for p in r.verification.perturbations:
        assert p.admissible is True
    assert any(abs(number(p.boundary_part)) > 1e-3 for p in r.verification.perturbations)


def test_higher_derivative_boundary_terms_are_verified_independently() -> None:
    r = run(
        "verify",
        r"\frac{m}{2}\dot{x}^2+\frac{\alpha}{2}\ddot{x}^2-V(x)",
        given_functions=[{"name": "V", "args": ["x"]}],
        definitions={"V": "x^4/4"},
        values={"m": "1.5", "alpha": "0.25"},
        paths=[{"analytic": {"x": r"\sin(2t)+t^3"}}],
    )
    assert r.status is Status.OK and r.verification is not None
    assert r.verification.verdict == "passed"


def test_user_prescribed_perturbation_is_flagged_when_it_breaks_a_fixed_condition() -> None:
    r = run(
        "verify",
        r"\frac12\dot{x}^2",
        conditions=[{"function": "x", "order": 0, "end": "lower", "kind": "fixed", "value": "0"}],
        paths=[{"analytic": {"x": "t"}}],
        perturbations=[
            {"label": "bad", "expressions": {"x": "1+t"}},
            {"label": "good", "expressions": {"x": "t*(1-t)"}},
        ],
        automatic_perturbations=False,
    )
    assert r.verification is not None
    by_label = {p.label: p for p in r.verification.perturbations}
    assert by_label["bad"].admissible is False and by_label["good"].admissible is True
    assert r.verification.verdict == "passed"  # the full variation holds for either


def test_sampled_path_uses_the_direct_form_and_says_so() -> None:
    ts = [i / 20 for i in range(21)]
    r = run(
        "verify",
        r"\frac12\dot{x}^2",
        paths=[{"sampled": {"x": {"t": ts, "values": [t * t for t in ts]}}}],
    )
    assert r.verification is not None
    assert r.verification.route == "direct"
    assert r.verification.verdict == "passed"
    assert any("direct form" in n for n in r.verification.notes)


def test_engine_round_trip(engine: Engine) -> None:
    r = engine.variation(
        VariationInput(
            action="evaluate",
            integrand=r"\frac12\dot{x}^2",
            functions=["x"],
            lower="0",
            upper="1",
            paths=[{"analytic": {"x": "t"}}],
        )
    )
    assert r.status is Status.OK and r.evaluation is not None
    assert r.evaluation.trials[0].action.startswith("0.5")


@pytest.mark.parametrize("bad", [{"x": "t", "y": "t"}, {}])
def test_a_path_must_give_every_function_once(bad: dict[str, str]) -> None:
    r = run("evaluate", r"\frac12\dot{x}^2", paths=[{"analytic": bad}])
    assert r.status is Status.PARSE_ERROR
