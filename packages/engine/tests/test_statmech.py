"""Canonical-ensemble statistical mechanics of explicit spectra: results, refusals, honesty.

Expected values are computed here with plain arithmetic (math/mpmath on the closed forms), never with
the code under test.
"""

import json
import math

import mpmath as mp
import pytest
import sympy as sp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from minerva_engine import Engine, Limits
from minerva_engine.models import Status
from minerva_engine.statmech import canonical
from minerva_engine.statmech.models import StatMechInput, StatMechResult
from minerva_engine.statmech.operation import statmech

LIMITS = Limits(timeout_seconds=60)


def run(**kwargs: object) -> StatMechResult:
    return statmech(StatMechInput(**kwargs), LIMITS)  # type: ignore[arg-type]


def two_level(**extra: object) -> StatMechResult:
    return run(
        system={"kind": "levels", "levels": [{"energy": "0"}, {"energy": "epsilon"}]},
        assumptions={"epsilon": ["positive"]},
        **extra,
    )


def value(result: StatMechResult, name: str, **symbols: float) -> float:
    q = next(q for q in result.quantities if q.name == name)
    names = {name: sp.Symbol(name) for name in (*symbols, "k_B", "beta", "T")}
    expr = sp.sympify(q.text, locals=names)
    return float(expr.subs({sp.Symbol(k): v for k, v in symbols.items()}))


def passed(result: StatMechResult, name: str) -> dict[str, bool | None]:
    q = next(q for q in result.quantities if q.name == name)
    return {c.name: c.passed for c in q.checks}


# -- known systems -------------------------------------------------------------------------
def test_two_level_system_matches_closed_forms() -> None:
    r = two_level(action="derive", quantities=["Z", "F", "U", "S", "Var_E"])
    assert r.status is Status.OK and r.verification.passed
    b, eps, kB = 0.7, 1.3, 2.0
    x = b * eps
    assert value(r, "Z", beta=b, epsilon=eps) == pytest.approx(1 + math.exp(-x))
    assert value(r, "F", beta=b, epsilon=eps) == pytest.approx(-math.log(1 + math.exp(-x)) / b)
    assert value(r, "U", beta=b, epsilon=eps) == pytest.approx(eps / (math.exp(x) + 1))
    assert value(r, "Var_E", beta=b, epsilon=eps) == pytest.approx(
        eps**2 * math.exp(x) / (1 + math.exp(x)) ** 2
    )
    u = eps / (math.exp(x) + 1)
    assert value(r, "S", beta=b, epsilon=eps, k_B=kB) == pytest.approx(
        kB * (math.log(1 + math.exp(-x)) + b * u)
    )


def test_every_displayed_quantity_has_independent_numeric_evidence() -> None:
    r = two_level(action="derive", quantities=["U", "S", "Var_E"])
    assert passed(r, "U")["enumeration"] is True
    assert passed(r, "U")["finite_difference"] is True
    assert passed(r, "S")["finite_difference"] is True  # S = -dF/dT, differentiated numerically
    assert passed(r, "Var_E")["nonnegative_variance"] is True
    assert r.verification.points_agreed == r.verification.points_checked >= 4
    assert "not a proof" in r.verification.note


def test_heat_capacity_is_a_derivative_with_the_other_variables_held_fixed() -> None:
    r = two_level(
        action="derive",
        quantities=["U"],
        partials=[{"of": "U", "variable": "T", "held_fixed": [], "label": "C"}],
    )
    assert r.status is Status.OK
    c = next(q for q in r.quantities if q.name == "C")
    assert c.held_fixed == []  # nothing else is a state variable here
    b, eps, kB = 0.7, 1.3, 2.0
    x = b * eps
    assert value(r, "C", beta=b, epsilon=eps, k_B=kB) == pytest.approx(
        kB * x**2 * math.exp(x) / (math.exp(x) + 1) ** 2
    )
    assert passed(r, "C")["finite_difference"] is True


def test_pressure_from_a_free_energy_derivative_needs_no_pressure_formula() -> None:
    # Levels E_i = a_i / V: dimensional scaling gives p V = U exactly.
    r = run(
        system={
            "kind": "levels",
            "levels": [{"energy": "1/V"}, {"energy": "2/V"}, {"energy": "5/V", "degeneracy": "2"}],
        },
        parameters=["V"],
        assumptions={"V": ["positive"]},
        action="derive",
        quantities=["U"],
        partials=[
            {"of": "F", "variable": "V", "held_fixed": ["T"], "scale": "-1", "label": "p"},
        ],
    )
    assert r.status is Status.OK, r.message
    p = next(q for q in r.quantities if q.name == "p")
    assert p.held_fixed == ["beta"]  # the temperature coordinate is the other state variable
    b, v = 0.9, 1.7
    assert value(r, "p", beta=b, V=v) * v == pytest.approx(value(r, "U", beta=b, V=v))


def test_spin_half_in_a_field_response_and_fluctuation() -> None:
    levels = [
        {"energy": "-mu*h", "observables": {"M": "mu"}},
        {"energy": "mu*h", "observables": {"M": "-mu"}},
    ]
    base = {
        "system": {"kind": "levels", "levels": levels},
        "parameters": ["h"],
        "assumptions": {"mu": ["positive"], "h": ["real"]},
        "coupling": {"parameter": "h", "observable": "M"},
        "action": "expectation",
        "observable": "M",
    }
    mean = run(**base, statistic="mean")
    assert mean.status is Status.OK, mean.message
    b, mu, h = 0.8, 1.1, 0.6
    assert value(mean, "mean_M", beta=b, mu=mu, h=h) == pytest.approx(mu * math.tanh(b * mu * h))
    assert passed(mean, "mean_M")["fluctuation_response"] is True
    var = run(**base, statistic="variance")
    assert value(var, "variance_M", beta=b, mu=mu, h=h) == pytest.approx(
        mu**2 / math.cosh(b * mu * h) ** 2
    )
    assert any(c.latex.startswith(r"\partial E_i") and c.established for c in var.conditions)


def test_cumulants_of_energy_are_derivatives_of_log_z() -> None:
    r = two_level(action="expectation", observable="E", statistic="cumulant", order=3)
    assert r.status is Status.OK
    assert passed(r, "cumulant_E")["fluctuation_response"] is True


def test_covariance_of_two_observables() -> None:
    r = run(
        system={
            "kind": "levels",
            "levels": [
                {"energy": "0", "observables": {"A": "1", "B": "2"}},
                {"energy": "1", "observables": {"A": "3", "B": "1"}},
            ],
        },
        action="expectation",
        observable="A",
        other_observable="B",
        statistic="covariance",
    )
    assert r.status is Status.OK
    q = value(r, "covariance_A", beta=1.0)
    p1 = math.exp(-1) / (1 + math.exp(-1))
    p0 = 1 - p1
    assert q == pytest.approx(p0 * 1 * 2 + p1 * 3 * 1 - (p0 * 1 + p1 * 3) * (p0 * 2 + p1 * 1))


def test_degenerate_levels_with_symbolic_degeneracy() -> None:
    r = run(
        system={
            "kind": "levels",
            "levels": [{"energy": "0"}, {"energy": "eps", "degeneracy": "g"}],
        },
        assumptions={"eps": ["positive"], "g": ["integer", "positive"]},
        action="derive",
        quantities=["Z", "U"],
    )
    assert r.status is Status.OK, r.message
    assert value(r, "Z", beta=0.5, eps=2.0, g=3) == pytest.approx(1 + 3 * math.exp(-1.0))


def test_probabilities_are_normalised() -> None:
    r = two_level(action="partition")
    assert r.status is Status.OK
    assert passed(r, "probabilities")["normalisation"] is True


def test_huge_energies_do_not_overflow() -> None:
    r = run(
        system={"kind": "levels", "levels": [{"energy": "0"}, {"energy": "1000"}]},
        action="derive",
        quantities=["F", "U", "S"],
    )
    assert r.status is Status.OK, r.message  # beta up to 4: exp(4000) is fine with log-sum-exp


def test_temperature_variable_t_gives_formulas_in_t() -> None:
    r = two_level(action="derive", quantities=["U"], temperature="T")
    assert r.status is Status.OK
    u = next(q for q in r.quantities if q.name == "U")
    assert "T" in u.text and "beta" not in u.text
    assert value(r, "U", T=0.5, epsilon=1.0, k_B=1.0) == pytest.approx(1.0 / (math.exp(2.0) + 1))


def test_runs_through_the_sandboxed_engine(engine: Engine) -> None:
    r = engine.statmech(
        StatMechInput(
            system={"kind": "levels", "levels": [{"energy": "0"}, {"energy": "epsilon"}]},
            assumptions={"epsilon": ["positive"]},
            action="derive",
            quantities=["U"],
        )
    )
    assert r.status is Status.OK and r.duration_ms > 0


# -- refusals ------------------------------------------------------------------------------
@pytest.mark.parametrize("degeneracy", ["0", "-2", "1/2", "g", "sqrt(2)"])
def test_invalid_degeneracies_are_refused(degeneracy: str) -> None:
    r = run(
        system={"kind": "levels", "levels": [{"energy": "0", "degeneracy": degeneracy}]},
        action="partition",
    )
    assert r.status is Status.CONDITIONS_NOT_SATISFIED and r.reason == "invalid_degeneracy"
    assert not r.quantities


def test_energies_must_be_provably_real() -> None:
    r = run(system={"kind": "levels", "levels": [{"energy": "epsilon"}]}, action="partition")
    assert r.reason == "assumptions_insufficient"
    r = run(system={"kind": "levels", "levels": [{"energy": "I"}]}, action="partition")
    assert r.reason == "assumptions_insufficient"


def test_energies_may_not_depend_on_the_temperature() -> None:
    r = run(system={"kind": "levels", "levels": [{"energy": "beta"}]}, action="partition")
    assert r.reason == "unsupported_system" and r.status is Status.UNSUPPORTED


def test_partial_with_the_wrong_held_fixed_set_is_refused() -> None:
    wrong = {"of": "F", "variable": "V", "held_fixed": []}  # T must be held fixed
    r = run(
        system={"kind": "levels", "levels": [{"energy": "1/V"}]},
        parameters=["V"],
        assumptions={"V": ["positive"]},
        action="derive",
        quantities=["U"],
        partials=[wrong],
    )
    assert r.reason == "ensemble_mismatch" and "fixed" in r.message
    other = {
        "of": "F",
        "variable": "S",
        "held_fixed": ["V"],
    }  # not a state variable of this ensemble
    r = run(
        system={"kind": "levels", "levels": [{"energy": "1/V"}]},
        parameters=["V"],
        assumptions={"V": ["positive"]},
        action="derive",
        partials=[other],
    )
    assert r.reason == "ensemble_mismatch"


def test_unknown_observable_and_nonlinear_coupling_are_refused() -> None:
    levels = [
        {"energy": "-h**2", "observables": {"M": "1"}},
        {"energy": "h**2", "observables": {"M": "-1"}},
    ]
    r = run(
        system={"kind": "levels", "levels": levels},
        parameters=["h"],
        assumptions={"h": ["real"]},
        action="expectation",
        observable="N",
    )
    assert r.reason == "unknown_observable"
    r = run(
        system={"kind": "levels", "levels": levels},
        parameters=["h"],
        assumptions={"h": ["real"]},
        action="expectation",
        observable="M",
        coupling={"parameter": "h", "observable": "M"},
    )
    assert r.reason == "coupling_not_linear"  # dE/dh = 2h is not -M


def test_a_wrong_formula_is_never_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    original = canonical.Canonical.shown

    def off_by_a_billionth(self: canonical.Canonical, expr: sp.Expr) -> sp.Expr:
        return original(self, expr) * (1 + sp.Rational(1, 10**9))

    monkeypatch.setattr(canonical.Canonical, "shown", off_by_a_billionth)
    r = two_level(action="derive", quantities=["U"])
    assert r.status is Status.NOT_VERIFIED and r.reason == "verification_failed"
    assert not r.quantities and not r.steps


def test_every_step_and_formula_is_renderable_latex() -> None:
    """A control word glued to a letter (\\quadE) is not a command: the UI shows it as marked source."""
    import re

    for result in (
        two_level(action="derive", quantities=["Z", "U", "S", "Var_E"]),
        two_level(action="partition"),
        two_level(action="expectation", observable="E", statistic="variance"),
    ):
        texts = [s.latex for s in result.steps] + [q.latex for q in result.quantities]
        assert texts
        for latex in texts:
            assert not re.search(r"\\(?:q?quad|,|;|:|!)?(?:quad)[A-Za-z]", latex), latex
            assert latex.count("{") == latex.count("}"), latex
            assert latex.count(r"\left") == latex.count(r"\right"), latex


def test_every_reason_maps_to_a_status() -> None:
    """A reason without a status would crash the refusal itself instead of reporting it."""
    from typing import get_args

    from minerva_engine.statmech.models import Reason
    from minerva_engine.statmech.operation import _STATUS_FOR_REASON

    assert set(get_args(Reason)) == set(_STATUS_FOR_REASON)
    assert all(status is not Status.OK for status in _STATUS_FOR_REASON.values())


def test_diagnostics_hold_no_expressions() -> None:
    r = two_level(action="derive", quantities=["U"])
    dumped = json.dumps(r.diagnostics.model_dump())
    assert "epsilon" not in dumped and "exp" not in dumped
    assert {e.stage for e in r.diagnostics.events} >= {"parse", "symbolic", "verify"}


def test_analysis_separates_applicable_from_implemented() -> None:
    r = two_level(action="analyze")
    assert r.status is Status.OK
    by_id = {m.id: m for m in r.methods}
    assert by_id["closed_form_sum"].implemented and by_id["closed_form_sum"].applies
    assert not by_id["transfer_matrix"].applies
    assert r.not_supported


# -- properties ----------------------------------------------------------------------------
PROFILE = settings(
    max_examples=12,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)


@st.composite
def spectra(draw: st.DrawFn) -> list[tuple[int, int]]:
    n = draw(st.integers(min_value=2, max_value=5))
    return [(draw(st.integers(-6, 6)), draw(st.integers(1, 4))) for _ in range(n)]


@PROFILE
@given(spectra())
def test_derivatives_of_ln_z_agree_with_direct_moments(spectrum: list[tuple[int, int]]) -> None:
    levels = [{"energy": f"{e}/2", "degeneracy": str(g)} for e, g in spectrum]
    r = run(
        system={"kind": "levels", "levels": levels},
        action="derive",
        quantities=["Z", "U", "Var_E", "S"],
    )
    assert r.status is Status.OK, r.message
    b, kB = 0.8, 1.0
    mp.mp.dps = 30
    weights = [g * mp.exp(-b * mp.mpf(e) / 2) for e, g in spectrum]
    z = sum(weights)
    u = sum(w * mp.mpf(e) / 2 for w, (e, _) in zip(weights, spectrum, strict=True)) / z
    var = sum(w * (mp.mpf(e) / 2 - u) ** 2 for w, (e, _) in zip(weights, spectrum, strict=True)) / z
    s = kB * sum(-(w / z) * mp.log(w / z / g) for w, (_, g) in zip(weights, spectrum, strict=True))
    assert value(r, "Z", beta=b) == pytest.approx(float(z))
    assert value(r, "U", beta=b) == pytest.approx(float(u), abs=1e-12)
    assert value(r, "Var_E", beta=b) == pytest.approx(float(var), abs=1e-12)
    assert value(r, "S", beta=b, k_B=kB) == pytest.approx(float(s), abs=1e-12)
    assert float(var) >= 0
