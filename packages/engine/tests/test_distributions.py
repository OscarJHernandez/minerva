"""Dirac delta and Heaviside calculus: analytic oracles first, regularised sequences as support."""

import time
from collections.abc import Callable
from typing import Any

import mpmath
import pytest
import sympy as sp
from minerva_engine import (
    DistributionInput,
    DistributionResult,
    Engine,
    ExpressionInput,
    Limits,
    Status,
)
from minerva_engine.distributions.pairing import verify_pairing

X = sp.Symbol("x", real=True)


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(Limits(timeout_seconds=30, max_concurrency=2))


def run(
    engine: Engine,
    source: str,
    action: str,
    *,
    lower: str | None = None,
    upper: str | None = None,
    endpoint: str | None = None,
    **assumptions: list[str],
) -> DistributionResult:
    declared = {"x": ["real"], **assumptions}
    return engine.distribution(
        DistributionInput(
            expression=ExpressionInput(input=source, format="text", assumptions=declared),
            action=action,  # type: ignore[arg-type]
            variable="x",
            lower=lower,
            upper=upper,
            endpoint=endpoint,  # type: ignore[arg-type]
        )
    )


def same(result: DistributionResult, expected: sp.Expr, **symbols: sp.Symbol) -> bool:
    """The result equals ``expected`` as a distribution (compared term by term, exactly)."""
    assert result.result_text is not None
    local = {"x": X, "DiracDelta": sp.DiracDelta, "Heaviside": sp.Heaviside, **symbols}
    got = sp.sympify(result.result_text, locals=local)  # tests only: never user input
    return sp.simplify(sp.expand(got - expected)) == 0


# --- delta scaling and translation -----------------------------------------------------------
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("DiracDelta(2*x - 6)", sp.DiracDelta(X - 3) / 2),
        ("DiracDelta(-3*x + 6)", sp.DiracDelta(X - 2) / 3),
        ("DiracDelta(x - 1) + DiracDelta(x + 1)", sp.DiracDelta(X - 1) + sp.DiracDelta(X + 1)),
        ("x*DiracDelta(x)", sp.Integer(0)),
        ("(x + 3)*DiracDelta(x - 1)", 4 * sp.DiracDelta(X - 1)),
        ("exp(x)*DiracDelta(x)", sp.DiracDelta(X)),
    ],
)
def test_scaling_translation_and_multipliers(
    engine: Engine, source: str, expected: sp.Expr
) -> None:
    result = run(engine, source, "simplify")
    assert result.status is Status.OK, result.message
    assert same(result, expected)
    assert result.equality == "distributional"
    assert result.verification is not None and result.verification.passed
    assert result.verification.conditions_established


def test_scaling_follows_the_assumption_ladder(engine: Engine) -> None:
    a = sp.Symbol("a", real=True, nonzero=True)
    undeclared = run(engine, "DiracDelta(a*x)", "simplify", a=["real"])
    assert undeclared.status is Status.CONDITIONS_NOT_SATISFIED
    assert undeclared.reason == "assumptions_insufficient"
    assert undeclared.result_latex is None
    assert undeclared.blocked and undeclared.blocked[0].entry_id == "distributions.delta_scaling"

    nonzero = run(engine, "DiracDelta(a*x)", "simplify", a=["real", "nonzero"])
    assert nonzero.status is Status.OK and "Abs(a)" in (nonzero.result_text or "")

    positive = run(engine, "DiracDelta(a*x)", "simplify", a=["positive"])
    assert same(
        positive, sp.DiracDelta(X) / sp.Symbol("a", positive=True), a=sp.Symbol("a", positive=True)
    )

    negative = run(engine, "DiracDelta(a*x)", "simplify", a=["negative"])
    assert same(
        negative, -sp.DiracDelta(X) / sp.Symbol("a", negative=True), a=sp.Symbol("a", negative=True)
    )
    del a


def test_a_complex_variable_is_refused(engine: Engine) -> None:
    result = engine.distribution(
        DistributionInput(
            expression=ExpressionInput(input="DiracDelta(x - 1)", format="text"),
            action="simplify",
            variable="x",
        )
    )
    assert result.status is Status.CONDITIONS_NOT_SATISFIED
    assert result.reason == "assumptions_insufficient"


# --- composite arguments ---------------------------------------------------------------------
def test_composite_argument_with_two_simple_roots(engine: Engine) -> None:
    result = run(engine, "DiracDelta(x**2 - 4)", "simplify")
    assert result.status is Status.OK
    assert same(result, (sp.DiracDelta(X - 2) + sp.DiracDelta(X + 2)) / 4)
    assert result.steps[0].entry_id == "distributions.delta_composite_simple_roots"
    assert all(c.established for step in result.steps for c in step.conditions)


def test_composite_argument_with_three_roots_and_a_multiplier(engine: Engine) -> None:
    result = run(engine, "(x**2 + 1)*DiracDelta(x**3 - x)", "simplify")
    assert result.status is Status.OK
    # g' = 3x^2 - 1: |g'(0)| = 1, |g'(+-1)| = 2; (x^2+1) = 1, 2, 2.
    assert same(result, sp.DiracDelta(X) + sp.DiracDelta(X - 1) + sp.DiracDelta(X + 1))


def test_parametric_roots_need_a_sign(engine: Engine) -> None:
    a = sp.Symbol("a", positive=True)
    ok = run(engine, "DiracDelta(x**2 - a**2)", "simplify", a=["positive"])
    assert ok.status is Status.OK
    assert same(ok, (sp.DiracDelta(X - a) + sp.DiracDelta(X + a)) / (2 * a), a=a)

    real_only = run(engine, "DiracDelta(x**2 - a**2)", "simplify", a=["real"])
    assert real_only.status is Status.CONDITIONS_NOT_SATISFIED  # a = 0 would be a double root
    assert real_only.result_latex is None


def test_a_polynomial_without_real_roots_has_no_delta(engine: Engine) -> None:
    result = run(engine, "DiracDelta(x**2 + 1) + 0*x", "simplify")
    assert result.status is Status.UNSUPPORTED  # SymPy already evaluates it to 0: nothing is left


# --- refusals: singular roots, unsupported forms, ambiguous products ---------------------------
@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("DiracDelta(x**2)", "singular_root"),
        ("DiracDelta((x - 1)**2*(x + 2))", "singular_root"),
        ("DiracDelta(sin(x))", "unsupported_distribution"),
        ("DiracDelta(exp(x) - 2)", "unsupported_distribution"),
        ("DiracDelta(x)**2", "unsupported_distribution"),
        ("DiracDelta(x)*DiracDelta(x - 1)", "unsupported_distribution"),
        ("DiracDelta(x)*Heaviside(x)", "unsupported_distribution"),
        ("DiracDelta(x)/x", "unsupported_distribution"),
        ("sin(DiracDelta(x))", "unsupported_distribution"),
        ("x + 1", "unsupported_distribution"),
    ],
)
def test_unsupported_and_singular_cases_are_refused(
    engine: Engine, source: str, reason: str
) -> None:
    result = run(engine, source, "simplify")
    assert result.status in (Status.UNSUPPORTED, Status.PARSE_ERROR), result.message
    assert result.reason == reason
    assert result.result_latex is None and result.result_text is None
    assert result.diagnostics.reason == reason
    assert result.diagnostics.events[-1].stage == "rejected"


def test_a_step_at_a_different_point_is_fine(engine: Engine) -> None:
    result = run(engine, "Heaviside(x - 3)*DiracDelta(x - 1)", "simplify")
    assert result.status is Status.OK
    assert result.result_text == "0"


# --- integration against deltas -------------------------------------------------------------
@pytest.mark.parametrize(
    ("source", "lower", "upper", "expected"),
    [
        ("x**2*DiracDelta(x - 3)", "-oo", "oo", sp.Integer(9)),
        ("x**2*DiracDelta(x**2 - 4)", "-oo", "oo", sp.Integer(2)),
        ("exp(x)*DiracDelta(x - 1)", "0", "3", sp.E),
        ("cos(x)*DiracDelta(2*x - 1)", "-oo", "oo", sp.cos(sp.Rational(1, 2)) / 2),
        ("sin(x)*DiracDelta(x**2 - 1)", "-oo", "oo", sp.Integer(0)),
        ("cos(x)*DiracDelta(x**2 - 1)", "-oo", "oo", sp.cos(1)),
        ("x**3*DiracDelta(x - 1)", "0", "5", sp.Integer(1)),
        ("DiracDelta(x - 1)", "2", "5", sp.Integer(0)),
        ("DiracDelta(x - 1)", "-oo", "0", sp.Integer(0)),
        ("DiracDelta(x - 1)", "-oo", "oo", sp.Integer(1)),
    ],
)
def test_integration_against_known_values(
    engine: Engine, source: str, lower: str, upper: str, expected: sp.Expr
) -> None:
    result = run(engine, source, "integrate", lower=lower, upper=upper)
    assert result.status is Status.OK, result.message
    assert result.equality == "value"
    assert result.result_text is not None
    assert sp.simplify(sp.sympify(result.result_text) - expected) == 0
    assert result.steps[-1].entry_id == "distributions.delta_sifting"
    assert result.verification is not None and result.verification.passed


def test_reversed_limits_flip_the_sign(engine: Engine) -> None:
    result = run(engine, "x*DiracDelta(x - 2)", "integrate", lower="5", upper="0")
    assert result.status is Status.OK
    assert result.result_text == "-2"


def test_integration_with_parameters_needs_the_position_decided(engine: Engine) -> None:
    decided = run(
        engine, "sin(x)*DiracDelta(x - a)", "integrate", lower="-1", upper="oo", a=["positive"]
    )
    assert decided.status is Status.OK and decided.result_text == "sin(a)"
    unknown = run(engine, "sin(x)*DiracDelta(x - a)", "integrate", lower="0", upper="1", a=["real"])
    assert unknown.status is Status.CONDITIONS_NOT_SATISFIED
    assert unknown.reason == "assumptions_insufficient"
    assert unknown.result_text is None


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [("half", "1/2"), ("include", "1"), ("exclude", "0")],
)
def test_a_delta_on_a_limit_needs_an_explicit_convention(
    engine: Engine, endpoint: str, expected: str
) -> None:
    ambiguous = run(engine, "DiracDelta(x - 1)", "integrate", lower="0", upper="1")
    assert ambiguous.status is Status.CONDITIONS_NOT_SATISFIED
    assert ambiguous.reason == "integration_domain_ambiguous"
    assert ambiguous.result_text is None

    chosen = run(engine, "DiracDelta(x - 1)", "integrate", lower="0", upper="1", endpoint=endpoint)
    assert chosen.status is Status.OK
    assert chosen.result_text == expected
    assert chosen.endpoint == endpoint
    assert any(endpoint in note for note in chosen.notes)


def test_a_convention_changes_nothing_for_a_delta_away_from_the_limits(engine: Engine) -> None:
    result = run(engine, "DiracDelta(x - 1)", "integrate", lower="0", upper="2", endpoint="exclude")
    assert result.result_text == "1"


@pytest.mark.parametrize(
    "source",
    ["DiracDelta(x) + x", "Heaviside(x)", "x**2"],
)
def test_integrands_that_mix_ordinary_terms_are_refused(engine: Engine, source: str) -> None:
    result = run(engine, source, "integrate", lower="-1", upper="1")
    assert result.status is Status.UNSUPPORTED
    assert result.result_text is None


def test_integration_needs_both_limits(engine: Engine) -> None:
    assert run(engine, "DiracDelta(x)", "integrate", lower="0").status is Status.UNSUPPORTED


# --- Heaviside derivatives -------------------------------------------------------------------
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Heaviside(x)", sp.DiracDelta(X)),
        ("Heaviside(x - 2)", sp.DiracDelta(X - 2)),
        ("Heaviside(-3*x)", -sp.DiracDelta(X)),
        ("Heaviside(2*x - 1)", sp.DiracDelta(X - sp.Rational(1, 2))),
        ("(x + 2)*Heaviside(x - 1)", sp.Heaviside(X - 1) + 3 * sp.DiracDelta(X - 1)),
        (
            "sin(x)*Heaviside(x) + x**2",
            sp.cos(X) * sp.Heaviside(X) + 2 * X,
        ),
        (
            "Heaviside(x) - Heaviside(x - 1)",
            sp.DiracDelta(X) - sp.DiracDelta(X - 1),
        ),
    ],
)
def test_heaviside_derivatives(engine: Engine, source: str, expected: sp.Expr) -> None:
    result = run(engine, source, "differentiate")
    assert result.status is Status.OK, result.message
    assert same(result, expected)
    assert result.steps[0].entry_id == "distributions.heaviside_derivative"
    assert result.equality == "distributional"
    assert result.verification is not None and result.verification.passed


def test_the_derivative_with_a_parameter_jump(engine: Engine) -> None:
    a = sp.Symbol("a", real=True)
    result = run(engine, "Heaviside(x - a)*exp(x)", "differentiate", a=["real"])
    assert result.status is Status.OK
    assert same(result, sp.exp(X) * sp.Heaviside(X - a) + sp.exp(a) * sp.DiracDelta(X - a), a=a)


@pytest.mark.parametrize(
    "source",
    [
        "Heaviside(x)**2",  # a power of a step
        "Heaviside(x)*Heaviside(x - 1)",
        "Heaviside(x**2 - 1)",  # not affine
        "Abs(x)*Heaviside(x)",  # the multiplier is not smooth
        "Heaviside(x)/x",
        "DiracDelta(x) + Heaviside(x)",  # would need the derivative of a delta
        "Abs(x)",  # a kink: no step function to differentiate here
    ],
)
def test_derivatives_outside_the_supported_form_are_refused(engine: Engine, source: str) -> None:
    result = run(engine, source, "differentiate")
    assert result.status is Status.UNSUPPORTED, result.message
    assert result.result_text is None


def test_a_step_slope_must_be_provably_nonzero(engine: Engine) -> None:
    result = run(engine, "Heaviside(a*x)", "differentiate", a=["real"])
    assert result.status is Status.CONDITIONS_NOT_SATISFIED
    assert result.reason == "assumptions_insufficient"
    ok = run(engine, "Heaviside(a*x)", "differentiate", a=["negative"])
    assert ok.status is Status.OK


# --- the verifier itself: wrong candidates must be rejected --------------------------------------
def pairing(
    original: sp.Expr,
    candidate: sp.Expr,
    action: str = "simplify",
    bounds: tuple[sp.Expr, sp.Expr] | None = None,
    endpoint: str | None = None,
) -> bool:
    return verify_pairing(
        original,
        candidate,
        X,
        action=action,  # type: ignore[arg-type]
        bounds=bounds,
        endpoint=endpoint,  # type: ignore[arg-type]
        deadline=time.monotonic() + 60,
    ).passed


def test_the_pairing_accepts_right_answers_and_rejects_wrong_ones() -> None:
    delta = sp.DiracDelta
    original = delta(X**2 - 4)
    right = (delta(X - 2) + delta(X + 2)) / 4
    assert pairing(original, right)
    assert not pairing(original, delta(X - 2) + delta(X + 2))  # forgot 1/|g'|
    assert not pairing(original, delta(X - 2) / 4)  # dropped a root
    assert not pairing(original, right * (1 + sp.Rational(1, 10**6)))  # a 1e-6 relative error
    assert not pairing(original, (delta(X - 2) + delta(X + 3)) / 4)  # moved a root
    assert not pairing(original, sp.Integer(0))  # the sample points alone cannot see this one


def test_the_pairing_catches_the_sign_of_a_scale_factor() -> None:
    a = sp.Symbol("a", negative=True)
    original = sp.DiracDelta(a * X)
    assert pairing(original, sp.DiracDelta(X) / sp.Abs(a))
    assert not pairing(original, sp.DiracDelta(X) / a)  # 1/a instead of 1/|a| for a < 0


def test_the_pairing_catches_a_wrong_jump_of_a_derivative() -> None:
    original = sp.Heaviside(X - 1)
    assert pairing(original, sp.DiracDelta(X - 1), "differentiate")
    assert not pairing(original, -sp.DiracDelta(X - 1), "differentiate")
    assert not pairing(original, 2 * sp.DiracDelta(X - 1), "differentiate")
    assert not pairing(original, sp.DiracDelta(X - 2), "differentiate")


def test_the_pairing_checks_integrals_and_endpoint_conventions() -> None:
    original = X**2 * sp.DiracDelta(X - 3)
    assert pairing(original, sp.Integer(9), "integrate", (sp.Integer(-5), sp.oo))
    assert not pairing(original, sp.Integer(3), "integrate", (sp.Integer(-5), sp.oo))
    assert not pairing(original, sp.Integer(9), "integrate", (sp.Integer(4), sp.oo))
    on_limit = sp.DiracDelta(X - 1)
    assert pairing(on_limit, sp.Rational(1, 2), "integrate", (sp.Integer(0), sp.Integer(1)), "half")
    assert not pairing(on_limit, sp.Integer(1), "integrate", (sp.Integer(0), sp.Integer(1)), "half")
    assert pairing(on_limit, sp.Integer(1), "integrate", (sp.Integer(0), sp.Integer(1)), "include")
    assert pairing(on_limit, sp.Integer(0), "integrate", (sp.Integer(0), sp.Integer(1)), "exclude")


# --- regularised delta sequences, computed independently of the module -----------------------
@pytest.mark.parametrize(
    ("g", "weights"),
    [
        (lambda t: 2 * t - 6, [(3, 0.5)]),
        (lambda t: t**2 - 4, [(2, 0.25), (-2, 0.25)]),
        (lambda t: t**3 - t, [(0, 1.0), (1, 0.5), (-1, 0.5)]),
    ],
)
def test_gaussian_sequences_approach_the_analytic_action(
    g: Callable[[Any], Any], weights: list[tuple[float, float]]
) -> None:
    """Supporting evidence only: tanh-sinh quadrature of a narrow Gaussian in g(x)."""
    mpmath.mp.dps = 25

    def phi(t: Any) -> Any:
        return mpmath.exp(-((t - 0.3) ** 2) / 4) * (1 + t)

    def regularised(eps: Any) -> Callable[[Any], Any]:
        def integrand(t: Any) -> Any:
            return mpmath.exp(-((g(t) / eps) ** 2)) / (eps * mpmath.sqrt(mpmath.pi)) * phi(t)

        return integrand

    exact = sum(w * phi(mpmath.mpf(r)) for r, w in weights)
    marks = sorted({-6, *[r + d for r, _ in weights for d in (-0.5, -0.1, 0, 0.1, 0.5)], 6})
    errors = [
        abs(mpmath.quad(regularised(mpmath.mpf(e)), marks) - exact)
        for e in ("0.04", "0.02", "0.01")
    ]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-3


# --- diagnostics, safety and limits ------------------------------------------------------------
def test_successful_runs_record_every_stage_without_the_expression(engine: Engine) -> None:
    marker = "marker_param_xyzzy"
    result = run(engine, f"{marker}*DiracDelta(x - 1)", "simplify", **{marker: ["positive"]})
    assert result.status is Status.OK
    stages = [e.stage for e in result.diagnostics.events]
    assert stages == [
        "parsed",
        "identities_identified",
        "transformation_selected",
        "assumptions_evaluated",
        "verification_attempted",
        "accepted",
        "completed",
    ]
    assert all(e.status == "ok" and e.duration_ms >= 0 for e in result.diagnostics.events)
    dumped = result.diagnostics.model_dump_json()
    assert marker not in dumped and "DiracDelta" not in dumped


def test_failed_runs_end_with_a_rejection_and_a_reason(engine: Engine) -> None:
    result = run(engine, "DiracDelta(x**2)", "simplify")
    last = result.diagnostics.events[-1]
    assert last.stage == "rejected" and last.status == "failed" and last.reason == "singular_root"


def test_hostile_and_oversized_inputs_are_stopped(engine: Engine) -> None:
    big = "+".join(f"DiracDelta(x - {i})*x**{i % 7}" for i in range(400))
    assert run(engine, big, "simplify").status in (Status.RESOURCE_LIMIT, Status.PARSE_ERROR)
    huge_power = run(engine, "DiracDelta(x)*x**(10**9)", "simplify")
    assert huge_power.status in (Status.RESOURCE_LIMIT, Status.PARSE_ERROR)
    high_degree = run(engine, "DiracDelta(x**9 - 1)", "simplify")
    assert high_degree.status is Status.RESOURCE_LIMIT
    assert high_degree.reason == "resource_limit"
    assert run(engine, "DiracDelta(x.__class__)", "simplify").status is Status.PARSE_ERROR


def test_a_small_time_budget_is_reported_as_a_resource_limit() -> None:
    tiny = Engine(Limits(timeout_seconds=0.05, max_concurrency=1))
    result = run(tiny, "DiracDelta(x**3 - x)", "simplify")
    assert result.status in (Status.RESOURCE_LIMIT, Status.OK)
    if result.status is Status.RESOURCE_LIMIT:
        assert result.result_latex is None


def test_results_read_back_through_the_strict_parser(engine: Engine) -> None:
    from minerva_engine import parse_latex, parse_text

    result = run(engine, "(x + 2)*Heaviside(x - 1)", "differentiate")
    assert result.result_latex and result.result_text
    again = parse_text(result.result_text, {"x": ["real"]})
    assert sp.simplify(again - parse_latex(result.result_latex, {"x": ["real"]})) == 0
