"""The integration strategy engine: detection on its own, each technique, refusals, agreement.

Detection is a pure function of the expression tree and is tested without any model or engine call.
Every result is compared with an independent reference: mpmath quadrature for a definite value, a
central difference for an antiderivative.
"""

from __future__ import annotations

from typing import Any

import mpmath as mp
import pytest
import sympy as sp
from minerva_engine import Engine, ExpressionInput
from minerva_engine.integration_strategy.detect import detect, frullani_pair, schwinger_factor
from minerva_engine.integration_strategy.models import StrategyReport
from minerva_engine.models import Status
from minerva_engine.textparse import parse_text

x = sp.Symbol("x", real=True)
a = sp.Symbol("a", positive=True)
b = sp.Symbol("b", positive=True)
n = sp.Symbol("n", positive=True)
oo = sp.oo
REAL = {"x": ["real"]}
POS = {"x": ["real"], "a": ["positive"], "b": ["positive"], "n": ["positive"]}


def techniques(f: sp.Expr, lo: Any = None, hi: Any = None) -> list[str]:
    lower = None if lo is None else sp.sympify(lo)
    upper = None if hi is None else sp.sympify(hi)
    return [s.technique for s in detect(f, x, lower, upper)]


def integrate(
    engine: Engine,
    src: str,
    lo: str | None = None,
    hi: str | None = None,
    strategy: str | None = None,
    assumptions: dict[str, list[str]] | None = None,
):  # type: ignore[no-untyped-def]
    data = ExpressionInput(input=src, format="text", assumptions=assumptions or REAL)
    return engine.integrate(data, "x", lo, hi, strategy)


def value_of(result: Any, assumptions: dict[str, list[str]]) -> sp.Expr:
    try:
        return parse_text(result.result_text, assumptions=assumptions)
    except Exception:  # a special function (erf, erfc) that the strict parser does not read back
        return sp.Float(result.numeric, 20)


def quad(src: str, lo: Any, hi: Any, **values: float) -> Any:
    mp.mp.dps = 25
    f = parse_text(src, assumptions=POS)
    fn = sp.lambdify([x, *sorted(values)], f, "mpmath")
    args = [mp.mpf(values[k]) for k in sorted(values)]

    def bound(text: str) -> Any:
        return {"oo": mp.inf, "-oo": -mp.inf}.get(text) or mp.mpf(str(sp.N(parse_text(text), 25)))

    lo_, hi_ = bound(lo), bound(hi)
    return mp.quad(lambda t: fn(t, *args), [lo_, 0, hi_] if lo_ < 0 < hi_ else [lo_, hi_])


def is_antiderivative(result: Any, src: str, assumptions: dict[str, list[str]]) -> bool:
    mp.mp.dps = 30
    F = sp.lambdify(x, value_of(result, assumptions), "mpmath")
    f = sp.lambdify(x, parse_text(src, assumptions=assumptions), "mpmath")
    h = mp.mpf("1e-10")
    return all(
        abs((F(t + h) - F(t - h)) / (2 * h) - f(t)) < 1e-7 * max(1, abs(f(t)))
        for t in (mp.mpf("0.31"), mp.mpf("0.77"), mp.mpf("1.9"))
    )


# -- detection, independent of any engine call or model --------------------------------------------
@pytest.mark.parametrize(
    ("f", "lo", "hi", "expected"),
    [
        (x / (1 + x**2) ** 2, None, None, "substitution"),
        (sp.cos(x) * sp.exp(sp.sin(x)), None, None, "substitution"),
        (x * sp.exp(x), None, None, "parts"),
        (x**2 * sp.log(x), None, None, "parts"),
        (sp.sin(x) ** 2 * sp.cos(x) ** 2, None, None, "trigonometric"),
        (1 / ((x + 1) ** 2 * (x**2 + 1)), None, None, "partial_fractions"),
        (x**3 * sp.exp(-(x**2)), -oo, oo, "parity"),
        (sp.exp(-(x**2)), -1, 1, "parity"),
        (sp.sin(x) ** n / (sp.sin(x) ** n + sp.cos(x) ** n), 0, sp.pi / 2, "reflection"),
        ((sp.exp(-a * x) - sp.exp(-b * x)) / x, 0, oo, "frullani"),
        (1 / (x**2 + a**2) ** 2, 0, oo, "schwinger"),
        (sp.exp(-a * x**2), -oo, oo, "parameter_differentiation"),
        (sp.exp(x**x), 0, 1, "numerical"),
    ],
)
def test_detection_names_the_promising_technique(
    f: sp.Expr, lo: Any, hi: Any, expected: str
) -> None:
    assert expected in techniques(f, lo, hi)


@pytest.mark.parametrize(
    ("f", "lo", "hi", "absent"),
    [
        (x**2, None, None, ["substitution", "parts", "trigonometric", "partial_fractions"]),
        (sp.exp(-x), 0, oo, ["parity", "reflection", "frullani", "schwinger"]),
        (sp.exp(-x), 0, 1, ["parity", "frullani"]),  # not a symmetric interval, not 0 to oo pair
        (x * sp.exp(-x), 0, oo, ["frullani", "schwinger", "parameter_differentiation"]),
        (1 / x, None, None, ["numerical", "parity", "reflection", "frullani"]),  # indefinite
    ],
)
def test_detection_does_not_suggest_what_cannot_apply(
    f: sp.Expr, lo: Any, hi: Any, absent: list[str]
) -> None:
    got = techniques(f, lo, hi)
    assert not set(absent) & set(got)


def test_detection_is_deterministic_and_pure() -> None:
    f = (sp.exp(-a * x) - sp.exp(-b * x)) / x
    first = detect(f, x, sp.Integer(0), oo)
    assert first == detect(f, x, sp.Integer(0), oo)
    assert next(s.why for s in first) == "difference_at_two_scalings_over_x"


def test_structural_matchers() -> None:
    f = (sp.exp(-a * x) - sp.exp(-b * x)) / x
    g, c1, c2 = frullani_pair(f, x) or (None,) * 3
    assert {c1, c2} == {a, b}
    assert (
        sp.simplify(g.subs(x, c1 * x) - g.subs(x, c2 * x) - f * x) == 0
    )  # x f = g(c1 x) - g(c2 x)
    assert frullani_pair(sp.exp(-a * x) / x, x) is None
    assert frullani_pair((sp.exp(-a * x) - sp.sin(x)) / x, x) is None
    rest, base, power = schwinger_factor(sp.exp(-(x**2)) / (1 + x**2), x) or (None,) * 3
    assert base == 1 + x**2 and power == 1 and rest == sp.exp(-(x**2))
    assert schwinger_factor(1 / x, x) is None  # a bare power of x is not a Schwinger factor


# -- the techniques, each result against an independent reference ------------------------------------
@pytest.mark.parametrize(
    ("src", "strategy"),
    [
        ("x/(1+x**2)**2", "substitution"),
        ("cos(x)*exp(sin(x))", "substitution"),
        ("x*exp(x)", "parts"),
        ("x**2*sin(x)", "parts"),
        ("sin(x)**2*cos(x)**2", "trigonometric"),
        ("cos(x)**4", "trigonometric"),
        ("1/(x**2-1)", "partial_fractions"),
        ("1/((x+1)**2*(x**2+1))", "partial_fractions"),
    ],
)
def test_antiderivative_techniques_name_themselves_and_verify(
    engine: Engine, src: str, strategy: str
) -> None:
    r = integrate(engine, src, strategy=strategy)
    assert r.status is Status.OK and r.strategy is not None
    assert r.strategy.technique == strategy and r.strategy.verification in (
        "proved",
        "numerically_checked",
    )
    assert r.strategy.steps and r.strategy.attempts[-1].outcome == "verified"
    assert is_antiderivative(r, src, REAL)


def test_the_substitution_example_shows_its_transformation(engine: Engine) -> None:
    r = integrate(engine, "x/(1+x**2)**2", strategy="substitution")
    assert r.strategy is not None
    steps = {s.what: s.latex for s in r.strategy.steps}
    assert steps["substitution"] == "u = x^{2} + 1"
    assert steps["differential"] == "\\mathrm{d}u = \\left(2 x\\right)\\,\\mathrm{d}x"
    assert steps["transformed integral"] == "\\int \\frac{1}{2 u^{2}}\\,\\mathrm{d}u"
    assert r.result_text == "-1/(2*(x**2 + 1))"


@pytest.mark.parametrize(
    ("src", "lo", "hi", "strategy", "assumptions", "values"),
    [
        ("x**3*exp(-x**2)", "-oo", "oo", "parity", REAL, {}),
        ("exp(-x**2)", "-oo", "oo", "parity", REAL, {}),
        ("x**2*exp(-x**2)", "-1", "1", "parity", REAL, {}),
        ("sin(x)**n/(sin(x)**n+cos(x)**n)", "0", "pi/2", "reflection", POS, {"n": 1.7}),
        ("(exp(-a*x)-exp(-b*x))/x", "0", "oo", "frullani", POS, {"a": 0.7, "b": 2.1}),
        ("(atan(a*x)-atan(b*x))/x", "0", "oo", "frullani", POS, {"a": 0.7, "b": 2.1}),
        ("1/(x**2+a**2)**2", "0", "oo", "schwinger", POS, {"a": 1.3}),
        ("exp(-x**2)/(1+x**2)", "-oo", "oo", "schwinger", REAL, {}),
        ("exp(-a*x**2)", "-oo", "oo", "parameter_differentiation", POS, {"a": 1.3}),
    ],
)
def test_definite_techniques_agree_with_quadrature(
    engine: Engine,
    src: str,
    lo: str,
    hi: str,
    strategy: str,
    assumptions: Any,
    values: dict[str, float],
) -> None:
    r = integrate(engine, src, lo, hi, strategy, assumptions)
    assert r.status is Status.OK and r.strategy and r.strategy.technique == strategy
    assert r.strategy.verification == "numerically_checked"  # a definite value is never "proved"
    value = value_of(r, assumptions)
    assert not value.free_symbols or all(k in values for k in (s.name for s in value.free_symbols))
    got = sp.lambdify(sorted(value.free_symbols, key=lambda s: s.name), value, "mpmath")(
        *[mp.mpf(values[s.name]) for s in sorted(value.free_symbols, key=lambda s: s.name)]
    )
    want = quad(src, lo, hi, **values)
    assert abs(got - want) < 1e-8 * max(1, abs(want))


def test_the_classic_king_integral_is_pi_over_four(engine: Engine) -> None:
    for power in ("1", "2", "3"):
        r = integrate(
            engine, f"sin(x)**{power}/(sin(x)**{power}+cos(x)**{power})", "0", "pi/2", "reflection"
        )
        assert r.status is Status.OK and r.result_text == "pi/4"


def test_frullani_value_is_the_log_ratio(engine: Engine) -> None:
    r = integrate(engine, "(exp(-a*x)-exp(-b*x))/x", "0", "oo", "frullani", POS)
    assert r.strategy is not None
    assert sp.simplify(value_of(r, POS) - sp.log(b / a)) == 0
    assert any("a > 0" in c or "> 0" in c for c in r.strategy.conditions_latex)


# -- conditions that are not established: nothing is applied -----------------------------------------
def test_frullani_needs_declared_positive_scales(engine: Engine) -> None:
    r = integrate(engine, "(exp(-a*x)-exp(-b*x))/x", "0", "oo", "frullani", REAL)
    assert r.status is Status.UNSUPPORTED and r.result_latex is None
    assert r.strategy is not None
    assert ("frullani", "blocked", "scales_not_provably_positive") in [
        (t.technique, t.outcome, t.detail) for t in r.strategy.attempts
    ]


def test_frullani_refuses_a_limit_that_does_not_exist(engine: Engine) -> None:
    r = integrate(engine, "(cos(a*x)-cos(b*x))/x", "0", "oo", "frullani", POS)
    assert r.status is not Status.OK and r.result_latex is None
    assert r.strategy is not None
    assert any(t.technique == "frullani" and t.outcome == "blocked" for t in r.strategy.attempts)


def test_symmetry_is_not_applied_when_the_half_range_diverges(engine: Engine) -> None:
    r = integrate(engine, "1/x", "-1", "1", "parity")
    assert r.status is not Status.OK and r.result_text is None  # never 0, never a principal value
    assert r.strategy is not None
    assert ("parity", "blocked", "convergence_not_established") in [
        (t.technique, t.outcome, t.detail) for t in r.strategy.attempts
    ]
    cubic = integrate(engine, "1/x**3", "-oo", "oo", "parity")
    assert cubic.status is not Status.OK and cubic.result_text is None


def test_schwinger_needs_a_provably_positive_base_and_a_nonnegative_rest(engine: Engine) -> None:
    undeclared = integrate(engine, "1/(x**2+a**2)**2", "0", "oo", "schwinger", REAL)
    assert undeclared.strategy is not None
    assert any(
        t.technique == "schwinger" and t.outcome == "blocked" for t in undeclared.strategy.attempts
    )
    # a sign-changing numerator: swapping the two integrals is not justified by Tonelli
    signed = integrate(engine, "sin(x)/(x**2+a**2)**2", "0", "oo", "schwinger", POS)
    assert signed.strategy is not None
    assert ("schwinger", "blocked", "interchange_not_justified") in [
        (t.technique, t.outcome, t.detail) for t in signed.strategy.attempts
    ]
    assert signed.result_text is None or signed.strategy.technique != "schwinger"


def test_reflection_is_not_suggested_when_the_average_is_not_simpler() -> None:
    assert "reflection" not in techniques(x**2 + 1, 0, 1)
    assert "reflection" in techniques(sp.sin(x) / (sp.sin(x) + sp.cos(x)), 0, sp.pi / 2)


# -- a strategy that fails, and strategies that agree -------------------------------------------------
def test_a_requested_strategy_that_does_not_apply_falls_back_to_the_direct_route(
    engine: Engine,
) -> None:
    r = integrate(engine, "1/(1+x**2)", strategy="parts")
    assert r.status is Status.OK and r.result_text == "atan(x)"
    assert r.strategy is not None and r.strategy.technique == "direct"
    assert (
        r.strategy.attempts[0].technique == "parts" and r.strategy.attempts[0].outcome != "verified"
    )


def test_two_techniques_give_antiderivatives_that_differ_by_a_constant(engine: Engine) -> None:
    src = "sin(x)*cos(x)"
    first = integrate(engine, src, strategy="substitution")
    second = integrate(engine, src, strategy="trigonometric")
    assert first.status is second.status is Status.OK
    assert is_antiderivative(first, src, REAL) and is_antiderivative(second, src, REAL)
    f1 = sp.lambdify(x, value_of(first, REAL), "mpmath")
    f2 = sp.lambdify(x, value_of(second, REAL), "mpmath")
    offsets = [f1(t) - f2(t) for t in (mp.mpf("0.2"), mp.mpf("0.9"), mp.mpf("1.4"))]
    assert max(offsets) - min(offsets) < 1e-12


# -- numerical fallback ------------------------------------------------------------------------------
def test_numerical_estimate_is_offered_but_is_not_a_result_unless_asked(engine: Engine) -> None:
    default = integrate(engine, "exp(x**x)", "0", "1")
    assert default.status is Status.UNSUPPORTED  # unchanged: no closed form
    assert default.result_kind == "numeric" and default.numeric and default.numeric_error
    assert default.result_latex is None
    asked = integrate(engine, "exp(x**x)", "0", "1", "numerical")
    assert asked.status is Status.OK and asked.result_kind == "numeric"
    assert asked.strategy is not None and asked.strategy.verification == "numeric_estimate"
    assert abs(float(asked.numeric or 0) - float(quad("exp(x**x)", "0", "1"))) < 1e-9


def test_divergent_and_unreliable_integrals_give_no_number(engine: Engine) -> None:
    for src, lo, hi in (("1/x", "0", "1"), ("1/x**2", "-1", "1"), ("exp(x)", "0", "oo")):
        r = integrate(engine, src, lo, hi, "numerical")
        assert r.status is not Status.OK and r.numeric is None, src


def test_numerical_is_refused_for_an_integral_with_free_symbols(engine: Engine) -> None:
    r = integrate(engine, "exp(-a*x**2)*cos(x**3)", "0", "oo", "numerical", POS)
    assert r.numeric is None and r.result_kind == "closed_form"
    assert r.strategy is not None
    assert ("numerical", "not_applicable") in [
        (t.technique, t.outcome) for t in r.strategy.attempts
    ]


# -- non-elementary, domain-sensitive and regression cases -------------------------------------------
def test_a_non_elementary_antiderivative_is_not_invented(engine: Engine) -> None:
    r = integrate(engine, "exp(x**x)", strategy="auto")
    assert r.status is not Status.OK and r.result_latex is None


def test_a_technique_that_does_not_apply_says_so_and_the_direct_result_keeps_its_form(
    engine: Engine,
) -> None:
    r = integrate(engine, "x**n", strategy="substitution", assumptions=POS)
    assert r.status is Status.OK and r.result_text == "x**(n + 1)/(n + 1)"
    assert r.strategy is not None and r.strategy.technique == "direct"
    assert ("substitution", "not_applicable") in [
        (t.technique, t.outcome) for t in r.strategy.attempts
    ]


@pytest.mark.parametrize(
    ("src", "lo", "hi", "text"),
    [
        ("x*cos(x)", None, None, "x*sin(x) + cos(x)"),
        ("sin(x)", "0", "pi", "2"),
        ("exp(-x)", "0", "oo", "1"),
        ("log(x)", "0", "1", "-1"),
        ("exp(-x**2)", None, None, "sqrt(pi)*erf(x)/2"),
    ],
)
def test_direct_results_are_unchanged_and_say_so(
    engine: Engine, src: str, lo: Any, hi: Any, text: str
) -> None:
    r = integrate(engine, src, lo, hi)
    assert r.status is Status.OK and r.result_text == text
    assert r.strategy is not None and r.strategy.technique == "direct"
    assert r.result_kind == "closed_form"


def test_an_unknown_strategy_name_is_refused(engine: Engine) -> None:
    assert integrate(engine, "x", strategy="magic").status is Status.PARSE_ERROR


def test_the_report_holds_fixed_vocabulary_not_input(engine: Engine) -> None:
    r = integrate(engine, "(exp(-a*x)-exp(-b*x))/x", "0", "oo", "frullani", REAL)
    assert r.strategy is not None
    dump = StrategyReport.model_validate(r.strategy.model_dump()).model_dump_json()
    assert "exp" not in "".join(a.detail for a in r.strategy.attempts)
    assert dump  # round-trips


def test_the_direct_route_cannot_starve_the_strategies(engine: Engine) -> None:
    # SymPy's own integrate of this does not finish within a short budget; the strategies still run.
    short = Engine(engine.limits.model_copy(update={"timeout_seconds": 20}))
    r = integrate(short, "sin(x)**n/(sin(x)**n+cos(x)**n)", "0", "pi/2", None, POS)
    assert r.status is Status.OK and r.result_text == "pi/4"
    assert r.strategy is not None and r.strategy.technique == "reflection"
