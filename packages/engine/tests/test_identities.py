import pytest
from minerva_engine import (
    Applicability,
    Engine,
    ExpressionInput,
    Rule,
    RuleCondition,
    verify_rule,
)

PYTH = Rule(id="trig.pythagorean", lhs="sin(a)**2 + cos(a)**2", rhs="1", variables=["a"])
LOG_PRODUCT = Rule(
    id="log.product",
    lhs="log(a*b)",
    rhs="log(a) + log(b)",
    variables=["a", "b"],
    conditions=[
        RuleCondition(variable="a", assumption="positive"),
        RuleCondition(variable="b", assumption="positive"),
    ],
)
BAD = Rule(id="bogus.sin_is_cos", lhs="sin(a)", rhs="cos(a)", variables=["a"])


def apply(engine: Engine, source: str, rules: list[Rule], **assumptions: list[str]):
    expr = ExpressionInput(input=source, format="latex", assumptions=assumptions)
    return {a.rule_id: a for a in engine.apply_rules(expr, rules).applications}


def test_unconditional_rule_applies_to_subexpression(engine: Engine) -> None:
    out = apply(engine, r"\sin^2 t + \cos^2 t + z", [PYTH])
    assert out["trig.pythagorean"].applicability is Applicability.APPLICABLE
    assert out["trig.pythagorean"].rewritten_latex == "z + 1"
    assert out["trig.pythagorean"].verification.passed  # type: ignore[union-attr]


def test_conditional_rule_blocked_without_assumptions(engine: Engine) -> None:
    out = apply(engine, r"\log(xy)", [LOG_PRODUCT])
    app = out["log.product"]
    assert app.applicability is Applicability.CONDITIONS_NOT_SATISFIED
    assert app.rewritten_latex is None
    assert {c.variable for c in app.unmet_conditions} == {"a", "b"}


def test_conditional_rule_applies_when_conditions_provable(engine: Engine) -> None:
    out = apply(engine, r"\log(xy)", [LOG_PRODUCT], x=["positive"], y=["positive"])
    app = out["log.product"]
    assert app.applicability is Applicability.APPLICABLE
    assert app.rewritten_latex is not None
    assert "\\log{\\left(x \\right)}" in app.rewritten_latex


def test_partial_assumptions_still_blocked(engine: Engine) -> None:
    out = apply(engine, r"\log(xy)", [LOG_PRODUCT], x=["positive"])
    assert out["log.product"].applicability is Applicability.CONDITIONS_NOT_SATISFIED


def test_no_match_is_reported(engine: Engine) -> None:
    out = apply(engine, r"x + 1", [PYTH, LOG_PRODUCT])
    assert all(a.applicability is Applicability.NO_MATCH for a in out.values())


def test_a_false_rule_is_never_applied(engine: Engine) -> None:
    out = apply(engine, r"\sin x", [BAD])
    assert out["bogus.sin_is_cos"].applicability is Applicability.NOT_VERIFIED
    assert out["bogus.sin_is_cos"].rewritten_latex is None


def test_verify_rule_accepts_true_and_rejects_false() -> None:
    assert verify_rule(PYTH).passed
    assert verify_rule(LOG_PRODUCT).passed
    assert not verify_rule(BAD).passed


def test_rule_with_undeclared_variable_is_rejected(engine: Engine) -> None:
    bad = Rule(id="bad", lhs="sin(a)*q", rhs="1", variables=["a"])
    out = apply(engine, r"\sin x", [bad])
    assert out["bad"].applicability is Applicability.NO_MATCH
    assert "could not be evaluated" in out["bad"].message


@pytest.mark.parametrize("expr", [r"\sin^2(2x)+\cos^2(2x)", r"\sin^2 (y z)+\cos^2 (y z)"])
def test_pattern_variable_binds_to_compound_arguments(engine: Engine, expr: str) -> None:
    assert apply(engine, expr, [PYTH])["trig.pythagorean"].rewritten_latex == "1"


def test_matches_that_invent_roots_are_discarded(engine: Engine) -> None:
    diff_squares = Rule(
        id="alg.dos", lhs="a**2 - b**2", rhs="(a - b)*(a + b)", variables=["a", "b"]
    )
    out = apply(engine, "x + y", [diff_squares])
    assert out["alg.dos"].applicability is Applicability.NO_MATCH
    out = apply(engine, "x^2 - y^2", [diff_squares])
    assert out["alg.dos"].applicability is Applicability.APPLICABLE


def test_trivial_instantiations_are_not_reported(engine: Engine) -> None:
    power = Rule(
        id="log.power",
        lhs="log(a**b)",
        rhs="b*log(a)",
        variables=["a", "b"],
        conditions=[
            RuleCondition(variable="a", assumption="positive"),
            RuleCondition(variable="b", assumption="real"),
        ],
    )
    out = apply(engine, r"\log(xy)", [power], x=["positive"], y=["positive"])
    assert out["log.power"].applicability is Applicability.NO_MATCH  # would be a=xy, b=1
    out = apply(engine, r"\log(x^3)", [power], x=["positive"])
    assert out["log.power"].applicability is Applicability.APPLICABLE


def test_verification_catches_a_rule_missing_its_conditions(engine: Engine) -> None:
    # log(x^3) = 3 log(x) is false for complex x; a rule that forgets its conditions is
    # still stopped by the numeric verification and never shown as a result.
    sloppy = Rule(id="log.power.sloppy", lhs="log(a**b)", rhs="b*log(a)", variables=["a", "b"])
    out = apply(engine, r"\log(x^3)", [sloppy])
    assert out["log.power.sloppy"].applicability is Applicability.NOT_VERIFIED
    assert out["log.power.sloppy"].rewritten_latex is None


def test_matches_that_invent_reciprocals_or_imaginary_units_are_discarded(engine: Engine) -> None:
    quotient = Rule(id="log.q", lhs="log(a/b)", rhs="log(a) - log(b)", variables=["a", "b"])
    out = apply(engine, r"\log(xy)", [quotient])
    assert out["log.q"].applicability is Applicability.NO_MATCH  # would need b = 1/y
    out = apply(engine, r"\log(\frac{x}{y})", [quotient])
    assert out["log.q"].rewritten_latex is not None or out["log.q"].applicability in (
        Applicability.NOT_VERIFIED,
        Applicability.APPLICABLE,
    )
    dos = Rule(id="dos", lhs="a**2 - b**2", rhs="(a - b)*(a + b)", variables=["a", "b"])
    assert apply(engine, "x^2 + y^2", [dos])["dos"].applicability is Applicability.NO_MATCH


def test_time_budget_returns_partial_results(engine: Engine) -> None:
    from minerva_engine import Limits

    tiny = Engine(Limits(timeout_seconds=0.001 + 0.0))
    # With a near-zero budget every rule is reported as skipped, never as a crash or fake match.
    out = {
        a.rule_id: a
        for a in tiny.apply_rules(
            ExpressionInput(input=r"\sin^2 t + \cos^2 t"), [PYTH, LOG_PRODUCT]
        ).applications
    }
    assert all(a.applicability is not Applicability.APPLICABLE for a in out.values()) or out == {}


# --- rules whose pattern is an integer multiple of a variable, e.g. sin(2a) ---------------------
DOUBLE_SIN = Rule(id="trig.double_sin", lhs="sin(2*a)", rhs="2*sin(a)*cos(a)", variables=["a"])
DOUBLE_COS = Rule(
    id="trig.double_cos", lhs="cos(2*a)", rhs="cos(a)**2 - sin(a)**2", variables=["a"]
)
SUM_SIN = Rule(
    id="trig.sum_sin", lhs="sin(a + b)", rhs="sin(a)*cos(b) + cos(a)*sin(b)", variables=["a", "b"]
)


@pytest.mark.parametrize("source", [r"\sin(2x)", r"\sin(6x)", r"\sin(2x+2y)", r"\sin(2xy)"])
def test_integer_multiple_pattern_applies_to_integer_multiples(engine: Engine, source: str) -> None:
    out = apply(engine, source, [DOUBLE_SIN])
    app = out["trig.double_sin"]
    assert app.applicability is Applicability.APPLICABLE
    assert app.verification.passed  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "source",
    [
        r"\sin(x)",  # would need a = x/2
        r"\sin(3x)",  # would need a = 3x/2
        r"\sin(x/2)",  # would need a = x/4
        r"\sin(x+y)",  # would need a = (x+y)/2
        r"\sin(x)\sin(y)",
        r"\frac{1}{2}\sin(x)",  # a coefficient elsewhere must not legitimise a = x/2
    ],
)
def test_integer_multiple_pattern_does_not_halve_the_argument(engine: Engine, source: str) -> None:
    out = apply(engine, source, [DOUBLE_SIN])
    assert out["trig.double_sin"].applicability is Applicability.NO_MATCH
    assert out["trig.double_sin"].rewritten_latex is None


def test_double_cos_is_also_not_applied_by_halving(engine: Engine) -> None:
    assert apply(engine, r"\cos(3x)", [DOUBLE_COS])["trig.double_cos"].applicability is (
        Applicability.NO_MATCH
    )
    assert apply(engine, r"\cos(4x)", [DOUBLE_COS])["trig.double_cos"].applicability is (
        Applicability.APPLICABLE
    )


def test_pattern_with_a_plain_variable_may_still_bind_fractions(engine: Engine) -> None:
    """Only wildcards that appear solely as n*w are restricted; `a` in sin(a + b) is not."""
    out = apply(engine, r"\sin(\frac{x}{2} + y)", [SUM_SIN])
    assert out["trig.sum_sin"].applicability is Applicability.APPLICABLE


def test_a_rewrite_found_by_halving_was_true_but_is_no_longer_offered(engine: Engine) -> None:
    """The halved rewrite is mathematically valid, so the rule itself is not wrong: only the match."""
    assert verify_rule(DOUBLE_SIN).passed
