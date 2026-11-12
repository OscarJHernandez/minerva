"""Chains of verified rewrites: combinations, nesting, conditions, and what must never happen.

Rules are defined inline (as in test_identities.py) so these tests do not depend on the handbook.
"""

import pytest
import sympy as sp
from minerva_engine import (
    Engine,
    ExpressionInput,
    Limits,
    Rule,
    RuleCondition,
    Status,
    TransformResult,
    parse_latex,
)
from minerva_engine.ops import can_reverse, rule_rewrites
from minerva_engine.transform import MAX_DEPTH, MAX_VERIFY_PER_STATE, MAX_WIDTH


def rule(
    rid: str,
    lhs: str,
    rhs: str,
    variables: list[str],
    conditions: list[tuple[str, str]] | None = None,
) -> Rule:
    return Rule(
        id=rid,
        lhs=lhs,
        rhs=rhs,
        variables=variables,
        conditions=[RuleCondition(variable=v, assumption=a) for v, a in conditions or []],
    )


PYTH = rule("pyth", "sin(a)**2 + cos(a)**2", "1", ["a"])
TAN_SEC = rule("tan_sec", "1 + tan(a)**2", "sec(a)**2", ["a"])
DIFF_SQ = rule("diff_sq", "a**2 - b**2", "(a - b)*(a + b)", ["a", "b"])
COS2 = rule("cos2", "cos(2*a)", "cos(a)**2 - sin(a)**2", ["a"])
COS2_SIN = rule("cos2_sin", "cos(2*a)", "1 - 2*sin(a)**2", ["a"])
SIN2 = rule("sin2", "sin(2*a)", "2*sin(a)*cos(a)", ["a"])
HYP = rule("hyp", "cosh(a)**2 - sinh(a)**2", "1", ["a"])
LOG_QUOT = rule(
    "log_quot",
    "log(a) - log(b)",
    "log(a/b)",
    ["a", "b"],
    [("a", "positive"), ("b", "positive")],
)
# Deliberately wrong: one is false outright, one is off by a relative 1e-6.
FALSE_RULE = rule("bogus.sin_is_cos", "sin(a)", "cos(a)", ["a"])
NEAR_RULE = rule("bogus.near", "sin(a)**2", "sin(a)**2*(1 + 1/1000000)", ["a"])
ALL = [PYTH, TAN_SEC, DIFF_SQ, COS2, COS2_SIN, SIN2, HYP, LOG_QUOT]


def run(
    engine: Engine,
    source: str,
    rules: list[Rule] | None = None,
    goal: str = "simplify",
    target: list[str] | None = None,
    **assumptions: list[str],
) -> TransformResult:
    expression = ExpressionInput(input=source, format="latex", assumptions=assumptions)
    return engine.transform(expression, ALL if rules is None else rules, goal, target)


def names(result: TransformResult) -> list[str]:
    return [s.name for s in result.steps]


def assert_chain_is_verified(result: TransformResult) -> None:
    """Every shown step and the end-to-end pair passed, and the steps connect."""
    assert result.status is Status.OK
    assert result.verification is not None and result.verification.passed
    assert result.verification.numeric_points_checked >= 4
    assert result.verification.numeric_points_agreed == result.verification.numeric_points_checked
    assert result.steps
    for step in result.steps:
        assert step.verification.passed
    for before, after in zip(result.steps, result.steps[1:], strict=False):
        assert before.after_latex == after.before_latex
    assert result.steps[0].before_latex == result.input_latex
    assert result.steps[-1].after_latex == result.result_latex


# --- combinations of identities ---------------------------------------------------------------
def test_two_identities_in_sequence(engine: Engine) -> None:
    """sin^2 + cos^2 -> 1 (pyth), then 1 + tan^2 -> sec^2 (tan_sec): neither alone gets there."""
    result = run(engine, r"\sin^2 x + \cos^2 x + \tan^2 x")
    assert_chain_is_verified(result)
    assert result.result_latex == r"\sec^{2}{\left(x \right)}"
    assert names(result) == ["pyth", "tan_sec"]
    assert all(s.kind == "rule" and s.direction == "forward" for s in result.steps)


def test_a_chain_is_reported_with_both_kinds_of_evidence(engine: Engine) -> None:
    result = run(engine, r"\sin^2 x + \cos^2 x + \tan^2 x")
    for step in result.steps:
        v = step.verification
        assert v.numeric_points_checked >= 4
        assert isinstance(v.symbolic_zero, bool)  # reported separately from the numeric check
    assert "not a proof" in result.message


def test_identity_applies_inside_nested_expressions(engine: Engine) -> None:
    result = run(engine, r"\log(\sin^2 t + \cos^2 t + y)", [PYTH])
    assert_chain_is_verified(result)
    assert names(result) == ["pyth"]
    assert result.result_latex == r"\log{\left(y + 1 \right)}"


def test_two_separate_places_are_both_rewritten(engine: Engine) -> None:
    source = r"\sin(\sin^2 x + \cos^2 x) + \cos(\sin^2 y + \cos^2 y)"
    result = run(engine, source, [PYTH])
    assert_chain_is_verified(result)
    assert names(result) == ["pyth", "pyth"]
    assert result.result_latex == sp.latex(sp.sin(1) + sp.cos(1))


def test_rule_rewrites_yields_every_place_a_rule_matches() -> None:
    expr = parse_latex(r"\sin(\sin^2 x + \cos^2 x) + \cos(\sin^2 y + \cos^2 y)", {}, Limits())
    places = list(rule_rewrites(expr, PYTH, strict=True))
    assert len(places) == 2
    assert all(not p.unmet for p in places)
    # The lenient default (used by find_identities) also lets a single term absorb the rest.
    assert len(list(rule_rewrites(expr, PYTH))) > 2


def test_exponential_of_a_hyperbolic_identity(engine: Engine) -> None:
    result = run(engine, r"\exp(\cosh^2 u - \sinh^2 u)", [HYP])
    assert_chain_is_verified(result)
    assert names(result) == ["hyp"]
    assert result.result_text == "E"


# --- right to left ----------------------------------------------------------------------------
def test_a_rule_is_read_right_to_left_when_that_serves_the_goal(engine: Engine) -> None:
    result = run(engine, r"1 - 2\sin^2 x", [COS2_SIN])
    assert_chain_is_verified(result)
    assert names(result) == ["cos2_sin"]
    assert result.steps[0].direction == "reverse"
    assert result.result_latex == r"\cos{\left(2 x \right)}"


def test_the_simpler_of_two_rules_is_cited(engine: Engine) -> None:
    """2 sin x cos x = sin 2x is the double angle formula; do not cite a two-variable rule."""
    sum_to_product = rule(
        "sum_to_product", "sin(a) + sin(b)", "2*sin(a/2 + b/2)*cos(a/2 - b/2)", ["a", "b"]
    )
    result = run(engine, r"2\sin x\cos x", [sum_to_product, SIN2])
    assert_chain_is_verified(result)
    assert names(result) == ["sin2"]
    assert result.steps[0].direction == "reverse"


def test_reverse_is_refused_when_the_right_side_cannot_bind_the_variables() -> None:
    assert not can_reverse(PYTH)  # rhs is the bare number 1
    assert not can_reverse(rule("lose", "sin(a)*b", "sin(a)", ["a", "b"]))  # b is not on the rhs
    assert not can_reverse(rule("ident", "a + 0", "a", ["a"]))  # bare symbol matches anything
    assert can_reverse(TAN_SEC)


def test_one_is_never_rewritten_as_sine_squared_plus_cosine_squared(engine: Engine) -> None:
    for goal in ("simplify", "expand", "factor"):
        result = run(engine, "1", [PYTH], goal)
        assert result.status is Status.UNCHANGED, goal
        assert result.steps == []


# --- conditional transformations --------------------------------------------------------------
def test_conditional_rule_is_blocked_and_reported_without_assumptions(engine: Engine) -> None:
    result = run(engine, r"\log x - \log y")
    assert result.status is Status.CONDITIONS_NOT_SATISFIED
    assert result.steps == []
    assert result.result_latex == result.input_latex  # nothing was changed
    [blocked] = result.blocked
    assert blocked.rule_id == "log_quot"
    assert {c.variable for c in blocked.unmet_conditions} == {"a", "b"}
    assert "not provable" in result.message


def test_partial_assumptions_keep_the_rule_blocked(engine: Engine) -> None:
    result = run(engine, r"\log x - \log y", x=["positive"])
    assert result.status is Status.CONDITIONS_NOT_SATISFIED
    [blocked] = result.blocked
    assert {c.variable for c in blocked.unmet_conditions} == {"b"}


def test_conditional_rule_applies_when_conditions_are_provable(engine: Engine) -> None:
    result = run(engine, r"\log x - \log y", x=["positive"], y=["positive"])
    assert_chain_is_verified(result)
    assert names(result) == ["log_quot"]
    assert result.blocked == []


def test_sqrt_of_a_square_is_not_simplified_without_assumptions(engine: Engine) -> None:
    result = run(engine, r"\sqrt{x^2}")
    assert result.status is Status.UNCHANGED
    assert result.result_latex == result.input_latex


# --- wrong rules never get in -------------------------------------------------------------------
@pytest.mark.parametrize(
    ("bad", "matching"),
    [(FALSE_RULE, r"\sin x"), (NEAR_RULE, r"\sin^2 x")],
    ids=lambda p: p.id if isinstance(p, Rule) else "",
)
def test_a_wrong_rule_never_appears_in_a_chain(engine: Engine, bad: Rule, matching: str) -> None:
    for source, goal in ((matching, "simplify"), (r"\sin^2 x + \cos^2 x", "simplify")):
        result = run(engine, source, [bad, PYTH], goal)
        assert bad.id not in names(result)
        for alternative in result.alternatives:
            assert bad.id not in [s.name for s in alternative.steps]
    only_bad = run(engine, matching, [bad])
    assert only_bad.status is Status.UNCHANGED
    assert only_bad.discarded_not_verified >= 1  # it matched, was checked, and was thrown away


def test_sine_is_not_turned_into_a_half_angle_product(engine: Engine) -> None:
    """sin(2a) must not match sin(x) by binding a = x/2 (the same guard as find_identities)."""
    result = run(engine, r"\sin x", [SIN2], "expand")
    assert result.status is Status.UNCHANGED


# --- goals ------------------------------------------------------------------------------------
def test_goals_rank_the_same_family_differently(engine: Engine) -> None:
    factored = run(engine, r"x^2 - y^2", [DIFF_SQ], "factor")
    assert_chain_is_verified(factored)
    assert factored.result_latex == r"\left(x - y\right) \left(x + y\right)"
    assert names(factored) == ["diff_sq"]

    expanded = run(engine, r"(x - y)(x + y)", [DIFF_SQ], "expand")
    assert_chain_is_verified(expanded)
    assert expanded.result_latex == r"x^{2} - y^{2}"

    # The same two inputs under the opposite goals have nothing to improve.
    assert run(engine, r"x^2 - y^2", [DIFF_SQ], "expand").status is Status.UNCHANGED
    assert run(engine, r"(x - y)(x + y)", [DIFF_SQ], "factor").status is Status.UNCHANGED


def test_rewrite_goal_removes_functions_outside_the_target(engine: Engine) -> None:
    result = run(engine, r"\cos(2x) + \sin^2 x", [COS2], "rewrite", ["cos"])
    assert_chain_is_verified(result)
    assert result.result_latex == r"\cos^{2}{\left(x \right)}"
    assert "sin" not in (result.result_text or "")


def test_rewrite_goal_needs_a_target(engine: Engine) -> None:
    assert run(engine, r"\sin x", [], "rewrite").status is Status.UNSUPPORTED


def test_unknown_goal_is_unsupported(engine: Engine) -> None:
    assert run(engine, r"\sin x", [], "integrate").status is Status.UNSUPPORTED


def test_an_input_that_cannot_be_read_is_a_parse_error(engine: Engine) -> None:
    assert run(engine, r"\foo{x}").status is Status.PARSE_ERROR


def test_nothing_to_improve_says_so_and_returns_the_input(engine: Engine) -> None:
    result = run(engine, r"\sin x")
    assert result.status is Status.UNCHANGED
    assert result.result_latex == result.input_latex
    assert "does not show that none exists" in result.message


# --- bounds -----------------------------------------------------------------------------------
def test_rules_that_undo_each_other_terminate(engine: Engine) -> None:
    result = run(
        engine, r"\sin(2x)", [SIN2, rule("sin2_back", "2*sin(a)*cos(a)", "sin(2*a)", ["a"])]
    )
    assert result.status in {Status.OK, Status.UNCHANGED}
    assert result.candidates_tried <= MAX_WIDTH * MAX_DEPTH * MAX_VERIFY_PER_STATE


def test_depth_one_allows_one_step(engine: Engine) -> None:
    expression = ExpressionInput(input=r"\sin^2 x + \cos^2 x + \tan^2 x", format="latex")
    one = engine.transform(expression, [PYTH, TAN_SEC], "simplify", None, width=4, depth=1)
    # Depth 1 may still find a one-step SymPy form, but never the two-identity chain.
    assert len(one.steps) <= 1
    assert one.result_latex != r"\sec^{2}{\left(x \right)}"


def test_caps_are_enforced(engine: Engine) -> None:
    expression = ExpressionInput(input=r"\sin^2 x + \cos^2 x + \tan^2 x", format="latex")
    result = engine.transform(expression, ALL, "simplify", None, width=1000, depth=1000)
    assert result.candidates_tried <= MAX_WIDTH * MAX_DEPTH * MAX_VERIFY_PER_STATE


def test_an_exploding_expansion_is_stopped_or_verified_never_guessed() -> None:
    short = Engine(Limits(timeout_seconds=4, max_concurrency=1))
    expression = ExpressionInput(input=r"(a+b+c+d+e+f+g)^{40}", format="latex")
    result = short.transform(expression, [], "expand")
    assert result.status in {Status.RESOURCE_LIMIT, Status.OK, Status.UNCHANGED}
    if result.status is Status.OK:
        assert result.verification is not None and result.verification.passed


def test_the_engine_result_never_contains_an_unverified_step(engine: Engine) -> None:
    for source in (r"\sin^2 x + \cos^2 x + \tan^2 x", r"\cos^2 x - \sin^2 x", r"1 - 2\sin^2 x"):
        result = run(engine, source, [*ALL, FALSE_RULE, NEAR_RULE])
        assert_chain_is_verified(result)
        for chain in result.alternatives:
            assert chain.verification.passed
            assert all(s.verification.passed for s in chain.steps)
