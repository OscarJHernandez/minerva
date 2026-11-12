"""Chains of identities from the real handbook: retrieval for chains, citations, honesty."""

import pytest
from minerva_engine import Engine, ExpressionInput, Status
from minerva_knowledge import KnowledgeBase
from minerva_knowledge.identities import chain_candidates, transform_expression


def run(kb: KnowledgeBase, engine: Engine, source: str, goal: str = "simplify", **kw):  # type: ignore[no-untyped-def]
    target = kw.pop("target", None)
    expression = ExpressionInput(input=source, assumptions=kw)
    return transform_expression(kb, engine, expression, goal, target)


def test_right_hand_side_features_let_a_chain_find_a_rule_to_read_backwards(
    kb: KnowledgeBase,
) -> None:
    features = {"f:sin", "f:cos", "op:quotient"}  # what sin(x)/cos(x) looks like
    forward = {e.id for e in kb.candidate_rules(features)}
    both = {e.id for e in kb.candidate_rules(features, sides="both")}
    assert "trig.tan_ratio" not in forward  # its left side is tan(a)
    assert "trig.tan_ratio" in both  # its right side is sin(a)/cos(a)
    assert forward <= both


def test_the_default_candidate_pool_is_unchanged_by_the_new_index(kb: KnowledgeBase) -> None:
    """find_identities must rank exactly as before: left-hand sides only."""
    features = {"f:sin", "f:cos", "op:square"}
    lhs = kb.candidate_rules(features)
    assert [e.id for e in lhs] == [e.id for e in kb.candidate_rules(features, sides="lhs")]
    assert all(e.rule is not None for e in lhs)


def test_target_functions_make_the_rules_that_introduce_them_candidates(
    kb: KnowledgeBase,
) -> None:
    plain = {e.id for e in chain_candidates(kb, ["f:tan"], [], "", 60)}
    aimed = {e.id for e in chain_candidates(kb, ["f:tan"], ["cos"], "", 60)}
    assert "trig.tan_ratio" in plain
    assert plain <= aimed or len(aimed) == 60


def test_a_two_identity_chain_cites_both_entries(kb: KnowledgeBase, engine: Engine) -> None:
    result = run(kb, engine, r"\sin^2 x + \cos^2 x + \tan^2 x")
    assert result.status is Status.OK
    assert [s.name for s in result.steps] == ["trig.pythagorean", "trig.tan_secant"]
    assert set(result.entries) == {"trig.pythagorean", "trig.tan_secant"}
    pyth = result.entries["trig.pythagorean"]
    assert pyth.verification == "symbolic"  # the handbook's own level, shown beside the step's
    assert pyth.reference is not None
    assert result.verification is not None and result.verification.passed
    assert all(s.verification.passed for s in result.steps)


def test_conditional_identity_is_blocked_and_the_entry_is_attached(
    kb: KnowledgeBase, engine: Engine
) -> None:
    result = run(kb, engine, r"\log x + \log y")
    assert result.status is Status.CONDITIONS_NOT_SATISFIED
    assert [b.rule_id for b in result.blocked] == ["log.product"]
    assert result.entries["log.product"].validity
    assert result.result_latex == result.input_latex


def test_assumptions_unlock_the_same_chain(kb: KnowledgeBase, engine: Engine) -> None:
    result = run(kb, engine, r"\log x + \log y", x=["positive"], y=["positive"])
    assert result.status is Status.OK
    assert [s.name for s in result.steps] == ["log.product"]
    assert result.steps[0].direction == "reverse"


def test_a_chain_is_never_longer_than_the_depth_asked_for(
    kb: KnowledgeBase, engine: Engine
) -> None:
    expression = ExpressionInput(input=r"\sin^2 x + \cos^2 x + \tan^2 x")
    result = transform_expression(kb, engine, expression, "simplify", None)
    assert len(result.steps) <= 6


def test_an_unreadable_input_is_a_parse_error(kb: KnowledgeBase, engine: Engine) -> None:
    assert run(kb, engine, r"\foo{x}").status is Status.PARSE_ERROR


@pytest.mark.parametrize("goal", ["simplify", "expand", "factor"])
def test_an_input_with_nothing_to_do_comes_back_unchanged(
    kb: KnowledgeBase, engine: Engine, goal: str
) -> None:
    result = run(kb, engine, "x", goal)
    assert result.status is Status.UNCHANGED
    assert result.steps == [] and result.entries == {}
