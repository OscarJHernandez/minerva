"""End to end: retrieval proposes, the engine checks applicability and conditions."""

from minerva_engine import Applicability, Engine, ExpressionInput, Status
from minerva_knowledge import KnowledgeBase
from minerva_knowledge.identities import find_identities


def run(kb: KnowledgeBase, engine: Engine, src: str, **assumptions: list[str]):  # type: ignore[no-untyped-def]
    resp = find_identities(kb, engine, ExpressionInput(input=src, assumptions=assumptions))
    return resp, {m.entry.id: m.application for m in resp.matches}


def test_pythagorean_found_and_applicable(kb: KnowledgeBase, engine: Engine) -> None:
    resp, by_id = run(kb, engine, r"\sin^2 x + \cos^2 x")
    assert resp.status is Status.OK
    assert by_id["trig.pythagorean"].applicability is Applicability.APPLICABLE
    assert by_id["trig.pythagorean"].rewritten_latex == "1"
    assert resp.matches[0].application.applicability is Applicability.APPLICABLE  # sorted first


def test_log_product_needs_assumptions(kb: KnowledgeBase, engine: Engine) -> None:
    _, plain = run(kb, engine, r"\log(xy)")
    assert plain["log.product"].applicability is Applicability.CONDITIONS_NOT_SATISFIED
    assert plain["log.product"].rewritten_latex is None
    _, positive = run(kb, engine, r"\log(xy)", x=["positive"], y=["positive"])
    assert positive["log.product"].applicability is Applicability.APPLICABLE


def test_sqrt_of_square_is_not_rewritten_without_assumptions(
    kb: KnowledgeBase, engine: Engine
) -> None:
    _, plain = run(kb, engine, r"\sqrt{x^2}")
    blocked = Applicability.CONDITIONS_NOT_SATISFIED
    assert plain["radical.sqrt_square_abs"].applicability is blocked
    assert plain["radical.sqrt_square_nonneg"].applicability is blocked
    assert plain["radical.sqrt_square_abs"].rewritten_latex is None
    # With x real, SymPy itself evaluates sqrt(x**2) to |x| while parsing; nothing is left to match.
    resp, real = run(kb, engine, r"\sqrt{x^2}", x=["real"])
    assert "|" in (resp.input_latex or "")
    assert "radical.sqrt_square_abs" not in real


def test_difference_of_squares_inside_larger_expression(kb: KnowledgeBase, engine: Engine) -> None:
    _, by_id = run(kb, engine, r"\frac{x^2 - y^2}{z}")
    assert by_id["alg.difference_of_squares"].applicability is Applicability.APPLICABLE


def test_unrelated_expression_reports_nothing_instead_of_inventing(
    kb: KnowledgeBase, engine: Engine
) -> None:
    resp, by_id = run(kb, engine, "x + y")
    assert resp.status is Status.OK
    assert not any(a.applicability is Applicability.APPLICABLE for a in by_id.values())


def test_parse_error_is_explicit(kb: KnowledgeBase, engine: Engine) -> None:
    resp, _ = run(kb, engine, r"\badcommand")
    assert resp.status is Status.PARSE_ERROR
    assert resp.matches == []


def test_default_candidate_pool_is_everything_that_shares_a_feature(kb: KnowledgeBase) -> None:
    feats = {"f:sin", "f:cos", "op:square"}
    default_pool = {e.id for e in kb.candidate_rules(feats)}
    everything = {e.id for e in kb.candidate_rules(feats, limit=200)}
    assert default_pool == everything  # a small cap would silently drop applicable rules
    assert "trig.pythagorean" in default_pool
