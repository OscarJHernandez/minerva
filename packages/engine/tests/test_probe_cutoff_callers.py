"""V-017: a verification whose targeted probes the clock cut short is incomplete, and no caller may accept it.

Load may turn an answer into a refusal, never a refusal into an answer. Every test below forces the probes'
wall-clock stop to zero (`verify.PROBE_SECONDS = 0`), so the search ends before its first point exactly as it would
on a badly overloaded machine, and checks what the caller does with that. Each has a twin with the stop untouched
that shows the same input is accepted when the search completes, so the zero budget is what made the difference.
The calls run in this process (not the sandbox) so the patched constant reaches them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
import sympy as sp
from minerva_engine import ExpressionInput, Rule, Status, parse_text
from minerva_engine import steps as steps_module
from minerva_engine import substitute as substitute_module
from minerva_engine import verify as verify_module
from minerva_engine.claim_evidence import _agree
from minerva_engine.claims import INCOMPLETE_DETAIL, check_claim
from minerva_engine.differentiate import DiffVariable, differentiate
from minerva_engine.identity_check import INCOMPLETE_MESSAGE, check_identity
from minerva_engine.indexed.instantiate import verify_equal_indexed
from minerva_engine.ode.analyze import solve_highest
from minerva_engine.ode.models import OdeInput
from minerva_engine.ode.reading import read_ode
from minerva_engine.ops import Applicability, apply_rule, op_simplify
from minerva_engine.stats.pmf_check import verify_moments
from minerva_engine.testing import budget_limits
from minerva_engine.transform import transform

L = budget_limits()
x = sp.Symbol("x")
PYTH = Rule(id="pyth", lhs="sin(a)**2 + cos(a)**2", rhs="1", variables=["a"])


@pytest.fixture
def no_probe_time(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(verify_module, "PROBE_SECONDS", 0.0)
    yield


def _zero_budget(fn: Callable[..., Any]) -> Callable[..., Any]:
    """`fn` with the probes' stop at zero for this one call only."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        saved = verify_module.PROBE_SECONDS
        verify_module.PROBE_SECONDS = 0.0
        try:
            return fn(*args, **kwargs)
        finally:
            verify_module.PROBE_SECONDS = saved

    return wrapped


# --- the verifier itself ---------------------------------------------------------------------------------------------
def test_the_outcome_of_the_search_is_one_closed_label() -> None:
    from minerva_engine.verify import verify_equal

    y = sp.Symbol("y")
    assert verify_equal(sp.exp(x) * sp.exp(y), sp.exp(x + y)).probe_outcome == "cap"
    assert verify_equal(sp.sin(x) ** 2 + sp.cos(x) ** 2, sp.Integer(1)).probe_outcome == "completed"
    assert (
        verify_equal(sp.Integer(2), sp.Integer(2)).probe_outcome == "completed"
    )  # one point, natural end
    assert (
        verify_equal(x, x + 1).probe_outcome == "not_run"
    )  # the random points disagreed: no search
    assert verify_equal(x, x, probe=False).probe_outcome == "not_run"
    found = verify_equal(sp.log(1 / x), -sp.log(x), symbolic=False)
    assert found.probe_outcome == "found" and not found.passed


def test_a_cut_short_search_is_incomplete_not_a_disagreement(no_probe_time: None) -> None:
    from minerva_engine.verify import verify_equal

    v = verify_equal(sp.sin(x) ** 2 + sp.cos(x) ** 2, sp.Integer(1))
    assert v.passed is False and v.probes_cut_short is True and v.probe_outcome == "cut_short"
    assert v.numeric_points_agreed == v.numeric_points_checked  # nothing disagreed
    assert v.probe_kind is None
    assert "incomplete" in v.note and "not a disagreement" in v.note and "differ" not in v.note


# --- claims and the identity check -----------------------------------------------------------------------------------
def test_a_claim_is_not_held_on_an_incomplete_check(no_probe_time: None) -> None:
    check = check_claim(r"\sin^2 x + \cos^2 x = 1", {}, L)
    assert check.status == "unverified" and check.detail == INCOMPLETE_DETAIL


def test_the_same_claim_holds_when_the_search_completes() -> None:
    assert check_claim(r"\sin^2 x + \cos^2 x = 1", {}, L).status == "holds"


def test_a_disagreement_found_is_still_one_when_the_probes_are_cut_short(
    no_probe_time: None,
) -> None:
    """The random points disagree before any probe runs: the clock changes nothing about that."""
    assert check_claim(r"\sin x = \cos x", {}, L).status != "holds"
    assert check_claim(r"x + 1 = x + 2", {}, L).status == "fails"


def test_an_identity_check_cut_short_in_the_claim_checker_is_not_verified(
    no_probe_time: None,
) -> None:
    result = check_identity(r"\sin^2 x + \cos^2 x = 1", {}, L)
    assert result.status is Status.NOT_VERIFIED and result.verdict is None
    assert result.message == INCOMPLETE_MESSAGE
    # the card's probe line reads this flag
    assert result.verification is not None and result.verification.probes_cut_short
    assert result.verification.passed is False


@pytest.mark.parametrize(
    "latex",
    [
        r"\sin^2 x + \cos^2 x = 1",  # the claim checker says `holds`
        r"\frac{x^2 - 1}{x - 1} = x + 1",  # `conditional`: equal as functions, not at the same points
    ],
)
def test_an_identity_check_cut_short_at_its_own_verification_is_not_verified(
    monkeypatch: pytest.MonkeyPatch, latex: str
) -> None:
    """The claim checker completed; only the identity check's own `verify_with_witness` is cut. A `conditional`
    claim used to fall through to the counterexample search and say "the sides disagree"."""
    import minerva_engine.identity_check as module

    assert check_identity(latex, {}, L).verdict in ("proved_equal", "numerically_equal")
    monkeypatch.setattr(module, "verify_with_witness", _zero_budget(module.verify_with_witness))
    result = check_identity(latex, {}, L)
    assert result.status is Status.NOT_VERIFIED and result.verdict is None
    assert result.counterexample is None and result.message == INCOMPLETE_MESSAGE
    assert result.verification is not None and result.verification.probes_cut_short


def test_claim_evidence_is_not_linked_on_an_incomplete_check(no_probe_time: None) -> None:
    assert _agree(sp.sin(x) ** 2 + sp.cos(x) ** 2, sp.Integer(1)) == (False, False)


def test_claim_evidence_links_when_the_search_completes() -> None:
    assert _agree(sp.sin(x) ** 2 + sp.cos(x) ** 2, sp.Integer(1))[0] is True


# --- simplify, rules and chains --------------------------------------------------------------------------------------
SIMPLIFY = {
    "limits": L.model_dump(),
    "expression": {"input": "sin(x)**2 + cos(x)**2 + x", "format": "text", "assumptions": {}},
}


def test_simplify_shows_nothing_on_an_incomplete_check(no_probe_time: None) -> None:
    result = op_simplify(SIMPLIFY)
    assert result["status"] == "not_verified" and result["result_latex"] is None
    assert (
        "incomplete" in result["message"] and result["verification"]["probe_outcome"] == "cut_short"
    )


def test_simplify_shows_the_result_when_the_search_completes() -> None:
    assert op_simplify(SIMPLIFY)["status"] == "ok"


def test_a_rule_is_not_applied_on_an_incomplete_check(no_probe_time: None) -> None:
    application = apply_rule(parse_text("sin(x)**2 + cos(x)**2 + x"), PYTH)
    assert application.applicability is Applicability.NOT_VERIFIED
    assert application.rewritten_latex is None


def test_the_rule_applies_when_the_search_completes() -> None:
    application = apply_rule(parse_text("sin(x)**2 + cos(x)**2 + x"), PYTH)
    assert application.applicability is Applicability.APPLICABLE


def _chain() -> Any:
    data = ExpressionInput(input=r"\sin^2 x + \cos^2 x + x", format="latex")
    return transform(data, [PYTH], "simplify", [], L)


def test_a_chain_is_not_shown_on_an_incomplete_check(no_probe_time: None) -> None:
    result = _chain()
    assert (
        result.status is not Status.OK
        and not result.steps
        and result.result_latex
        in (
            None,
            result.input_latex,
        )
    )


def test_the_chain_is_shown_when_the_search_completes() -> None:
    assert _chain().status is Status.OK


# --- controlled steps ------------------------------------------------------------------------------------------------
REWRITE = {
    "limits": L.model_dump(),
    "expression": {"input": "(x + 1)**2 + y", "format": "text", "assumptions": {}},
    "action": "expand",
    "part": "(x + 1)**2",
}


def test_a_rewrite_step_is_not_shown_on_an_incomplete_check(no_probe_time: None) -> None:
    result = steps_module.op_rewrite_at(REWRITE)
    assert result["status"] == "not_verified" and result["result_latex"] is None
    assert "incomplete" in result["message"]


def test_the_rewrite_step_is_shown_when_the_search_completes() -> None:
    assert steps_module.op_rewrite_at(REWRITE)["status"] == "ok"


SUBSTITUTE = {
    "limits": L.model_dump(),
    "expression": {"input": "sin(x)**2 + cos(x)**2 + y", "format": "text", "assumptions": {}},
    "relations": ["sin(x)**2 = 1 - cos(x)**2"],
}


def _cut_nth_verify(monkeypatch: pytest.MonkeyPatch, which: int) -> list[int]:
    """Cut the probes of the `which`-th `_verify` call of one substitution (1: the back check of the relation,
    2: the unconditional check of the result)."""
    calls: list[int] = []
    real = steps_module._verify

    def counted(*args: Any) -> Any:
        calls.append(1)
        return (_zero_budget(real) if len(calls) == which else real)(*args)

    monkeypatch.setattr(substitute_module, "_verify", counted)
    return calls


def test_a_substitution_that_is_an_identity_says_so_when_every_check_completes() -> None:
    result = substitute_module.op_substitute(SUBSTITUTE)
    assert result["status"] == "ok" and result["evidence"] == "identity"


def test_a_cut_unconditional_check_gives_conditional_never_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _cut_nth_verify(monkeypatch, 2)
    result = substitute_module.op_substitute(SUBSTITUTE)
    assert len(calls) == 2
    assert (
        result["status"] == "ok" and result["evidence"] == "conditional"
    )  # the weaker claim, still true
    assert "incomplete" in result["message"] and "not failed" in result["message"]
    assert (
        result["verification"]["passed"] is True
    )  # the per-relation checks, each of which completed
    assert all(s["verification"]["probe_outcome"] != "cut_short" for s in result["steps"])


def test_a_cut_back_check_gives_not_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _cut_nth_verify(monkeypatch, 1)
    result = substitute_module.op_substitute(SUBSTITUTE)
    assert len(calls) == 1
    assert result["status"] == "not_verified" and result["result_latex"] is None
    assert (
        "incomplete" in result["message"] and result["verification"]["probe_outcome"] == "cut_short"
    )


# --- other verifications that read `.passed` -------------------------------------------------------------------------
def _derivative() -> Any:
    data = ExpressionInput(input="(x**2 - 1)/(x - 1)", format="text")
    return differentiate(data, [DiffVariable(name="x")], L)


def test_a_derivative_keeps_its_untidied_form_on_an_incomplete_check(no_probe_time: None) -> None:
    """The derivative has its own check (an independent numerical derivative); only the tidy-up is refused."""
    result = _derivative()
    assert result.status is Status.OK and result.result_latex != "1"


def test_the_derivative_is_tidied_when_the_search_completes() -> None:
    result = _derivative()
    assert result.status is Status.OK and result.result_latex == "1"


def test_an_ode_solved_form_is_not_confirmed_on_an_incomplete_check(no_probe_time: None) -> None:
    problem = read_ode(OdeInput(equations=["y'' + y = 0"], independent="t"), L)
    solved, reason = solve_highest(problem)
    assert solved is None and "could not be confirmed" in reason


def test_the_ode_solved_form_is_confirmed_when_the_search_completes() -> None:
    problem = read_ode(OdeInput(equations=["y'' + y = 0"], independent="t"), L)
    assert solve_highest(problem)[0] is not None


SUMS = {"n": ["integer", "positive"]}


def _sum(text: str) -> sp.Expr:
    return parse_text(text, SUMS, allow_sums=True)


def test_an_instance_check_is_incomplete_not_different(no_probe_time: None) -> None:
    v = verify_equal_indexed(
        _sum("Sum(x[i] + y[i], (i, 1, n))"), _sum("Sum(x[i], (i, 1, n)) + Sum(y[i], (i, 1, n))")
    )
    assert v.passed is False and "incomplete" in v.note and "differs" not in v.note


def test_the_instance_check_passes_when_the_search_completes() -> None:
    v = verify_equal_indexed(
        _sum("Sum(x[i] + y[i], (i, 1, n))"), _sum("Sum(x[i], (i, 1, n)) + Sum(y[i], (i, 1, n))")
    )
    assert v.passed


def _moments() -> Any:
    def P(text: str) -> sp.Expr:
        return parse_text(text, {"a": ["real"]}, allow_sums=True, random_variables=["X", "Y"])

    return verify_moments(P("E[a*X + Y]"), P("a*E[X] + E[Y]"), frozenset({"X", "Y"}), [])


def test_a_moment_check_is_incomplete_not_different(no_probe_time: None) -> None:
    v = _moments()
    assert v.passed is False and "incomplete" in v.note and "differs" not in v.note


def test_the_moment_check_passes_when_the_search_completes() -> None:
    assert _moments().passed
