"""The vector identities of the handbook: their level is honest, a wrong identity fails its check,
and the engine can match and verify every one of them on an expression it appears in."""

import itertools

import pytest
import sympy as sp
from minerva_engine import FieldDecl, Limits, Status, VectorInput
from minerva_engine.vector import vector_calculus
from minerva_knowledge import KnowledgeBase
from minerva_knowledge.checks import run_check
from minerva_knowledge.identities import vector_identity_rules
from minerva_knowledge.models import Entry, VectorIdentityCheck
from pydantic import ValidationError

KB = KnowledgeBase.load()
VECTOR = [e for e in KB.entries if e.chapter == "math.vector_calculus"]
CHECKED = [e for e in VECTOR if isinstance(e.check, VectorIdentityCheck)]
L = Limits(timeout_seconds=60)


def clone(entry: Entry, **check_over) -> Entry:  # type: ignore[no-untyped-def]
    assert isinstance(entry.check, VectorIdentityCheck)
    check = entry.check.model_copy(update=check_over)
    return entry.model_copy(update={"check": check})


def test_the_chapter_has_checked_and_stated_entries() -> None:
    assert len(CHECKED) >= 20
    assert {e.id for e in VECTOR if e.verification == "stated"} == {
        "vector.epsilon_delta",
        "vector.epsilon_two_contraction",
        "vector.epsilon_full_contraction",
    }
    assert all(e.rule is None for e in VECTOR)  # the rule matcher cannot express operators
    assert not next(c for c in KB.chapters if c.id == "math.vector_calculus").planned


def test_the_symbolic_level_needs_a_vector_check_and_stated_entries_carry_none() -> None:
    entry = CHECKED[0]
    with pytest.raises(ValidationError):
        Entry(**{**entry.model_dump(), "verification": "stated"})
    with pytest.raises(ValidationError):
        Entry(**{**entry.model_dump(), "verification": "dimensional"})


@pytest.mark.parametrize("entry", CHECKED, ids=lambda e: e.id)
def test_every_checked_identity_is_proved_in_three_coordinate_systems(entry: Entry) -> None:
    report = run_check(entry)
    assert report.passed, report.detail
    assert report.symbolic_proof
    assert "Cartesian, cylindrical and spherical" in report.detail


def _mutations(entry: Entry) -> list[tuple[str, str]]:
    assert isinstance(entry.check, VectorIdentityCheck)
    lhs, rhs = entry.check.lhs, entry.check.rhs
    variables = entry.check.fields
    out: list[tuple[str, str]] = []
    scalar = next((n for n, k in variables.items() if k == "scalar"), None)
    vector = next((n for n, k in variables.items() if k == "vector"), None)
    if rhs.strip() == "0":
        # a nonzero right side of the right kind
        out.append(("nonzero", f"grad({scalar})" if scalar and "curl" in lhs else f"div({vector})"))
    else:
        out.append(("doubled", f"2*({rhs})"))
        if " + " in rhs:
            out.append(("sign", rhs.replace(" + ", " - ", 1)))
        if " - " in rhs:
            out.append(("sign", rhs.replace(" - ", " + ", 1)))
    return out


@pytest.mark.parametrize("entry", CHECKED, ids=lambda e: e.id)
def test_a_mutated_identity_fails_its_check(entry: Entry) -> None:
    assert isinstance(entry.check, VectorIdentityCheck)
    for name, rhs in _mutations(entry):
        kind_ok = True
        try:
            report = run_check(clone(entry, rhs=rhs))
        except Exception:  # a mutation of the wrong kind is refused before it is checked
            kind_ok = False
            report = None
        if kind_ok:
            assert report is not None and not report.passed, (entry.id, name, rhs)


def test_an_identity_that_needs_c2_but_does_not_say_so_does_not_pass() -> None:
    entry = next(e for e in CHECKED if e.id == "vector.curl_grad")
    report = run_check(clone(entry, requires=[]))
    assert not report.passed  # without declared smoothness the difference needs mixed partials


def test_the_rules_handed_to_the_engine_come_only_from_checked_entries() -> None:
    rules = vector_identity_rules(KB)
    assert {r.id for r in rules} == {e.id for e in CHECKED}
    by_id = {r.id: r for r in rules}
    assert by_id["vector.curl_curl"].requires == ["c2"]
    assert by_id["vector.grad_product"].requires == []
    assert by_id["vector.curl_curl"].variables == {"A": "vector"}


def test_retrieval_puts_the_identities_that_mention_the_operators_first() -> None:
    first = vector_identity_rules(KB, {"vector:curl", "vector:grad"})[0]
    entry = KB.get(first.id)
    assert entry is not None and {"vector:curl", "vector:grad"} & set(entry.applies_to)


# -- the engine applies every identity of the handbook to an expression it occurs in --------------
INSTANCES = {
    "vector.curl_grad": "curl(grad(f))",
    "vector.div_curl": "div(curl(A))",
    "vector.curl_curl": "curl(curl(A))",
    "vector.laplacian_scalar": "lap(f)",
    "vector.grad_product": "grad(f*g)",
    "vector.div_scalar_vector": "div(f*A)",
    "vector.curl_scalar_vector": "curl(f*A)",
    "vector.div_cross": "div(cross(A, B))",
    "vector.curl_cross": "curl(cross(A, B))",
    "vector.grad_dot": "grad(dot(A, B))",
    "vector.grad_square": "grad(dot(A, A))",
    "vector.laplacian_product": "lap(f*g)",
    "vector.adv_scalar": "adv(A, f)",
    "vector.grad_sum": "grad(f + g)",
    "vector.div_sum": "div(A + B)",
    "vector.curl_sum": "curl(A + B)",
    "vector.time_curl": "dt(curl(A))",
    "vector.time_div": "dt(div(A))",
    "vector.time_grad": "dt(grad(f))",
    "vector.triple_product_expansion": "cross(A, cross(B, C))",
    "vector.scalar_triple_cyclic": "dot(A, cross(B, C))",
    "vector.cross_anticommutes": "cross(A, B)",
}


def test_every_checked_identity_has_an_instance_in_this_test() -> None:
    assert set(INSTANCES) == {e.id for e in CHECKED}


@pytest.mark.parametrize("rule_id", sorted(INSTANCES))
def test_the_engine_applies_each_identity_and_verifies_the_whole_expression(rule_id: str) -> None:
    rules = vector_identity_rules(KB)
    rule = next(r for r in rules if r.id == rule_id)
    fields = [
        FieldDecl(
            name=n,
            kind=k,
            smoothness="c2" if rule.requires else None,
            depends_on=["x", "y", "z", "t"] if "dt(" in rule.lhs else None,
        )
        for n, k in rule.variables.items()
    ]
    result = vector_calculus(
        VectorInput(
            expression=INSTANCES[rule_id],
            format="text",
            coordinates="cartesian",
            fields=fields,
            rules=rules,
            action="apply_identity",
            rule_id=rule_id,
            path=[],
        ),
        L,
    )
    assert result.status is Status.OK, (rule_id, result.message)
    (step,) = result.steps
    assert step.evidence == "identity" and step.symbolic_zero is True
    assert step.verification is not None and step.verification.numeric_passed


# -- the Levi-Civita and Kronecker delta entries are stated: check the statements by enumeration ----
def test_the_stated_epsilon_identities_hold_by_enumeration() -> None:
    eps = sp.LeviCivita
    delta = sp.KroneckerDelta
    r = range(3)
    for j, k, lo, m in itertools.product(r, r, r, r):
        lhs = sum(eps(i, j, k) * eps(i, lo, m) for i in r)
        assert lhs == delta(j, lo) * delta(k, m) - delta(j, m) * delta(k, lo)
    for k, lo in itertools.product(r, r):
        assert sum(eps(i, j, k) * eps(i, j, lo) for i in r for j in r) == 2 * delta(k, lo)
    assert sum(eps(i, j, k) ** 2 for i in r for j in r for k in r) == 6
