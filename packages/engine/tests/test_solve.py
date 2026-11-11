"""Polynomial solving: closed forms, solvability by radicals, honest limits, verification."""

import pytest
from minerva_engine import Engine, Limits, Status
from minerva_engine.solve import solve_polynomial

L = Limits()


def solve(eq: str, var: str | None = None, **assumptions: list[str]):  # type: ignore[no-untyped-def]
    return solve_polynomial(eq, var, assumptions, L)


def test_reducible_cubic_factors_and_gives_exact_roots() -> None:
    r = solve("x^3+x^2+x+1=0")
    assert r.status is Status.OK and r.degree == 3
    assert sorted(f.degree for f in r.factors) == [1, 2]
    exact = {root.exact_latex for root in r.roots}
    assert exact == {"-1", "i", "- i"}
    assert [root.real for root in r.roots].count(True) == 1
    v = r.verification
    assert v and v.roots_substituted_back and v.vieta_holds and v.count_matches_degree
    assert v.exact_roots_match_numeric is True


def test_quintic_without_formula_is_reported_as_such() -> None:
    r = solve("x^5-x-1=0")
    assert r.status is Status.OK
    (factor,) = r.factors
    assert factor.radicals == "no" and "not solvable" in factor.radicals_reason
    assert len(r.roots) == 5 and all(root.exact_latex is None for root in r.roots)
    assert "poly:not-solvable-by-radicals" in r.features
    assert any("no formula in radicals" in n for n in r.notes)
    assert r.verification and r.verification.roots_substituted_back


def test_solvable_quintic_formula_exists_but_is_not_produced() -> None:
    r = solve("x^5-5x+12=0")
    assert r.factors[0].radicals == "yes"
    assert any("cannot construct" in n for n in r.notes)
    assert len(r.roots) == 5


def test_cubic_with_three_real_roots_notes_casus_irreducibilis() -> None:
    r = solve("x^3-3x+1=0")
    assert sum(root.real for root in r.roots) == 3
    assert "poly:casus-irreducibilis" in r.features


def test_roots_have_multiplicities_through_factors() -> None:
    r = solve("x^2-2x+1=0")
    assert r.factors[0].multiplicity == 2 and len(r.roots) == 2
    assert all(root.numeric == "1" for root in r.roots)


def test_symbolic_coefficients_quadratic_has_a_formula_higher_degrees_do_not() -> None:
    q = solve("ax^2+bx+c=0", "x")
    assert q.status is Status.OK and q.symbolic_coefficients
    assert len(q.factors[0].exact_roots_latex) == 2 and q.roots == []
    quintic = solve("ax^5+bx+c=0", "x")
    assert (
        quintic.factors[0].radicals == "no" and "Abel-Ruffini" in quintic.factors[0].radicals_reason
    )
    cubic = solve("ax^3+bx+c=0", "x")
    assert cubic.factors[0].radicals == "yes" and "poly:general-formula-exists" in cubic.features


def test_degree_above_six_is_not_decided() -> None:
    r = solve("x^7-x-1=0")
    assert r.factors[0].radicals == "undetermined" and len(r.roots) == 7


@pytest.mark.parametrize(
    ("eq", "status"),
    [
        (r"\sin x = x", Status.UNSUPPORTED),
        ("x^2=y", Status.UNSUPPORTED),  # which unknown?
        ("x^2+1", Status.PARSE_ERROR),  # no equals sign
        (r"\foo = 1", Status.PARSE_ERROR),
        ("3 = 3", Status.UNSUPPORTED),
        (r"\sqrt{2}x^2 = 1", Status.UNSUPPORTED),  # irrational coefficients
        ("x^{100} = 1", Status.RESOURCE_LIMIT),
    ],
)
def test_unsupported_inputs_are_explicit(eq: str, status: Status) -> None:
    assert solve(eq).status is status


def test_choosing_the_unknown_resolves_ambiguity() -> None:
    r = solve("x^2=y", "x")
    assert r.status is Status.OK and r.symbolic_coefficients
    assert len(r.factors[0].exact_roots_latex) == 2


def test_engine_runs_solve_in_the_sandbox() -> None:
    engine = Engine(Limits(timeout_seconds=30))
    r = engine.solve("x^4+x+1=0")
    assert r.status is Status.OK and len(r.roots) == 4 and r.duration_ms >= 0
    assert engine.solve("x^2-1=0", "z").status is Status.PARSE_ERROR
