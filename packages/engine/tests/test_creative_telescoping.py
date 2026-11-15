"""Creative telescoping: discovery, verification, boundary analysis, evaluation, refusals."""

from __future__ import annotations

import time
from typing import Any

import pytest
import sympy as sp
from minerva_engine import Limits, Status, TelescopingInput
from minerva_engine.creative_telescoping import boundary as bd
from minerva_engine.creative_telescoping import certificate as cert
from minerva_engine.creative_telescoping import evaluate as ev
from minerva_engine.creative_telescoping import ode
from minerva_engine.creative_telescoping import operation as op_module
from minerva_engine.creative_telescoping.integrand import analyse
from minerva_engine.creative_telescoping.models import Refusal, ValueVerification
from minerva_engine.creative_telescoping.operation import creative_telescoping
from minerva_engine.creative_telescoping.operators import DifferentialOperator
from minerva_engine.creative_telescoping.search import discover

x = sp.Symbol("x", real=True)
a = sp.Symbol("a", positive=True)
n = sp.Symbol("n", positive=True)
POSITIVE = {"a": ["positive"], "x": ["real"]}
LIMITS = Limits(timeout_seconds=60)


def run(expr: str, lower: str | None, upper: str | None, assumptions: Any = None, **kw: Any):  # type: ignore[no-untyped-def]
    request = TelescopingInput(
        input=expr,
        assumptions=POSITIVE if assumptions is None else assumptions,
        variable="x",
        parameter=kw.pop("parameter", "a"),
        lower=lower,
        upper=upper,
        **kw,
    )
    return creative_telescoping(request, LIMITS)


def find(f: sp.Expr, max_order: int = 3, max_degree: int = 6):  # type: ignore[no-untyped-def]
    h = analyse(f, x, a)

    def accept(c: Any) -> bool:
        return cert.verify(f, x, c.operator, c.y * f).passed

    found, families, rejected = discover(
        h,
        max_order=max_order,
        max_degree=max_degree,
        deadline=time.monotonic() + 60,
        accept=accept,
    )
    return found, families, rejected


# -- discovery: nontrivial operators, minimal order, exact proof -------------------------------------
@pytest.mark.parametrize(
    ("f", "order"),
    [
        (sp.exp(-a * x**2), 1),
        (sp.exp(-(x**2) + a * x), 1),
        (x**n * sp.exp(-a * x), 1),
        (sp.exp(-a * x) / (1 + x), 1),
        (1 / (x**2 + a**2), 1),
        (sp.exp(-(x**2) - a**2 / x**2), 2),
        (sp.exp(sp.I * a * x) / sp.sqrt(1 - x**2), 2),
        (sp.exp(-(x**3) / 3 - a * x), 2),
        (sp.exp(-a * x), 0),
    ],
)
def test_discovers_a_proved_relation_of_minimal_order(f: sp.Expr, order: int) -> None:
    found, families, rejected = find(f)
    assert found is not None and rejected == 0
    assert found.order == order
    assert [fam.order for fam in families] == sorted(
        fam.order for fam in families
    )  # cheapest first
    check = cert.verify(f, x, found.operator, found.y * f)
    assert check.passed and check.symbolic_zero
    assert not found.operator.is_zero


def test_the_operator_found_for_the_gaussian_is_the_known_one() -> None:
    found, _, _ = find(sp.exp(-a * x**2))
    assert found is not None
    p0, p1 = found.operator.coefficients
    # 1 + 2a d/da annihilates a^(-1/2): the relation behind sqrt(pi/a), found rather than looked up
    assert sp.simplify(p0 * a ** sp.Rational(-1, 2) + p1 * sp.diff(a ** sp.Rational(-1, 2), a)) == 0


def test_the_family_order_is_reported_and_an_empty_search_says_what_it_searched() -> None:
    found, families, _ = find(sp.exp(-(x**3) / 3 - a * x), max_order=1, max_degree=3)
    assert found is None
    assert {(f.order, f.degree) for f in families} == {(o, d) for o in (0, 1) for d in range(4)}
    assert all(f.outcome == "none" for f in families)


# -- verification rejects what is not an exact identity -----------------------------------------------
GAUSS = sp.exp(-a * x**2)


@pytest.mark.parametrize(
    ("coefficients", "G"),
    [
        ((1, 2 * a), -x * GAUSS),  # wrong sign
        ((1 + sp.Rational(1, 10**6), 2 * a), x * GAUSS),  # coefficient off by 1e-6
        ((1, 2 * a), x * GAUSS * (1 + sp.Rational(1, 10**6))),  # G off by 1e-6
        ((0, 2 * a), x * GAUSS),  # dropped term
        ((0, 0), sp.Integer(0)),  # trivial relation
        ((1, 2 * a), x * sp.exp(-2 * a * x**2)),  # wrong exponent
    ],
)
def test_forged_certificates_are_rejected(coefficients: tuple[Any, ...], G: sp.Expr) -> None:
    operator = DifferentialOperator(a, tuple(sp.sympify(c) for c in coefficients))
    assert not cert.verify(GAUSS, x, operator, G).passed


def test_the_correct_pair_is_accepted_and_reports_each_leg() -> None:
    check = cert.verify(GAUSS, x, DifferentialOperator(a, (sp.Integer(1), 2 * a)), x * GAUSS)
    assert check.passed and check.symbolic_zero
    assert check.numeric_points_agreed == check.numeric_points_checked > 0
    assert check.derivative_points_agreed == check.derivative_points_checked > 0


def test_a_numerical_disagreement_vetoes_even_an_exact_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cert, "numeric_detail", lambda *args, **kw: (10, 9, 0))
    check = cert.verify(GAUSS, x, DifferentialOperator(a, (sp.Integer(1), 2 * a)), x * GAUSS)
    assert check.symbolic_zero and not check.passed


# -- eligibility ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("f", "reason"),
    [
        (sp.sin(a * x), "not_hyperexponential"),
        (sp.exp(-a * x) + sp.exp(-2 * a * x), "not_hyperexponential"),
        (x**a * sp.exp(-x), "not_hyperexponential"),
        (sp.log(x) * sp.exp(-a * x), "not_hyperexponential"),
        (sp.exp(-x), "parameter_missing"),
        (sp.exp(-a), "variable_missing"),
    ],
)
def test_integrands_outside_the_class_are_refused_with_a_reason(f: sp.Expr, reason: str) -> None:
    with pytest.raises(Refusal) as info:
        analyse(f, x, a)
    assert info.value.reason == reason


# -- boundary analysis -----------------------------------------------------------------------------------
def test_signs_on_an_interval_come_from_coefficient_certificates() -> None:
    p = sp.Symbol("p", positive=True)
    q = sp.Symbol("q", real=True)
    assert bd.sign_on_interval(1 - x**2, x, sp.Integer(-1), sp.Integer(1)) == 1
    assert bd.sign_on_interval(x**2 + p**2, x, -sp.oo, sp.oo) == 1
    assert bd.sign_on_interval(x - 1, x, sp.Integer(0), sp.Integer(1)) == -1
    assert bd.sign_on_interval(x - 1, x, sp.Integer(0), sp.oo) is None  # changes sign
    assert bd.sign_on_interval(x**2 + q, x, sp.Integer(0), sp.oo) is None  # q undecided
    assert bd.rational_sign((x + 1) / (x**2 + 1), x, sp.Integer(0), sp.oo) == 1


def test_parameter_domain_follows_the_declaration_only() -> None:
    assert bd.parameter_domain(sp.Symbol("t", positive=True)) == (0, sp.oo)
    assert bd.parameter_domain(sp.Symbol("t", negative=True)) == (-sp.oo, 0)
    assert bd.parameter_domain(sp.Symbol("t", real=True)) == (-sp.oo, sp.oo)
    assert bd.parameter_domain(sp.Symbol("t")) is None


def test_decay_is_decided_from_the_sign_of_the_leading_coefficient() -> None:
    f = sp.exp(-a * x)
    h = analyse(f, x, a)
    ok = bd.analyse(h, [sp.Integer(1), -x], -1 / a, sp.Integer(0), sp.oo)
    assert ok.sound and ok.values["upper"] == 0
    b = sp.Symbol("b", real=True)
    g = analyse(sp.exp(-b * x) * (1 + 0 * a * x) * sp.exp(-a), x, a)  # b of unknown sign
    unknown = bd.analyse(g, [sp.Integer(1), -sp.Integer(1)], -1 / b, sp.Integer(0), sp.oo)
    assert not unknown.sound and unknown.problems[0].reason == "convergence_not_established"


# -- the operation ---------------------------------------------------------------------------------------
def test_gaussian_is_discovered_proved_and_evaluated() -> None:
    r = run(r"\exp(-a x^2)", "-oo", "oo")
    assert r.status is Status.OK and r.outcome == "evaluated"
    assert r.result_latex == r"\frac{\sqrt{\pi}}{\sqrt{a}}"
    assert r.certificate_check and r.certificate_check.symbolic_zero
    assert r.equation and r.equation.homogeneous and r.boundary and r.boundary.vanishes
    assert r.value_verification and r.value_verification.passed
    assert [s.name for s in r.diagnostics.stages] == [
        "analyse",
        "search",
        "verify",
        "boundary",
        "equation",
        "solve",
        "value",
    ]
    assert r.solution and r.solution.constants_latex


def test_nonzero_boundary_gives_an_inhomogeneous_equation_and_stops_where_values_are_missing() -> (
    None
):
    r = run(r"\frac{\exp(-a x)}{1+x}", "0", "1")
    assert r.status is Status.OK and r.outcome == "equation"
    assert r.equation and not r.equation.homogeneous
    assert r.result_latex is None and r.solution is None
    assert r.unresolved and r.reason == "initial_value_unavailable"


def test_an_undeclared_parameter_sign_is_never_assumed() -> None:
    r = run(r"\exp(-a x)", "0", "oo", {"x": ["real"]})
    assert r.status is Status.CONDITIONS_NOT_SATISFIED and r.outcome == "certificate"
    assert r.reason == "convergence_not_established"
    assert r.certificate_check and r.certificate_check.passed and r.result_latex is None
    assert any(not c.established for c in r.conditions) or r.unresolved


def test_divergence_and_interior_poles_are_reported_not_evaluated() -> None:
    d = run(r"x^{-2}\exp(-a x)", "0", "oo")
    assert d.status is Status.UNSUPPORTED and d.reason == "divergent" and d.operator is None
    p = run(r"\frac{\exp(-a x)}{x^2-1}", "0", "oo")
    assert p.status is Status.CONDITIONS_NOT_SATISFIED
    assert p.reason == "singularity_not_excluded" and p.result_latex is None
    g = run(r"\exp(a x)", "0", "oo")
    assert g.reason == "divergent"


def test_oscillation_is_not_called_divergence() -> None:
    r = run(r"\exp(\imaginaryI a x)", "0", "oo")
    assert r.reason != "divergent" and r.result_latex is None


def test_no_bounds_gives_the_certificate_only() -> None:
    r = run(r"\exp(-a x^2)", None, None)
    assert r.status is Status.OK and r.outcome == "certificate"
    assert r.equation is None and r.boundary is None and r.result_latex is None


def test_input_errors_have_their_own_reasons() -> None:
    assert run(r"\exp(-a x)", "0", None).status is Status.PARSE_ERROR
    assert run(r"\exp(-a x)", "1", "0").status is Status.PARSE_ERROR
    assert run(r"\exp(-a x)", "0", "oo", parameter="x").status is Status.PARSE_ERROR
    assert run(r"\sin(a x)", "0", "1").reason == "not_hyperexponential"


def test_a_search_that_hits_its_limits_says_so() -> None:
    r = run(r"\exp(-x^3/3 - a x)", "0", "oo", max_order=1, max_degree=2)
    assert r.status is Status.UNSUPPORTED and r.reason == "no_telescoper_found"
    assert "does not prove" in r.message or "nothing about other families" in r.message
    assert r.families and all(f.outcome == "none" for f in r.families)


# -- nothing unverified is ever shown ----------------------------------------------------------------------
def test_a_candidate_that_fails_verification_is_dropped_and_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = cert.verify

    def broken(f, var, operator, certificate):  # type: ignore[no-untyped-def]
        check = real(f, var, operator, certificate)
        return check.model_copy(update={"passed": False, "note": "forced"})

    monkeypatch.setattr(op_module.cert, "verify", broken)
    r = run(r"\exp(-a x^2)", "-oo", "oo")
    assert r.reason == "certificate_rejected" and r.status is Status.NOT_VERIFIED
    assert r.operator is None and r.certificate is None and r.diagnostics.candidates_rejected > 0


def test_a_value_that_fails_its_checks_is_withheld_with_its_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = ValueVerification(note="forced failure", passed=False)
    monkeypatch.setattr(op_module.ev, "accept", lambda *args, **kw: failed)
    r = run(r"\exp(-a x^2)", "-oo", "oo")
    assert r.outcome == "equation" and r.reason == "value_not_verified"
    assert r.result_latex is None and r.result_text is None and r.solution is None
    assert r.equation is not None


def test_a_solution_that_misses_the_inhomogeneous_term_is_rejected() -> None:
    # SymPy's dsolve returns only the homogeneous Airy solution for I'' + a I = 1
    operator = DifferentialOperator(a, (a, sp.Integer(0), sp.Integer(1)))
    general = ode.general_solution(operator, sp.Integer(1))
    assert general is None or not ode.satisfies(operator, sp.Integer(1), general)
    good = DifferentialOperator(a, (sp.Integer(1), 2 * a))
    assert ode.satisfies(good, sp.Integer(0), sp.sqrt(sp.pi) / sp.sqrt(a))
    assert not ode.satisfies(good, sp.Integer(0), sp.sqrt(sp.pi) / a)


def test_quadrature_must_confirm_the_value() -> None:
    operator = DifferentialOperator(a, (sp.Integer(1), 2 * a))
    initial = ev.Initial(sp.Integer(1), [sp.sqrt(sp.pi)])
    wrong = sp.sqrt(sp.pi) * (1 + sp.Rational(1, 10**6)) / sp.sqrt(a)  # fails the equation too
    check = ev.accept(
        GAUSS, x, a, operator, sp.Integer(0), wrong, initial, -sp.oo, sp.oo, time.monotonic() + 30
    )
    assert not check.passed
    right = ev.accept(
        GAUSS,
        x,
        a,
        operator,
        sp.Integer(0),
        sp.sqrt(sp.pi) / sp.sqrt(a),
        initial,
        -sp.oo,
        sp.oo,
        time.monotonic() + 30,
    )
    assert right.passed and right.quadrature_points_agreed == right.quadrature_points_checked > 0


def test_diagnostics_and_log_fields_hold_no_expression() -> None:
    r = run(r"\exp(-a x^2)", "-oo", "oo")
    dump = r.diagnostics.model_dump_json()
    assert "exp" not in dump and "sqrt" not in dump and "x^" not in dump
    assert set(r.diagnostics.log_fields()) == {
        "stages",
        "highest_order",
        "points_checked",
        "reason",
    }


def test_it_runs_through_the_sandbox(engine) -> None:  # type: ignore[no-untyped-def]
    r = engine.creative_telescoping(
        TelescopingInput(
            input=r"\exp(-a x^2)",
            assumptions=POSITIVE,
            variable="x",
            parameter="a",
            lower="-oo",
            upper="oo",
        )
    )
    assert r.status is Status.OK and r.outcome == "evaluated" and r.duration_ms > 0


def test_input_validation_is_strict() -> None:
    with pytest.raises(ValueError, match="unknown assumption"):
        TelescopingInput(input="x", variable="x", parameter="a", assumptions={"a": ["magic"]})
    with pytest.raises(ValueError, match="noncommutative"):
        TelescopingInput(
            input="x", variable="x", parameter="a", assumptions={"a": ["noncommutative"]}
        )
    with pytest.raises(ValueError):
        TelescopingInput(input="x", variable="x", parameter="a", max_order=99)
