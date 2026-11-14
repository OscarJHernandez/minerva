"""First variation: closed-form references, boundary analysis, and deliberately wrong candidates.

The references below are written by hand (not produced by Minerva) in the convention
EL_i = Σ_k (-1)^k D^k ∂L/∂u_i^(k) and δS = ∫ EL·η dt + [Σ Π η^(j)]_a^b.
"""

from __future__ import annotations

from typing import Any

import pytest
import sympy as sp
from minerva_engine import Engine
from minerva_engine.models import Limits, Status
from minerva_engine.variational import VariationInput
from minerva_engine.variational.firstvar import (
    FirstVariationData,
    Stage,
    first_variation,
)
from minerva_engine.variational.setup import Problem, read_problem
from minerva_engine.variational.verify import (
    VariationForm,
    check_el_pointwise,
    check_first_variation,
    check_ibp_step,
    sympy_euler,
)


def problem(integrand: str, **kw: Any) -> Problem:
    kw.setdefault("functions", ["x"])
    kw.setdefault("lower", "0")
    kw.setdefault("upper", "1")
    return read_problem(VariationInput(integrand=integrand, action="first", **kw), Limits())


def same(a: sp.Expr, b: sp.Expr) -> bool:
    return bool(sp.simplify(a - b) == 0)


# -- closed forms -------------------------------------------------------------------------------
def test_free_particle() -> None:
    p = problem(r"\frac{m}{2}\dot{x}^2")
    d = first_variation(p)
    m = sp.Symbol("m")
    x1, x2 = p.jets.u[0][1], p.jets.u[0][2]
    assert same(d.euler[0], -m * x2)
    assert same(d.pi[(0, 0)], m * x1)
    assert same(d.coefficients[("upper", 0, 0)], m * x1)
    assert same(d.coefficients[("lower", 0, 0)], -m * x1)


def test_time_dependent_uses_the_total_derivative() -> None:
    p = problem(r"\frac{t}{2}\dot{x}^2", lower="1", upper="2")
    t = p.variable
    x1, x2 = p.jets.u[0][1], p.jets.u[0][2]
    d = first_variation(p)
    assert same(d.euler[0], -(t * x2 + x1))  # not -t*x2: d/dt of t*x' contains x'
    assert same(d.coefficients[("upper", 0, 0)], 2 * x1)  # t replaced by the end


def test_higher_derivative_lagrangian_matches_the_specification() -> None:
    p = problem(
        r"\frac{m}{2}\dot{x}^2+\frac{\alpha}{2}\ddot{x}^2-V(x)",
        given_functions=[{"name": "V", "args": ["x"]}],
    )
    m, alpha = sp.symbols("m alpha")
    x0, x1, x2, x3, x4 = p.jets.u[0][:5]
    V = sp.Function("V", real=True)
    d = first_variation(p)
    assert same(d.euler[0], alpha * x4 - m * x2 - sp.Derivative(V(x0), x0))
    assert same(d.pi[(0, 0)], m * x1 - alpha * x3)
    assert same(d.pi[(0, 1)], alpha * x2)
    # the boundary form is [(m x' - alpha x''') eta + alpha x'' eta']
    eta, eta1 = p.jets.eta[0][:2]
    assert same(d.boundary, (m * x1 - alpha * x3) * eta + alpha * x2 * eta1)
    assert d.el_orders == [4]


def test_integration_by_parts_stages_are_listed_one_derivative_at_a_time() -> None:
    p = problem(r"\frac{1}{2}\ddot{x}^2")
    d = first_variation(p)
    assert [(s.function, s.order) for s in d.stages] == [(-1, 0), (0, 2), (0, 1)]
    # after the first step the second derivative is gone and nothing was discarded
    eta = p.jets.eta[0]
    x2, x3 = p.jets.u[0][2], p.jets.u[0][3]
    assert same(d.stages[1].interior, -x3 * eta[1])
    assert same(d.stages[1].boundary, x2 * eta[1])
    assert same(d.stages[2].boundary, x2 * eta[1] - x3 * eta[0])
    assert same(d.stages[2].interior, p.jets.u[0][4] * eta[0])


def test_cantilever_natural_conditions() -> None:
    p = problem(
        r"\frac12\ddot{x}^2-q x",
        conditions=[
            {"function": "x", "order": 0, "end": "lower", "kind": "fixed", "value": "0"},
            {"function": "x", "order": 1, "end": "lower", "kind": "fixed", "value": "0"},
            {"function": "x", "order": 0, "end": "upper", "kind": "free"},
            {"function": "x", "order": 1, "end": "upper", "kind": "free"},
        ],
    )
    from minerva_engine.variational import boundary

    d = first_variation(p)
    x2, x3, x4 = p.jets.u[0][2], p.jets.u[0][3], p.jets.u[0][4]
    q = sp.Symbol("q")
    assert same(d.euler[0], x4 - q)
    entries = {(e.end, e.order): e for e in boundary.classify(p, d)}
    assert entries[("lower", 0)].effect == entries[("lower", 1)].effect == "removed_by_fixed"
    assert entries[("upper", 0)].effect == entries[("upper", 1)].effect == "natural_condition"
    assert same(entries[("upper", 0)].coefficient, -x3)  # x'''(1) = 0
    assert same(entries[("upper", 1)].coefficient, x2)  # x''(1) = 0


def test_fixing_the_value_does_not_fix_the_slope() -> None:
    p = problem(
        r"\frac12\ddot{x}^2",
        conditions=[
            {"function": "x", "order": 0, "end": "lower", "kind": "fixed", "value": "0"},
            {"function": "x", "order": 0, "end": "upper", "kind": "fixed", "value": "1"},
        ],
    )
    from minerva_engine.variational import boundary

    entries = {(e.end, e.order): e for e in boundary.classify(p, first_variation(p))}
    assert entries[("lower", 0)].effect == "removed_by_fixed"
    assert entries[("lower", 1)].effect == "kept_unspecified"  # still there, still undecided
    assert entries[("upper", 1)].effect == "kept_unspecified"


def test_endpoint_term_adds_a_natural_condition() -> None:
    p = problem(
        r"\frac12\dot{x}^2",
        upper_term=r"\frac{\kappa}{2}x^2-c x",
        conditions=[
            {"function": "x", "order": 0, "end": "lower", "kind": "fixed", "value": "0"},
            {"function": "x", "order": 0, "end": "upper", "kind": "free"},
        ],
    )
    d = first_variation(p)
    kappa, c = sp.symbols("kappa c")
    x0, x1 = p.jets.u[0][:2]
    assert same(d.coefficients[("upper", 0, 0)], x1 + kappa * x0 - c)
    assert d.coefficients[("lower", 0, 0)] == -x1


def test_endpoint_term_of_order_beyond_the_integrand_is_reported() -> None:
    p = problem(r"\frac12\dot{x}^2", lower_term=r"g\dot{x}")
    d = first_variation(p)
    # g * x' at the lower end multiplies eta' there, which the integrand never produces
    assert ("lower", 0, 1) in d.coefficients
    assert same(d.coefficients[("lower", 0, 1)], sp.Symbol("g"))


def test_null_lagrangians_have_pure_boundary_variations() -> None:
    for source in (r"x\dot{x}", r"x\ddot{x}+\dot{x}^2"):
        p = problem(source)
        d = first_variation(p)
        assert d.null and d.euler == [0]
    p = problem(r"x\dot{x}")
    d = first_variation(p)
    assert same(d.pi[(0, 0)], p.jets.u[0][0])


def test_mixed_orders_phase_space() -> None:
    p = problem(r"p\dot{x}-\frac12(p^2+x^2)", functions=["x", "p"])
    d = first_variation(p)
    x, px = p.jets.u[0][0], p.jets.u[1][0]
    xd, pd = p.jets.u[0][1], p.jets.u[1][1]
    assert same(d.euler[0], -pd - x)
    assert same(d.euler[1], xd - px)
    assert same(d.pi[(0, 0)], px)
    assert (1, 0) not in d.pi  # p has no derivative in L: no boundary term for p


def test_several_functions_with_different_orders() -> None:
    p = problem(r"\frac12\ddot{x}^2+\frac12\dot{y}^2+xy", functions=["x", "y"])
    d = first_variation(p)
    x, y = p.jets.u[0][0], p.jets.u[1][0]
    assert same(d.euler[0], p.jets.u[0][4] + y)
    assert same(d.euler[1], -p.jets.u[1][2] + x)
    assert d.el_orders[0] == 4 and d.el_orders[1] == 2


def test_nonlinear_arc_length() -> None:
    p = problem(r"\sqrt{1+\dot{x}^2}")
    d = first_variation(p)
    x1, x2 = p.jets.u[0][1], p.jets.u[0][2]
    assert same(d.euler[0], -x2 / (1 + x1**2) ** sp.Rational(3, 2))


# -- the checkers pass the truth and fail forged candidates --------------------------------------
@pytest.fixture(scope="module")
def beam() -> tuple[Problem, FirstVariationData]:
    p = problem(r"\frac{m}{2}\dot{x}^2+\frac{\alpha}{2}\ddot{x}^2-\frac{k}{2}x^2+t x")
    return p, first_variation(p)


def test_truth_passes_every_leg(beam: tuple[Problem, FirstVariationData]) -> None:
    p, d = beam
    form = VariationForm.of(d)
    check = check_first_variation(p, form)
    assert check.verdict == "passed" and check.points_agreed == check.points_checked >= 4
    assert check_el_pointwise(p, form.interior).passed
    assert all(
        check_ibp_step(p, d.stages[i], d.stages[i + 1]).passed for i in range(len(d.stages) - 1)
    )
    assert sympy_euler(p, form.interior) is True


def forge(form: VariationForm, **changes: Any) -> VariationForm:
    return VariationForm(
        changes.get("interior", form.interior), changes.get("boundary", dict(form.boundary))
    )


def test_wrong_integration_by_parts_sign_is_rejected(
    beam: tuple[Problem, FirstVariationData],
) -> None:
    p, d = beam
    form = VariationForm.of(d)
    jets = p.jets
    # D(P_2) enters Π_0 with a minus sign; the plus sign is the classic mistake.
    m, alpha = sp.symbols("m alpha")
    wrong_pi0 = m * jets.u[0][1] + alpha * jets.u[0][3]
    boundary = dict(form.boundary)
    boundary[("upper", 0, 0)] = wrong_pi0.subs(jets.variable, p.upper)
    boundary[("lower", 0, 0)] = -wrong_pi0.subs(jets.variable, p.lower)
    assert check_first_variation(p, forge(form, boundary=boundary)).verdict == "failed"


def test_missing_boundary_term_is_rejected(beam: tuple[Problem, FirstVariationData]) -> None:
    p, d = beam
    form = VariationForm.of(d)
    boundary = dict(form.boundary)
    boundary[("upper", 0, 1)] = sp.Integer(0)  # dropped alpha x'' eta'(b)
    assert check_first_variation(p, forge(form, boundary=boundary)).verdict == "failed"


def test_lower_end_sign_is_rejected(beam: tuple[Problem, FirstVariationData]) -> None:
    p, d = beam
    form = VariationForm.of(d)
    boundary = {k: (-v if k[0] == "lower" else v) for k, v in form.boundary.items()}
    assert check_first_variation(p, forge(form, boundary=boundary)).verdict == "failed"


def test_partial_derivative_where_the_total_derivative_is_meant_is_rejected() -> None:
    p = problem(r"\frac{t}{2}\dot{x}^2", lower="1", upper="2")
    d = first_variation(p)
    t = p.variable
    wrong = tuple(-t * p.jets.u[0][2] + 0 * e for e in d.euler)  # misses the d/dt of t
    form = forge(VariationForm.of(d), interior=wrong)
    assert check_first_variation(p, form).verdict == "failed"
    assert not check_el_pointwise(p, wrong).passed


def test_a_wrong_euler_lagrange_sign_is_rejected(beam: tuple[Problem, FirstVariationData]) -> None:
    p, d = beam
    form = VariationForm.of(d)
    wrong = tuple(-e for e in form.interior)
    assert check_first_variation(p, forge(form, interior=wrong)).verdict == "failed"


def test_a_tiny_error_in_a_coefficient_is_still_rejected() -> None:
    p = problem(r"\frac12\dot{x}^2")
    d = first_variation(p)
    form = VariationForm.of(d)
    boundary = dict(form.boundary)
    boundary[("upper", 0, 0)] = boundary[("upper", 0, 0)] * (1 + sp.Rational(1, 10**7))
    assert check_first_variation(p, forge(form, boundary=boundary)).verdict == "failed"


def test_a_forged_integration_by_parts_stage_is_rejected(
    beam: tuple[Problem, FirstVariationData],
) -> None:
    p, d = beam
    before, after = d.stages[0], d.stages[1]
    forged = Stage(after.index, after.function, after.order, after.interior, -after.boundary, 0)
    assert check_ibp_step(p, before, after).passed
    assert not check_ibp_step(p, before, forged).passed


def test_complex_parameters_do_not_break_the_identity() -> None:
    p = problem(r"\frac{m}{2}\dot{x}^2-\frac{k}{2}x^2")  # m, k complex: S is complex-valued
    d = first_variation(p)
    assert check_first_variation(p, VariationForm.of(d)).verdict == "passed"


# -- through the engine ----------------------------------------------------------------------------
def test_engine_returns_the_derivation_with_the_specified_boundary_form(engine: Engine) -> None:
    r = engine.variation(
        VariationInput(
            action="first",
            integrand=r"\frac{m}{2}\dot{x}^2+\frac{\alpha}{2}\ddot{x}^2-V(x)",
            functions=["x"],
            given_functions=[{"name": "V", "args": ["x"]}],
            lower="a",
            upper="b",
        )
    )
    assert r.status is Status.OK and r.first is not None
    f = r.first
    assert f.equations[0].equation_latex == r"\alpha x^{(4)} - m \ddot{x} - V'\left(x\right) = 0"
    assert "\\ddot{x}" in f.boundary_form_latex and f.boundary_form_latex.endswith("_{a}^{b}")
    assert [s.moved_order for s in f.stages] == [None, 2, 1]
    assert all(s.checked for s in f.stages[1:])
    assert f.checks.deciding == "passed" and f.verification.passed
    assert "numerical evidence" in f.verification.note.lower()
    assert {b.effect for b in f.boundary} == {"kept_unspecified"}  # nothing was declared
    assert not f.stationarity_latex or all("= 0" in s for s in f.stationarity_latex)


def test_engine_flags_boundary_terms_that_stay(engine: Engine) -> None:
    r = engine.variation(
        VariationInput(
            action="first",
            integrand=r"\frac12\ddot{x}^2",
            functions=["x"],
            lower="0",
            upper="1",
            conditions=[
                {"function": "x", "order": 0, "end": "lower", "kind": "fixed", "value": "0"},
                {"function": "x", "order": 0, "end": "upper", "kind": "fixed", "value": "1"},
            ],
        )
    )
    assert r.first is not None
    kept = [(b.end, b.order) for b in r.first.boundary if b.effect == "kept_unspecified"]
    assert kept == [("lower", 1), ("upper", 1)]
    assert any("kept" in n for n in r.first.notes)


def test_a_null_lagrangian_goes_through_the_whole_action(engine: Engine) -> None:
    r = engine.variation(
        VariationInput(action="first", integrand=r"x\dot{x}", functions=["x"], lower="0", upper="1")
    )
    assert r.status is Status.OK and r.first is not None
    assert r.first.null_lagrangian and r.first.equations[0].expression_latex == "0"
    assert r.first.stationarity_latex == []  # nothing interior to satisfy
    assert r.first.verification.symbolic_zero is False  # a second derivation route is not a proof
