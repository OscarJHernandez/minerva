"""The brachistochrone functional in the form chat sends it: x as a function of y, written with the
Leibniz derivative, bounds not given. It used to be read as sqrt(1 + x^2/y^2) / sqrt(2 g y) (the d's
cancelled) and, once read correctly, its internal check sampled an interval across the singularity at
y = 0 and reported "did not pass" for what was only inconclusive."""

from __future__ import annotations

import sympy as sp
from minerva_engine import Engine
from minerva_engine.models import Status
from minerva_engine.testing import budget_limits
from minerva_engine.variational import VariationInput
from minerva_engine.variational.firstvar import first_variation
from minerva_engine.variational.setup import read_problem
from minerva_engine.variational.verify import (
    VariationForm,
    check_el_pointwise,
    check_first_variation,
    check_ibp_step,
    regular_numerics,
)

LEIBNIZ = r"\frac{\sqrt{1 + \left(\frac{dx}{dy}\right)^2}}{\sqrt{2 g y}}"
PRIME = r"\frac{\sqrt{1 + (x')^2}}{\sqrt{2 g y}}"


def make(integrand: str, **kw: object):  # type: ignore[no-untyped-def]
    data = VariationInput(
        integrand=integrand,
        action="first",
        variable="y",
        functions=["x"],
        **kw,  # type: ignore[arg-type]
    )
    return read_problem(data, budget_limits())


def test_the_leibniz_spelling_is_the_derivative_not_a_ratio() -> None:
    p = make(LEIBNIZ)
    xdot = p.jets.u[0][1]
    names = {s.name for s in p.lagrangian.free_symbols}
    assert xdot in p.lagrangian.free_symbols  # the derivative is there at all
    assert "x" not in names  # not the ratio x/y that the cancelled d's used to give
    assert make(PRIME).lagrangian == p.lagrangian


def test_euler_lagrange_matches_the_hand_written_expression(engine: Engine) -> None:
    # EL = -D[ x' / (sqrt(2 g y) sqrt(1 + x'^2)) ] (the project's sign convention)
    p = make(LEIBNIZ, assumptions={"g": ["positive"], "y": ["positive"]})
    derived = first_variation(p)
    y = next(s for s in p.lagrangian.free_symbols if s.name == "y")
    g = next(s for s in p.lagrangian.free_symbols if s.name == "g")
    xd, xdd = p.jets.u[0][1], p.jets.u[0][2]
    momentum = xd / (sp.sqrt(2 * g * y) * sp.sqrt(1 + xd**2))
    expected = -(
        sp.diff(momentum, y) + sp.diff(momentum, xd) * xdd  # D = d/dy + x'' d/dx'
    )
    assert sp.simplify(derived.euler[0] - expected) == 0


def run(engine: Engine, **kw: object):  # type: ignore[no-untyped-def]
    return engine.variation(
        VariationInput(
            action="first",
            integrand=LEIBNIZ,
            variable="y",
            functions=["x"],
            **kw,  # type: ignore[arg-type]
        )
    )


def test_without_bounds_it_is_verified_on_a_regular_interval(engine: Engine) -> None:
    for assumptions in ({}, {"g": ["positive"]}):
        r = run(engine, assumptions=assumptions)
        assert r.status == Status.OK, r.message
        assert r.first is not None and r.first.checks.deciding == "passed"
        assert r.input_latex is not None and "dot" in r.input_latex.replace("\\", "")


def test_with_bounds_across_the_singularity_it_is_never_ok(engine: Engine) -> None:
    r = run(engine, lower="-1", upper="1", assumptions={"g": ["positive"]})
    assert r.status == Status.NOT_VERIFIED
    assert "could not be checked" in r.message and "bounds" in r.message
    assert r.input_latex is not None  # the input as read is shown on every status


def test_a_regular_interval_is_chosen_for_every_leg() -> None:
    p = make(LEIBNIZ)
    for seed in (20260301, 20260302, 20260303, 20260501):
        n = regular_numerics(p, seed)
        assert n.interval in {"positive", "negative"} or (n.lower > 0 or n.upper < 0)
        assert not (n.lower < 0 < n.upper)  # never across the singularity at y = 0


def test_every_leg_agrees_once_the_interval_is_regular() -> None:
    p = make(LEIBNIZ, assumptions={"g": ["positive"]})
    derived = first_variation(p)
    form = VariationForm.of(derived)
    assert check_first_variation(p, form).passed
    assert check_el_pointwise(p, form.interior).passed
    assert all(
        check_ibp_step(p, derived.stages[k], derived.stages[k + 1]).passed
        for k in range(len(derived.stages) - 1)
    )


def test_a_wrong_candidate_is_still_failed_on_the_regular_interval() -> None:
    p = make(LEIBNIZ, assumptions={"g": ["positive"]})
    derived = first_variation(p)
    good = VariationForm.of(derived)
    wrong = VariationForm(
        tuple(e * (1 + sp.Rational(1, 10**7)) for e in good.interior), good.boundary
    )
    assert check_first_variation(p, good).passed
    assert check_first_variation(p, wrong).verdict == "failed"


def test_user_numbers_are_never_replaced() -> None:
    p = make(LEIBNIZ, lower="1", upper="2", assumptions={"g": ["positive"]})
    n = regular_numerics(p, 20260301)
    assert (float(n.lower), float(n.upper)) == (1.0, 2.0)
