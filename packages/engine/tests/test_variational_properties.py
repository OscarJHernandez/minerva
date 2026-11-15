"""Bounded random families of functionals whose first variation is known independently."""

from __future__ import annotations

import sympy as sp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from minerva_engine.models import Limits
from minerva_engine.variational import VariationInput
from minerva_engine.variational.firstvar import first_variation
from minerva_engine.variational.jets import JetSpace
from minerva_engine.variational.setup import Problem, read_problem
from minerva_engine.variational.verify import VariationForm, check_first_variation

SETTINGS = settings(
    max_examples=8,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)

NAMES = ["x", "x_dot", "x_ddot"]


@st.composite
def polynomials(draw: st.DrawFn, top: int = 2) -> str:
    """A polynomial in x, x', ..., x^(top) and t, as text, with small integer coefficients."""
    terms = []
    for _ in range(draw(st.integers(min_value=1, max_value=3))):
        coefficient = draw(st.integers(min_value=-3, max_value=3).filter(lambda c: c != 0))
        factors = []
        for name in [*NAMES[: top + 1], "t"]:
            power = draw(st.integers(min_value=0, max_value=2 if name != "t" else 1))
            if power:
                factors.append(f"{name}**{power}")
        terms.append(f"{coefficient}*" + ("*".join(factors) if factors else "1"))
    return " + ".join(terms)


def build(source: str, upper: str = "1") -> Problem:
    data = VariationInput(
        integrand=source, format="text", functions=["x"], lower="0", upper=upper, action="first"
    )
    return read_problem(data, Limits())


@SETTINGS
@given(polynomials())
def test_random_polynomial_lagrangians_satisfy_the_variational_identity(source: str) -> None:
    problem = build(source)
    derived = first_variation(problem)
    check = check_first_variation(problem, VariationForm.of(derived))
    assert check.verdict == "passed", source


@SETTINGS
@given(polynomials(top=1))
def test_total_derivatives_have_no_interior_equation_and_a_pure_boundary_variation(
    source: str,
) -> None:
    jets = JetSpace.build("t", ("x",), (2,), 4)
    F = sp.sympify(
        source, locals={n: s for n, s in zip(NAMES, jets.u[0], strict=False)} | {"t": jets.variable}
    )
    total = jets.D(F)
    if total == 0:
        return
    problem = build(str(total))
    derived = first_variation(problem)
    assert derived.euler == [0]
    # δ∫ D F dt = [δF]_a^b: the coefficient of η^(j) at the upper end is ∂F/∂u^(j) there.
    for j in range(0, 2):
        expected = sp.diff(F, jets.u[0][j]).subs(jets.variable, problem.upper)
        got = derived.coefficients.get(("upper", 0, j), sp.Integer(0))
        assert sp.simplify(got - expected) == 0, (source, j)
