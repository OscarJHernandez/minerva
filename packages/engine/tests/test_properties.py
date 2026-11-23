"""Property and metamorphic tests: general mathematical statements checked on generated objects.

Each test states an invariant and checks it against an oracle that does not share the computation under test
(exact noncommutative expansion, textbook curvature of the sphere, a Legendre transformation done by hand, an
mpmath finite difference). Generators are bounded (`strategies/`); the pr profile runs 8 derandomized examples,
the extended profile (`MINERVA_HYPOTHESIS_PROFILE=extended`) many more with a reproduction blob.
"""

import mpmath
import pytest
import sympy as sp
from hypothesis import assume, example, given
from hypothesis import strategies as st
from minerva_engine import Engine, ExpressionInput, Limits, Status
from minerva_engine.counterexample import Budget, check_counterexample, find_counterexample
from minerva_engine.geometry import GeometryInput, geometry
from minerva_engine.lagrangian import analyze_lagrangian
from minerva_engine.latexparse import parse_latex
from minerva_engine.matrix import MatrixCalculusInput, MatrixDecl
from minerva_engine.ncverify import verify_equal_nc
from minerva_engine.testing import budget_limits
from minerva_engine.textparse import parse_text
from minerva_engine.verify import verify_equal
from strategies.algebra import X, cancelled_rationals, polynomials
from strategies.calculus import antiderivable
from strategies.geometry import sphere_charts
from strategies.matrices import A, B, explicit_matrices, nc_polynomials
from strategies.mechanics import anharmonic_lagrangians

LIMITS = Limits(timeout_seconds=60)


# --- integration and differentiation ------------------------------------------------------------------------
@example((X * sp.exp(X) + 8 * sp.exp(4 * X) + sp.exp(X), X * sp.exp(X) + 2 * sp.exp(4 * X)))
@given(antiderivable())
def test_an_antiderivative_differentiates_back_to_the_integrand(
    engine: Engine, case: tuple[sp.Expr, sp.Expr]
) -> None:
    integrand, F = case
    assume(integrand != 0)
    result = engine.integrate(ExpressionInput(input=str(integrand), format="text"), "x")
    assert result.status is Status.OK, result.message
    assert result.verification is not None and result.verification.passed
    assert result.result_text is not None
    found = parse_text(result.result_text)
    assert verify_equal(sp.diff(found, X), integrand, symbolic=False).passed
    # Two antiderivatives of one integrand differ by a constant: their derivatives agree everywhere.
    # Compared side against side, not as a difference against 0: for exp(4x) the difference of two large
    # derivatives (about e^200 at |x| = 50) is cancellation noise that the sampler read as a disagreement
    # on the nightly runs of 2026-11-18 and 2026-11-22 (pinned by the @example above).
    assert verify_equal(sp.diff(found, X), sp.diff(F, X), symbolic=False).passed


# --- restrictions propagate -----------------------------------------------------------------------------------
@given(cancelled_rationals())
def test_a_cancelled_factor_is_reported_as_a_restriction(
    engine: Engine, case: tuple[str, int]
) -> None:
    text, root = case
    result = engine.simplify(ExpressionInput(input=text, format="text"))
    assert result.status in (Status.OK, Status.UNCHANGED), result.message
    assert f"x \\neq {root}" in result.restrictions_latex  # the point the cancelled factor excluded
    assert result.restrictions_complete


# --- the numerical verifier itself --------------------------------------------------------------------------
@given(polynomials(max_degree=4), st.integers(5, 12))
def test_the_verifier_accepts_equivalent_forms_and_rejects_small_perturbations(
    p: sp.Expr, k: int
) -> None:
    assert verify_equal(p, sp.factor(p), symbolic=False).passed
    assert verify_equal(p, sp.expand(p * (1 + X) - p * X), symbolic=False).passed
    assert not verify_equal(
        p, p + sp.Rational(1, 10**k), symbolic=False
    ).passed  # absolute error 10^-k
    assert not verify_equal(p, p * (1 + sp.Rational(1, 10**k)), symbolic=False).passed  # relative


# --- counterexample search ----------------------------------------------------------------------------------
@given(polynomials(max_degree=3), polynomials(max_degree=3), st.integers(1, 3))
def test_the_search_finds_a_difference_of_polynomials_and_reports_a_real_one(
    p: sp.Expr, q: sp.Expr, c: int
) -> None:
    other = p + c * q  # q is a nonzero polynomial, so the two sides differ almost everywhere
    result = find_counterexample(p, other, budget=Budget(points=96, seconds=20.0))
    found = result.counterexample
    assert found is not None
    assert check_counterexample(p, other, found.values)  # re-checked independently of the search
    assert (p - other).xreplace(found.values) != 0  # exact arithmetic agrees


@given(polynomials(max_degree=4))
def test_the_search_never_reports_a_counterexample_to_an_identity(p: sp.Expr) -> None:
    for equal in (sp.expand(p), sp.factor(p), sp.expand(p * (X + 1)) / (X + 1) if p != 0 else p):
        result = find_counterexample(p, equal, budget=Budget(points=96, seconds=20.0))
        assert result.counterexample is None, result.counterexample


# --- noncommuting algebra ----------------------------------------------------------------------------------
@st.composite
def nc_pairs(draw: st.DrawFn) -> tuple[sp.Expr, sp.Expr]:
    p = draw(nc_polynomials())
    kind = draw(st.sampled_from(["random", "same", "swapped", "commuted"]))
    if kind == "same":
        return p, sp.expand(p)
    if kind == "swapped":
        return p, p.xreplace({A: B, B: A})
    if kind == "commuted":  # every product with its factors put in alphabetical order
        return p, _alphabetical(p)
    return p, draw(nc_polynomials())


def _alphabetical(expr: sp.Expr) -> sp.Expr:
    """Each product with its factors sorted: equal to `expr` only if the symbols happened to commute there."""
    terms = []
    for term in sp.Add.make_args(sp.expand(expr)):
        coeff, rest = term.as_coeff_Mul()
        factors = sorted(sp.Mul.make_args(rest), key=lambda f: str(f))
        terms.append(coeff * sp.Mul(*factors))
    return sp.Add(*terms)


@given(nc_pairs())
def test_the_noncommutative_verifier_agrees_with_exact_expansion(
    pair: tuple[sp.Expr, sp.Expr],
) -> None:
    p, q = pair
    truth = sp.expand(p - q) == 0  # exact: sympy keeps the order of noncommuting factors
    assert verify_equal_nc(p, q).passed == truth


# --- differential geometry ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("radius", "expected"),
    [("a", "2/a**2"), ("2 a", "1/(2*a**2)"), (r"\frac{3}{2}", "8/9")],
)
def test_the_scalar_curvature_of_a_sphere_does_not_depend_on_the_chart(
    radius: str, expected: str
) -> None:
    """R = 2 / r^2 is a textbook value; the angle chart and the stereographic chart must both give it."""
    values = []
    for coordinates, metric in sphere_charts(radius):
        data = GeometryInput(
            coordinates=coordinates,
            metric=metric,
            quantities=["ricci_scalar", "kretschmann"],
            assumptions={"a": ["positive"]},
        )
        report = geometry(data, LIMITS)
        assert report.status is Status.OK, report.message
        scalars = {q.kind: q.scalar_text for q in report.quantities}
        assert all(
            q.verification is not None and q.verification.numeric_passed for q in report.quantities
        )
        values.append(scalars)
    a = sp.Symbol("a", positive=True)
    r2 = sp.sympify(expected, locals={"a": a})
    for scalars in values:
        assert scalars["ricci_scalar"] is not None and scalars["kretschmann"] is not None
        assert sp.simplify(sp.sympify(scalars["ricci_scalar"], locals={"a": a}) - r2) == 0
        # in two dimensions Riemann is fixed by R: Kretschmann = R^2
        assert sp.simplify(sp.sympify(scalars["kretschmann"], locals={"a": a}) - r2**2) == 0


# --- mechanics ----------------------------------------------------------------------------------------------
@given(anharmonic_lagrangians())
def test_the_accelerations_match_the_legendre_transformation_done_by_hand(
    case: tuple[str, sp.Expr, sp.Expr],
) -> None:
    latex, m, V = case
    x, v, p = sp.symbols("x v p")
    # Independent route: p = dL/dv, H = p v - L, Hamilton's equations xdot = dH/dp, pdot = -dH/dx.
    L = m * v**2 / 2 - V
    momentum = sp.diff(L, v)
    H = sp.simplify((p * v - L).subs(v, sp.solve(sp.Eq(p, momentum), v)[0]))
    acceleration = (-sp.diff(H, x)).subs(p, m * v) / m  # xddot = pdot / m
    report = analyze_lagrangian(latex, ["x"], "t", {}, budget_limits())
    assert report.status is Status.OK
    rhs = report.accelerations[0].latex.split("=", 1)[1]
    assert verify_equal(parse_latex(rhs), acceleration, symbolic=False).passed
    assert any(c.kind == "energy" and c.verified for c in report.conserved)


# --- matrix calculus ----------------------------------------------------------------------------------------
@given(
    st.sampled_from(["any", "symmetric", "singular", "nilpotent", "defective"]).flatmap(
        lambda kind: explicit_matrices(kind=kind)
    )
)
def test_the_derivative_of_a_matrix_exponential_matches_a_finite_difference(
    engine: Engine, matrix: sp.Matrix
) -> None:
    request = MatrixCalculusInput(
        expression=ExpressionInput(input=r"\exp(tA)", assumptions={"t": ["real"]}),
        matrices=[
            MatrixDecl(name="A", entries=[[str(e) for e in matrix.row(i)] for i in range(2)])
        ],
        action="differentiate",
        variable="t",
    )
    result = engine.matrix_calculus(request)
    assert result.status is Status.OK, result.message
    assert result.verification is not None and result.verification.passed
    # A is a single declared matrix, so the engine's answer is the closed form A exp(tA). Whatever it says,
    # it must equal a finite difference of mpmath's own matrix exponential (nothing shared with the engine).
    assert result.result_latex == r"A \exp\left(t A\right)"
    t0, h = mpmath.mpf(1) / 3, mpmath.mpf(10) ** -10
    with mpmath.workdps(40):
        M = mpmath.matrix(matrix.tolist())
        reference = (mpmath.expm((t0 + h) * M) - mpmath.expm((t0 - h) * M)) / (2 * h)
        claimed = M * mpmath.expm(t0 * M)
        for i in range(2):
            for j in range(2):
                assert abs(claimed[i, j] - reference[i, j]) <= 1e-8 * max(1, abs(reference[i, j]))
