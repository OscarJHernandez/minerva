"""Noncommutative matrix calculus: exponentials, derivatives, integrals, Sylvester reduction."""

from __future__ import annotations

import random
from typing import Any

import mpmath as mp
import pytest
import sympy as sp
from minerva_engine import Engine, ExpressionInput, Limits, Status
from minerva_engine.errors import ParseError
from minerva_engine.latexparse import parse_latex
from minerva_engine.matrix import MatrixCalculusInput, MatrixCalculusResult, MatrixDecl
from minerva_engine.matrix.expressions import Commutation, canonicalize, to_terms
from minerva_engine.matrix.verify import (
    Evaluator,
    Setup,
    check_derivative,
    check_equal,
    close,
    make_env,
    series_terms_equal,
)
from minerva_engine.textparse import parse_text

t = sp.Symbol("t")


def abstract(name: str, rows: int, cols: int | None = None) -> MatrixDecl:
    return MatrixDecl(name=name, rows=rows, cols=cols)


def explicit(name: str, rows: list[list[str]]) -> MatrixDecl:
    return MatrixDecl(name=name, entries=rows)


def mc(
    engine: Engine, source: str, matrices: list[MatrixDecl], fmt: str = "latex", **kw: Any
) -> MatrixCalculusResult:
    request = MatrixCalculusInput(
        expression=ExpressionInput(input=source, format=fmt), matrices=matrices, **kw
    )
    return engine.matrix_calculus(request)


def entries(result: MatrixCalculusResult) -> sp.Matrix:
    assert result.result_entries is not None
    return sp.Matrix([[sp.sympify(e) for e in row] for row in result.result_entries])


def same(a: sp.Matrix, b: sp.Matrix) -> bool:
    return a.shape == b.shape and all(sp.simplify(x - y) == 0 for x, y in zip(a, b, strict=True))


AB = [abstract("A", 2), abstract("B", 2)]
DIAG = explicit("A", [["1", "0"], ["0", "2"]])
NILPOTENT = explicit("B", [["0", "1"], ["0", "0"]])


# -- 1, 2, 8: explicit matrix exponentials ----------------------------------------------------
def test_exponential_of_a_diagonal_matrix(engine: Engine) -> None:
    r = mc(engine, r"\exp(tA)", [DIAG], action="evaluate")
    assert r.status is Status.OK and r.result_kind == "explicit" and r.evaluated
    assert same(entries(r), sp.Matrix([[sp.exp(t), 0], [0, sp.exp(2 * t)]]))
    assert r.verification is not None and r.verification.passed


def test_exponential_of_a_non_diagonalisable_matrix(engine: Engine) -> None:
    r = mc(engine, r"\exp(tB)", [NILPOTENT], action="evaluate")
    assert r.status is Status.OK
    assert same(entries(r), sp.Matrix([[1, t], [0, 1]]))
    jordan = explicit("B", [["2", "1"], ["0", "2"]])
    r = mc(engine, r"\exp(B)", [jordan], action="evaluate")
    assert r.status is Status.OK
    assert same(entries(r), sp.Matrix([[sp.exp(2), sp.exp(2)], [0, sp.exp(2)]]))


def test_ordered_product_of_explicit_exponentials(engine: Engine) -> None:
    r = mc(engine, r"\exp(A)\exp(B)", [DIAG, NILPOTENT], action="evaluate")
    assert r.status is Status.OK
    assert same(entries(r), sp.Matrix([[sp.E, sp.E], [0, sp.exp(2)]]))
    swapped = mc(engine, r"\exp(B)\exp(A)", [DIAG, NILPOTENT], action="evaluate")
    assert same(entries(swapped), sp.Matrix([[sp.E, sp.exp(2)], [0, sp.exp(2)]]))


def test_evaluate_needs_every_entry(engine: Engine) -> None:
    r = mc(engine, r"\exp(A)", [abstract("A", 2)], action="evaluate")
    assert r.status is Status.UNSUPPORTED and r.result_latex is None


# -- 3: differentiation -----------------------------------------------------------------------
def test_derivative_of_an_abstract_exponential(engine: Engine) -> None:
    r = mc(engine, r"\exp(tA)", [abstract("A", 3)], action="differentiate", variable="t")
    assert r.status is Status.OK and r.method == "exp_derivative"
    assert r.result_latex == r"A \exp\left(t A\right)"
    assert r.verification is not None and r.verification.passed
    assert r.verification.symbolic_zero is True


def test_derivative_of_a_product_keeps_the_order(engine: Engine) -> None:
    r = mc(engine, r"\exp(tA)\exp(sB)", AB, action="differentiate", variable="t")
    assert r.status is Status.OK
    # d/dt e^{tA} e^{sB} = A e^{tA} e^{sB}: A stays to the left, B is not touched
    assert r.result_latex == r"A \exp\left(t A\right) \exp\left(s B\right)"
    r = mc(engine, r"\exp(tA)\exp(tB)", AB, action="differentiate", variable="t")
    assert r.status is Status.OK
    assert r.result_latex == (
        r"A \exp\left(t A\right) \exp\left(t B\right) + \exp\left(t A\right) B "
        r"\exp\left(t B\right)"
    )


def test_derivative_of_a_scalar_function_in_the_exponent(engine: Engine) -> None:
    r = mc(engine, r"\exp(t^2 A)", [abstract("A", 2)], action="differentiate", variable="t")
    assert r.status is Status.OK
    assert r.result_latex is not None and r.result_latex.startswith("2 t A")


def test_derivative_of_a_noncommuting_exponent_is_the_frechet_integral(engine: Engine) -> None:
    r = mc(engine, r"\exp(t^2 A + t B)", AB, action="differentiate", variable="t")
    assert r.status is Status.OK and r.method == "exp_frechet_derivative"
    # an integral representation: never presented as an evaluated matrix, never as X' exp(X)
    assert r.result_kind == "representation" and not r.evaluated
    assert r.result_latex is not None and r.result_latex.startswith(r"\int_0^1")
    assert any(b.method == "exp_derivative" for b in r.blocked)
    assert r.verification is not None and r.verification.passed
    assert r.verification.symbolic_zero is True
    # declared commutation makes it a different problem, with the familiar answer
    r = mc(
        engine,
        r"\exp(t^2 A + t B)",
        AB,
        action="differentiate",
        variable="t",
        commuting=[("A", "B")],
    )
    assert r.status is Status.OK and r.method == "exp_derivative"
    assert r.result_kind == "expression" and r.evaluated


# -- 4, 5: integration of one exponential -----------------------------------------------------
def test_integral_of_one_exponential_with_an_invertible_matrix(engine: Engine) -> None:
    r = mc(
        engine,
        r"\exp(tA)",
        [abstract("A", 2)],
        action="integrate",
        variable="t",
        lower="a",
        upper="b",
        invertible=["A"],
    )
    assert r.status is Status.OK and r.method == "exp_integral_invertible"
    assert r.result_latex == r"A^{-1} \exp\left(b A\right) - A^{-1} \exp\left(a A\right)"
    assert any("invertible" in a for a in r.assumptions_used)
    assert r.verification is not None and r.verification.passed
    assert r.verification.symbolic_zero is True


def test_integral_of_one_exponential_without_invertibility_is_a_reduction(engine: Engine) -> None:
    r = mc(
        engine,
        r"\exp(tA)",
        [abstract("A", 2)],
        action="integrate",
        variable="t",
        lower="0",
        upper="1",
    )
    assert r.status is Status.OK and r.method == "block_exponential"
    assert r.result_kind == "representation" and not r.evaluated
    assert r.equation_latex == r"A X = \exp\left(A\right) - I"
    assert any(b.method == "exp_integral_invertible" for b in r.blocked)


def test_integral_with_a_singular_matrix(engine: Engine) -> None:
    singular = explicit("A", [["0", "1"], ["0", "0"]])
    r = mc(engine, r"\exp(tA)", [singular], action="integrate", variable="t", lower="0", upper="1")
    assert r.status is Status.OK and r.result_kind == "explicit"
    assert same(entries(r), sp.Matrix([[1, sp.Rational(1, 2)], [0, 1]]))
    # declaring a singular matrix invertible is an error, not an assumption
    r = mc(
        engine,
        r"\exp(tA)",
        [singular],
        action="integrate",
        variable="t",
        lower="0",
        upper="1",
        invertible=["A"],
    )
    assert r.status is Status.PARSE_ERROR


def test_integral_of_a_diagonal_exponential_uses_the_invertible_closed_form(engine: Engine) -> None:
    r = mc(engine, r"\exp(tA)", [DIAG], action="integrate", variable="t", lower="0", upper="1")
    assert r.status is Status.OK and r.method == "exp_integral_invertible"
    assert r.verification is not None and r.verification.passed


def test_indefinite_integral_and_polynomial_scalars(engine: Engine) -> None:
    r = mc(engine, r"3 t^2 A + B", AB, action="integrate", variable="t")
    assert r.status is Status.OK and r.method == "integrate_polynomial"
    assert r.result_latex == r"t^{3} A + t B"
    r = mc(
        engine,
        r"\exp(2 t A)",
        [abstract("A", 2)],
        action="integrate",
        variable="t",
        invertible=["A"],
    )
    assert r.status is Status.OK
    assert r.result_latex == r"\frac{1}{2} A^{-1} \exp\left(2 t A\right)"
    # no closed antiderivative without invertibility: say so, never guess
    r = mc(engine, r"\exp(tA)", [abstract("A", 2)], action="integrate", variable="t")
    assert r.status is Status.UNSUPPORTED and r.result_latex is None


# -- 6, 10: noncommuting products -------------------------------------------------------------
def test_products_of_noncommuting_exponentials_are_not_merged(engine: Engine) -> None:
    r = mc(engine, r"\exp(A)\exp(B)", AB, action="simplify")
    assert r.status is Status.CONDITIONS_NOT_SATISFIED
    assert r.result_latex is None
    assert [b.needs for b in r.blocked] == ["AB = BA (not declared)"]


def test_merge_requires_a_declared_commutation(engine: Engine) -> None:
    r = mc(engine, r"\exp(A)\exp(B)", AB, action="simplify", commuting=[("A", "B")])
    assert r.status is Status.OK
    assert r.result_latex == r"\exp\left(A + B\right)"
    assert "AB = BA (declared)" in r.assumptions_used
    assert r.verification is not None and r.verification.symbolic_zero is True


def test_commutation_is_read_off_explicit_entries(engine: Engine) -> None:
    other = explicit("B", [["3", "0"], ["0", "4"]])
    r = mc(engine, r"\exp(A)\exp(B)", [DIAG, other], action="simplify")
    assert r.status is Status.OK
    assert "AB = BA (from the entries)" in r.assumptions_used
    # entries that do not commute cannot be declared commuting
    r = mc(engine, r"\exp(A)\exp(B)", [DIAG, NILPOTENT], action="simplify", commuting=[("A", "B")])
    assert r.status is Status.PARSE_ERROR


def test_same_matrix_always_merges_and_nothing_else_moves(engine: Engine) -> None:
    r = mc(engine, r"\exp(tA) A \exp(sA)", [abstract("A", 2)], action="simplify")
    assert r.status is Status.OK
    assert r.result_latex == r"A \exp\left(\left(s + t\right) A\right)"
    # A and B do not commute, so B never moves past exp(A), and the order is kept
    r = mc(engine, r"\exp(A) B \exp(A)", AB, action="simplify")
    assert r.status is Status.UNCHANGED


def test_commutation_must_be_declared_for_every_pair(engine: Engine) -> None:
    mats = [abstract("A", 2), abstract("B", 2), abstract("C", 2)]
    r = mc(engine, r"\exp(A)", mats, action="simplify", commuting=[("A", "B"), ("B", "C")])
    assert r.status is Status.UNSUPPORTED
    assert "every pair" in r.message


# -- 7, 9: Sylvester reduction ----------------------------------------------------------------
def test_sylvester_reduction_of_abstract_matrices(engine: Engine) -> None:
    mats = [abstract("P", 2), abstract("Q", 3), abstract("C", 2, 3)]
    r = mc(
        engine,
        r"\exp(tP) C \exp(tQ)",
        mats,
        action="integrate",
        variable="t",
        lower="a",
        upper="b",
    )
    assert r.status is Status.OK and not r.evaluated
    assert r.result_kind == "representation"
    assert r.equation_latex is not None
    assert r.equation_latex.startswith(r"P \mathcal{X}") or r.equation_latex.startswith("P X")
    assert "feature" not in r.message
    assert "matrix:sylvester-form" in r.features
    assert r.verification is not None and r.verification.passed
    assert r.verification.symbolic_zero is True
    assert [s.method for s in r.steps][:2] == ["sylvester_reduction", "sylvester_reduction"]


def test_the_headline_problem_with_abstract_noncommuting_matrices(engine: Engine) -> None:
    r = mc(
        engine,
        r"\exp(-(x-y)(y-w)A)\exp(-(w-z)(z-q)B)",
        AB,
        action="integrate",
        variable="w",
        lower="a",
        upper="b",
    )
    assert r.status is Status.OK
    assert r.result_kind == "representation" and not r.evaluated
    assert r.method == "block_exponential"
    assert r.equation_latex is not None
    assert r"\left(x - y\right) A" in r.equation_latex
    assert r"\left(q - z\right)" in r.equation_latex
    assert r.verification is not None and r.verification.passed
    assert r.verification.symbolic_zero is True
    assert any("not an evaluated" in n for n in r.notes)
    assert "commutation is not assumed for AB" in r.structure


def test_the_regression_problem(engine: Engine) -> None:
    """The integral of exp(wA) exp(-wB) over [0, 1], A = diag(1, 2), B = [[0, 1], [0, 0]]."""
    r = mc(
        engine,
        r"\exp(wA)\exp(-wB)",
        [DIAG, NILPOTENT],
        action="integrate",
        variable="w",
        lower="0",
        upper="1",
    )
    assert r.status is Status.OK and r.result_kind == "explicit" and r.evaluated
    expected = sp.Matrix([[sp.E - 1, -1], [0, (sp.exp(2) - 1) / 2]])
    assert same(entries(r), expected)
    assert r.method == "sylvester_reduction"
    assert r.equation_latex is not None
    assert r.verification is not None
    assert r.verification.passed and r.verification.symbolic_zero is True
    assert r.verification.points_agreed == r.verification.points_checked >= 4


def test_the_regression_problem_in_the_other_order_is_a_different_matrix(engine: Engine) -> None:
    r = mc(
        engine,
        r"\exp(-wB)\exp(wA)",
        [DIAG, NILPOTENT],
        action="integrate",
        variable="w",
        lower="0",
        upper="1",
    )
    assert r.status is Status.OK
    assert not same(entries(r), sp.Matrix([[sp.E - 1, -1], [0, (sp.exp(2) - 1) / 2]]))


def test_a_singular_sylvester_operator_still_gets_an_evaluation(engine: Engine) -> None:
    # eigenvalues of P and -Q coincide, so P X + X Q = RHS does not determine X
    p = explicit("P", [["1", "0"], ["0", "2"]])
    q = explicit("Q", [["-1", "0"], ["0", "0"]])
    r = mc(
        engine, r"\exp(tP)\exp(tQ)", [p, q], action="integrate", variable="t", lower="0", upper="1"
    )
    assert r.status is Status.OK and r.result_kind == "explicit"
    assert r.method == "block_exponential"
    assert same(entries(r), sp.Matrix([[1, 0], [0, (sp.exp(2) - 1) / 2]]))
    assert r.verification is not None and r.verification.passed


def test_the_sylvester_form_with_constant_matrices_around(engine: Engine) -> None:
    mats = [
        abstract("A", 2),
        abstract("B", 2),
        abstract("C", 3, 2),
        abstract("D", 2),
        abstract("F", 2, 4),
    ]
    r = mc(
        engine,
        r"3 C \exp(tA) D \exp(tB) F",
        mats,
        action="integrate",
        variable="t",
        lower="0",
        upper="s",
    )
    assert r.status is Status.OK and r.verification is not None and r.verification.passed
    assert r.result_latex is not None and r.result_latex.startswith("3 C")


def test_symbolic_entries_stay_symbolic(engine: Engine) -> None:
    a = explicit("A", [["a", "0"], ["0", "2"]])
    r = mc(
        engine,
        r"\exp(tA)\exp(-tB)",
        [a, NILPOTENT],
        action="integrate",
        variable="t",
        lower="0",
        upper="1",
    )
    assert r.status is Status.OK
    m = entries(r)
    assert m.free_symbols == {sp.Symbol("a")}
    assert sp.simplify(m[1, 1] - (sp.exp(2) - 1) / 2) == 0


def test_the_equation_is_never_called_an_evaluation(engine: Engine) -> None:
    r = mc(
        engine,
        r"\exp(tP) C \exp(tQ)",
        [abstract("P", 2), abstract("Q", 2), abstract("C", 2)],
        action="integrate",
        variable="t",
        lower="0",
        upper="1",
    )
    assert r.status is Status.OK and not r.evaluated and r.result_entries is None
    assert "not an evaluated" in " ".join(r.notes) or "representation" in r.message


# -- 11, 12: invalid input, limits ------------------------------------------------------------
def test_dimension_mismatches(engine: Engine) -> None:
    wide = [abstract("A", 2, 3), abstract("B", 2)]
    for source in (r"AB", r"A+B", r"\exp(A)"):
        r = mc(engine, source, wide, action="simplify")
        assert r.status is Status.PARSE_ERROR, source
        assert "dimension" in r.message and r.result_latex is None
    # a rectangular matrix times a compatible one is fine
    assert mc(engine, r"BA", wide, action="simplify").status is Status.UNCHANGED


def test_invalid_expressions(engine: Engine) -> None:
    cases = {
        r"e^{tA}": Status.PARSE_ERROR,  # a bare e is an ordinary symbol
        r"\exp(tA": Status.PARSE_ERROR,
        r"A^{-1}": Status.UNSUPPORTED,
        r"\sin(A)": Status.UNSUPPORTED,
        r"\frac{1}{A}": Status.UNSUPPORTED,
        r"\exp(A+1)": Status.UNSUPPORTED,
        r"A+1": Status.UNSUPPORTED,
        r"\exp(AB)": Status.UNSUPPORTED,
        r"x+1": Status.UNSUPPORTED,
    }
    for source, status in cases.items():
        r = mc(engine, source, AB, action="simplify")
        assert r.status is status, source
        assert r.result_latex is None


def test_invalid_declarations_and_requests(engine: Engine) -> None:
    one = [abstract("A", 2)]
    assert mc(engine, "A", one + one, action="simplify").status is Status.PARSE_ERROR
    assert mc(engine, "A", [abstract("E", 2)], action="simplify").status is Status.PARSE_ERROR
    assert mc(engine, r"\exp(tA)", one, action="differentiate").status is Status.PARSE_ERROR
    assert (
        mc(engine, r"\exp(tA)", one, action="integrate", variable="t", lower="0").status
        is Status.PARSE_ERROR
    )
    assert (
        mc(engine, r"\exp(tA)", one, action="integrate", variable="A", lower="0", upper="1").status
        is Status.PARSE_ERROR
    )
    assert (
        mc(engine, r"\exp(tA)", one, action="simplify", variable="t").status is Status.PARSE_ERROR
    )
    assert (
        mc(engine, r"\exp(tA)", one, action="simplify", commuting=[("A", "Z")]).status
        is Status.PARSE_ERROR
    )
    # a matrix cannot be used as a scalar bound
    r = mc(engine, r"\exp(tA)", one, action="integrate", variable="t", lower="0", upper="A")
    assert r.status is Status.PARSE_ERROR
    with pytest.raises(ValueError, match="entries or a size"):
        MatrixDecl(name="A", rows=2, entries=[["1"]])
    with pytest.raises(ValueError, match="same length"):
        MatrixDecl(name="A", entries=[["1", "2"], ["3"]])


def test_unsupported_structures_are_reported_as_such(engine: Engine) -> None:
    one = [abstract("A", 2)]
    kw: dict[str, Any] = {"action": "integrate", "variable": "t", "lower": "0", "upper": "1"}
    assert mc(engine, r"\exp(t^2 A)", one, **kw).status is Status.UNSUPPORTED
    assert mc(engine, r"t \exp(tA)", one, **kw).status is Status.UNSUPPORTED
    assert mc(engine, r"\exp(tA)\exp(tA)", one, **kw).status is Status.OK  # merges to exp(2tA)
    three = [abstract("A", 2), abstract("B", 2), abstract("C", 2)]
    r = mc(engine, r"\exp(tA)\exp(tB)\exp(tC)", three, **kw)
    assert r.status is Status.UNSUPPORTED and "one or two" in r.message
    r = mc(engine, r"\exp(t(A+B))", AB, **kw)
    assert r.status is Status.UNSUPPORTED
    # infinite bounds (text syntax has oo)
    r = mc(
        engine, "exp(t*A)", one, fmt="text", action="integrate", variable="t", lower="0", upper="oo"
    )
    assert r.status is Status.UNSUPPORTED


def test_an_undeclared_capital_letter_is_flagged_as_a_scalar(engine: Engine) -> None:
    r = mc(engine, r"\exp(tA)\exp(sC)", [abstract("A", 2)], action="simplify")
    assert any("C is not a declared matrix" in n for n in r.notes)


def test_resource_limits(engine: Engine) -> None:
    r = mc(engine, r"(A+B)^{12}", AB, action="simplify")
    assert r.status is Status.RESOURCE_LIMIT and r.result_latex is None
    r = mc(engine, "A" * 3_000, AB, action="simplify")
    assert r.status is Status.RESOURCE_LIMIT
    small = Engine(Limits(timeout_seconds=1, max_concurrency=1))
    r = mc(
        small,
        r"\exp(-(x-y)(y-w)A)\exp(-(w-z)(z-q)B)",
        AB,
        action="integrate",
        variable="w",
        lower="a",
        upper="b",
    )
    assert r.status in (Status.OK, Status.RESOURCE_LIMIT)
    if r.status is not Status.OK:
        assert r.result_latex is None


def test_assumptions_on_scalars_are_kept(engine: Engine) -> None:
    request = MatrixCalculusInput(
        expression=ExpressionInput(input=r"\exp(tA)", assumptions={"t": ["real"]}),
        matrices=[abstract("A", 2)],
        action="differentiate",
        variable="t",
    )
    r = engine.matrix_calculus(request)
    assert r.status is Status.OK
    assert r.verification is not None and r.verification.method.count("imaginary") == 0
    bad = request.model_copy(
        update={"expression": ExpressionInput(input="tA", assumptions={"A": ["real"]})}
    )
    assert engine.matrix_calculus(bad).status is Status.PARSE_ERROR


# -- the verifiers must reject wrong candidates -----------------------------------------------
def _setup(*names: str) -> tuple[Setup, Evaluator]:
    shapes = {n: (3, 3) for n in names}
    return Setup(shapes), Evaluator([t, sp.Symbol("s")])


def _terms(source: str, *names: str):  # type: ignore[no-untyped-def]
    return to_terms(parse_latex(source, noncommutative=names), frozenset(names))


def test_the_verifiers_reject_a_wrong_merge() -> None:
    setup, ev = _setup("A", "B")
    both = _terms(r"\exp(A)\exp(B)", "A", "B")
    merged = _terms(r"\exp(A+B)", "A", "B")
    count = check_equal(
        setup,
        ev,
        lambda env: ev.terms(both, env),
        lambda env: ev.terms(merged, env),
        deadline=1e18,
    )
    assert count.checked >= 4 and count.agreed == 0
    # the exact leg refutes it too, and agrees once the commutation is declared
    assert series_terms_equal(both, merged, Commutation()) is False
    assert series_terms_equal(both, merged, Commutation(frozenset({frozenset("AB")}))) is True


def test_the_verifiers_reject_a_reordered_product() -> None:
    setup, ev = _setup("A", "B")
    ab = _terms(r"\exp(tA)\exp(sB)", "A", "B")
    ba = _terms(r"\exp(sB)\exp(tA)", "A", "B")
    count = check_equal(
        setup, ev, lambda env: ev.terms(ab, env), lambda env: ev.terms(ba, env), deadline=1e18
    )
    assert count.agreed == 0
    declared = Setup(
        {"A": (3, 3), "B": (3, 3)},
        comm=Commutation(frozenset({frozenset("AB")})),
        cliques=[["A", "B"]],
    )
    count = check_equal(
        declared, ev, lambda env: ev.terms(ab, env), lambda env: ev.terms(ba, env), deadline=1e18
    )
    assert count.passed


def test_the_derivative_check_rejects_wrong_candidates() -> None:
    setup, ev = _setup("A")
    f = _terms(r"\exp(tA)", "A")
    good = _terms(r"A\exp(tA)", "A")
    wrong = _terms(r"2A\exp(tA)", "A")
    sign = _terms(r"-A\exp(tA)", "A")
    for candidate, passes in ((good, True), (wrong, False), (sign, False)):
        count, _ = check_derivative(
            setup,
            ev,
            t,
            lambda env: ev.terms(f, env),
            lambda env, c=candidate: ev.terms(c, env),
            deadline=1e18,
        )
        assert count.passed is passes


def test_a_one_in_a_million_error_is_rejected() -> None:
    setup, ev = _setup("A")
    f = _terms(r"\exp(A)", "A")

    def nudged(env: Any) -> mp.matrix:
        return ev.terms(f, env) * (1 + mp.mpf("1e-6"))

    count = check_equal(setup, ev, lambda env: ev.terms(f, env), nudged, deadline=1e18)
    assert count.agreed == 0
    env = make_env(setup, ev, random.Random(1))
    assert close(ev.terms(f, env), ev.terms(f, env))


# -- the normal form --------------------------------------------------------------------------
def test_parsers_keep_the_order_of_declared_matrices_only() -> None:
    ordered = parse_latex(r"AB", noncommutative=["A", "B"])
    assert ordered != parse_latex(r"BA", noncommutative=["A", "B"])
    assert parse_latex("ab") == parse_latex("ba")  # unchanged behaviour without declarations
    assert parse_text("A*B", noncommutative=["A", "B"]) != parse_text(
        "B*A", noncommutative=["A", "B"]
    )
    with pytest.raises(ParseError):
        parse_latex("A", {"A": ["real"]}, noncommutative=["A"])


def test_canonical_form_orders_commuting_matrices_and_nothing_else() -> None:
    terms = to_terms(parse_latex(r"BA", noncommutative=["A", "B"]), frozenset("AB"))
    assert canonicalize(terms, Commutation()).terms == terms
    swapped = canonicalize(terms, Commutation(frozenset({frozenset("AB")})))
    assert swapped.terms != terms and swapped.assumptions == ["AB = BA"]
