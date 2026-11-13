"""Derivatives of exp(X(t)) when X does not commute with X': the Frechet derivative."""

from __future__ import annotations

import random
from typing import Any

import mpmath as mp
import pytest
import sympy as sp
from minerva_engine import Engine, ExpressionInput, Status
from minerva_engine.latexparse import parse_latex
from minerva_engine.matrix import MatrixCalculusInput, MatrixCalculusResult, MatrixDecl
from minerva_engine.matrix.expressions import Commutation, Exp, Frechet, Term, to_terms
from minerva_engine.matrix.verify import (
    Evaluator,
    Setup,
    check_derivative,
    expand_dependent,
    make_env,
    series_of_terms,
)

t = sp.Symbol("t")


def decl(name: str, rows: int, **kw: Any) -> MatrixDecl:
    return MatrixDecl(name=name, rows=rows, **kw)


def exp_decl(name: str, rows: list[list[str]]) -> MatrixDecl:
    return MatrixDecl(name=name, entries=rows)


def diff(
    engine: Engine, source: str, matrices: list[MatrixDecl], **kw: Any
) -> MatrixCalculusResult:
    return engine.matrix_calculus(
        MatrixCalculusInput(
            expression=ExpressionInput(input=source, format="latex"),
            matrices=matrices,
            action="differentiate",
            variable="t",
            **kw,
        )
    )


def numeric_derivative(
    matrices: dict[str, list[list[int]]], coeffs: dict[str, str], at: float
) -> mp.matrix:
    """d/dt exp(sum c_i(t) M_i) at t = `at`, by a central difference of mpmath.expm."""

    def value(x: mp.mpf) -> mp.matrix:
        total = None
        for name, c in coeffs.items():
            piece = sp.lambdify(t, sp.sympify(c), "mpmath")(x) * mp.matrix(matrices[name])
            total = piece if total is None else total + piece
        return mp.expm(total)

    with mp.workdps(30):
        h = mp.mpf("1e-10")
        x = mp.mpf(at)
        return (value(x + h) - value(x - h)) / (2 * h)


def entries_at(result: MatrixCalculusResult, at: float) -> mp.matrix:
    assert result.result_entries is not None
    rows = result.result_entries
    out = mp.matrix(len(rows), len(rows[0]))
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            out[i, j] = mp.mpmathify(complex(sp.sympify(text).subs(t, at).evalf(30)))
    return out


# -- the engine ---------------------------------------------------------------------------------
def test_abstract_noncommuting_exponent_gives_a_verified_representation(engine: Engine) -> None:
    r = diff(engine, r"\exp(t^2 A + t B)", [decl("A", 2), decl("B", 3 - 1)])
    assert r.status is Status.OK and r.method == "exp_frechet_derivative"
    assert r.result_kind == "representation" and not r.evaluated and r.result_entries is None
    assert r.result_latex is not None and r"\exp\left(\sigma" in r.result_latex
    v = r.verification
    assert v is not None and v.passed and v.symbolic_zero is True
    assert v.points_checked >= 8  # central difference and quadrature points
    # the familiar form is reported as blocked, with its condition
    blocked = [b for b in r.blocked if b.method == "exp_derivative"]
    assert blocked and "X'" in blocked[0].needs
    assert any("not an evaluated matrix" in n for n in r.notes)


def test_a_single_constant_matrix_keeps_the_familiar_form(engine: Engine) -> None:
    r = diff(engine, r"\exp(t^3 A)", [decl("A", 2)])
    assert r.method == "exp_derivative" and r.result_kind == "expression" and r.evaluated
    assert r.result_latex is not None and r.result_latex.startswith("3 t^{2} A")


def test_declared_commutation_restores_the_familiar_form(engine: Engine) -> None:
    r = diff(engine, r"\exp(t^2 A + t B)", [decl("A", 2), decl("B", 2)], commuting=[("A", "B")])
    assert r.method == "exp_derivative" and r.evaluated
    assert any("AB = BA" in a for a in r.assumptions_used)


def test_commutation_visible_in_the_entries_is_used(engine: Engine) -> None:
    a = exp_decl("A", [["1", "0"], ["0", "2"]])
    b = exp_decl("B", [["3", "0"], ["0", "5"]])
    r = diff(engine, r"\exp(t^2 A + t B)", [a, b])
    assert r.method == "exp_derivative"
    assert any("from the entries" in x for x in r.assumptions_used)


@pytest.mark.parametrize(
    ("a", "b", "label"),
    [
        ([[0, 1], [0, 0]], [[0, 0], [1, 0]], "nilpotent pair"),
        ([[2, 1], [0, 2]], [[0, 0], [1, 0]], "defective with a repeated eigenvalue"),
        ([[1, 0], [0, 0]], [[0, 1], [1, 0]], "singular and diagonal"),
        ([[0, 1, 0], [0, 0, 1], [0, 0, 0]], [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "3x3 shifts"),
    ],
)
def test_explicit_frechet_derivative_matches_an_independent_derivative(
    engine: Engine, a: list[list[int]], b: list[list[int]], label: str
) -> None:
    def as_text(m: list[list[int]]) -> list[list[str]]:
        return [[str(x) for x in row] for row in m]

    r = diff(engine, r"\exp(tA + t^2 B)", [exp_decl("A", as_text(a)), exp_decl("B", as_text(b))])
    assert r.status is Status.OK, label
    if r.result_kind == "explicit":
        assert r.evaluated
        for x in (0.6, 1.3):
            want = numeric_derivative({"A": a, "B": b}, {"A": "t", "B": "t**2"}, x)
            got = entries_at(r, x)
            assert mp.mnorm(got - want, 1) < 1e-6 * max(1, mp.mnorm(want, 1)), (label, x)
    else:  # SymPy found no closed form: an honest representation, still verified
        assert r.result_kind == "representation" and not r.evaluated


# -- time-dependent matrices --------------------------------------------------------------------
def test_time_dependent_abstract_matrix_is_a_representation(engine: Engine) -> None:
    r = diff(engine, r"\exp(A(t))", [decl("A", 2, depends_on="t")])
    assert r.status is Status.OK and r.method == "exp_frechet_derivative"
    assert r.result_kind == "representation" and not r.evaluated
    assert r.result_latex is not None and "A'" in r.result_latex
    assert r.verification is not None and r.verification.symbolic_zero is True
    assert any("A depends on t" in a for a in r.assumptions_used)


def test_declared_self_commutation_gives_the_familiar_form(engine: Engine) -> None:
    r = diff(engine, r"\exp(A(t))", [decl("A", 3, depends_on="t")], self_commuting=["A"])
    assert r.method == "exp_derivative" and r.result_kind == "expression" and r.evaluated
    assert r.result_latex == r"A' \exp\left(A\right)"
    assert any("at all times (declared)" in a for a in r.assumptions_used)


def test_time_dependent_explicit_entries(engine: Engine) -> None:
    noncommuting = exp_decl("A", [["0", "1"], ["t", "0"]])
    r = diff(engine, r"\exp(A(t))", [noncommuting])
    assert r.status is Status.OK and r.method == "exp_frechet_derivative"
    assert r.result_kind == "explicit"
    # d/dt exp([[0,1],[t,0]]) at t = 0.4 against an independent central difference
    for x in (0.4, 1.1):
        with mp.workdps(30):
            h = mp.mpf("1e-10")
            f = lambda s: mp.expm(mp.matrix([[0, 1], [s, 0]]))  # noqa: E731
            want = (f(mp.mpf(x) + h) - f(mp.mpf(x) - h)) / (2 * h)
        assert mp.mnorm(entries_at(r, x) - want, 1) < 1e-6
    # entries that commute with themselves at all times are recognised
    commuting = exp_decl("A", [["0", "t"], ["0", "0"]])
    r = diff(engine, r"\exp(A(t))", [commuting])
    assert r.method == "exp_derivative"
    assert any("from the entries" in a for a in r.assumptions_used)


def test_a_wrong_self_commutation_declaration_is_an_error(engine: Engine) -> None:
    noncommuting = exp_decl("A", [["0", "1"], ["t", "0"]])
    assert diff(engine, r"\exp(A(t))", [noncommuting], self_commuting=["A"]).status is (
        Status.PARSE_ERROR
    )
    r = diff(engine, r"\exp(tB)", [decl("B", 2)], self_commuting=["B"])
    assert r.status is Status.PARSE_ERROR  # B is not time dependent


def test_two_time_dependent_matrices_need_a_declared_commutation(engine: Engine) -> None:
    mats = [decl("A", 2, depends_on="t"), decl("B", 2, depends_on="t")]
    r = diff(engine, r"\exp(A(t) + B(t))", mats, self_commuting=["A", "B"])
    assert r.method == "exp_frechet_derivative"  # A(t)B(s) = B(s)A(t) was not declared
    r = diff(engine, r"\exp(A(t) + B(t))", mats, self_commuting=["A", "B"], commuting=[("A", "B")])
    assert r.method == "exp_derivative" and r.evaluated


def test_product_with_a_time_dependent_matrix_uses_the_ordered_product_rule(
    engine: Engine,
) -> None:
    r = diff(engine, r"A(t) B", [decl("A", 2, depends_on="t"), decl("B", 2)])
    assert r.status is Status.OK and r.result_latex == "A' B" and r.evaluated
    r = diff(engine, r"B A(t)^2", [decl("A", 2, depends_on="t"), decl("B", 2)])
    assert r.status is Status.OK and r.verification is not None and r.verification.passed
    assert r.result_latex is not None and "A'" in r.result_latex


def test_a_matrix_followed_by_the_variable_is_never_read_as_a_product(engine: Engine) -> None:
    r = diff(engine, r"\exp(A(t))", [decl("A", 2)])
    assert r.status is Status.PARSE_ERROR and "depending on t" in r.message
    r = diff(engine, r"A \left(t\right)", [decl("A", 2)])
    assert r.status is Status.PARSE_ERROR
    # the explicit product is fine
    assert diff(engine, r"\exp(tA)", [decl("A", 2)]).status is Status.OK


def test_time_dependence_must_name_the_operation_variable(engine: Engine) -> None:
    r = diff(engine, r"\exp(A)", [decl("A", 2, depends_on="s")])
    assert r.status is Status.PARSE_ERROR
    r = engine.matrix_calculus(
        MatrixCalculusInput(
            expression=ExpressionInput(input=r"\exp(A)", format="latex"),
            matrices=[decl("A", 2, depends_on="t")],
            action="simplify",
        )
    )
    assert r.status is Status.PARSE_ERROR  # depends on t, but no variable is given


def test_integrals_and_simplification_of_time_dependent_matrices_are_unsupported(
    engine: Engine,
) -> None:
    r = engine.matrix_calculus(
        MatrixCalculusInput(
            expression=ExpressionInput(input=r"A(t)", format="latex"),
            matrices=[decl("A", 2, depends_on="t")],
            action="integrate",
            variable="t",
        )
    )
    assert r.status is Status.UNSUPPORTED and "time-dependent" in r.message


def test_dimension_errors_are_reported(engine: Engine) -> None:
    r = diff(engine, r"\exp(tA + B)", [decl("A", 2), decl("B", 3)])
    assert r.status is Status.PARSE_ERROR and "dimension" in r.message
    r = diff(engine, r"\exp(tA)", [decl("A", 2, cols=3)])
    assert r.status is Status.PARSE_ERROR


# -- the verifiers reject wrong candidates ------------------------------------------------------
def _setup(*names: str, dependent: tuple[str, ...] = ()) -> tuple[Setup, Evaluator]:
    shapes = {n: (3, 3) for n in names}
    setup = Setup(shapes, var=t, dependent=frozenset(dependent))
    return setup, Evaluator([t])


def _terms(source: str, *names: str):  # type: ignore[no-untyped-def]
    return to_terms(parse_latex(source, noncommutative=names), frozenset(names))


def test_x_prime_exp_x_is_rejected_when_the_exponent_does_not_commute() -> None:
    setup, ev = _setup("A", "B")
    f = _terms(r"\exp(t^2 A + t B)", "A", "B")
    (exp_factor,) = f[0].factors
    assert isinstance(exp_factor, Exp)
    # the familiar (wrong) answer: X' exp(X) with X' = 2tA + B
    wrong = (
        Term(2 * t, (_sym("A"), exp_factor)),
        Term(sp.Integer(1), (_sym("B"), exp_factor)),
    )
    right = (
        Term(sp.Integer(1), (Frechet(exp_factor, (("A", 0, 2 * t), ("B", 0, sp.Integer(1)))),)),
    )
    count, _ = check_derivative(
        setup, ev, t, lambda env: ev.terms(f, env), lambda env: ev.terms(wrong, env), deadline=1e18
    )
    assert count.checked >= 4 and count.agreed == 0
    reversed_order = (
        Term(2 * t, (exp_factor, _sym("A"))),
        Term(sp.Integer(1), (exp_factor, _sym("B"))),
    )
    count, _ = check_derivative(
        setup,
        ev,
        t,
        lambda env: ev.terms(f, env),
        lambda env: ev.terms(reversed_order, env),
        deadline=1e18,
    )
    assert count.agreed == 0
    count, _ = check_derivative(
        setup, ev, t, lambda env: ev.terms(f, env), lambda env: ev.terms(right, env), deadline=1e18
    )
    assert count.passed


def test_a_one_in_a_million_error_in_the_derivative_is_rejected() -> None:
    setup, ev = _setup("A", "B")
    f = _terms(r"\exp(t^2 A + t B)", "A", "B")
    (exp_factor,) = f[0].factors
    assert isinstance(exp_factor, Exp)
    direction = (("A", 0, 2 * t), ("B", 0, sp.Integer(1)))
    exact = (Term(sp.Integer(1), (Frechet(exp_factor, direction),)),)
    nudged = (Term(sp.Rational(1000001, 1000000), (Frechet(exp_factor, direction),)),)
    count, _ = check_derivative(
        setup, ev, t, lambda env: ev.terms(f, env), lambda env: ev.terms(nudged, env), deadline=1e18
    )
    assert count.checked >= 4 and count.agreed == 0
    count, _ = check_derivative(
        setup, ev, t, lambda env: ev.terms(f, env), lambda env: ev.terms(exact, env), deadline=1e18
    )
    assert count.passed


def test_time_dependent_sampling_honours_the_declared_commutation() -> None:
    setup, ev = _setup("A", "B", dependent=("A",))
    setup = Setup(
        setup.shapes,
        comm=Commutation(frozenset({frozenset("AB")})),
        cliques=[["A", "B"]],
        var=t,
        dependent=frozenset({"A"}),
    )
    rng = random.Random(7)
    with mp.workdps(30):
        _check_commuting_sample(setup, ev, rng)


def _check_commuting_sample(setup: Setup, ev: Evaluator, rng: random.Random) -> None:
    env = make_env(setup, ev, rng)
    for shift in (mp.mpf("0.3"), mp.mpc("0.2", "0.5")):
        moved = env.at(t, shift)
        commutator = moved.mats["A"] * moved.mats["B"] - moved.mats["B"] * moved.mats["A"]
        assert mp.mnorm(commutator, 1) < 1e-20
        derivative = moved.mats["A'"] * moved.mats["B"] - moved.mats["B"] * moved.mats["A'"]
        assert mp.mnorm(derivative, 1) < 1e-20


def test_the_exact_leg_refutes_the_familiar_form() -> None:
    comm = Commutation()
    original = _terms(r"\exp(A)", "A")
    wrong = (Term(sp.Integer(1), (_sym("A", deriv=1), original[0].factors[0])),)
    dependent = frozenset({"A"})
    expanded, c = expand_dependent(original, t, dependent, comm)
    candidate, _ = expand_dependent(wrong, t, dependent, comm)
    from minerva_engine.matrix.verify import poly_diff, poly_equal

    assert not poly_equal(poly_diff(series_of_terms(expanded), t), series_of_terms(candidate), c)


def _sym(name: str, deriv: int = 0):  # type: ignore[no-untyped-def]
    from minerva_engine.matrix.expressions import Sym

    return Sym(name, 1, deriv)


# -- randomized small matrices ------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_random_integer_matrices_agree_with_an_independent_derivative(
    engine: Engine, seed: int
) -> None:
    rng = random.Random(seed)
    size = 2 if seed % 2 == 0 else 3
    a = [[rng.randint(-2, 2) for _ in range(size)] for _ in range(size)]
    b = [[rng.randint(-2, 2) for _ in range(size)] for _ in range(size)]
    texts = [[[str(x) for x in row] for row in m] for m in (a, b)]
    r = diff(engine, r"\exp(tA + t^2 B)", [exp_decl("A", texts[0]), exp_decl("B", texts[1])])
    assert r.status in (Status.OK,), r.message
    assert r.verification is not None and r.verification.passed
    if r.result_kind == "explicit":
        want = numeric_derivative({"A": a, "B": b}, {"A": "t", "B": "t**2"}, 0.7)
        assert mp.mnorm(entries_at(r, 0.7) - want, 1) < 1e-6 * max(1, mp.mnorm(want, 1))
    else:
        assert r.result_kind in ("representation", "expression") and r.method is not None
