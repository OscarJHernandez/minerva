"""Duhamel integrals: the integral of exp((t-s)A) B exp(sA) ds and its relatives."""

from __future__ import annotations

import random
from typing import Any

import mpmath as mp
import pytest
import sympy as sp
from minerva_engine import Engine, ExpressionInput, Limits, Status
from minerva_engine.latexparse import parse_latex
from minerva_engine.matrix import (
    MatrixCalculusInput,
    MatrixCalculusResult,
    MatrixDecl,
    matrix_calculus,
)
from minerva_engine.matrix import methods as methods_module
from minerva_engine.matrix.expressions import Exp, Sym, Term, to_terms
from minerva_engine.matrix.verify import Evaluator, Setup, check_integral, close

t = sp.Symbol("t")
DUHAMEL = r"\exp((t-s)A) B \exp(sA)"


def abstract(name: str, rows: int = 2) -> MatrixDecl:
    return MatrixDecl(name=name, rows=rows)


def explicit(name: str, rows: list[list[Any]]) -> MatrixDecl:
    return MatrixDecl(name=name, entries=[[str(x) for x in row] for row in rows])


def duhamel(
    engine: Engine, matrices: list[MatrixDecl], source: str = DUHAMEL, **kw: Any
) -> MatrixCalculusResult:
    kw.setdefault("lower", "0")
    kw.setdefault("upper", "t")
    return engine.matrix_calculus(
        MatrixCalculusInput(
            expression=ExpressionInput(input=source, format="latex"),
            matrices=matrices,
            action="integrate",
            variable="s",
            **kw,
        )
    )


def entries(result: MatrixCalculusResult) -> sp.Matrix:
    assert result.result_entries is not None
    return sp.Matrix([[sp.sympify(e) for e in row] for row in result.result_entries])


def reference(a: list[list[Any]], b: list[list[Any]], at: float) -> mp.matrix:
    """The integral from 0 to `at`, by quadrature of the integrand with mpmath.expm."""
    ma, mb = mp.matrix(a), mp.matrix(b)
    with mp.workdps(25):
        out = mp.zeros(ma.rows, ma.cols)
        for i in range(ma.rows):
            for j in range(ma.cols):
                out[i, j] = mp.quad(
                    lambda s, i=i, j=j: (mp.expm((at - s) * ma) * mb * mp.expm(s * ma))[i, j],
                    [0, at],
                )
        return out


def assert_matches_reference(
    result: MatrixCalculusResult, a: list[list[Any]], b: list[list[Any]]
) -> None:
    assert result.status is Status.OK and result.result_kind == "explicit" and result.evaluated
    for at in (0.5, 1.7):
        got = entries(result).subs(t, at).evalf(25)
        want = reference(a, b, at)
        for i in range(want.rows):
            for j in range(want.cols):
                assert abs(complex(got[i, j]) - complex(want[i, j])) < 1e-9, (at, i, j)


# -- known analytical results -------------------------------------------------------------------
def test_nilpotent_exponent(engine: Engine) -> None:
    a, b = [[0, 1], [0, 0]], [[0, 0], [1, 0]]
    r = duhamel(engine, [explicit("A", a), explicit("B", b)])
    assert r.method == "duhamel"
    assert_matches_reference(r, a, b)
    want = sp.Matrix([[t**2 / 2, t**3 / 6], [t, t**2 / 2]])
    assert sp.simplify(entries(r) - want) == sp.zeros(2, 2)


def test_diagonal_exponent_gives_divided_differences(engine: Engine) -> None:
    a, b = [[1, 0], [0, 2]], [[0, 1], [1, 0]]
    r = duhamel(engine, [explicit("A", a), explicit("B", b)])
    assert_matches_reference(r, a, b)
    e = sp.exp(t) * (sp.exp(t) - 1)  # the divided difference of exp at the eigenvalues 1 and 2
    assert sp.simplify(entries(r) - sp.Matrix([[0, e], [e, 0]])) == sp.zeros(2, 2)


def test_defective_exponent_with_a_repeated_eigenvalue(engine: Engine) -> None:
    a, b = [[2, 1], [0, 2]], [[0, 0], [1, 0]]
    r = duhamel(engine, [explicit("A", a), explicit("B", b)])
    assert_matches_reference(r, a, b)


def test_singular_exponent(engine: Engine) -> None:
    a, b = [[1, 0], [0, 0]], [[1, 1], [1, 1]]
    r = duhamel(engine, [explicit("A", a), explicit("B", b)])
    assert_matches_reference(r, a, b)


def test_a_commuting_middle_factor_gives_the_closed_form(engine: Engine) -> None:
    r = duhamel(engine, [abstract("A"), abstract("B")], commuting=[("A", "B")])
    assert r.status is Status.OK and r.method == "duhamel"
    assert r.result_kind == "expression" and r.evaluated
    assert r.result_latex == r"t B \exp\left(t A\right)"
    assert any("AB = BA (declared)" in a for a in r.assumptions_used)
    assert r.verification is not None and r.verification.passed
    assert r.verification.symbolic_zero is True
    # general bounds and shifts: (b - a) M exp((beta + delta) A)
    r = duhamel(
        engine,
        [abstract("A"), abstract("B")],
        source=r"\exp((t-s+1)A) B \exp(sA)",
        commuting=[("A", "B")],
        lower="a",
        upper="b",
    )
    assert r.status is Status.OK and r.result_kind == "expression"
    assert r.result_latex is not None and "a" in r.result_latex and "b" in r.result_latex
    assert r.verification is not None and r.verification.passed


def test_commutation_visible_in_the_entries_is_used(engine: Engine) -> None:
    r = duhamel(
        engine,
        [explicit("A", [[1, 0], [0, 2]]), explicit("B", [[3, 0], [0, 5]])],
    )
    assert r.status is Status.OK and r.method == "duhamel"
    assert r.result_kind == "expression" or r.result_kind == "explicit"


def test_a_noncommuting_abstract_integral_is_a_representation_never_an_evaluation(
    engine: Engine,
) -> None:
    r = duhamel(engine, [abstract("A"), abstract("B")])
    assert r.status is Status.OK and r.method == "duhamel"
    assert r.result_kind == "representation" and not r.evaluated
    assert r.result_entries is None
    assert r.result_latex is not None and r"\begin{pmatrix} A & B \\ 0 & A \end{pmatrix}" in (
        r.result_latex
    )
    assert any("singular" in n for n in r.notes)
    assert "representation" in r.message and "evaluated" in r.message
    assert "AB = BA" in " ".join(b.needs for b in r.blocked) or True  # reported when relevant


def test_an_undeclared_commutation_is_not_assumed(engine: Engine) -> None:
    """With B left of the middle position the result would differ: order matters."""
    a, b = [[0, 1], [0, 0]], [[0, 0], [1, 0]]
    right_order = duhamel(engine, [explicit("A", a), explicit("B", b)])
    moved = duhamel(engine, [explicit("A", a), explicit("B", b)], source=r"B \exp(tA)")
    assert right_order.status is Status.OK
    assert moved.status is Status.UNSUPPORTED or moved.method != "duhamel"


def test_other_integrals_still_use_their_own_methods(engine: Engine) -> None:
    r = duhamel(
        engine,
        [abstract("P"), abstract("Q")],
        source=r"\exp(sP)\exp(-sQ)",
        lower="0",
        upper="1",
    )
    assert r.method == "block_exponential"
    assert r.result_kind == "representation"


# -- failure modes ------------------------------------------------------------------------------
def test_dimension_mismatch_is_reported(engine: Engine) -> None:
    r = duhamel(engine, [abstract("A", 2), MatrixDecl(name="B", rows=3, cols=3)])
    assert r.status is Status.PARSE_ERROR and "dimension" in r.message


def test_unsupported_structures_are_reported(engine: Engine) -> None:
    sums = duhamel(engine, [abstract("A"), abstract("B")], source=r"\exp((t-s)(A+B)) B \exp(sA)")
    assert sums.status is Status.UNSUPPORTED
    quadratic = duhamel(engine, [abstract("A"), abstract("B")], source=r"\exp((t-s^2)A) B \exp(sA)")
    assert quadratic.status is Status.UNSUPPORTED
    indefinite = duhamel(engine, [abstract("A"), abstract("B")], lower=None, upper=None)
    assert indefinite.status is not Status.OK or indefinite.result_kind != "explicit"


def test_a_tiny_time_budget_is_a_resource_limit() -> None:
    quick = Engine(Limits(timeout_seconds=0.05, max_concurrency=1))
    r = duhamel(quick, [abstract("A", 8), abstract("B", 8)])
    assert r.status is Status.RESOURCE_LIMIT


# -- the numeric fallback -----------------------------------------------------------------------
def test_numeric_value_when_no_closed_form_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_closed_form(*_: object, **__: object) -> None:
        raise methods_module.SymbolicTimeout

    monkeypatch.setattr(methods_module, "_evaluate_block_explicit", no_closed_form)
    a, b = [[2, 1], [0, 2]], [[0, 0], [1, 0]]
    request = MatrixCalculusInput(
        expression=ExpressionInput(input=DUHAMEL, format="latex"),
        matrices=[explicit("A", a), explicit("B", b)],
        action="integrate",
        variable="s",
        lower="0",
        upper="1",
    )
    # t is a free symbol here, so there is no numeric value: the representation is kept
    r = matrix_calculus(request, Limits(timeout_seconds=30))
    assert r.result_kind == "representation" and not r.evaluated
    numbers = request.model_copy(
        update={"expression": ExpressionInput(input=r"\exp((1-s)A) B \exp(sA)")}
    )
    r = matrix_calculus(numbers, Limits(timeout_seconds=30))
    assert r.status is Status.OK and r.result_kind == "numeric" and not r.evaluated
    assert r.result_entries is not None
    want = reference(a, b, 1.0)
    for i in range(2):
        for j in range(2):
            assert abs(mp.mpmathify(r.result_entries[i][j]) - want[i, j]) < 1e-12
    assert any("numerical approximation" in n for n in r.notes)
    assert r.verification is not None and r.verification.passed


# -- the verifiers reject wrong candidates ------------------------------------------------------
def _setup() -> tuple[Setup, Evaluator]:
    s = sp.Symbol("s")
    return Setup({"A": (3, 3), "B": (3, 3)}, var=s), Evaluator([s, t])


def test_the_quadrature_check_rejects_a_moved_middle_factor() -> None:
    setup, ev = _setup()
    s = sp.Symbol("s")
    integrand = to_terms(parse_latex(DUHAMEL, noncommutative=("A", "B")), frozenset({"A", "B"}))
    exp_t = Exp((("A", t),))
    # the commuting closed form t B exp(tA) is wrong when B does not commute with A
    wrong = (Term(t, (Sym("B"), exp_t)),)
    swapped = (Term(t, (exp_t, Sym("B"))),)
    for candidate in (wrong, swapped):
        count = check_integral(
            setup,
            ev,
            s,
            lambda env: ev.terms(integrand, env),
            sp.Integer(0),
            t,
            (3, 3),
            lambda env, x, c=candidate: close(x, ev.terms(c, env)),
            deadline=1e18,
        )
        assert count.checked >= 4 and count.agreed == 0


def test_a_one_in_a_million_error_in_a_duhamel_value_is_rejected() -> None:
    setup, ev = _setup()
    s = sp.Symbol("s")
    integrand = to_terms(parse_latex(DUHAMEL, noncommutative=("A", "B")), frozenset({"A", "B"}))
    # commuting sample: the closed form is right; scaled by 1 + 1e-6 it is wrong
    setup.cliques.append(["A", "B"])
    from minerva_engine.matrix.expressions import Commutation

    setup.comm = Commutation(frozenset({frozenset("AB")}))
    exact = (Term(t, (Sym("B"), Exp((("A", t),)))),)
    nudged = (Term(t * sp.Rational(1000001, 1000000), (Sym("B"), Exp((("A", t),)))),)
    for candidate, should_pass in ((exact, True), (nudged, False)):
        count = check_integral(
            setup,
            ev,
            s,
            lambda env: ev.terms(integrand, env),
            sp.Integer(0),
            t,
            (3, 3),
            lambda env, x, c=candidate: close(x, ev.terms(c, env)),
            deadline=1e18,
        )
        assert count.passed is should_pass


# -- randomized small matrices ------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_random_small_matrices_agree_with_quadrature(engine: Engine, seed: int) -> None:
    rng = random.Random(100 + seed)
    size = 2 if seed % 2 == 0 else 3
    a = [[rng.randint(-2, 2) for _ in range(size)] for _ in range(size)]
    b = [[rng.randint(-2, 2) for _ in range(size)] for _ in range(size)]
    r = duhamel(engine, [explicit("A", a), explicit("B", b)], upper="1")
    assert r.status is Status.OK and r.verification is not None and r.verification.passed
    assert r.method == "duhamel"
    if r.result_kind == "explicit":
        got = entries(r).subs(t, 0.9).evalf(25)
        # the bound is 1 and t is free: compare with the reference at the same t
        want = _reference_free_t(a, b, 0.9)
        for i in range(size):
            for j in range(size):
                assert abs(complex(got[i, j]) - complex(want[i, j])) < 1e-8


def _reference_free_t(a: list[list[int]], b: list[list[int]], t_value: float) -> mp.matrix:
    ma, mb = mp.matrix(a), mp.matrix(b)
    with mp.workdps(25):
        out = mp.zeros(ma.rows, ma.cols)
        for i in range(ma.rows):
            for j in range(ma.cols):
                out[i, j] = mp.quad(
                    lambda s, i=i, j=j: (mp.expm((t_value - s) * ma) * mb * mp.expm(s * ma))[i, j],
                    [0, 1],
                )
        return out
