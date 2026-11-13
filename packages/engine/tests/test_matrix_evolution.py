"""Evolution operators dU/dt = A(t) U, U(t0) = I."""

from __future__ import annotations

import random
from typing import Any

import mpmath as mp
import pytest
import sympy as sp
from minerva_engine import Engine, ExpressionInput, Limits, Status
from minerva_engine.matrix import MatrixCalculusInput, MatrixCalculusResult, MatrixDecl
from minerva_engine.matrix.verify import Evaluator, Setup, check_derivative

t = sp.Symbol("t")


def abstract(name: str, rows: int = 2, **kw: Any) -> MatrixDecl:
    return MatrixDecl(name=name, rows=rows, **kw)


def explicit(name: str, rows: list[list[Any]]) -> MatrixDecl:
    return MatrixDecl(name=name, entries=[[str(x) for x in row] for row in rows])


def evolve(
    engine: Engine, source: str, matrices: list[MatrixDecl], **kw: Any
) -> MatrixCalculusResult:
    return engine.matrix_calculus(
        MatrixCalculusInput(
            expression=ExpressionInput(input=source, format="latex"),
            matrices=matrices,
            action="evolve",
            variable="t",
            **kw,
        )
    )


def entries(result: MatrixCalculusResult) -> sp.Matrix:
    assert result.result_entries is not None
    return sp.Matrix([[sp.sympify(e) for e in row] for row in result.result_entries])


def numeric_entries(result: MatrixCalculusResult) -> mp.matrix:
    assert result.result_entries is not None
    rows = result.result_entries
    out = mp.matrix(len(rows), len(rows[0]))
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            out[i, j] = mp.mpmathify(complex(text.replace(" i", "j").replace(" ", "")))
    return out


def reference_by_taylor(rows: list[list[str]], end: float) -> mp.matrix:
    """U(end) from mpmath's Taylor-series ODE solver, which shares no code with the engine."""
    n = len(rows)
    fns = [[sp.lambdify(t, sp.sympify(e), "mpmath") for e in row] for row in rows]
    out = mp.zeros(n, n)
    with mp.workdps(30):
        for col in range(n):

            def rhs(x: mp.mpf, y: list[mp.mpf]) -> list[mp.mpf]:
                return [sum(fns[i][j](x) * y[j] for j in range(n)) for i in range(n)]

            f = mp.odefun(rhs, 0, [1 if i == col else 0 for i in range(n)])
            values = f(end)
            for i in range(n):
                out[i, col] = values[i]
    return out


# -- closed forms --------------------------------------------------------------------------------
def test_a_constant_generator_gives_the_exponential(engine: Engine) -> None:
    r = evolve(engine, "A", [abstract("A", 3)])
    assert r.status is Status.OK and r.method == "evolution_commuting"
    assert r.result_kind == "expression" and r.evaluated
    assert r.result_latex == r"\exp\left(t A\right)"
    assert r.verification is not None and r.verification.passed
    assert r.verification.symbolic_zero is True


def test_a_nonzero_initial_time(engine: Engine) -> None:
    r = evolve(engine, "A", [abstract("A")], lower="s")
    assert r.status is Status.OK
    assert r.result_latex is not None and "s" in r.result_latex and "t" in r.result_latex
    assert r.verification is not None and r.verification.passed


def test_scalar_functions_times_commuting_matrices(engine: Engine) -> None:
    r = evolve(engine, r"tA + t^2 B", [abstract("A"), abstract("B")], commuting=[("A", "B")])
    assert r.status is Status.OK and r.result_kind == "expression"
    assert r.result_latex == r"\exp\left(\frac{t^{2}}{2} A + \frac{t^{3}}{3} B\right)"
    assert any("AB = BA (declared)" in a for a in r.assumptions_used)


def test_explicit_diagonal_generator(engine: Engine) -> None:
    r = evolve(engine, "A(t)", [explicit("A", [["t", 0], [0, "2t"]])])
    assert r.status is Status.OK and r.method == "evolution_commuting"
    assert r.result_kind == "explicit"
    want = sp.diag(sp.exp(t**2 / 2), sp.exp(t**2))
    assert sp.simplify(entries(r) - want) == sp.zeros(2, 2)
    assert r.verification is not None and r.verification.symbolic_zero is True


def test_explicit_nilpotent_generator_with_time_dependence(engine: Engine) -> None:
    r = evolve(engine, "A(t)", [explicit("A", [[0, "t"], [0, 0]])])
    assert r.status is Status.OK and r.result_kind == "explicit"
    assert sp.simplify(entries(r) - sp.Matrix([[1, t**2 / 2], [0, 1]])) == sp.zeros(2, 2)
    assert any("from the entries" in a for a in r.assumptions_used)


def test_a_self_commuting_abstract_generator_is_a_representation(engine: Engine) -> None:
    r = evolve(engine, "A(t)", [abstract("A", 3, depends_on="t")], self_commuting=["A"])
    assert r.status is Status.OK and r.method == "evolution_commuting"
    assert r.result_kind == "representation" and not r.evaluated and r.result_entries is None
    assert r.result_latex is not None and r"\int_{0}^{t}" in r.result_latex
    assert r.verification is not None and r.verification.passed
    assert any("integral" in n for n in r.notes)
    # a scalar factor in the generator stays inside the integral
    r = evolve(engine, "t A(t)", [abstract("A", depends_on="t")], self_commuting=["A"])
    assert r.status is Status.OK and r.result_latex is not None and "\\tau" in r.result_latex


# -- numerical solutions -------------------------------------------------------------------------
def test_airy_equation_matches_its_closed_form(engine: Engine) -> None:
    """y'' = -t y: the columns of U are combinations of Ai(-t) and Bi(-t)."""
    r = evolve(engine, "A(t)", [explicit("A", [[0, 1], ["-t", 0]])], upper="1.5")
    assert r.status is Status.OK and r.method == "evolution_numeric"
    assert r.result_kind == "numeric" and not r.evaluated
    assert r.verification is not None and r.verification.passed
    assert r.verification.symbolic_zero is None  # never described as proved
    end = mp.mpf("1.5")
    with mp.workdps(30):
        ai0, aip0 = mp.airyai(0), mp.airyai(0, derivative=1)
        bi0, bip0 = mp.airybi(0), mp.airybi(0, derivative=1)

        def column(y0: float, dy0: float) -> tuple[mp.mpf, mp.mpf]:
            alpha, beta = mp.lu_solve(mp.matrix([[ai0, bi0], [-aip0, -bip0]]), mp.matrix([y0, dy0]))
            y = alpha * mp.airyai(-end) + beta * mp.airybi(-end)
            dy = -alpha * mp.airyai(-end, derivative=1) - beta * mp.airybi(-end, derivative=1)
            return y, dy

        first, second = column(1, 0), column(0, 1)
    got = numeric_entries(r)
    for i, (a, b) in enumerate(zip(first, second, strict=True)):
        assert abs(got[i, 0] - a) < 1e-9 and abs(got[i, 1] - b) < 1e-9


def test_numeric_result_states_what_it_is(engine: Engine) -> None:
    r = evolve(engine, "A(t)", [explicit("A", [[0, "t"], [-1, 0]])], upper="1")
    assert r.status is Status.OK and r.result_kind == "numeric" and not r.evaluated
    assert "numerical approximation" in r.message and "closed form" in r.message
    assert any("measurement on that span" in n for n in r.notes)
    assert r.steps[0].title.startswith("Numerical solution")
    assert r.verification is not None and r.verification.points_checked >= 8


@pytest.mark.parametrize("seed", range(4))
def test_random_generators_match_an_independent_taylor_solution(engine: Engine, seed: int) -> None:
    rng = random.Random(seed)
    size = 2 if seed % 2 == 0 else 3
    coefficients = [
        [(rng.randint(-2, 2), rng.randint(-2, 2)) for _ in range(size)] for _ in range(size)
    ]
    rows = [[f"{a} + {b} * t" for a, b in row] for row in coefficients]
    latex_rows = [[f"{a} + {b} t" for a, b in row] for row in coefficients]
    r = evolve(engine, "A(t)", [explicit("A", latex_rows)], upper="1")
    assert r.status is Status.OK, r.message
    assert r.result_kind in ("numeric", "explicit")
    if r.result_kind == "numeric":
        want = reference_by_taylor(rows, 1.0)
        got = numeric_entries(r)
        assert mp.mnorm(got - want, 1) < 1e-8 * max(1, mp.mnorm(want, 1))


# -- what is declined ------------------------------------------------------------------------------
def test_an_abstract_noncommuting_generator_is_not_approximated(engine: Engine) -> None:
    r = evolve(engine, "tA + t^2 B", [abstract("A"), abstract("B")])
    assert r.status is Status.UNSUPPORTED and r.result_latex is None
    assert "time-ordered" in r.message and "Magnus" in r.message
    assert any("AB = BA" in b.needs for b in r.blocked)
    r = evolve(engine, "A(t)", [abstract("A", depends_on="t")])
    assert r.status is Status.UNSUPPORTED
    assert any("at all times" in b.needs for b in r.blocked)


def test_the_numeric_method_never_picks_the_span(engine: Engine) -> None:
    r = evolve(engine, "A(t)", [explicit("A", [[0, "t"], [-1, 0]])])
    assert r.status is Status.UNSUPPORTED and "end time" in r.message
    assert r.result_entries is None


def test_free_parameters_need_numbers(engine: Engine) -> None:
    r = evolve(engine, "A(t)", [explicit("A", [[0, "w t"], [-1, 0]])], upper="1")
    assert r.status is Status.UNSUPPORTED and "w" in r.message


def test_invalid_requests(engine: Engine) -> None:
    assert evolve(engine, "A", [abstract("A", 2, cols=3)]).status is Status.UNSUPPORTED
    r = engine.matrix_calculus(
        MatrixCalculusInput(
            expression=ExpressionInput(input="A", format="latex"),
            matrices=[abstract("A")],
            action="evolve",
        )
    )
    assert r.status is Status.PARSE_ERROR  # no time variable
    assert evolve(engine, "A", [abstract("A")], lower="1", upper="1").status is (Status.PARSE_ERROR)
    assert evolve(engine, r"\sin(A)", [abstract("A")]).status is Status.UNSUPPORTED
    big = explicit("A", [[0] * 4 for _ in range(4)])
    assert evolve(engine, "A", [big]).status is Status.OK  # 4x4 is the supported maximum


def test_a_stiff_generator_reaches_a_resource_limit(engine: Engine) -> None:
    r = evolve(
        engine,
        "A(t)",
        [explicit("A", [[0, "1000000 t"], [-1, 0]])],
        upper="100",
    )
    assert r.status is Status.RESOURCE_LIMIT


def test_a_tiny_time_budget(engine: Engine) -> None:
    quick = Engine(Limits(timeout_seconds=0.05, max_concurrency=1))
    r = evolve(quick, "A(t)", [explicit("A", [[0, "t"], [-1, 0]])], upper="1")
    assert r.status is Status.RESOURCE_LIMIT


# -- the verifiers reject wrong candidates ------------------------------------------------------------
def test_exp_of_the_integral_is_rejected_for_a_noncommuting_generator() -> None:
    """U = exp(integral of A) is the commuting formula: it fails the evolution equation here."""
    entries_a = sp.Matrix([[0, 1], [-t, 0]])
    setup = Setup({"A": (2, 2)}, explicit={"A": entries_a}, var=t, dependent=frozenset({"A"}))
    ev = Evaluator([t])
    wrong = sp.Matrix([[0, t], [-(t**2) / 2, 0]])
    wrong_u = sp.lambdify(t, wrong, "mpmath")

    def u(env):  # type: ignore[no-untyped-def]
        value = mp.matrix(wrong_u(env.scalars[t]).tolist())
        return mp.expm(value)

    count, _ = check_derivative(setup, ev, t, u, lambda env: env.mats["A"] * u(env), deadline=1e18)
    assert count.checked >= 4 and count.agreed == 0
    # for a generator that does commute with itself the same formula passes
    commuting = sp.Matrix([[0, t], [0, 0]])
    setup = Setup({"A": (2, 2)}, explicit={"A": commuting}, var=t, dependent=frozenset({"A"}))
    good = sp.lambdify(t, sp.Matrix([[0, t**2 / 2], [0, 0]]), "mpmath")
    count, _ = check_derivative(
        setup,
        ev,
        t,
        lambda env: mp.expm(mp.matrix(good(env.scalars[t]).tolist())),
        lambda env: env.mats["A"] * mp.expm(mp.matrix(good(env.scalars[t]).tolist())),
        deadline=1e18,
    )
    assert count.passed
