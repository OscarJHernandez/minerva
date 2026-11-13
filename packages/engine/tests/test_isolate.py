"""Isolating a variable: every solution, every condition, nothing dropped, nothing extraneous."""

import random

import pytest
import sympy as sp
from minerva_engine import Engine, ExpressionInput, Status, parse_text
from minerva_engine import isolate as isolate_module
from minerva_engine.models import Limits


def eq(text: str, assumptions: dict[str, list[str]] | None = None) -> ExpressionInput:
    return ExpressionInput(input=text, assumptions=assumptions or {})


def values(result: object) -> list[sp.Expr]:
    return [parse_text(s.text) for s in result.solutions]  # type: ignore[attr-defined]


def same_set(got: list[sp.Expr], wanted: list[sp.Expr]) -> bool:
    """Equal as sets of numbers (compared at a generic point, symbols given fixed values)."""
    symbols = sorted(set().union(*(e.free_symbols for e in [*got, *wanted])), key=lambda s: s.name)
    point = {
        s: sp.Rational(3 + 2 * i, 7) + sp.I * sp.Rational(1 + i, 5) for i, s in enumerate(symbols)
    }
    key = lambda z: (round(z.real, 9), round(z.imag, 9))  # noqa: E731
    a = sorted((complex(sp.N(e.subs(point), 30)) for e in got), key=key)
    b = sorted((complex(sp.N(e.subs(point), 30)) for e in wanted), key=key)
    return len(a) == len(b) and all(abs(u - v) < 1e-9 for u, v in zip(a, b, strict=True))


# --- one occurrence: peeling ------------------------------------------------------------------------------------
def test_linear_equation_reports_the_condition_and_the_exceptional_value(engine: Engine) -> None:
    result = engine.isolate(eq("a x + b = c"), "x")
    assert result.status is Status.OK and result.method == "inversion"
    assert [s.text for s in result.solutions] == ["(-b + c)/a"]
    assert result.conditions_latex == [r"a \neq 0"]
    assert result.exceptional == ["a = 0"]
    assert result.complete and "returned by some solution" in result.completeness_note
    assert [step.name for step in result.steps] == ["subtract", "divide"]


def test_a_condition_the_assumptions_prove_is_not_reported(engine: Engine) -> None:
    result = engine.isolate(eq("a x + b = c", {"a": ["nonzero"]}), "x")
    assert result.conditions_latex == [] and result.exceptional == []


def test_an_even_power_has_both_roots_and_an_odd_one_all_three(engine: Engine) -> None:
    assert same_set(
        values(engine.isolate(eq("(x + 1)^2 = 4"), "x")), [sp.Integer(1), sp.Integer(-3)]
    )
    cube = values(engine.isolate(eq("x^3 = 2"), "x"))
    assert len(cube) == 3
    assert same_set(cube, [2 ** sp.Rational(1, 3) * sp.root(1, 3, k) for k in range(3)])


def test_the_inverse_square_law_has_a_root_of_each_sign(engine: Engine) -> None:
    result = engine.isolate(eq(r"F = \frac{G m n}{r^2}"), "r")
    F, G, m, n = sp.symbols("F G m n")
    root = sp.sqrt(G * m * n / F)
    assert same_set(values(result), [root, -root])
    assert any("F" in c for c in result.conditions_latex)


def test_an_exponential_is_a_family_of_solutions_and_says_which_integer(engine: Engine) -> None:
    result = engine.isolate(eq(r"N = M \exp(-r t)"), "t")
    assert result.status is Status.OK and len(result.solutions) == 1
    (solution,) = result.solutions
    assert "k_1" in solution.text and solution.branch == "k_1 any integer"
    # Independent check: every integer k gives a time at which the original equation holds.
    N, M, r, t = sp.symbols("N M r t")
    k = sp.Symbol("k_1", integer=True)
    candidate = parse_text(solution.text, {"k_1": ["integer"]})
    point = {N: sp.Rational(2, 3), M: sp.Rational(7, 5), r: sp.Rational(3, 2)}
    for kk in (-3, 0, 2):
        value = candidate.subs(point).subs(k, kk)
        assert (
            abs(sp.N((M * sp.exp(-r * t)).subs(point).subs(t, value) - N.subs(point), 30)) < 1e-12
        )


def test_a_logarithm_needs_the_principal_strip_unless_the_quantity_is_real(engine: Engine) -> None:
    open_ = engine.isolate(eq(r"y = a \log x + b"), "x")
    assert any(
        r"\operatorname{Im}" in c for s in open_.solutions for c in s.conditions_latex
    ) or any(r"\operatorname{Im}" in c for c in open_.conditions_latex)
    real = engine.isolate(
        eq(r"y = a \log x + b", {"a": ["real"], "y": ["real"], "b": ["real"]}), "x"
    )
    assert not any(r"\operatorname{Im}" in c for c in real.conditions_latex)
    assert not any(r"\operatorname{Im}" in c for s in real.solutions for c in s.conditions_latex)


def test_a_principal_root_can_make_the_only_candidate_extraneous(engine: Engine) -> None:
    """Squaring sqrt(x) = -2 gives x = 4, which does not satisfy it."""
    result = engine.isolate(eq(r"\sqrt{x} = -2"), "x")
    assert result.status is Status.OK and result.solutions == []
    assert result.diagnostics.rejected == {"extraneous": 1}
    assert "no solution" in result.message.lower()


def test_the_branch_condition_of_a_root_is_reported_or_proved_away(engine: Engine) -> None:
    open_ = engine.isolate(eq(r"v = \sqrt{2 g h}"), "h")
    assert any(
        r"\arg" in c for c in [*open_.conditions_latex, *open_.solutions[0].conditions_latex]
    )
    proved = engine.isolate(eq(r"v = \sqrt{2 g h}", {"v": ["positive"]}), "h")
    assert not any(
        r"\arg" in c for c in [*proved.conditions_latex, *proved.solutions[0].conditions_latex]
    )


def test_a_base_not_known_positive_is_refused(engine: Engine) -> None:
    refused = engine.isolate(eq("y = c^x"), "x")
    assert refused.status is Status.UNSUPPORTED and refused.solutions == []
    allowed = engine.isolate(eq("y = c^x", {"c": ["positive"]}), "x")
    assert allowed.status is Status.OK and "k_1" in allowed.solutions[0].text


# --- several occurrences: polynomials and rational equations -------------------------------------------------------
def test_a_quadratic_with_symbolic_coefficients_has_both_roots(engine: Engine) -> None:
    result = engine.isolate(eq(r"s = u t + \frac{1}{2} a t^2"), "t")
    s, u, a = sp.symbols("s u a")
    roots = [(-u + sp.sqrt(u**2 + 2 * a * s)) / a, (-u - sp.sqrt(u**2 + 2 * a * s)) / a]
    assert result.method == "polynomial" and same_set(values(result), roots)
    assert r"a \neq 0" in result.conditions_latex and result.exceptional == ["a = 0"]


def test_numeric_polynomials_give_all_their_roots(engine: Engine) -> None:
    assert same_set(
        values(engine.isolate(eq("x^2 - 5x + 6 = 0"), "x")), [sp.Integer(2), sp.Integer(3)]
    )
    assert same_set(
        values(engine.isolate(eq("x^3 - x = 0"), "x")),
        [sp.Integer(-1), sp.Integer(0), sp.Integer(1)],
    )


def test_a_repeated_root_is_listed_once(engine: Engine) -> None:
    result = engine.isolate(eq("x^2 - 2x + 1 = 0"), "x")
    assert values(result) == [sp.Integer(1)]


def test_a_rational_equation_lists_what_the_denominator_excludes(engine: Engine) -> None:
    result = engine.isolate(eq(r"\frac{1}{x} + \frac{1}{x+1} = 1"), "x")
    assert same_set(values(result), [(1 + sp.sqrt(5)) / 2, (1 - sp.sqrt(5)) / 2])
    assert sorted(result.excluded_latex) == sorted([r"x \neq 0", r"x + 1 \neq 0"])


def test_a_root_that_zeroes_a_denominator_is_removed_not_reported(engine: Engine) -> None:
    result = engine.isolate(eq(r"\frac{x^2 - 1}{x - 1} = 2"), "x")
    assert result.status is Status.OK and result.solutions == []
    assert result.diagnostics.rejected == {"extraneous": 1}
    assert result.excluded_latex == [r"x - 1 \neq 0"]


def test_a_parameter_dependent_denominator_becomes_a_condition(engine: Engine) -> None:
    result = engine.isolate(eq(r"\frac{x}{x - a} = 2"), "x")
    assert same_set(values(result), [2 * sp.Symbol("a")])
    assert any("a" in c for c in [*result.conditions_latex, *result.solutions[0].conditions_latex])


def test_an_equation_with_no_solution_is_ok_and_empty(engine: Engine) -> None:
    result = engine.isolate(eq(r"\frac{1}{x} = 0"), "x")
    assert result.status is Status.OK and result.solutions == []


# --- honest refusals ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        r"x + \log x = 1",  # the unknown twice, not polynomial
        r"x \exp(x) = 1",  # Lambert
        r"x^5 + a x + 1 = 0",  # no closed form
        r"\sin x = a",  # periodic inverse
        r"|x| = a",  # absolute value
        r"x^a = 2",  # symbolic exponent
        r"x^{7} = 2",  # root of too high an order
    ],
)
def test_unsupported_forms_give_no_partial_answer(engine: Engine, text: str) -> None:
    result = engine.isolate(eq(text.replace("|x|", r"\left|x\right|")), "x")
    assert result.status is Status.UNSUPPORTED and result.solutions == []


def test_a_quartic_with_symbolic_coefficients_is_refused_a_numeric_one_is_solved(
    engine: Engine,
) -> None:
    assert engine.isolate(eq("x^4 + a x + 1 = 0"), "x").status is Status.UNSUPPORTED
    solved = engine.isolate(eq("x^4 - 5 x^2 + 4 = 0"), "x")
    assert same_set(values(solved), [sp.Integer(v) for v in (-2, -1, 1, 2)])


def test_malformed_requests(engine: Engine) -> None:
    assert engine.isolate(eq("a = b"), "x").status is Status.PARSE_ERROR  # x does not occur
    assert engine.isolate(eq("x + 1"), "x").status is Status.PARSE_ERROR  # not an equation
    assert engine.isolate(eq("x = y = z"), "x").status is Status.PARSE_ERROR
    assert engine.isolate(eq("x = 1"), "1x").status is Status.PARSE_ERROR  # not a variable name
    assert engine.isolate(eq("x = 1"), "x; import os").status is Status.PARSE_ERROR
    assert engine.isolate(eq("x = x"), "x").status is Status.UNSUPPORTED  # holds for every x


def test_noncommuting_symbols_are_unsupported(engine: Engine) -> None:
    nc = {"A": ["noncommutative"], "B": ["noncommutative"]}
    assert engine.isolate(eq("A x = B", nc), "x").status is Status.UNSUPPORTED


def test_an_oversized_equation_is_a_resource_limit(engine: Engine) -> None:
    long = "+".join(["x"] * 1500) + "=1"
    assert engine.isolate(eq(long), "x").status is Status.RESOURCE_LIMIT


def test_the_assumption_on_the_unknown_is_not_used_and_says_so(engine: Engine) -> None:
    result = engine.isolate(eq("x^2 = 4", {"x": ["positive"]}), "x")
    assert len(result.solutions) == 2 and "complex" in result.assumption_note


def test_diagnostics_hold_no_expression(engine: Engine) -> None:
    result = engine.isolate(eq("zzmarkerzz x = 3"), "x")
    dumped = result.diagnostics.model_dump_json() + str(result.diagnostics.log_fields())
    assert (
        "zzmarkerzz" not in dumped and result.diagnostics.verification_method == "back_substitution"
    )


# --- the checks catch wrong candidates ---------------------------------------------------------------------------------------
def run(
    text: str, variable: str, assumptions: dict[str, list[str]] | None = None
) -> dict[str, object]:
    payload = {
        "limits": Limits(timeout_seconds=30).model_dump(),
        "expression": {"input": text, "format": "text", "assumptions": assumptions or {}},
        "variable": variable,
    }
    return isolate_module.op_isolate(payload)


def test_a_dropped_root_is_caught_by_the_root_count(monkeypatch: pytest.MonkeyPatch) -> None:
    real = isolate_module.solve_rational

    def drop_one(*args: object, **kwargs: object) -> isolate_module.Solved:
        solved = real(*args, **kwargs)  # type: ignore[arg-type]
        solved.states = solved.states[:-1]
        return solved

    monkeypatch.setattr(isolate_module, "solve_rational", drop_one)
    result = run("x**2 - 5*x + 6 = 0", "x")
    assert result["status"] == "not_verified" and result["solutions"] == []


def test_an_extraneous_root_is_caught_by_substituting_back(monkeypatch: pytest.MonkeyPatch) -> None:
    real = isolate_module.solve_rational

    def add_one(*args: object, **kwargs: object) -> isolate_module.Solved:
        solved = real(*args, **kwargs)  # type: ignore[arg-type]
        solved.states = [*solved.states, isolate_module.State(sp.Integer(7))]
        return solved

    monkeypatch.setattr(isolate_module, "solve_rational", add_one)
    result = run("x**2 - 5*x + 6 = 0", "x")
    assert result["status"] == "not_verified" and result["solutions"] == []


def test_a_root_of_the_denominator_kept_as_a_solution_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = isolate_module.solve_rational

    def keep_pole(*args: object, **kwargs: object) -> isolate_module.Solved:
        solved = real(*args, **kwargs)  # type: ignore[arg-type]
        solved.states = [isolate_module.State(sp.Integer(1))]
        return solved

    monkeypatch.setattr(isolate_module, "solve_rational", keep_pole)
    assert run("(x**2 - 1)/(x - 1) = 2", "x")["status"] == "not_verified"


def test_a_missing_square_root_branch_is_caught_by_the_forward_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every back-substitution passes with only +sqrt, but a random value of the unknown is returned by
    +sqrt only half the time: the solution set is incomplete."""
    monkeypatch.setattr(isolate_module, "_roots_of", lambda h, n: [h ** sp.Rational(1, n)])
    result = run("x**2 = a", "x")
    assert result["status"] == "not_verified" and result["solutions"] == []


def test_a_missing_integer_family_is_caught_by_wide_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the 2 pi i k term the principal value still solves the equation (back-substitution passes),
    but unknowns with a large imaginary part are not returned: only wide probes see that."""
    real = isolate_module._apply_layer

    def principal_only(
        kind: str, extra: sp.Expr, state: object, taken: set[str], integer: list[sp.Symbol]
    ):  # type: ignore[no-untyped-def]
        if kind == "exp":
            return [isolate_module._nonzero_state(state, sp.log(state.h), state.h)], "logarithm"  # type: ignore[attr-defined]
        return real(kind, extra, state, taken, integer)  # type: ignore[arg-type]

    monkeypatch.setattr(isolate_module, "_apply_layer", principal_only)
    assert run("n = m*exp(-r*t)", "t")["status"] == "not_verified"


def test_a_wrong_formula_is_caught_by_substituting_back(monkeypatch: pytest.MonkeyPatch) -> None:
    real = isolate_module._apply_layer

    def off_by_one(
        kind: str,
        extra: sp.Expr,
        state: isolate_module.State,
        taken: set[str],
        integer: list[sp.Symbol],
    ):  # type: ignore[no-untyped-def]
        results, name = real(kind, extra, state, taken, integer)
        if kind == "add":
            return [isolate_module.State(r.h + 1, r.conds) for r in results if r is not None], name
        return results, name

    monkeypatch.setattr(isolate_module, "_apply_layer", off_by_one)
    assert run("x + b = c", "x")["status"] == "not_verified"


def test_a_tiny_push_forward_value_is_not_mistaken_for_a_zero(engine: Engine) -> None:
    """Regression: `exp(-r t)` at a large t is about 1e-17, which an absolute test called zero and so
    counted a complete solution as incomplete."""
    for assumptions in (
        {},
        {"n": ["positive"], "m": ["positive"], "r": ["positive"], "t": ["positive"]},
    ):
        result = engine.isolate(eq(r"n = m \exp(-r t)", assumptions), "t")
        assert result.status is Status.OK and result.complete


# --- randomized -----------------------------------------------------------------------------------------------------------------------
def test_planted_roots_are_recovered(engine: Engine) -> None:
    rng = random.Random(20261113)
    x = sp.Symbol("x")
    for _ in range(8):
        roots = rng.sample(range(-6, 7), rng.randint(1, 4))
        polynomial = sp.expand(sp.Mul(*[(x - r) for r in roots]) * rng.choice([1, 2, 3]))
        result = engine.isolate(
            ExpressionInput(input=f"{sp.sstr(polynomial)} = 0", format="text"), "x"
        )
        assert result.status is Status.OK
        assert same_set(values(result), [sp.Integer(r) for r in roots]), (roots, result.solutions)


def test_planted_solutions_of_linear_equations(engine: Engine) -> None:
    rng = random.Random(7)
    for _ in range(8):
        a, b, c = (rng.choice([n for n in range(-9, 10) if n]) for _ in range(3))
        result = engine.isolate(eq(f"{a} x + {b} = {c}"), "x")
        assert values(result) == [sp.Rational(c - b, a)]
