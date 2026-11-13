"""Substituting relations: premises are recorded, the order is never guessed, nothing unverified is shown."""

import random

import pytest
import sympy as sp
from minerva_engine import Engine, ExpressionInput, Limits, Reason, Status, parse_text
from minerva_engine import substitute as substitute_module

NC = {"A": ["noncommutative"], "B": ["noncommutative"], "D": ["noncommutative"]}


def expr(
    text: str, assumptions: dict[str, list[str]] | None = None, fmt: str = "latex"
) -> ExpressionInput:
    return ExpressionInput(input=text, format=fmt, assumptions=assumptions or {})  # type: ignore[arg-type]


# --- the worked example ---------------------------------------------------------------------------------
def test_the_pythagorean_example_is_recorded_as_a_step_with_its_premise(engine: Engine) -> None:
    """E = x^2 + y^2 with y^2 = 1 - x^2: the substitution, then the value it combines to."""
    result = engine.substitute(expr("x^2 + y^2"), ["y^2 = 1 - x^2"])
    assert result.status is Status.OK and result.result_text == "1"
    assert result.evidence == "conditional"
    assert [p.text for p in result.given] == ["y**2 = 1 - x**2"]
    (step,) = result.steps
    assert step.action == "substitute" and step.evidence == "conditional"
    assert step.display_latex is not None and "1 - x^{2}" in step.display_latex
    assert step.after_latex == "1"
    assert step.given == [result.given[0].latex]
    assert "did not check" in result.message and "proved" not in result.message


def test_a_result_that_holds_without_the_relation_says_so(engine: Engine) -> None:
    """The relation here is itself an identity, so the premise is not needed."""
    result = engine.substitute(expr(r"\sin^2 x + \cos^2 x + z"), [r"\sin^2 x = 1 - \cos^2 x"])
    assert result.status is Status.OK and result.evidence == "identity"
    assert "without the relation" in result.message


def test_the_relation_is_never_assumed_true_so_the_unconditional_check_can_fail(
    engine: Engine,
) -> None:
    result = engine.substitute(expr("x^2 + y^2"), ["y^2 = 1 - x^2"])
    assert result.verification is not None and result.verification.passed
    assert result.evidence == "conditional", "x^2 + y^2 is not 1 for all x, y"


# --- what is matched -------------------------------------------------------------------------------------------
def test_occurrences_inside_functions_and_sums(engine: Engine) -> None:
    nested = engine.substitute(expr(r"\sin(y^2) + \cos(y^2)"), ["y^2 = 1 - x^2"])
    assert nested.result_text is not None
    # SymPy writes sin(1 - x^2) as -sin(x^2 - 1): compare the expressions, not the spelling.
    assert parse_text(nested.result_text) == parse_text("sin(1 - x**2) + cos(1 - x**2)")
    assert engine.substitute(expr("a + b + c"), ["a + b = 5"]).result_text == "c + 5"


def test_a_relation_read_the_other_way_round(engine: Engine) -> None:
    """`2 = x` replaces x by 2: the side with symbols is the pattern."""
    assert engine.substitute(expr("x^2 + x"), ["2 = x"]).result_text == "6"


def test_a_pattern_that_does_not_occur_changes_nothing(engine: Engine) -> None:
    result = engine.substitute(expr("x + 1"), ["z = 2"])
    assert result.status is Status.UNCHANGED and result.reason is Reason.PATTERN_NOT_FOUND
    assert result.result_text == "x + 1"


def test_text_input_works_too(engine: Engine) -> None:
    result = engine.substitute(expr("x**2 + y**2", fmt="text"), ["y**2 = 1 - x**2"])
    assert result.status is Status.OK and result.result_text == "1"


def test_a_domain_restriction_is_kept(engine: Engine) -> None:
    result = engine.substitute(expr(r"\frac{y}{x}"), ["y = x^2"])
    assert result.status is Status.OK and result.result_text == "x"
    assert result.restrictions_latex == [r"x \neq 0"]


# --- several relations: the order is part of the answer ---------------------------------------------------------
def test_relations_that_do_not_interact_need_no_order(engine: Engine) -> None:
    result = engine.substitute(expr("a b + c"), ["a = 2 c", "b = 3"])
    assert result.status is Status.OK and result.result_text == "7*c"
    assert result.order_dependent is False and len(result.steps) == 2


def test_an_order_dependent_pair_is_refused_until_the_order_is_stated(engine: Engine) -> None:
    refused = engine.substitute(expr("x + y"), ["x = y", "y = 1"])
    assert refused.status is Status.UNSUPPORTED and refused.reason is Reason.ORDER_DEPENDENT
    assert refused.result_text is None and refused.order_dependent is True
    given = engine.substitute(expr("x + y"), ["x = y", "y = 1"], order="given")
    assert given.result_text == "2" and given.order_dependent is True
    assert "different expression" in given.message
    reversed_order = engine.substitute(expr("x + y"), ["y = 1", "x = y"], order="given")
    assert reversed_order.result_text == "y + 1", "the stated order is what is applied"
    together = engine.substitute(expr("x + y"), ["x = y", "y = 1"], order="simultaneous")
    assert together.result_text == "y + 1"


def test_relations_that_rewrite_each_other_are_called_cyclic(engine: Engine) -> None:
    result = engine.substitute(expr("x + 1"), ["x = y", "y = x"])
    assert result.status is Status.UNSUPPORTED and result.reason is Reason.CYCLIC


def test_relations_are_applied_once_each_so_a_cycle_cannot_loop(engine: Engine) -> None:
    result = engine.substitute(expr("x + 1"), ["x = y", "y = x"], order="given")
    assert result.status is Status.UNCHANGED and result.result_text == "x + 1"
    assert len(result.steps) == 2, "both relations were applied; they cancelled out"
    assert "original expression" in result.message


# --- relations that cannot be used ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("relation", "status"),
    [
        ("x", Status.PARSE_ERROR),  # no equals sign
        ("x = y = z", Status.PARSE_ERROR),  # two
        ("x = ", Status.PARSE_ERROR),  # empty side
        ("2 = 3", Status.UNSUPPORTED),  # numbers only
        ("x = x + 1", Status.UNSUPPORTED),  # can never hold
        ("__import__('os') = 1", Status.PARSE_ERROR),
    ],
)
def test_unusable_relations_are_refused_not_guessed(
    engine: Engine, relation: str, status: Status
) -> None:
    result = engine.substitute(expr("x + 1", fmt="text"), [relation])
    assert result.status is status and result.result_text is None


def test_zero_or_too_many_relations(engine: Engine) -> None:
    assert engine.substitute(expr("x"), []).status is Status.PARSE_ERROR
    assert engine.substitute(expr("x"), ["x = 1"] * 5).status is Status.PARSE_ERROR
    assert engine.substitute(expr("x"), ["x = 1"], order="whatever").status is Status.PARSE_ERROR


def test_an_equation_is_not_an_expression_to_substitute_into(engine: Engine) -> None:
    assert engine.substitute(expr("x = y"), ["x = 1"]).status is Status.PARSE_ERROR


def test_an_oversized_relation_is_a_resource_limit(engine: Engine) -> None:
    long = "x = " + "+".join(["y"] * 1500)
    assert engine.substitute(expr("x"), [long]).status is Status.RESOURCE_LIMIT


# --- noncommuting symbols ---------------------------------------------------------------------------------------------
def test_a_product_of_noncommuting_symbols_is_matched_in_order(engine: Engine) -> None:
    replaced = engine.substitute(expr("A^2 B", NC), ["A B = D"])
    assert replaced.status is Status.OK and replaced.result_text == "A*D"
    assert replaced.diagnostics.verification_method == "congruence"
    swapped = engine.substitute(expr("B A", NC), ["A B = D"])
    assert swapped.status is Status.UNCHANGED, "B A is not A B"


# --- a wrong substitution is never shown -------------------------------------------------------------------------------
def test_a_misidentified_occurrence_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the placeholder stood for something that is not the pattern, putting the pattern back would
    not reproduce the original expression, and the verification refuses the step."""

    def wrong(expression: sp.Expr, pattern: sp.Expr) -> tuple[sp.Expr, sp.Dummy]:
        placeholder = sp.Dummy("p")
        return (
            expression + placeholder - placeholder * 2,
            placeholder,
        )  # not the original with p := pattern

    monkeypatch.setattr(substitute_module, "_marked", wrong)
    payload = {
        "limits": Limits().model_dump(),
        "expression": {"input": "x + y", "format": "text", "assumptions": {}},
        "relations": ["y = 2"],
    }
    result = substitute_module.op_substitute(payload)
    assert result["status"] == "not_verified" and result["result_text"] is None
    assert result["diagnostics"]["rejected"] == {"not_verified": 1}


# --- randomized ----------------------------------------------------------------------------------------------------
def test_substituting_a_polynomial_relation_agrees_with_direct_evaluation(engine: Engine) -> None:
    """For random E(x, y^2) and y^2 = g(x): the result at x0 equals E at (x0, y0) for y0 = sqrt(g(x0))."""
    rng = random.Random(20261113)
    x, y = sp.symbols("x y")
    for _ in range(12):
        terms = [
            rng.randint(-3, 3) * x ** rng.randint(0, 2) * y ** (2 * rng.randint(0, 2))
            for _ in range(3)
        ]
        e = sp.Add(*terms) + 1
        g = rng.randint(-3, 3) * x + rng.randint(1, 4)
        if not e.has(y):
            continue
        result = engine.substitute(
            ExpressionInput(input=str(e).replace("**", "^"), format="text"), [f"y**2 = {g}"]
        )
        assert result.status is Status.OK and result.result_text is not None
        got = parse_text(result.result_text)
        for x0 in (sp.Rational(1, 3), sp.Integer(2), sp.Rational(-5, 4)):
            y0 = sp.sqrt(g.subs(x, x0))
            assert sp.simplify(got.subs(x, x0) - e.subs({x: x0, y: y0})) == 0


def test_other_orders_are_listed_only_when_debugging_is_requested(engine: Engine) -> None:
    plain = engine.substitute(expr("x + y"), ["x = y", "y = 1"])
    assert plain.diagnostics.detail == [], (
        "by default nothing that could hold an expression is kept"
    )
    debug = engine.substitute(expr("x + y"), ["x = y", "y = 1"], debug=True)
    assert any(line.startswith("order [1, 2]: 2") for line in debug.diagnostics.detail)
    assert any(line.startswith("order [2, 1]:") for line in debug.diagnostics.detail)
    assert debug.status is Status.UNSUPPORTED, "debugging never changes the outcome"
