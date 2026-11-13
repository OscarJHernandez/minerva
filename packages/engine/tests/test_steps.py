"""Rewriting one chosen part: exact placement, honest refusals, whole-expression verification."""

import random

import pytest
import sympy as sp
from minerva_engine import (
    Engine,
    ExpressionInput,
    Limits,
    Reason,
    Rule,
    Status,
    parse_text,
)
from minerva_engine import steps as steps_module
from minerva_engine.steps import Chosen, find_matches, node_at, splice, walk

NC = {"A": ["noncommutative"], "B": ["noncommutative"], "C": ["noncommutative"]}

DOUBLE_SIN = Rule(id="trig.double_sin", lhs="sin(2*a)", rhs="2*sin(a)*cos(a)", variables=["a"])
PYTHAGORAS = Rule(id="trig.pythagorean", lhs="sin(a)**2 + cos(a)**2", rhs="1", variables=["a"])
LOG_PRODUCT = Rule(
    id="log.product",
    lhs="log(a*b)",
    rhs="log(a) + log(b)",
    variables=["a", "b"],
    conditions=[
        {"variable": "a", "assumption": "positive"},
        {"variable": "b", "assumption": "positive"},
    ],
)
DIFFERENCE_OF_SQUARES = Rule(
    id="alg.difference_of_squares", lhs="a**2 - b**2", rhs="(a - b)*(a + b)", variables=["a", "b"]
)


def expr(text: str, assumptions: dict[str, list[str]] | None = None) -> ExpressionInput:
    return ExpressionInput(input=text, assumptions=assumptions or {})


def same(got: str | None, wanted: str, assumptions: dict[str, list[str]] | None = None) -> bool:
    """Same expression tree (not merely an equal function)."""
    assert got is not None
    nc = bool(assumptions) and any("noncommutative" in v for v in (assumptions or {}).values())
    read = lambda t: parse_text(t, assumptions, allow_noncommutative=nc)  # noqa: E731
    return bool(read(got) == read(wanted))


# --- exactly one place changes -------------------------------------------------------------------------
def test_expand_changes_only_the_chosen_factor(engine: Engine) -> None:
    result = engine.rewrite_at(expr(r"(x+1)^2 (y+1)^2"), "expand", part=r"(x+1)^2")
    assert result.status is Status.OK
    assert same(result.result_text, "(x**2 + 2*x + 1)*(y + 1)**2")
    assert result.steps[0].matched_latex == "\\left(x + 1\\right)^{2}"
    assert result.verification is not None and result.verification.passed


def test_a_sub_sum_can_be_chosen_and_the_rest_is_untouched(engine: Engine) -> None:
    result = engine.rewrite_at(expr("x^2 - 1 + y^2"), "factor", part="x^2 - 1")
    assert result.status is Status.OK
    assert same(result.result_text, "(x - 1)*(x + 1) + y**2")


def test_a_part_inside_a_function_argument(engine: Engine) -> None:
    result = engine.rewrite_at(expr(r"\sqrt{x^2+2x+1} + x"), "factor", part="x^2 + 2 x + 1")
    assert result.status is Status.OK
    assert same(result.result_text, "x + sqrt((x + 1)**2)")


def test_only_the_chosen_one_of_two_equal_places_changes(engine: Engine) -> None:
    text = r"\sin(2x) + y\sin(2x)"
    first = engine.rewrite_at(expr(text), "rule", part=r"\sin(2x)", occurrence=1, rule=DOUBLE_SIN)
    second = engine.rewrite_at(expr(text), "rule", part=r"\sin(2x)", occurrence=2, rule=DOUBLE_SIN)
    assert same(first.result_text, "2*y*sin(x)*cos(x) + sin(2*x)")
    assert same(second.result_text, "y*sin(2*x) + 2*sin(x)*cos(x)")
    assert first.result_text != second.result_text


def test_the_rule_acts_on_the_chosen_term_not_on_the_first_match(engine: Engine) -> None:
    result = engine.rewrite_at(
        expr(r"\sin(2x) + \sin(2y)"), "rule", part=r"\sin(2y)", rule=DOUBLE_SIN
    )
    assert same(result.result_text, "2*sin(y)*cos(y) + sin(2*x)")


def test_an_identity_on_a_sub_sum(engine: Engine) -> None:
    result = engine.rewrite_at(
        expr(r"\sin^2 x + \cos^2 x + y"), "rule", part=r"\sin^2 x + \cos^2 x", rule=PYTHAGORAS
    )
    assert result.status is Status.OK and same(result.result_text, "y + 1")


def test_a_nested_part_is_found_and_rewritten(engine: Engine) -> None:
    result = engine.rewrite_at(
        expr(r"\cos(\sin^2 x + \cos^2 x)"), "rule", part=r"\sin^2 x + \cos^2 x", rule=PYTHAGORAS
    )
    assert same(result.result_text, "cos(1)")


# --- choosing: lists, paths, stale and ambiguous selections -------------------------------------------
def test_the_list_of_parts_boxes_each_one_and_numbers_equal_ones(engine: Engine) -> None:
    listed = engine.subterms(expr(r"\sin(2x) + y\sin(2x)"), find=r"\sin(2x)")
    assert [s.occurrence for s in listed.subterms] == [1, 2]
    assert all(s.occurrences == 2 for s in listed.subterms)
    assert all(r"\boxed{" in s.highlight_latex for s in listed.subterms)


def test_a_part_chosen_from_the_list_can_be_applied_by_its_path(engine: Engine) -> None:
    text = r"\sin(2x) + \sin(2y)"
    listed = engine.subterms(expr(text), rule=DOUBLE_SIN)
    assert [s.latex for s in listed.subterms] == [
        r"\sin{\left(2 x \right)}",
        r"\sin{\left(2 y \right)}",
    ]
    pick = listed.subterms[1]
    result = engine.rewrite_at(
        expr(text),
        "rule",
        rule=DOUBLE_SIN,
        selection={"path": pick.path, "args": pick.args, "expected_latex": pick.latex},
    )
    assert same(result.result_text, "2*sin(y)*cos(y) + sin(2*x)")


def test_a_stale_selection_is_refused_not_applied_somewhere_else(engine: Engine) -> None:
    listed = engine.subterms(expr(r"\sin(2x) + \sin(2y)"), find=r"\sin(2y)")
    pick = listed.subterms[0]
    selection = {"path": pick.path, "args": pick.args, "expected_latex": pick.latex}
    other = engine.rewrite_at(expr(r"\sin(2x) + \cos(2y) + z"), "expand", selection=selection)
    assert other.status is Status.PARSE_ERROR and other.reason is Reason.STALE_SELECTION
    assert other.result_text is None
    out_of_range = engine.rewrite_at(
        expr("x + y"), "expand", selection={"path": [5, 5], "expected_latex": "x"}
    )
    assert out_of_range.reason is Reason.STALE_SELECTION


def test_a_part_that_occurs_twice_needs_an_occurrence(engine: Engine) -> None:
    result = engine.rewrite_at(
        expr(r"\sin(2x) + y\sin(2x)"), "rule", part=r"\sin(2x)", rule=DOUBLE_SIN
    )
    assert result.status is Status.PARSE_ERROR and result.reason is Reason.AMBIGUOUS_SELECTION
    assert len(result.subterms) == 2
    beyond = engine.rewrite_at(
        expr(r"\sin(2x) + y\sin(2x)"), "rule", part=r"\sin(2x)", occurrence=3, rule=DOUBLE_SIN
    )
    assert beyond.reason is Reason.STALE_SELECTION


def test_a_part_that_is_not_there(engine: Engine) -> None:
    result = engine.rewrite_at(expr(r"\sin x"), "expand", part=r"\tan x")
    assert result.status is Status.PARSE_ERROR and result.reason is Reason.STALE_SELECTION


def test_the_list_is_bounded(engine: Engine) -> None:
    big = "+".join(f"\\sin(x_{i}+{i})" for i in range(30))
    listed = engine.subterms(expr(big))
    assert len(listed.subterms) == 40 and listed.truncated


# --- identities: conditions, no match, structure that is not in the expression -----------------------
def test_an_identity_whose_conditions_are_unproven_is_reported_and_not_applied(
    engine: Engine,
) -> None:
    result = engine.rewrite_at(
        expr(r"\log(x y) + \log(x z)"), "rule", part=r"\log(x y)", rule=LOG_PRODUCT
    )
    assert result.status is Status.CONDITIONS_NOT_SATISFIED
    assert result.result_text is None
    assert {(c.variable, c.assumption) for c in result.unmet_conditions} == {
        ("a", "positive"),
        ("b", "positive"),
    }


def test_the_same_identity_applies_once_the_assumptions_are_declared(engine: Engine) -> None:
    result = engine.rewrite_at(
        expr(r"\log(x y) + \log(x z)", {"x": ["positive"], "y": ["positive"]}),
        "rule",
        part=r"\log(x y)",
        rule=LOG_PRODUCT,
    )
    assert result.status is Status.OK
    assert same(
        result.result_text, "log(x) + log(y) + log(x*z)", {"x": ["positive"], "y": ["positive"]}
    )


def test_an_identity_that_does_not_match_the_part_leaves_the_expression_alone(
    engine: Engine,
) -> None:
    result = engine.rewrite_at(expr(r"\sin(3x) + y"), "rule", part=r"\sin(3x)", rule=DOUBLE_SIN)
    assert result.status is Status.UNCHANGED and result.reason is Reason.NO_MATCH
    assert same(result.result_text, "sin(3*x) + y")


def test_a_match_that_only_works_by_inventing_roots_is_still_discarded(engine: Engine) -> None:
    """`a**2 - b**2` matches `x + y` with a = sqrt(x), b = sqrt(-y): true, but nobody means it."""
    result = engine.rewrite_at(expr("x + y + z"), "rule", part="x + y", rule=DIFFERENCE_OF_SQUARES)
    assert result.status is Status.UNCHANGED and result.reason is Reason.NO_MATCH


def test_a_rule_cannot_be_read_backwards_when_its_right_side_does_not_determine_it(
    engine: Engine,
) -> None:
    result = engine.rewrite_at(expr(r"\sin^2 x + \cos^2 x"), "rule", rule=PYTHAGORAS, reverse=True)
    assert result.status is Status.UNSUPPORTED


# --- collect --------------------------------------------------------------------------------------------
def test_collect_by_a_variable(engine: Engine) -> None:
    result = engine.rewrite_at(expr("a x^2 y + b x y^2 + c x^2 + d y^2"), "collect", by=["x"])
    assert result.status is Status.OK
    assert same(result.result_text, "b*x*y**2 + d*y**2 + x**2*(a*y + c)")


def test_collect_by_a_power_and_by_a_function(engine: Engine) -> None:
    assert same(
        engine.rewrite_at(expr("a x^2 + b x^2 y + x"), "collect", by=["x^2"]).result_text,
        "x**2*(a + b*y) + x",
    )
    assert same(
        engine.rewrite_at(
            expr(r"a \sin x + b \sin x + c \cos x"), "collect", by=[r"\sin x"]
        ).result_text,
        "c*cos(x) + (a + b)*sin(x)",
    )


def test_collect_by_something_that_is_not_there_changes_nothing(engine: Engine) -> None:
    result = engine.rewrite_at(expr("x + 1"), "collect", by=["z"])
    assert result.status is Status.UNCHANGED and result.reason is Reason.PATTERN_NOT_FOUND


@pytest.mark.parametrize("by", [[], ["x", "y", "z", "u", "w"], ["3"]])
def test_collect_needs_between_one_and_four_variables_or_parts(
    engine: Engine, by: list[str]
) -> None:
    assert engine.rewrite_at(expr("x + y"), "collect", by=by).status is Status.PARSE_ERROR


# --- domain restrictions and verification ---------------------------------------------------------------
def test_a_cancelled_denominator_is_reported(engine: Engine) -> None:
    result = engine.rewrite_at(expr(r"\frac{x^2-1}{x-1}"), "cancel")
    assert result.status is Status.OK and same(result.result_text, "x + 1")
    assert result.restrictions_latex == [r"x \neq 1"]


def test_a_wrong_candidate_is_never_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a move produced a different expression, verification of the whole expression refuses it."""
    x = sp.Symbol("x")
    monkeypatch.setattr(
        steps_module,
        "_builtin",
        lambda *a, **k: steps_module._Move(x + 1, "builtin", "sympy.fake"),
    )
    payload = {
        "limits": Limits().model_dump(),
        "expression": {"input": "x + 2*y", "format": "text", "assumptions": {}},
        "action": "expand",
        "part": "x",
    }
    result = steps_module.op_rewrite_at(payload)
    assert result["status"] == "not_verified"
    assert result["result_text"] is None and result["result_latex"] is None
    assert result["diagnostics"]["rejected"] == {"not_verified": 1}


# --- noncommuting symbols ------------------------------------------------------------------------------------
def test_expanding_a_square_keeps_the_order_of_factors(engine: Engine) -> None:
    result = engine.rewrite_at(expr("(A + B)^2 + x", NC), "expand", part="(A + B)^2")
    assert result.status is Status.OK
    assert same(result.result_text, "A**2 + A*B + B*A + B**2 + x", NC)
    assert result.diagnostics.verification_method == "matrix"
    assert result.verification is not None and result.verification.passed


def test_the_expanded_part_is_placed_inside_a_product_in_order(engine: Engine) -> None:
    result = engine.rewrite_at(expr("(A + B)^2 C", NC), "expand", part="(A + B)^2")
    assert same(result.result_text, "(A**2 + A*B + B*A + B**2)*C", NC)


@pytest.mark.parametrize("action", ["simplify", "factor", "cancel"])
def test_moves_that_may_reorder_noncommuting_factors_are_unsupported(
    engine: Engine, action: str
) -> None:
    assert engine.rewrite_at(expr("A B - B A", NC), action).status is Status.UNSUPPORTED


def test_collect_by_a_noncommuting_symbol_is_unsupported_and_by_a_commuting_one_keeps_order(
    engine: Engine,
) -> None:
    assert (
        engine.rewrite_at(expr("A B + B A", NC), "collect", by=["A"]).status is Status.UNSUPPORTED
    )
    result = engine.rewrite_at(expr("x A B + y B A + x B A", NC), "collect", by=["x"])
    assert same(result.result_text, "x*(A*B + B*A) + y*B*A", NC)


def test_a_function_of_a_noncommuting_symbol_is_unsupported(engine: Engine) -> None:
    result = engine.rewrite_at(expr(r"\sin(A) + x", NC), "expand")
    assert result.status is Status.UNSUPPORTED


def test_matching_respects_order_for_noncommuting_factors() -> None:
    a, b, c = sp.symbols("A B C", commutative=False)
    assert [m.args for m in find_matches(a * b * c, a * b)] == [(0, 1)]
    assert find_matches(a * b * c, a * c) == [], "A C is not a run of A B C"
    x, y, z = sp.symbols("x y z")
    assert [m.args for m in find_matches(x * y * z, x * z)] == [(0, 2)]


# --- hostile and malformed input -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "part",
    ["__import__('os').system('true')", "(lambda: 1)()", "x__class__", "x ==", "9**9**9", ""],
)
def test_a_hostile_or_malformed_part_is_an_error_not_code(engine: Engine, part: str) -> None:
    result = engine.rewrite_at(ExpressionInput(input="x + y", format="text"), "expand", part=part)
    assert result.status in {Status.PARSE_ERROR, Status.RESOURCE_LIMIT}
    assert result.result_text is None


def test_an_equation_is_not_an_expression_here(engine: Engine) -> None:
    result = engine.rewrite_at(expr("x = y + 1"), "expand")
    assert result.status is Status.PARSE_ERROR


def test_an_unknown_move_is_refused(engine: Engine) -> None:
    assert engine.rewrite_at(expr("x + 1"), "integrate").status is Status.PARSE_ERROR


def test_an_exploding_expansion_is_a_resource_limit_not_a_guess() -> None:
    small = Engine(Limits(timeout_seconds=2))
    result = small.rewrite_at(expr("(x+1)^{9000} (y+1)^{9000}"), "expand", part="(x+1)^{9000}")
    assert result.status is Status.RESOURCE_LIMIT and result.result_text is None


def test_diagnostics_hold_no_expression(engine: Engine) -> None:
    result = engine.rewrite_at(
        expr("zzmarkerzz^2 + (zzmarkerzz+1)^2"), "expand", part="(zzmarkerzz+1)^2"
    )
    assert result.status is Status.OK
    dumped = result.diagnostics.model_dump_json() + str(result.diagnostics.log_fields())
    assert "zzmarkerzz" not in dumped
    assert result.diagnostics.verification_method == "numeric"
    assert result.diagnostics.verification_outcome == "passed"
    assert result.diagnostics.parse_ms >= 0 and result.diagnostics.candidates >= 1


# --- randomized: replacing a part is exact -------------------------------------------------------------------
def random_expression(rng: random.Random, depth: int = 3) -> sp.Expr:
    symbols = sp.symbols("x y z")
    if depth == 0 or rng.random() < 0.25:
        return rng.choice([*symbols, sp.Integer(rng.randint(2, 5))])
    kind = rng.choice(["add", "mul", "pow", "sin"])
    first, second = random_expression(rng, depth - 1), random_expression(rng, depth - 1)
    if kind == "add":
        return first + second
    if kind == "mul":
        return first * second
    if kind == "pow":
        return first ** rng.randint(2, 3)
    return sp.sin(first)


def test_splicing_a_part_back_in_changes_nothing_and_a_marker_lands_exactly_there() -> None:
    rng = random.Random(20261113)
    checked = 0
    for _ in range(150):
        e = random_expression(rng)
        for path, node in walk(e):
            chosen = Chosen(path, None, node, node)
            assert splice(e, chosen, node) == e
            marker = sp.Dummy("m")
            rebuilt = splice(e, chosen, marker)
            # The marker occurs exactly once, unless SymPy folded it away with a neighbour.
            if rebuilt.has(marker):
                assert rebuilt.count(marker) == 1
            checked += 1
    assert checked > 200


def test_every_listed_path_leads_to_the_listed_node() -> None:
    rng = random.Random(7)
    for _ in range(40):
        e = random_expression(rng)
        for path, node in walk(e):
            assert node_at(e, path) == node


def test_the_rejected_candidate_is_kept_only_when_debugging_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sp.Symbol("x")
    monkeypatch.setattr(
        steps_module,
        "_builtin",
        lambda *a, **k: steps_module._Move(x + 1, "builtin", "sympy.fake"),
    )

    def call(debug: bool) -> dict:  # type: ignore[type-arg]
        return steps_module.op_rewrite_at(
            {
                "limits": Limits().model_dump(),
                "expression": {"input": "x + 2*y", "format": "text", "assumptions": {}},
                "action": "expand",
                "part": "x",
                "debug": debug,
            }
        )

    assert call(False)["diagnostics"]["detail"] == []
    detail = call(True)["diagnostics"]["detail"]
    assert detail and "failed verification" in detail[0]
    assert call(True)["status"] == "not_verified"
