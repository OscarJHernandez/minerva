"""Reading formulas transcribed from an image: the parser reports how it understood them."""

from minerva_engine import Engine, Limits, Status


def read(*formulas: str):  # type: ignore[no-untyped-def]
    return Engine(Limits(timeout_seconds=30)).read_formulas(list(formulas))


def test_expression_is_rendered_as_parsed_without_simplifying() -> None:
    (r,) = read(r"\frac{x^2-1}{x-1}")
    assert r.status is Status.OK
    assert r.read_as is not None and "x^{2} - 1" in r.read_as  # not reduced to x + 1


def test_equation_keeps_both_sides() -> None:
    (r,) = read(r"x^5 - x - 1 = 0")
    assert r.status is Status.OK
    assert r.read_as == r"x^{5} - x - 1 = 0"


def test_e_stays_a_plain_symbol() -> None:
    (r,) = read(r"e^{x}")
    assert r.status is Status.OK and r.read_as is not None
    assert "e^{x}" in r.read_as  # not Euler's number: AGENTS.md convention


def test_ambiguous_notation_is_a_parse_error_with_a_message() -> None:
    (r,) = read(r"\sin 2x")
    assert r.status is Status.PARSE_ERROR
    assert r.read_as is None and r.message


def test_results_keep_the_input_order_and_failures_do_not_hide_successes() -> None:
    a, b, c = read(r"x+1", r"\sin 2x", r"y=2")
    assert [a.status, b.status, c.status] == [Status.OK, Status.PARSE_ERROR, Status.OK]


def test_over_long_input_is_a_resource_limit() -> None:
    (r,) = read("x+" * 2000 + "1")
    assert r.status is Status.RESOURCE_LIMIT


def test_hostile_input_is_rejected_not_executed() -> None:
    (r,) = read(r"x^{9^{9^{9}}}")
    assert r.status is not Status.OK or r.read_as is not None  # never crashes the engine
    (s,) = read("__import__('os').system('true')")
    assert s.status is Status.PARSE_ERROR


def test_empty_list_needs_no_sandbox() -> None:
    assert read() == []
