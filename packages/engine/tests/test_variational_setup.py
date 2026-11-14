"""Reading a functional: notation, refusals, orders, regularity, boundary slots, next steps."""

from __future__ import annotations

import pytest
import sympy as sp
from minerva_engine import Engine, Limits
from minerva_engine.build import JetBinding
from minerva_engine.errors import ParseError
from minerva_engine.latexparse import parse_latex
from minerva_engine.models import Status
from minerva_engine.textparse import parse_text
from minerva_engine.variational import VariationInput
from minerva_engine.variational.jets import JetSpace
from pydantic import ValidationError

HIGHER = r"\frac{m}{2}\dot{x}^2+\frac{\alpha}{2}\ddot{x}^2-V(x)"


def analyze(engine: Engine, integrand: str, **kw: object):  # type: ignore[no-untyped-def]
    kw.setdefault("functions", ["x"])
    return engine.variation(VariationInput(integrand=integrand, action="analyze", **kw))  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def jets() -> JetSpace:
    return JetSpace.build("t", ("x", "y"), (2, 1), 4)


def test_total_derivative_is_not_the_partial_derivative(jets: JetSpace) -> None:
    x, xd, xdd = jets.u[0][0], jets.u[0][1], jets.u[0][2]
    f = x * xd + jets.variable * x
    assert jets.partial(f, 0, 0) == xd + jets.variable
    # D(x xd + t x) = xd^2 + x xdd + x + t xd
    assert sp.expand(jets.D(f) - (xd**2 + x * xdd + x + jets.variable * xd)) == 0


def test_every_derivative_spelling_is_one_jet(jets: JetSpace) -> None:
    binding = jets.binding()
    a = parse_latex(r"\dot{x}\,\ddot{x}\,x^{(3)}\,\ddddot{x}", jets=binding)
    b = parse_text("x_dot*x_ddot*x_d3*x_d4", jets=binding)
    c = parse_latex(r"\dot{x}\,\ddot{x}\,\dddot{x}\,x^{(4)}", jets=binding)
    assert a == b == c == jets.u[0][1] * jets.u[0][2] * jets.u[0][3] * jets.u[0][4]
    assert parse_latex(r"x(t)", jets=binding) == jets.u[0][0]
    assert parse_latex(r"x^{(2)}(t)", jets=binding) == jets.u[0][2]
    assert parse_latex(r"\dot{y}", jets=binding) == jets.u[1][1]


@pytest.mark.parametrize(
    "source",
    [r"x(s)", r"\dot{\dot{x}}", r"x_{d5}", r"\dot{z}", r"x^{(5)}", r"x^{(2)}(s)"],
)
def test_misreadings_are_refused_not_guessed(jets: JetSpace, source: str) -> None:
    with pytest.raises(ParseError):
        parse_latex(source, jets=jets.binding())


def test_without_the_jet_binding_the_old_behaviour_is_unchanged() -> None:
    # The shared parsers must keep refusing \dddot and keep reading x^{(4)} as a power.
    with pytest.raises(ParseError):
        parse_latex(r"\dddot{x}")
    assert parse_latex(r"x^{(4)}") == sp.Symbol("x") ** 4


def test_jet_binding_type_is_what_the_parsers_take(jets: JetSpace) -> None:
    assert isinstance(jets.binding(), JetBinding)


def test_analysis_of_the_higher_derivative_lagrangian(engine: Engine) -> None:
    r = analyze(
        engine,
        HIGHER,
        given_functions=[{"name": "V", "args": ["x"]}],
        lower="0",
        upper="1",
    )
    assert r.status is Status.OK
    a = r.analysis
    assert a is not None
    assert a.highest_order == 2
    assert a.functions[0].order == 2
    assert sorted(a.parameters) == ["alpha", "m"]
    assert a.given_functions == ["V"]
    # Four boundary slots: (lower, upper) x (x, xdot), none specified, nothing assumed.
    assert {(c.end, c.order, c.state) for c in a.conditions} == {
        (e, j, "unspecified") for e in ("lower", "upper") for j in (0, 1)
    }
    assert a.unspecified == 4
    assert a.conditions_needed_per_end == 2
    assert a.regularity == "conditional"  # alpha may be 0: nothing declared about it
    assert a.hessian_det_latex == r"\alpha"
    ids = [s.id for s in a.next_steps]
    assert ids[:2] == ["first_variation", "euler_lagrange"] and "declare_conditions" in ids
    assert any(s.id == "solve_stationary" and "parameter_values" in s.needs for s in a.next_steps)


def test_assumption_decides_regularity(engine: Engine) -> None:
    r = analyze(
        engine,
        HIGHER,
        given_functions=[{"name": "V", "args": ["x"]}],
        assumptions={"alpha": ["positive"], "m": ["positive"]},
    )
    assert r.analysis is not None and r.analysis.regularity == "regular"


def test_degenerate_and_mixed_order_problems(engine: Engine) -> None:
    r = analyze(engine, r"x\dot{x}")
    assert r.analysis is not None and r.analysis.regularity == "degenerate"
    # p x' - H(x, p): orders (1, 0), regular because W = [[0, 1], [1, -1]]
    r = analyze(engine, r"p\dot{x}-\frac12(p^2+x^2)", functions=["x", "p"], lower="0", upper="1")
    assert r.status is Status.OK and r.analysis is not None
    assert [f.order for f in r.analysis.functions] == [1, 0]
    assert r.analysis.regularity == "regular" and r.analysis.hessian_det_latex == "-1"
    assert r.analysis.conditions_needed_per_end == 1


def test_undeclared_call_is_an_error_not_a_product(engine: Engine) -> None:
    r = analyze(engine, r"\frac12\dot{x}^2-V(x)")
    assert r.status is Status.PARSE_ERROR and "V" in r.message


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"integrand": r"|\dot{x}|"}, Status.UNSUPPORTED),
        ({"integrand": r"\dot{x}^2", "lower": "1", "upper": "0"}, Status.PARSE_ERROR),
        ({"integrand": r"\dot{x}^2", "lower": "0", "upper": r"\infty"}, Status.PARSE_ERROR),
        ({"integrand": r"\dot{x}^2", "lower": "0", "upper": "t"}, Status.PARSE_ERROR),
        ({"integrand": r"\dot{x}^2+\dot{y}^2"}, Status.PARSE_ERROR),
        ({"integrand": r"\ddddot{x}^2"}, Status.OK),
        ({"integrand": r"x^{(5)}^2"}, Status.PARSE_ERROR),
        (
            {
                "integrand": r"\dot{x}^2",
                "conditions": [{"function": "z", "order": 0, "end": "lower", "kind": "fixed"}],
            },
            Status.PARSE_ERROR,
        ),
    ],
)
def test_refusals(engine: Engine, kwargs: dict[str, object], status: Status) -> None:
    kw = dict(kwargs)
    integrand = str(kw.pop("integrand"))
    assert analyze(engine, integrand, **kw).status is status


def test_declared_conditions_are_reported(engine: Engine) -> None:
    r = analyze(
        engine,
        r"\frac12\ddot{x}^2",
        conditions=[
            {"function": "x", "order": 0, "end": "lower", "kind": "fixed", "value": "0"},
            {"function": "x", "order": 1, "end": "lower", "kind": "fixed", "value": "0"},
            {"function": "x", "order": 0, "end": "upper", "kind": "free"},
        ],
    )
    assert r.analysis is not None
    states = {(c.end, c.order): c.state for c in r.analysis.conditions}
    assert states == {
        ("lower", 0): "fixed",
        ("lower", 1): "fixed",
        ("upper", 0): "free",
        ("upper", 1): "unspecified",
    }
    assert r.analysis.unspecified == 1


def test_default_interval_is_symbolic_and_clash_is_refused(engine: Engine) -> None:
    r = analyze(engine, r"\frac12\dot{x}^2")
    assert r.analysis is not None
    assert (r.analysis.lower_latex, r.analysis.upper_latex) == ("a", "b")
    assert analyze(engine, r"\frac{a}{2}\dot{x}^2").status is Status.PARSE_ERROR
    assert analyze(engine, r"\frac{a}{2}\dot{x}^2", lower="0", upper="1").status is Status.OK


def test_time_dependence_and_limits(engine: Engine) -> None:
    r = analyze(engine, r"\frac{t}{2}\dot{x}^2", lower="1", upper="2")
    assert r.analysis is not None and r.analysis.explicit_in_variable
    with pytest.raises(ValidationError):
        VariationInput(integrand="a1", format="text", functions=["a1", "a2", "a3", "a4", "a5"])


def test_engine_limits_exist() -> None:
    assert Limits().timeout_seconds > 0
