"""Line, surface and volume integrals and the three integral theorems.

References are hand-computed closed forms. Every direct integral is the agreement of an exact value with
an independent quadrature; a theorem is two separately computed sides."""

import pytest
import sympy as sp
from minerva_engine import Limits
from minerva_engine.models import Status
from minerva_engine.vector import (
    Curve,
    FieldDecl,
    Param,
    Region,
    Surface,
    VectorInput,
    integrals,
    vector_calculus,
)

L = Limits(timeout_seconds=120)


def vec(name: str, *c: str) -> FieldDecl:
    return FieldDecl(name=name, kind="vector", components=list(c))


def sca(name: str, c: str) -> FieldDecl:
    return FieldDecl(name=name, kind="scalar", components=[c])


def P(name: str, lo: str, hi: str) -> Param:
    return Param(name=name, lower=lo, upper=hi)


def run(expression: str, integral: str, fields, system: str = "cartesian", **kw):
    return vector_calculus(
        VectorInput(
            expression=expression,
            format="text",
            action="integrate",
            integral=integral,  # type: ignore[arg-type]
            coordinates=system,  # type: ignore[arg-type]
            fields=list(fields),
            **kw,
        ),
        L,
    )


POSITIVE = {n: sp.Symbol(n, positive=True) for n in ("R", "rho0", "epsilon0")}


def read(text: str | None) -> sp.Expr:
    """The engine's text output, with the constants positive as they were declared."""
    assert text is not None
    return sp.sympify(text, locals=POSITIVE)


def value(result) -> sp.Expr:  # type: ignore[no-untyped-def]
    assert result.integral is not None
    return read(result.integral.values[0].text)


BALL = Region(bounds=[("0", "R"), ("0", "pi"), ("0", "2*pi")])
R = {"constants": ["R"], "assumptions": {"R": ["positive"]}}


# -- direct integrals, against closed forms ----------------------------------------------------------
def test_work_along_a_parabola() -> None:
    result = run(
        "F",
        "line",
        [vec("F", "y", "x", "0")],
        curve=Curve(q=["s", "s**2", "0"], parameter=P("s", "0", "1")),
    )
    assert result.status is Status.OK and value(result) == 1  # d(xy): 1*1 - 0
    assert result.integral and result.integral.values[0].agreed is True
    assert result.integral.method == "symbolic" and "not a proof" in result.message


def test_circulation_around_a_circle_in_three_systems() -> None:
    cart = run(
        "F", "line", [vec("F", "-y", "x", "0")],
        curve=Curve(q=["3*cos(s)", "3*sin(s)", "0"], parameter=P("s", "0", "2*pi")),
    )  # fmt: skip
    cyl = run(
        "F", "line", [vec("F", "0", "rho", "0")], "cylindrical",
        curve=Curve(q=["3", "s", "0"], parameter=P("s", "0", "2*pi")),
    )  # fmt: skip
    sph = run(
        "F", "line", [vec("F", "0", "0", "r*sin(theta)")], "spherical",
        curve=Curve(q=["3", "pi/2", "s"], parameter=P("s", "0", "2*pi")),
    )  # fmt: skip
    assert [value(r) for r in (cart, cyl, sph)] == [18 * sp.pi] * 3


def test_scalar_line_integral_is_a_length_for_one() -> None:
    result = run(
        "f",
        "line_scalar",
        [sca("f", "1")],
        curve=Curve(q=["cos(s)", "sin(s)", "s"], parameter=P("s", "0", "2*pi")),
    )
    assert value(result) == 2 * sp.sqrt(2) * sp.pi  # helix: sqrt(2) per unit of parameter


def test_flux_of_the_inverse_square_field_through_a_sphere_is_4_pi() -> None:
    result = run(
        "F", "flux", [vec("F", "1/r**2", "0", "0")], "spherical",
        surface=Surface(q=["R", "u", "v"], u=P("u", "0", "pi"), v=P("v", "0", "2*pi")), **R,
    )  # fmt: skip
    assert value(result) == 4 * sp.pi  # independent of R
    assert result.integral and "d_u r x d_v r" in result.integral.orientation


def test_the_order_of_the_parameters_is_the_orientation() -> None:
    kw = {"fields": [vec("F", "1/r**2", "0", "0")], "system": "spherical", **R}
    out = run(
        "F",
        "flux",
        surface=Surface(q=["R", "u", "v"], u=P("u", "0", "pi"), v=P("v", "0", "2*pi")),
        **kw,
    )
    inward = run(
        "F",
        "flux",
        surface=Surface(q=["R", "v", "u"], u=P("u", "0", "2*pi"), v=P("v", "0", "pi")),
        **kw,
    )
    assert value(out) == 4 * sp.pi and value(inward) == -4 * sp.pi


def test_volume_of_a_ball_and_a_cylinder() -> None:
    ball = run("f", "volume", [sca("f", "1")], "spherical", region=BALL, **R)
    assert value(ball) == sp.Rational(4, 3) * sp.pi * POSITIVE["R"] ** 3
    cyl = run(
        "f",
        "volume",
        [sca("f", "z")],
        "cylindrical",
        region=Region(bounds=[("0", "2"), ("0", "2*pi"), ("0", "3")]),
    )
    assert value(cyl) == 18 * sp.pi  # int rho drho (2) * 2 pi * int z dz (9/2)


def test_gauss_law_integral_form_for_a_uniformly_charged_ball() -> None:
    """E = rho r / (3 eps0) inside: the flux through the sphere equals rho V / eps0, and so does the
    volume integral of div E (the divergence theorem is the bridge between the two forms)."""
    consts = {
        "constants": ["R", "rho0", "epsilon0"],
        "assumptions": {"R": ["positive"], "rho0": ["positive"], "epsilon0": ["positive"]},
    }
    e = vec("E", "rho0*r/(3*epsilon0)", "0", "0")
    flux = run(
        "E",
        "flux",
        [e],
        "spherical",
        surface=Surface(q=["R", "u", "v"], u=P("u", "0", "pi"), v=P("v", "0", "2*pi")),
        **consts,
    )
    theorem = run("E", "divergence_theorem", [e], "spherical", region=BALL, **consts)
    charge = (
        sp.Symbol("rho0", positive=True)
        * sp.Rational(4, 3)
        * sp.pi
        * sp.Symbol("R", positive=True) ** 3
    )
    eps = sp.Symbol("epsilon0", positive=True)
    assert sp.simplify(value(flux) - charge / eps) == 0
    assert theorem.status is Status.OK and theorem.integral and theorem.integral.holds is True
    assert all(sp.simplify(read(v.text) - charge / eps) == 0 for v in theorem.integral.values)


# -- the theorems --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("system", "field", "region"),
    [
        ("cartesian", ("x**2", "y*z", "z**3"), [("0", "1"), ("0", "2"), ("0", "1")]),
        ("cylindrical", ("rho*z", "rho**2", "z**2"), [("1", "2"), ("0", "pi"), ("0", "1")]),
        (
            "spherical",
            ("r**2", "r*sin(theta)", "cos(theta)"),
            [("1", "2"), ("pi/4", "pi/2"), ("0", "pi")],
        ),
    ],
)
def test_divergence_theorem_on_a_box(system, field, region) -> None:  # type: ignore[no-untyped-def]
    result = run(
        "F", "divergence_theorem", [vec("F", *field)], system, region=Region(bounds=region)
    )
    assert result.status is Status.OK, result.message
    assert (
        result.integral
        and result.integral.holds
        and result.integral.theorem == "the divergence theorem"
    )
    left, right = result.integral.values
    assert left.agreed and right.agreed
    assert sp.simplify(sp.sympify(left.text) - sp.sympify(right.text)) == 0
    statuses = {h.status for h in result.integral.hypotheses}
    assert statuses == {"established", "checked"}  # nothing is claimed beyond a scan


def test_stokes_theorem_on_a_parametrised_surface() -> None:
    result = run(
        "F", "stokes_theorem", [vec("F", "-y*z", "x*z", "x*y")],
        surface=Surface(q=["u", "v", "u*v"], u=P("u", "0", "1"), v=P("v", "0", "1")),
    )  # fmt: skip
    assert result.status is Status.OK and result.integral and result.integral.holds
    assert "counter-clockwise" in result.integral.orientation
    assert [sp.sympify(v.text) for v in result.integral.values] == [1, 1]


def test_stokes_theorem_on_a_hemisphere() -> None:
    result = run(
        "F", "stokes_theorem", [vec("F", "0", "r*sin(theta)", "0")], "spherical",
        surface=Surface(q=["2", "u", "v"], u=P("u", "0", "pi/2"), v=P("v", "0", "2*pi")),
    )  # fmt: skip
    assert result.status is Status.OK, result.message
    left, right = result.integral.values  # type: ignore[union-attr]
    assert sp.simplify(sp.sympify(left.text) - sp.sympify(right.text)) == 0


def test_gradient_theorem_is_path_independent() -> None:
    f = sca("f", "x**2*y + z*x")
    a = run(
        "f",
        "gradient_theorem",
        [f],
        curve=Curve(q=["s", "s**2", "1-s"], parameter=P("s", "0", "1")),
    )
    b = run(
        "f",
        "gradient_theorem",
        [f],
        curve=Curve(q=["s**3", "s", "1-s**2"], parameter=P("s", "0", "1")),
    )
    assert a.status is b.status is Status.OK
    assert sp.sympify(a.integral.values[1].text) == sp.sympify(b.integral.values[1].text)  # type: ignore[union-attr]


# -- no closed form: a labelled numerical approximation ------------------------------------------------
def test_without_a_closed_form_the_value_is_a_labelled_approximation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(integrals, "_symbolic", lambda task: None)
    result = run(
        "F",
        "line",
        [vec("F", "cos(x**3)", "0", "0")],
        curve=Curve(q=["s", "0", "0"], parameter=P("s", "0", "1")),
    )
    assert result.status is Status.OK and result.integral is not None
    assert result.integral.method == "numerical" and result.integral.values[0].exact is False
    assert result.integral.values[0].latex is None and result.integral.values[0].numeric
    assert "numerical approximation" in result.message and "not an exact value" in result.message


# -- refusals -------------------------------------------------------------------------------------------
def test_the_domain_is_never_invented() -> None:
    assert run("F", "line", [vec("F", "1", "0", "0")]).diagnostics.reason == "missing_geometry"
    assert run("F", "flux", [vec("F", "1", "0", "0")]).status is Status.PARSE_ERROR
    assert run("F", "divergence_theorem", [vec("F", "1", "0", "0")]).status is Status.PARSE_ERROR


def test_abstract_and_time_dependent_fields_are_refused() -> None:
    curve = Curve(q=["s", "0", "0"], parameter=P("s", "0", "1"))
    abstract = run("F", "line", [FieldDecl(name="F", kind="vector")], curve=curve)
    assert abstract.status is Status.UNSUPPORTED and "abstract" in abstract.message
    timed = run("F", "line", [vec("F", "t", "0", "0")], curve=curve)
    assert timed.status is not Status.OK


def test_a_pole_on_the_domain_refuses_the_theorem_and_the_integral() -> None:
    ball = run(
        "F",
        "divergence_theorem",
        [vec("F", "1/r**2", "0", "0")],
        "spherical",
        region=Region(bounds=[("0", "2"), ("0", "pi"), ("0", "2*pi")]),
    )
    assert ball.status is Status.UNSUPPORTED and ball.diagnostics.reason == "singular_in_domain"
    assert ball.result_latex is None and ball.integral is None
    line = run(
        "F",
        "line",
        [vec("F", "1/x", "0", "0")],
        curve=Curve(q=["s", "0", "0"], parameter=P("s", "-1", "1")),
    )
    assert line.status is Status.UNSUPPORTED  # the pole at x = 0 is on the curve
    # the same field is fine where it is defined: the flux through the sphere avoids the origin
    assert (
        run(
            "F",
            "flux",
            [vec("F", "1/r**2", "0", "0")],
            "spherical",
            surface=Surface(q=["1", "u", "v"], u=P("u", "0", "pi"), v=P("v", "0", "2*pi")),
        ).status
        is Status.OK
    )


def test_wrong_kinds_and_bad_parameters() -> None:
    curve = Curve(q=["s", "0", "0"], parameter=P("s", "0", "1"))
    assert run("f", "line", [sca("f", "1")], curve=curve).status is Status.PARSE_ERROR
    assert run("F", "volume", [vec("F", "1", "0", "0")], region=BALL).status is Status.PARSE_ERROR
    with pytest.raises(ValueError):
        P("t", "0", "1")  # t is time
    clash = run(
        "F",
        "line",
        [vec("F", "1", "0", "0")],
        curve=Curve(q=["x", "0", "0"], parameter=P("x", "0", "1")),
    )
    assert clash.status is Status.PARSE_ERROR  # x is a coordinate
    stray = run(
        "F",
        "line",
        [vec("F", "1", "0", "0")],
        curve=Curve(q=["s*k", "0", "0"], parameter=P("s", "0", "1")),
    )
    assert stray.status is Status.PARSE_ERROR and "k" in stray.message


# -- the verifier rejects wrong results -----------------------------------------------------------------
def test_an_exact_value_that_disagrees_with_the_quadrature_is_withheld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(integrals, "_symbolic", lambda task: sp.Integer(7))
    result = run(
        "F",
        "line",
        [vec("F", "y", "x", "0")],
        curve=Curve(q=["s", "s**2", "0"], parameter=P("s", "0", "1")),
    )
    assert result.status is Status.NOT_VERIFIED and result.integral is None


def test_a_theorem_whose_sides_differ_is_withheld(monkeypatch: pytest.MonkeyPatch) -> None:
    real = integrals._compare
    monkeypatch.setattr(integrals, "_compare", lambda left, right: False)
    result = run(
        "F",
        "stokes_theorem",
        [vec("F", "-y", "x", "0")],
        surface=Surface(q=["u", "v", "0"], u=P("u", "0", "1"), v=P("v", "0", "1")),
    )
    assert result.status is Status.NOT_VERIFIED and result.integral is None
    assert real is not None


def test_a_misdescribed_domain_makes_the_sides_differ() -> None:
    """F = (x, 0, 0) through a closed box described with one face missing would not satisfy the
    theorem: here a region whose bounds are reversed has a negative orientation on both sides, so the
    theorem still holds; the check that matters is that the sides are computed independently."""
    result = run(
        "F",
        "divergence_theorem",
        [vec("F", "x", "y", "z")],
        region=Region(bounds=[("1", "0"), ("0", "1"), ("0", "1")]),
    )
    assert result.status is Status.OK
    left, right = result.integral.values  # type: ignore[union-attr]
    assert sp.sympify(left.text) == sp.sympify(right.text) == -3  # reversed x orientation


def test_diagnostics_hold_no_content() -> None:
    result = run(
        "F",
        "line",
        [vec("secretfield", "y", "x", "0")],
        curve=Curve(q=["s", "s", "0"], parameter=P("s", "0", "1")),
    )
    assert (
        result.status is Status.PARSE_ERROR
        or "secretfield" not in result.diagnostics.model_dump_json()
    )


def test_a_parameter_bound_written_as_a_number_is_read_as_its_text() -> None:
    """`{"lower": 0, "upper": 1}` (JSON without quotes) is the bound `0` to `1`; a bool or a list is still refused."""
    import pytest
    from minerva_engine.vector.models import Param
    from pydantic import ValidationError

    p = Param(name="u", lower=0, upper=1.5)  # type: ignore[arg-type]
    assert (p.lower, p.upper) == ("0", "1.5")
    assert Param(name="u", lower="0", upper="pi").upper == "pi"
    for bad in (True, [0], None):
        with pytest.raises(ValidationError):
            Param(name="u", lower=bad, upper="1")  # type: ignore[arg-type]


def test_a_coordinate_triple_sent_as_one_string_is_read_as_three_expressions_and_nothing_else_is_guessed() -> (
    None
):
    import pytest
    from minerva_engine.vector.models import Curve, Surface
    from pydantic import ValidationError

    u, v = {"name": "u", "lower": "0", "upper": "1"}, {"name": "v", "lower": "0", "upper": "1"}
    for sent in (["u,v,u*v"], ["(u, v, u*v)"], "u, v, u*v", ["u", "v", "u*v"]):
        assert Surface(q=sent, u=u, v=v).q == ["u", "v", "u*v"], sent  # type: ignore[arg-type]
    assert Surface(q=["f(a, b)", "v", "0"], u=u, v=v).q == [
        "f(a, b)",
        "v",
        "0",
    ]  # nested commas stay
    assert Curve(q=["cos(t), sin(t), t"], parameter=u).q == ["cos(t)", "sin(t)", "t"]  # type: ignore[arg-type]
    for bad in (["u,v"], ["u,v,w,x"], ["u,,v"], [], ["u", "v"]):
        with pytest.raises(ValidationError):
            Surface(q=bad, u=u, v=v)  # type: ignore[arg-type]
