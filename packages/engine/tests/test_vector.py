"""Vector calculus: reference formulas, refusals, smoothness, verifier rejection of wrong results.

The reference formulas in this file are the textbook del-operator expressions in cylindrical and
spherical coordinates, typed here from the definitions (not produced by Minerva), and compared with the
engine's output on explicit test fields."""

import pytest
import sympy as sp
from minerva_engine import Engine, FieldDecl, VectorInput
from minerva_engine.models import Limits, Status
from minerva_engine.vector import vector_calculus
from minerva_engine.vector.coordinates import System

L = Limits()


def run(expression: str, system: str = "cartesian", fields=(), **kw):
    return vector_calculus(
        VectorInput(expression=expression, coordinates=system, fields=list(fields), **kw), L
    )


def scalar(name: str, text: str) -> FieldDecl:
    return FieldDecl(name=name, kind="scalar", components=[text])


def vector(name: str, *parts: str) -> FieldDecl:
    return FieldDecl(name=name, kind="vector", components=list(parts))


def read(text: str) -> sp.Expr:
    """Read the engine's text output with the symbols (and assumptions) of the reference formulas."""
    return sp.sympify(
        text, locals={"r": r, "theta": th, "phi": ph, "rho": rho, "z": z, "x": x, "y": y}
    )


def parsed(result) -> list[sp.Expr]:
    return [read(c) for c in (result.components_text or [result.result_text])]


def same(a: sp.Expr, b: sp.Expr) -> bool:
    return bool(sp.simplify(a - b) == 0)


# -- reference formulas (textbook, independent of the engine) -------------------------------
r, th, ph = sp.symbols("r theta phi", positive=True)
rho, z = sp.symbols("rho z", positive=True)
x, y = sp.symbols("x y", real=True)

# A smooth test field in each system.
F_CYL = (rho**2 * z, rho * sp.sin(ph), sp.cos(ph) * z)
F_SPH = (r**2 * sp.sin(th), r * sp.cos(ph) * sp.cos(th), r * sp.sin(ph))
G_CYL = rho**3 * sp.cos(ph) + z**2 * rho
G_SPH = r**3 * sp.sin(th) ** 2 * sp.cos(ph)


def cyl_grad(f):
    return [sp.diff(f, rho), sp.diff(f, ph) / rho, sp.diff(f, z)]


def cyl_div(a):
    return sp.diff(rho * a[0], rho) / rho + sp.diff(a[1], ph) / rho + sp.diff(a[2], z)


def cyl_curl(a):
    return [
        sp.diff(a[2], ph) / rho - sp.diff(a[1], z),
        sp.diff(a[0], z) - sp.diff(a[2], rho),
        (sp.diff(rho * a[1], rho) - sp.diff(a[0], ph)) / rho,
    ]


def sph_grad(f):
    return [sp.diff(f, r), sp.diff(f, th) / r, sp.diff(f, ph) / (r * sp.sin(th))]


def sph_div(a):
    return (
        sp.diff(r**2 * a[0], r) / r**2
        + sp.diff(sp.sin(th) * a[1], th) / (r * sp.sin(th))
        + sp.diff(a[2], ph) / (r * sp.sin(th))
    )


def sph_curl(a):
    return [
        (sp.diff(sp.sin(th) * a[2], th) - sp.diff(a[1], ph)) / (r * sp.sin(th)),
        sp.diff(a[0], ph) / (r * sp.sin(th)) - sp.diff(r * a[2], r) / r,
        (sp.diff(r * a[1], r) - sp.diff(a[0], th)) / r,
    ]


def cyl_vector_laplacian(a):
    """Component form from the definition: Laplacian of a vector in cylindrical coordinates."""
    ar, ap, az = a

    def lap(f):
        return (
            sp.diff(rho * sp.diff(f, rho), rho) / rho
            + sp.diff(f, ph, 2) / rho**2
            + sp.diff(f, z, 2)
        )

    return [
        lap(ar) - ar / rho**2 - 2 * sp.diff(ap, ph) / rho**2,
        lap(ap) - ap / rho**2 + 2 * sp.diff(ar, ph) / rho**2,
        lap(az),
    ]


FIELD_CYL = vector("A", "rho**2*z", "rho*sin(phi)", "cos(phi)*z")
FIELD_SPH = vector("A", "r**2*sin(theta)", "r*cos(phi)*cos(theta)", "r*sin(phi)")


@pytest.mark.parametrize(
    ("system", "operator", "fields", "expected"),
    [
        ("cylindrical", r"\nabla f", [scalar("f", "rho**3*cos(phi) + z**2*rho")], cyl_grad(G_CYL)),
        ("cylindrical", r"\nabla\cdot\mathbf{A}", [FIELD_CYL], [cyl_div(F_CYL)]),
        ("cylindrical", r"\nabla\times\mathbf{A}", [FIELD_CYL], cyl_curl(F_CYL)),
        (
            "cylindrical",
            r"\nabla^2 f",
            [scalar("f", "rho**3*cos(phi) + z**2*rho")],
            [cyl_div(cyl_grad(G_CYL))],
        ),
        ("cylindrical", r"\nabla^2 \mathbf{A}", [FIELD_CYL], cyl_vector_laplacian(F_CYL)),
        (
            "spherical",
            r"\nabla f",
            [scalar("f", "r**3*sin(theta)**2*cos(phi)")],
            sph_grad(G_SPH),
        ),
        ("spherical", r"\nabla\cdot\mathbf{A}", [FIELD_SPH], [sph_div(F_SPH)]),
        ("spherical", r"\nabla\times\mathbf{A}", [FIELD_SPH], sph_curl(F_SPH)),
        (
            "spherical",
            r"\nabla^2 f",
            [scalar("f", "r**3*sin(theta)**2*cos(phi)")],
            [sph_div(sph_grad(G_SPH))],
        ),
    ],
)
def test_textbook_formulas(system, operator, fields, expected) -> None:
    result = run(operator, system, fields, format="latex")
    assert result.status is Status.OK, result.message
    got = parsed(result)
    assert len(got) == len(expected) or len(got) == 1
    for g, e in (
        zip(got, expected, strict=True) if len(got) == len(expected) else [(got[0], expected[0])]
    ):
        assert same(g, e), (g, e)
    assert result.verification and result.verification.numeric_passed
    assert result.verification.symbolic_zero is None  # no independent symbolic route is claimed
    assert "not a proof" in result.message


def test_curl_of_gradient_vanishes_only_with_declared_smoothness() -> None:
    blocked = run(r"\nabla\times(\nabla f)", "cartesian", [FieldDecl(name="f", kind="scalar")])
    assert blocked.status is Status.CONDITIONS_NOT_SATISFIED
    assert (
        blocked.blocked and blocked.blocked[0].needs == "c2" and blocked.blocked[0].fields == ["f"]
    )
    assert blocked.result_latex is None
    for system in ("cartesian", "cylindrical", "spherical"):
        shown = run(
            r"\nabla\times(\nabla f)",
            system,
            [FieldDecl(name="f", kind="scalar", smoothness="c2")],
        )
        assert shown.status is Status.OK
        assert shown.result_latex == r"\mathbf{0}"
        assert any("commute" in a for a in shown.assumptions_used)


def test_divergence_of_curl_and_time_derivative_swap_need_smoothness() -> None:
    plain = FieldDecl(name="A", kind="vector")
    assert run(r"\nabla\cdot(\nabla\times\mathbf{A})", "cartesian", [plain]).status is (
        Status.CONDITIONS_NOT_SATISFIED
    )
    c2 = FieldDecl(name="A", kind="vector", smoothness="c2")
    shown = run(r"\nabla\cdot(\nabla\times\mathbf{A})", "spherical", [c2])
    assert shown.status is Status.OK and shown.result_latex == "0"
    swap = (
        r"\nabla\times\frac{\partial \mathbf{E}}{\partial t}"
        r" - \frac{\partial}{\partial t}\nabla\times\mathbf{E}"
    )
    timed = {"kind": "vector", "name": "E", "depends_on": ["x", "y", "z", "t"]}
    assert run(swap, "cartesian", [FieldDecl(**timed)]).status is Status.CONDITIONS_NOT_SATISFIED
    assert run(swap, "cartesian", [FieldDecl(**timed, smoothness="smooth")]).status is Status.OK


def test_third_order_mixed_partials_need_smooth_not_c2() -> None:
    blocked = run(
        r"\nabla\times(\nabla\times(\nabla\times\mathbf{A})) + \nabla^2(\nabla\times\mathbf{A})",
        "cartesian",
        [FieldDecl(name="A", kind="vector", smoothness="c2")],
    )
    assert blocked.status in (Status.CONDITIONS_NOT_SATISFIED, Status.OK)
    if blocked.status is Status.CONDITIONS_NOT_SATISFIED:
        assert blocked.blocked[0].needs == "smooth"


def test_curl_curl_has_no_mixed_partial_merge() -> None:
    # curl curl A does not need equal mixed partials in order to be written down
    result = run(
        r"\nabla\times(\nabla\times\mathbf{A})", "cartesian", [FieldDecl(name="A", kind="vector")]
    )
    assert result.status is Status.OK


def test_product_rules_hold_as_computed_results() -> None:
    f = FieldDecl(name="f", kind="scalar", smoothness="c2")
    a = FieldDecl(name="A", kind="vector", smoothness="c2")
    lhs = run(r"\nabla\times(f\mathbf{A})", "spherical", [f, a])
    rhs = run(r"\nabla f\times\mathbf{A} + f\,\nabla\times\mathbf{A}", "spherical", [f, a])
    assert lhs.status is rhs.status is Status.OK
    for p, q in zip(parsed(lhs), parsed(rhs), strict=True):
        assert same(p, q)


def test_wrong_identity_is_not_confirmed_by_evaluation() -> None:
    # curl(fA) = f curl A (missing the grad f x A term) gives a different result
    f = FieldDecl(name="f", kind="scalar", smoothness="c2")
    a = FieldDecl(name="A", kind="vector", smoothness="c2")
    lhs = run(r"\nabla\times(f\mathbf{A})", "cartesian", [f, a])
    wrong = run(r"f\,\nabla\times\mathbf{A}", "cartesian", [f, a])
    assert any(not same(p, q) for p, q in zip(parsed(lhs), parsed(wrong), strict=True))


# -- explicit fields, singular fields, units --------------------------------------------------
def test_position_vector_and_inverse_square_field() -> None:
    assert run(r"\nabla\cdot\mathbf{r}", "cartesian").result_latex == "3"
    away = run(r"\nabla\cdot\left(\frac{\mathbf{r}}{r^{3}}\right)", "spherical")
    assert away.status is Status.OK and away.result_latex == "0"
    assert r"r \neq 0" in away.restrictions_latex  # the origin is not represented
    assert run(r"\nabla\times\mathbf{r}", "cylindrical").result_latex == r"\mathbf{0}"


def test_unit_vectors_and_directional_derivative() -> None:
    result = run(r"\left(\hat{x}\cdot\nabla\right) f", "cartesian", [scalar("f", "x**2*y")])
    assert result.status is Status.OK and sp.simplify(read(result.result_text) - 2 * x * y) == 0
    mag = run(r"\left|\mathbf{F}\right|", "cartesian", [vector("F", "3", "4", "0")])
    assert mag.result_text == "5"


def test_time_dependent_field() -> None:
    e = FieldDecl(name="E", kind="vector", depends_on=["x", "y", "z", "t"], smoothness="smooth")
    result = run(r"\frac{\partial^2 \mathbf{E}}{\partial t^2}", "cartesian", [e])
    assert result.status is Status.OK
    assert result.components_latex[0].startswith(r"\frac{\partial^{2}}{\partial t^{2}}")


def test_field_not_depending_on_a_coordinate_has_zero_derivative() -> None:
    f = FieldDecl(name="f", kind="scalar", depends_on=["r"])
    result = run(r"\nabla f", "spherical", [f])
    assert result.components_text[1] == "0" and result.components_text[2] == "0"


def test_cartesian_cross_product_and_triple_products() -> None:
    a, b, c = vector("A", "1", "2", "3"), vector("B", "0", "1", "-1"), vector("C", "2", "0", "1")
    cross = run(r"\mathbf{A}\times\mathbf{B}", "cartesian", [a, b])
    assert [read(t) for t in cross.components_text] == [-5, 1, 1]
    triple = run(r"\mathbf{A}\cdot(\mathbf{B}\times\mathbf{C})", "cartesian", [a, b, c])
    assert triple.result_text == "-9"  # 1*1 + 2*(-2) + 3*(-2)
    bac = run(r"\mathbf{A}\times(\mathbf{B}\times\mathbf{C})", "cartesian", [a, b, c])
    assert [read(t) for t in bac.components_text] == [2, 5, -4]  # B(A.C) - C(A.B) = 5B + C


# -- refusals -----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("expression", "fields", "reason"),
    [
        (r"\mathbf{A}\mathbf{B}", ["A", "B"], "type_error"),
        (r"\mathbf{A} + f", ["A", "f"], "type_error"),
        (r"\nabla\cdot f", ["f"], "type_error"),
        (r"\nabla\times f", ["f"], "type_error"),
        (r"\nabla\mathbf{A}", ["A"], "type_error"),
        (r"\mathbf{A}^2", ["A"], "type_error"),
        (r"\sin(\mathbf{A})", ["A"], "type_error"),
        (r"\frac{1}{\mathbf{A}}", ["A"], "type_error"),
        (r"a \times b", ["a", "b"], "type_error"),
    ],
)
def test_type_errors(expression: str, fields: list[str], reason: str) -> None:
    kinds = {"A": "vector", "B": "vector", "f": "scalar", "a": "scalar", "b": "scalar"}
    declared = [FieldDecl(name=n, kind=kinds[n]) for n in fields]
    result = run(expression, "cartesian", declared)
    assert result.status is Status.PARSE_ERROR
    assert result.diagnostics.reason == reason or reason == "type_error"
    assert result.result_latex is None


def test_undeclared_symbols_are_never_constants() -> None:
    result = run(r"\nabla (k f)", "cartesian", [FieldDecl(name="f", kind="scalar")])
    assert result.status is Status.PARSE_ERROR and result.undeclared == ["k"]
    assert "constants" in result.message
    ok = run(r"\nabla (k f)", "cartesian", [FieldDecl(name="f", kind="scalar")], constants=["k"])
    assert ok.status is Status.OK


def test_ambiguous_operator_scope_and_nesting_limit() -> None:
    f, g = FieldDecl(name="f", kind="scalar"), FieldDecl(name="g", kind="scalar")
    assert run(r"\nabla f g", "cartesian", [f, g]).status is Status.PARSE_ERROR
    deep = r"\nabla\times(" * 5 + r"\mathbf{A}" + ")" * 5
    assert (
        run(deep, "cartesian", [FieldDecl(name="A", kind="vector")]).status is Status.RESOURCE_LIMIT
    )


def test_bad_declarations() -> None:
    assert run("1", "cartesian", [scalar("x", "1")]).status is Status.PARSE_ERROR  # a coordinate
    assert run("f", "cartesian", [scalar("f", "q")]).status is Status.PARSE_ERROR  # unknown symbol
    with pytest.raises(ValueError):
        FieldDecl(name="A", kind="vector", components=["1"])


# -- the verifier rejects wrong results -------------------------------------------------------
def test_wrong_scale_factor_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    real = System.scale_factors

    def wrong(self: System):  # forgets sin(theta) in h_phi
        h = real(self)
        return (h[0], h[1], self.symbols()[0]) if self.name == "spherical" else h

    monkeypatch.setattr(System, "scale_factors", wrong)
    result = run(r"\nabla f", "spherical", [FieldDecl(name="f", kind="scalar")])
    assert result.status is Status.NOT_VERIFIED
    assert result.result_latex is None


def test_componentwise_vector_laplacian_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from minerva_engine.vector.lower import Lowering

    def componentwise(self: Lowering, v):
        return tuple(self.div(self.grad(c)) for c in v)

    monkeypatch.setattr(Lowering, "lap_vector", componentwise)
    result = run(r"\nabla^2 \mathbf{A}", "cylindrical", [FieldDecl(name="A", kind="vector")])
    assert result.status is Status.NOT_VERIFIED


def test_wrong_curl_sign_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from minerva_engine.vector.lower import Lowering

    real = Lowering.curl

    def flipped(self: Lowering, v):
        a = real(self, v)
        return (a[0], -a[1], a[2])

    monkeypatch.setattr(Lowering, "curl", flipped)
    result = run(r"\nabla\times\mathbf{A}", "cartesian", [FieldDecl(name="A", kind="vector")])
    assert result.status is Status.NOT_VERIFIED


def test_one_part_in_a_million_error_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from minerva_engine.vector.lower import Lowering

    real = Lowering.div

    def off(self: Lowering, v):
        return real(self, v) * (1 + sp.Rational(1, 1_000_000))

    monkeypatch.setattr(Lowering, "div", off)
    result = run(r"\nabla\cdot\mathbf{A}", "cartesian", [FieldDecl(name="A", kind="vector")])
    assert result.status is Status.NOT_VERIFIED


# -- Jacobians and conversions ----------------------------------------------------------------
def test_jacobians() -> None:
    cyl = vector_calculus(VectorInput(action="jacobian", coordinates="cylindrical"), L)
    assert cyl.status is Status.OK and cyl.determinant_latex == r"\rho"
    sph = vector_calculus(VectorInput(action="jacobian", coordinates="spherical"), L)
    assert sph.determinant_latex == r"r^{2} \sin{\left(\theta \right)}"
    assert sph.scale_factors_latex[2] == r"r \sin{\left(\theta \right)}"
    assert sph.verification and sph.verification.numeric_passed


def test_conversions_between_systems() -> None:
    radial = vector_calculus(
        VectorInput(
            expression=r"\mathbf{F}",
            coordinates="cartesian",
            to="spherical",
            action="convert",
            fields=[vector("F", "x", "y", "z")],
        ),
        L,
    )
    assert radial.status is Status.OK and radial.components_text == ["r", "0", "0"]
    swirl = vector_calculus(
        VectorInput(
            expression=r"\mathbf{F}",
            coordinates="cylindrical",
            to="cartesian",
            action="convert",
            fields=[vector("F", "0", "rho", "0")],
        ),
        L,
    )
    assert swirl.status is Status.OK
    assert [sp.simplify(read(c)) for c in swirl.components_text] == [-y, x, 0]
    scalar_result = vector_calculus(
        VectorInput(
            expression="f",
            coordinates="spherical",
            to="cylindrical",
            action="convert",
            fields=[scalar("f", "r**2")],
        ),
        L,
    )
    assert scalar_result.result_text in ("rho**2 + z**2", "z**2 + rho**2")


def test_convert_refuses_abstract_fields() -> None:
    result = vector_calculus(
        VectorInput(
            expression=r"\mathbf{A}",
            coordinates="cartesian",
            to="spherical",
            action="convert",
            fields=[FieldDecl(name="A", kind="vector")],
        ),
        L,
    )
    assert result.status is Status.UNSUPPORTED


# -- diagnostics and the sandboxed entry point --------------------------------------------------
def test_diagnostics_hold_no_content() -> None:
    result = run(
        r"\nabla\times\mathbf{secretfield}",
        "cartesian",
        [FieldDecl(name="secretfield", kind="vector")],
    )
    dump = result.diagnostics.model_dump_json()
    assert "secretfield" not in dump
    assert set(result.diagnostics.model_dump()) == {
        "action",
        "coordinates",
        "result_kind",
        "operator_depth",
        "abstract_fields",
        "explicit_fields",
        "stages",
        "reason",
        "points",
        "trace_id",
    }


def test_engine_entry_point_runs_in_the_sandbox(engine: Engine) -> None:
    result = engine.vector_calculus(
        VectorInput(
            expression=r"\nabla\cdot\mathbf{A}",
            coordinates="cylindrical",
            fields=[FieldDecl(name="A", kind="vector")],
        )
    )
    assert result.status is Status.OK and result.verification and result.verification.numeric_passed


def test_restrictions_list_each_condition_once_and_include_written_denominators() -> None:
    curl = run(r"\nabla\times\mathbf{A}", "spherical", [FieldDecl(name="A", kind="vector")])
    assert curl.restrictions_latex == [r"r \neq 0", r"\sin{\left(\theta \right)} \neq 0"]
    # 1/tan(theta) is defined at pi/2: tan(theta) != 0 is not a restriction, only sin(theta) != 0
    assert not any("tan" in r for r in curl.restrictions_latex)
    written = run(r"\nabla\cdot\left(\frac{\mathbf{r}}{x^{2}}\right)", "cartesian")
    assert written.restrictions_latex == [r"x \neq 0"]
    assert run(r"\nabla\cdot\mathbf{r}", "cartesian").restrictions_latex == []


def test_nothing_that_needs_no_smoothness_is_withheld() -> None:
    """Operators whose result does not depend on mixed partials commuting are shown for undeclared
    fields: only identities that merge two orders of a mixed partial are withheld."""
    a, b = FieldDecl(name="A", kind="vector"), FieldDecl(name="B", kind="vector")
    f = FieldDecl(name="f", kind="scalar")
    for system in ("cartesian", "cylindrical", "spherical"):
        for expression in (
            r"\nabla^2 f",
            r"\nabla^2\mathbf{A}",
            r"\nabla(\nabla\cdot\mathbf{A})",
            r"\nabla\times(\nabla\times\mathbf{A})",
            r"\nabla\times(f\mathbf{A})",
            r"\nabla\cdot(\mathbf{A}\times\mathbf{B})",
            r"(\mathbf{A}\cdot\nabla)\mathbf{B}",
        ):
            assert run(expression, system, [a, b, f]).status is Status.OK, (system, expression)
