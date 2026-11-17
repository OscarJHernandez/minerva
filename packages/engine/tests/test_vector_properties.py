"""Property tests: identities that hold for every smooth field, on random explicit fields.

The fields are random polynomials in the coordinates; every identity is computed by the engine as
two separate calculations (never by applying the identity), and the results are compared."""

import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st
from minerva_engine import FieldDecl, VectorInput
from minerva_engine.models import Status
from minerva_engine.testing import budget_limits
from minerva_engine.vector import vector_calculus

L = budget_limits()
SETTINGS = settings(max_examples=6, deadline=None, derandomize=True)
SYSTEMS = {
    "cartesian": ("x", "y", "z"),
    "cylindrical": ("rho", "phi", "z"),
    "spherical": ("r", "theta", "phi"),
}


def polynomial(names: tuple[str, ...], draw) -> str:
    terms = []
    for name in names:
        c1, c2 = draw(st.integers(-3, 3)), draw(st.integers(-2, 2))
        terms.append(f"({c1})*{name} + ({c2})*{name}**2")
    cross = draw(st.integers(-2, 2))
    terms.append(f"({cross})*{names[0]}*{names[1]}*{names[2]}")
    return " + ".join(terms)


@st.composite
def case(draw):
    system = draw(st.sampled_from(sorted(SYSTEMS)))
    names = SYSTEMS[system]
    scalar = FieldDecl(name="f", kind="scalar", components=[polynomial(names, draw)])
    scalar2 = FieldDecl(name="g", kind="scalar", components=[polynomial(names, draw)])
    a = FieldDecl(name="A", kind="vector", components=[polynomial(names, draw) for _ in range(3)])
    b = FieldDecl(name="B", kind="vector", components=[polynomial(names, draw) for _ in range(3)])
    return system, [scalar, scalar2, a, b]


def evaluate(expression: str, system: str, fields):
    result = vector_calculus(
        VectorInput(expression=expression, coordinates=system, fields=fields), L
    )
    assert result.status is Status.OK, (expression, system, result.message)
    return result


def values(result) -> list[sp.Expr]:
    texts = result.components_text or [result.result_text]
    return [sp.sympify(t) for t in texts]


def equal(a, b) -> bool:
    return all(sp.simplify(p - q) == 0 for p, q in zip(a, b, strict=True))


@SETTINGS
@given(case())
def test_curl_of_gradient_and_divergence_of_curl_vanish(c) -> None:
    system, fields = c
    assert all(v == 0 for v in values(evaluate(r"\nabla\times(\nabla f)", system, fields)))
    assert values(evaluate(r"\nabla\cdot(\nabla\times\mathbf{A})", system, fields)) == [0]


@SETTINGS
@given(case())
def test_product_rules(c) -> None:
    system, fields = c
    grad = evaluate(r"\nabla(f g)", system, fields)
    expanded = evaluate(r"f\,\nabla g + g\,\nabla f", system, fields)
    assert equal(values(grad), values(expanded))
    curl = evaluate(r"\nabla\times(f\mathbf{A})", system, fields)
    rule = evaluate(r"\nabla f\times\mathbf{A} + f\,\nabla\times\mathbf{A}", system, fields)
    assert equal(values(curl), values(rule))
    div = evaluate(r"\nabla\cdot(f\mathbf{A})", system, fields)
    rule = evaluate(r"\nabla f\cdot\mathbf{A} + f\,\nabla\cdot\mathbf{A}", system, fields)
    assert equal(values(div), values(rule))


@SETTINGS
@given(case())
def test_cross_product_divergence_and_curl_curl(c) -> None:
    system, fields = c
    div = evaluate(r"\nabla\cdot(\mathbf{A}\times\mathbf{B})", system, fields)
    rule = evaluate(
        r"\mathbf{B}\cdot(\nabla\times\mathbf{A}) - \mathbf{A}\cdot(\nabla\times\mathbf{B})",
        system,
        fields,
    )
    assert equal(values(div), values(rule))
    curlcurl = evaluate(r"\nabla\times(\nabla\times\mathbf{A})", system, fields)
    rule = evaluate(r"\nabla(\nabla\cdot\mathbf{A}) - \nabla^2\mathbf{A}", system, fields)
    assert equal(values(curlcurl), values(rule))


@SETTINGS
@given(case())
def test_gradient_of_dot_product(c) -> None:
    system, fields = c
    lhs = evaluate(r"\nabla(\mathbf{A}\cdot\mathbf{B})", system, fields)
    rhs = evaluate(
        r"(\mathbf{A}\cdot\nabla)\mathbf{B} + (\mathbf{B}\cdot\nabla)\mathbf{A}"
        r" + \mathbf{A}\times(\nabla\times\mathbf{B}) + \mathbf{B}\times(\nabla\times\mathbf{A})",
        system,
        fields,
    )
    assert equal(values(lhs), values(rhs))


@SETTINGS
@given(case())
def test_conversion_preserves_the_magnitude(c) -> None:
    """|A|^2 read in the target system equals |A|^2 read in the source system at the same point.

    The source side is evaluated at the target point through the change of coordinates (typed here
    from the definitions of the two systems, not taken from the engine)."""
    system, fields = c
    vector = fields[2]
    target = "spherical" if system != "spherical" else "cylindrical"
    converted = vector_calculus(
        VectorInput(
            expression="A", coordinates=system, to=target, action="convert", fields=[vector]
        ),
        L,
    )
    assert converted.status is Status.OK
    names = {"cartesian": "x y z", "cylindrical": "rho phi z", "spherical": "r theta phi"}
    q = dict(zip(names[target].split(), sp.symbols(names[target], real=True), strict=True))
    point = {sym: sp.Rational(7 + 3 * i, 5) for i, sym in enumerate(q.values())}
    if target == "spherical":
        cart = {
            "x": q["r"] * sp.sin(q["theta"]) * sp.cos(q["phi"]),
            "y": q["r"] * sp.sin(q["theta"]) * sp.sin(q["phi"]),
            "z": q["r"] * sp.cos(q["theta"]),
        }
    else:
        cart = {"x": q["rho"] * sp.cos(q["phi"]), "y": q["rho"] * sp.sin(q["phi"]), "z": q["z"]}
    if system == "cartesian":
        at = cart
    elif system == "cylindrical":
        at = {
            "rho": sp.sqrt(cart["x"] ** 2 + cart["y"] ** 2),
            "phi": sp.atan2(cart["y"], cart["x"]),
            "z": cart["z"],
        }
    else:
        rr = sp.sqrt(cart["x"] ** 2 + cart["y"] ** 2 + cart["z"] ** 2)
        at = {"r": rr, "theta": sp.acos(cart["z"] / rr), "phi": sp.atan2(cart["y"], cart["x"])}
    local = {
        k: v
        for k, v in zip(names[system].split(), sp.symbols(names[system], real=True), strict=True)
    }
    source = [sp.sympify(t, locals=local) for t in vector.components or []]
    here = sum(comp**2 for comp in source).subs({local[k]: v for k, v in at.items()})
    there = sum(
        sp.sympify(t, locals={k: v for k, v in q.items()}) ** 2 for t in converted.components_text
    )
    assert abs(sp.N((here - there).subs(point), 30)) < 1e-12 * max(
        1, abs(sp.N(here.subs(point), 30))
    )
