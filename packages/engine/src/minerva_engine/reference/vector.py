"""Vector calculus, electromagnetism, integral theorems, electrostatics and field tensors.

Expected values are hand-typed with a locator (Griffiths, Jackson, Arfken and Weber, Marsden and Tromba), or come
from `mpmath` quadrature, plain SymPy on explicit components, or plain arithmetic. Nothing is read from the
engine's own derivation. Results that rest on field equations the user adopted are `conditional`.
"""

from __future__ import annotations

from typing import Any

import mpmath as mp
import sympy as sp

from ..service import Engine
from ..vector import FieldDecl, VectorInput
from .core import Problem, verdict


def _vector_field(name: str, parts: list[str]) -> FieldDecl:
    return FieldDecl(name=name, kind="vector", components=parts)


def _cartesian_symbols() -> dict[str, sp.Symbol]:
    return {n: sp.Symbol(n, real=True) for n in ("x", "y", "z")}


def _laplacian_across_coordinates(engine: Engine) -> tuple[str, str]:
    """The vector Laplacian computed in cylindrical coordinates (grad div - curl curl in the
    cylindrical frame) must equal the componentwise Cartesian Laplacian, which plain SymPy gives."""
    parts = ["x**2*y", "y*z**2 + x", "x*z + y**3"]
    sym = _cartesian_symbols()
    field = [sp.sympify(p, locals=sym) for p in parts]
    truth = [sum(sp.diff(c, v, 2) for v in sym.values()) for c in field]
    to_cyl = engine.vector_calculus(
        VectorInput(expression="F", coordinates="cartesian", to="cylindrical", action="convert",
                    fields=[_vector_field("F", parts)])
    )  # fmt: skip
    if to_cyl.status.value != "ok":
        return verdict("correct", False, None, f"conversion: {to_cyl.status.value}")
    lap = engine.vector_calculus(
        VectorInput(expression=r"\nabla^2 \mathbf{G}", coordinates="cylindrical",
                    fields=[_vector_field("G", to_cyl.components_text)])
    )  # fmt: skip
    if lap.status.value != "ok":
        return verdict("correct", False, None, f"laplacian: {lap.status.value}")
    back = engine.vector_calculus(
        VectorInput(expression="H", coordinates="cylindrical", to="cartesian", action="convert",
                    fields=[_vector_field("H", lap.components_text)])
    )  # fmt: skip
    if back.status.value != "ok":
        return verdict("correct", False, None, f"back-conversion: {back.status.value}")
    got = [sp.sympify(t, locals=sym) for t in back.components_text]
    agrees = all(sp.simplify(g - t) == 0 for g, t in zip(got, truth, strict=True))
    return verdict(
        "correct", True, agrees, "cylindrical result rotated back vs Cartesian componentwise"
    )


def _div_position(engine: Engine) -> tuple[str, str]:
    """div r = 3 in every coordinate system (the dimension of space)."""
    values = []
    for system in ("cartesian", "cylindrical", "spherical"):
        result = engine.vector_calculus(
            VectorInput(expression=r"\nabla\cdot\mathbf{r}", coordinates=system)
        )
        if result.status.value != "ok":
            return verdict("correct", False, None, f"{system}: {result.status.value}")
        values.append(result.result_text)
    return verdict("correct", True, values == ["3", "3", "3"], "div r = 3")


def _point_charge_field(engine: Engine) -> tuple[str, str]:
    """div(r_vec/r^3) = 0 away from the origin, with r != 0 stated (the origin carries the delta)."""
    result = engine.vector_calculus(
        VectorInput(expression=r"\nabla\cdot\left(\frac{\mathbf{r}}{r^{3}}\right)",
                    coordinates="spherical")
    )  # fmt: skip
    if result.status.value != "ok":
        return verdict("correct", False, None, f"status {result.status.value}")
    agrees = result.result_text == "0" and r"r \neq 0" in result.restrictions_latex
    return verdict("correct", True, agrees, "zero away from the origin; origin excluded")


def _div_curl_without_smoothness(engine: Engine) -> tuple[str, str]:
    """div curl A = 0 needs equal mixed partials: with no smoothness declared nothing may be shown."""
    result = engine.vector_calculus(
        VectorInput(expression=r"\nabla\cdot(\nabla\times\mathbf{A})", coordinates="spherical",
                    fields=[FieldDecl(name="A", kind="vector")])
    )  # fmt: skip
    return verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _cross_of_scalars(engine: Engine) -> tuple[str, str]:
    """a x b is not defined for scalars: it must not be read as a product."""
    result = engine.vector_calculus(
        VectorInput(expression=r"a \times b", fields=[FieldDecl(name="a", kind="scalar"),
                                                         FieldDecl(name="b", kind="scalar")])
    )  # fmt: skip
    return verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _em_request(
    template_id: str, premises: list[str] | None = None, smoothness: str | None = "c2"
) -> Any:
    """The request the `electromagnetism` tool makes, with the handbook's identities."""
    from minerva_knowledge import KnowledgeBase
    from minerva_knowledge.identities import vector_identity_rules

    from ..vector import Premise
    from ..vector.em import TEMPLATES

    t = TEMPLATES[template_id]
    return VectorInput(
        expression="",
        format="text",
        fields=[
            FieldDecl(
                name=n,
                kind=k,
                depends_on=["x", "y", "z", "t"],
                smoothness=smoothness if n in t.smooth else None,
            )
            for n, k in t.fields
        ],
        constants=list(t.constants),
        premises=[Premise(equation=p) for p in (t.needs if premises is None else premises)],
        rules=vector_identity_rules(KnowledgeBase.load()),
        action="em_derive",
        template=template_id,
    )


def _plane_wave() -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr, dict[str, sp.Symbol]]:
    """E and B of a plane wave in vacuum (textbook: Griffiths chapter 9), written by hand."""
    x, y, z, t = (sp.Symbol(n, real=True) for n in "xyzt")
    eps, c = sp.Rational(7, 5), sp.Rational(5, 2)
    k = (sp.Integer(2), sp.Integer(1), sp.Integer(2))  # |k| = 3
    omega = c * 3
    e0 = (
        sp.Integer(1),
        sp.Integer(-2),
        sp.Rational(3, 2),
    )  # k . e0 = 2 - 2 + 3 = 3 ... fixed below
    e0 = (sp.Integer(1), sp.Integer(0), sp.Integer(-1))  # k . e0 = 0
    phase = sum(a * v for a, v in zip(k, (x, y, z), strict=True)) - omega * t
    kxe = (
        k[1] * e0[2] - k[2] * e0[1],
        k[2] * e0[0] - k[0] * e0[2],
        k[0] * e0[1] - k[1] * e0[0],
    )
    E = tuple(a * sp.cos(phase) for a in e0)
    B = tuple(a / omega * sp.cos(phase) for a in kxe)
    mu = 1 / (eps * c**2)
    return E, B, eps, mu, omega, {"x": x, "y": y, "z": z, "t": t}


def _em_wave(engine: Engine) -> tuple[str, str]:
    """The wave equation for E, derived from Maxwell's equations as premises, in the textbook form;
    the textbook equation is also checked on a plane wave with plain SymPy (an oracle of its own)."""
    result = engine.vector_calculus(_em_request("wave_equation_e"))
    if result.status.value != "ok":
        return verdict("correct", False, None, f"status {result.status.value}")
    E, _B, eps, mu, _omega, v = _plane_wave()
    x, y, z, t = v["x"], v["y"], v["z"], v["t"]
    lap = [sum(sp.diff(e, a, 2) for a in (x, y, z)) for e in E]
    residual = [sp.simplify(lap[i] - mu * eps * sp.diff(E[i], t, 2)) for i in range(3)]
    expected = r"\nabla^{2} \mathbf{E} = \epsilon_{0} \mu_{0} \frac{\partial^{2} \mathbf{E}}{\partial t^{2}}"
    agrees = result.equations_latex == [expected] and all(r == 0 for r in residual)
    return verdict("correct", True, agrees, "textbook form; plane wave satisfies it")


def _em_poynting(engine: Engine) -> tuple[str, str]:
    """div S + du/dt = 0 for a plane wave (J = 0), by plain SymPy, and the engine derives the theorem."""
    result = engine.vector_calculus(_em_request("poynting"))
    if result.status.value != "ok":
        return verdict("correct", False, None, f"status {result.status.value}")
    E, B, eps, mu, _omega, v = _plane_wave()
    x, y, z, t = v["x"], v["y"], v["z"], v["t"]
    cross = (
        E[1] * B[2] - E[2] * B[1],
        E[2] * B[0] - E[0] * B[2],
        E[0] * B[1] - E[1] * B[0],
    )
    div_s = sum(sp.diff(c / mu, a) for c, a in zip(cross, (x, y, z), strict=True))
    u = eps * sum(e**2 for e in E) / 2 + sum(b**2 for b in B) / (2 * mu)
    expected = (
        r"\mathbf{J} \cdot \mathbf{E} + \frac{\partial \left(\frac{\epsilon_{0} \left(\mathbf{E} \cdot \mathbf{E}\right)}{2}"
        r" + \frac{\mathbf{B} \cdot \mathbf{B}}{2 \mu_{0}}\right)}{\partial t}"
        r" + \frac{\nabla \cdot \left(\mathbf{E} \times \mathbf{B}\right)}{\mu_{0}} = 0"
    )
    agrees = (
        sp.simplify(div_s + sp.diff(u, t)) == 0
        and result.evidence == "conditional"
        and result.equations_latex == [expected]
    )
    return verdict(
        "correct", True, agrees, "plane wave conserves energy; derivation is conditional"
    )


def _em_gauge(engine: Engine) -> tuple[str, str]:
    """E and B are invariant under A -> A + grad chi, phi -> phi - dchi/dt (plain SymPy on explicit
    potentials), and Minerva derives it as an identity with no premise."""
    result = engine.vector_calculus(_em_request("gauge_invariance", []))
    if result.status.value != "ok":
        return verdict("correct", False, None, f"status {result.status.value}")
    x, y, z, t = (sp.Symbol(n, real=True) for n in "xyzt")
    phi = x * y + t * z**2
    a = (y * t, x**2 * z, sp.sin(x * t))
    chi = x**2 * y * t + z * sp.exp(t)

    def grad(f: sp.Expr) -> list[sp.Expr]:
        return [sp.diff(f, v) for v in (x, y, z)]

    e = [-g - sp.diff(ai, t) for g, ai in zip(grad(phi), a, strict=True)]
    new_phi = phi - sp.diff(chi, t)
    new_a = [ai + g for ai, g in zip(a, grad(chi), strict=True)]
    e2 = [-g - sp.diff(ai, t) for g, ai in zip(grad(new_phi), new_a, strict=True)]
    expected = [
        r"\nabla \times \left(\mathbf{A} + \nabla \chi\right) = \nabla \times \mathbf{A}",
        r"- \nabla \left(\phi - \frac{\partial \chi}{\partial t}\right)"
        r" - \frac{\partial \left(\mathbf{A} + \nabla \chi\right)}{\partial t}"
        r" = - \nabla \phi - \frac{\partial \mathbf{A}}{\partial t}",
    ]
    agrees = all(sp.simplify(p - q) == 0 for p, q in zip(e, e2, strict=True)) and (
        result.evidence == "identity" and not result.given and result.equations_latex == expected
    )
    return verdict("correct", True, agrees, "E invariant on explicit potentials; an identity")


def _em_missing_premise(engine: Engine) -> tuple[str, str]:
    """Without Gauss's law as a premise, continuity cannot follow: refuse and ask, never assume."""
    from ..vector.em import TEMPLATES

    needs = TEMPLATES["continuity"].needs
    result = engine.vector_calculus(_em_request("continuity", [needs[0]]))
    return verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _em_flipped_sign(engine: Engine) -> tuple[str, str]:
    """A premise with the wrong sign is not the one the derivation needs (it would give the wrong
    wave equation): refused as a missing premise."""
    result = engine.vector_calculus(
        _em_request(
            "wave_equation_e",
            ["div(E) = 0", "curl(E) = dt(B)", "curl(B) = mu0*epsilon0*dt(E)"],
        )
    )
    return verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _integral_request(**kw: Any) -> Any:
    from ..vector import Curve, Param, Region, Surface

    del Curve, Param, Region, Surface
    return VectorInput(format="text", action="integrate", **kw)


def _int_flux(engine: Engine) -> tuple[str, str]:
    """The flux of r-hat / r^2 through a sphere is 4 pi (Gauss, textbook), whatever the radius."""
    from ..vector import Param, Surface

    result = engine.vector_calculus(
        _integral_request(
            expression="F",
            integral="flux",
            coordinates="spherical",
            fields=[FieldDecl(name="F", kind="vector", components=["1/r**2", "0", "0"])],
            surface=Surface(
                q=["3", "u", "v"],
                u=Param(name="u", lower="0", upper="pi"),
                v=Param(name="v", lower="0", upper="2*pi"),
            ),
        )
    )
    if result.status.value != "ok" or result.integral is None:
        return verdict("correct", False, None, f"status {result.status.value}")
    agrees = result.integral.values[0].text == "4*pi"
    return verdict("correct", True, agrees, "4 pi, as Gauss's law gives")


def _int_line_against_mpmath(engine: Engine) -> tuple[str, str]:
    """A line integral along a helix, against nested mpmath quadrature written by hand in Cartesian form
    (`F . r'(s)` with r' written out: none of Minerva's integral code is used)."""
    from ..vector import Curve, Param

    result = engine.vector_calculus(
        _integral_request(
            expression="F",
            integral="line",
            fields=[FieldDecl(name="F", kind="vector", components=["-y**2", "x*z", "x*y + z"])],
            curve=Curve(
                q=["cos(s)", "sin(s)", "s/3"], parameter=Param(name="s", lower="0", upper="2*pi")
            ),
        )
    )
    if result.status.value != "ok" or result.integral is None:
        return verdict("correct", False, None, f"status {result.status.value}")

    def integrand(s: Any) -> Any:
        x, y, z = mp.cos(s), mp.sin(s), s / 3
        dx, dy, dz = -mp.sin(s), mp.cos(s), mp.mpf(1) / 3
        return (-(y**2)) * dx + (x * z) * dy + (x * y + z) * dz

    truth = mp.quad(integrand, [0, 2 * mp.pi])
    value = result.integral.values[0]
    got = mp.mpf(value.numeric) if value.numeric else mp.mpf(sp.N(sp.sympify(value.text), 20))
    return verdict("correct", True, abs(got - truth) < 1e-9, "helix, nested mpmath quadrature")


def _int_pole_refused(engine: Engine) -> tuple[str, str]:
    """The divergence theorem needs F defined on the whole ball; 1/r^2 has a pole at its centre (the
    true flux is 4 pi while div F = 0 elsewhere): refuse rather than report two unequal sides."""
    from ..vector import Region

    result = engine.vector_calculus(
        _integral_request(
            expression="F",
            integral="divergence_theorem",
            coordinates="spherical",
            fields=[FieldDecl(name="F", kind="vector", components=["1/r**2", "0", "0"])],
            region=Region(bounds=[("0", "2"), ("0", "pi"), ("0", "2*pi")]),
        )
    )
    return verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _estat_request(problem: Any) -> Any:
    from ..vector import VectorInput

    return VectorInput(expression="", format="text", action="em_compute", em=problem)


def _estat_centre(result: Any) -> Any:
    if result.status.value != "ok" or result.em is None:
        return None
    item = next((i for i in result.em.items if i.label.startswith("phi(0.5, 0.5)")), None)
    return mp.mpf(item.numeric) if item is not None and item.numeric else None


def _estat_laplace(engine: Engine) -> tuple[str, str]:
    """phi(x, y) = sin(pi x) sinh(pi y) / sinh(pi) solves Laplace's equation with phi = sin(pi x) on the top
    edge and 0 on the others: the oracle is that closed form at the centre, 1 / (2 cosh(pi / 2))."""
    from ..vector import BoundaryProblem

    result = engine.vector_calculus(
        _estat_request(BoundaryProblem(kind="boundary_value", a="1", b="1", top="sin(pi*x)"))
    )
    got = _estat_centre(result)
    truth = 1 / (2 * mp.cosh(mp.pi / 2))
    return verdict(
        "correct", got is not None, got is not None and abs(got - truth) < 1e-6, "centre value"
    )


def _estat_poisson(engine: Engine) -> tuple[str, str]:
    """laplacian(phi) = -1 on the unit square with zero edges: the double sine series summed by mpmath
    (odd p, q only), written by hand, not the engine's coefficients."""
    from ..vector import BoundaryProblem

    result = engine.vector_calculus(
        _estat_request(BoundaryProblem(kind="boundary_value", a="1", b="1", source="1"))
    )
    got = _estat_centre(result)
    truth = sum(
        16 * (-1) ** ((p - 1) // 2 + (q - 1) // 2) / (mp.pi**4 * p * q * (p * p + q * q))
        for p in range(1, 400, 2)
        for q in range(1, 400, 2)
    )
    return verdict(
        "correct", got is not None, got is not None and abs(got - truth) < 3e-5, "centre value"
    )


def _estat_point_charge_refused(engine: Engine) -> tuple[str, str]:
    from ..vector import FieldProblem, PointCharge

    result = engine.vector_calculus(
        _estat_request(
            FieldProblem(
                kind="field",
                charges=[PointCharge(q="1", position=("0", "0", "0"))],
                point=("0", "0", "0"),
            )
        )
    )
    return verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _field_tensor_request(quantity: str, components: list[str]) -> Any:
    from ..geometry import GeometryInput
    from ..geometry.models import TensorField

    metric = [
        [("-1" if i == j == 0 else "1") if i == j else "0" for j in range(4)] for i in range(4)
    ]
    return GeometryInput(
        coordinates=["t", "x", "y", "z"],
        metric=metric,
        format="text",
        quantities=[quantity],
        tensor=TensorField(variance=["down", "down"], components=components),
    )


def _field_invariant_numbers(engine: Engine) -> tuple[str, str]:
    """F_{mu nu} F^{mu nu} = 2 (B^2 - E^2) for E = (4, 5, 6), B = (1, 2, 3) in signature (-+++), c = 1:
    2 (14 - 77) = -126, the right-hand side typed by hand (Jackson, Classical Electrodynamics, section 11.9)."""
    ex, ey, ez, bx, by, bz = 4, 5, 6, 1, 2, 3
    f = [
        [0, -ex, -ey, -ez],
        [ex, 0, bz, -by],
        [ey, -bz, 0, bx],
        [ez, by, -bx, 0],
    ]
    result = engine.geometry(
        _field_tensor_request("field_invariant", [str(v) for row in f for v in row]),
    )
    q = result.quantities[0]
    ok = q.status.value == "ok" and q.scalar_text is not None
    return verdict("correct", ok, ok and sp.sympify(q.scalar_text) == -126, "invariant")


def _field_not_antisymmetric_refused(engine: Engine) -> tuple[str, str]:
    result = engine.geometry(
        _field_tensor_request("field_invariant", ["0", "1", "0", "0", "1", *["0"] * 11]),
    )
    return verdict("refused", result.quantities[0].status.value == "ok", None, "symmetric input")


PROBLEMS: list[Problem] = [
    Problem(
        "vector-laplacian-across-coordinates",
        "vector",
        "Arfken and Weber, Mathematical Methods for Physicists (vector Laplacian; Cartesian components)",
        "correct",
        _laplacian_across_coordinates,
        oracle="closed_form",
        evidence="numeric_agreement",
    ),
    Problem(
        "divergence-of-position-vector",
        "vector",
        "Griffiths, Introduction to Electrodynamics, chapter 1 (div r = 3)",
        "correct",
        _div_position,
        oracle="closed_form",
        evidence="numeric_agreement",
    ),
    Problem(
        "point-charge-field-away-from-origin",
        "vector",
        "Griffiths, Introduction to Electrodynamics, section 1.5 (divergence of r-hat / r^2)",
        "correct",
        _point_charge_field,
        oracle="closed_form",
        evidence="numeric_agreement",
    ),
    Problem(
        "div-curl-without-smoothness-refused",
        "vector",
        "Marsden and Tromba, Vector Calculus (div curl = 0 for C^2 fields)",
        "refused",
        _div_curl_without_smoothness,
        oracle="closed_form",
        evidence="refusal",
    ),
    Problem(
        "cross-product-of-scalars-refused",
        "vector",
        "definition of the cross product (two vectors)",
        "refused",
        _cross_of_scalars,
        oracle="closed_form",
        evidence="refusal",
    ),
    Problem(
        "em-wave-equation-from-maxwell",
        "electromagnetism",
        "Griffiths, Introduction to Electrodynamics, chapter 9 (wave equation in vacuum; plane waves)",
        "correct",
        _em_wave,
        oracle="closed_form",
        evidence="conditional",
    ),
    Problem(
        "em-poynting-theorem-plane-wave",
        "electromagnetism",
        "Griffiths, Introduction to Electrodynamics, chapter 8 (Poynting's theorem)",
        "correct",
        _em_poynting,
        oracle="closed_form",
        evidence="conditional",
    ),
    Problem(
        "em-gauge-invariance-of-the-fields",
        "electromagnetism",
        "Griffiths, Introduction to Electrodynamics, chapter 10 (gauge transformations)",
        "correct",
        _em_gauge,
        oracle="closed_form",
        evidence="numeric_agreement",
    ),
    Problem(
        "em-continuity-without-gauss-refused",
        "electromagnetism",
        "Griffiths, Introduction to Electrodynamics, chapter 7 (continuity needs Gauss's law and Ampere-Maxwell)",
        "refused",
        _em_missing_premise,
        oracle="closed_form",
        evidence="refusal",
    ),
    Problem(
        "em-wave-equation-flipped-faraday-refused",
        "electromagnetism",
        "Griffiths, Introduction to Electrodynamics, chapter 7 (the sign of Faraday's law)",
        "refused",
        _em_flipped_sign,
        oracle="closed_form",
        evidence="refusal",
    ),
    Problem(
        "integral-flux-inverse-square",
        "integral",
        "Griffiths, Introduction to Electrodynamics, section 1.3 (flux of r-hat over r squared)",
        "correct",
        _int_flux,
        oracle="closed_form",
        evidence="numeric_agreement",
    ),
    Problem(
        "integral-line-against-mpmath",
        "integral",
        "direct quadrature of F . dr along the helix (mpmath), written by hand",
        "correct",
        _int_line_against_mpmath,
        oracle="mpmath_quad",
        evidence="numeric_agreement",
    ),
    Problem(
        "integral-divergence-theorem-pole-refused",
        "integral",
        "Griffiths, Introduction to Electrodynamics, section 1.5 (the delta function at the origin)",
        "refused",
        _int_pole_refused,
        oracle="closed_form",
        evidence="refusal",
    ),
    Problem(
        "estat-laplace-closed-form",
        "electrostatics",
        "separation of variables on a rectangle, Griffiths, Introduction to Electrodynamics, section 3.3",
        "correct",
        _estat_laplace,
        oracle="closed_form",
        evidence="numeric_agreement",
    ),
    Problem(
        "estat-poisson-double-series",
        "electrostatics",
        "double sine series, summed by mpmath written by hand (Griffiths, section 3.3)",
        "correct",
        _estat_poisson,
        oracle="closed_form",
        evidence="numeric_agreement",
    ),
    Problem(
        "estat-field-at-a-point-charge-refused",
        "electrostatics",
        "Coulomb's law is singular at the charge (Griffiths, section 2.1)",
        "refused",
        _estat_point_charge_refused,
        oracle="closed_form",
        evidence="refusal",
    ),
    Problem(
        "field-tensor-invariant",
        "electromagnetic tensor",
        "Jackson, Classical Electrodynamics, section 11.9 (F_{mu nu}F^{mu nu} = 2(B^2 - E^2/c^2))",
        "correct",
        _field_invariant_numbers,
        oracle="arithmetic",
        evidence="numeric_agreement",
    ),
    Problem(
        "field-tensor-symmetric-refused",
        "electromagnetic tensor",
        "a field strength is antisymmetric (Jackson, section 11.9)",
        "refused",
        _field_not_antisymmetric_refused,
        oracle="closed_form",
        evidence="refusal",
    ),
]
