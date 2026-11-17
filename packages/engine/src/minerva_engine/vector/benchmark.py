"""Reproducible benchmark of the vector calculus engine.

A fixed corpus of problems with independently known answers: textbook del-operator formulas in
cylindrical and spherical coordinates typed from their definitions, closed forms such as
``div(r_vec) = 3``, and cases that must be refused (type errors, undeclared symbols, results that
need smoothness nobody declared, singular fields, abstract fields that cannot be converted).

It measures correctness against the references, coverage by group (valid, consistency, conditional,
singular, invalid, unsupported), the verification evidence (independent sample points) and latency.
Nothing is inflated: a refusal counts as correct only for the status the corpus expects.

``python -m minerva_engine.vector.benchmark`` writes ``docs/benchmarks/vector.json`` and
``docs/vector-benchmark.md``. The JSON has a ``deterministic`` section (statuses, correctness, point
counts: the same on every machine) and a ``measurements`` section (timings). ``--check`` recomputes the
deterministic section and compares it with the stored one.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sympy as sp

from ..latexparse import parse_latex
from ..models import Limits, Status
from .compute import vector_calculus
from .models import FieldDecl, Premise, VectorInput, VectorResult

LIMITS = Limits(timeout_seconds=60)
# Listing identities and replacing a premise by its other side compute nothing to check numerically.
NO_NUMERIC_CHECK = ("identities", "substitute", "em_derive", "integrate", "em_compute")
GROUPS = (
    "valid",
    "consistency",
    "conditional",
    "singular",
    "invalid",
    "unsupported",
    "identities",
    "derivations",
    "electromagnetism",
    "integrals",
    "electrostatics",
    "distributional",
)

# Symbols of the references (the engine's text output is read back with these).
r, th, ph = sp.symbols("r theta phi", positive=True)
rho, z = sp.symbols("rho z", positive=True)
x, y = sp.symbols("x y", real=True)
LOCALS: dict[str, Any] = {
    "r": r,
    "theta": th,
    "phi": ph,
    "rho": rho,
    "z": z,
    "x": x,
    "y": y,
}


def scalar(name: str, text: str) -> FieldDecl:
    return FieldDecl(name=name, kind="scalar", components=[text])


def vector(name: str, *parts: str) -> FieldDecl:
    return FieldDecl(name=name, kind="vector", components=list(parts))


def abstract(name: str, kind: str = "scalar", smoothness: str | None = None) -> FieldDecl:
    return FieldDecl(name=name, kind=kind, smoothness=smoothness)


# -- textbook references (definitions, not Minerva output) -------------------------------------
def cart_grad(f: sp.Expr) -> list[sp.Expr]:
    return [sp.diff(f, v) for v in (x, y, z)]


def cart_div(a: list[sp.Expr]) -> sp.Expr:
    return sum((sp.diff(c, v) for c, v in zip(a, (x, y, z), strict=True)), sp.Integer(0))


def cart_curl(a: list[sp.Expr]) -> list[sp.Expr]:
    return [
        sp.diff(a[2], y) - sp.diff(a[1], z),
        sp.diff(a[0], z) - sp.diff(a[2], x),
        sp.diff(a[1], x) - sp.diff(a[0], y),
    ]


def cyl_grad(f: sp.Expr) -> list[sp.Expr]:
    return [sp.diff(f, rho), sp.diff(f, ph) / rho, sp.diff(f, z)]


def cyl_div(a: list[sp.Expr]) -> sp.Expr:
    return sp.diff(rho * a[0], rho) / rho + sp.diff(a[1], ph) / rho + sp.diff(a[2], z)


def cyl_curl(a: list[sp.Expr]) -> list[sp.Expr]:
    return [
        sp.diff(a[2], ph) / rho - sp.diff(a[1], z),
        sp.diff(a[0], z) - sp.diff(a[2], rho),
        (sp.diff(rho * a[1], rho) - sp.diff(a[0], ph)) / rho,
    ]


def cyl_vector_laplacian(a: list[sp.Expr]) -> list[sp.Expr]:
    def lap(f: sp.Expr) -> sp.Expr:
        return (
            sp.diff(rho * sp.diff(f, rho), rho) / rho
            + sp.diff(f, ph, 2) / rho**2
            + sp.diff(f, z, 2)
        )

    return [
        lap(a[0]) - a[0] / rho**2 - 2 * sp.diff(a[1], ph) / rho**2,
        lap(a[1]) - a[1] / rho**2 + 2 * sp.diff(a[0], ph) / rho**2,
        lap(a[2]),
    ]


def sph_grad(f: sp.Expr) -> list[sp.Expr]:
    return [sp.diff(f, r), sp.diff(f, th) / r, sp.diff(f, ph) / (r * sp.sin(th))]


def sph_div(a: list[sp.Expr]) -> sp.Expr:
    return (
        sp.diff(r**2 * a[0], r) / r**2
        + sp.diff(sp.sin(th) * a[1], th) / (r * sp.sin(th))
        + sp.diff(a[2], ph) / (r * sp.sin(th))
    )


def sph_curl(a: list[sp.Expr]) -> list[sp.Expr]:
    return [
        (sp.diff(sp.sin(th) * a[2], th) - sp.diff(a[1], ph)) / (r * sp.sin(th)),
        sp.diff(a[0], ph) / (r * sp.sin(th)) - sp.diff(r * a[2], r) / r,
        (sp.diff(r * a[1], r) - sp.diff(a[0], th)) / r,
    ]


CART_F = x**2 * y + z
CART_A = [x * y, y * z**2, x + z]
CYL_F = rho**3 * sp.cos(ph) + z**2 * rho
CYL_A = [rho**2 * z, rho * sp.sin(ph), sp.cos(ph) * z]
SPH_F = r**3 * sp.sin(th) ** 2 * sp.cos(ph)
SPH_A = [r**2 * sp.sin(th), r * sp.cos(ph) * sp.cos(th), r * sp.sin(ph)]
TXT: dict[str, Any] = {
    "cart_f": "x**2*y + z",
    "cart_a": ("x*y", "y*z**2", "x + z"),
    "cyl_f": "rho**3*cos(phi) + z**2*rho",
    "cyl_a": ("rho**2*z", "rho*sin(phi)", "cos(phi)*z"),
    "sph_f": "r**3*sin(theta)**2*cos(phi)",
    "sph_a": ("r**2*sin(theta)", "r*cos(phi)*cos(theta)", "r*sin(phi)"),
}
Oracle = Callable[[], list[sp.Expr]]


def _apply(fn: Callable[[Any], Any], arg: Any, single: bool = False) -> Oracle:
    """An oracle that applies a textbook operator to a reference field."""

    def oracle() -> list[sp.Expr]:
        out = fn(arg)
        return [out] if single else list(out)

    return oracle


def _laplacian(grad: Callable[[Any], Any], div: Callable[[Any], Any], f: Any) -> Oracle:
    def oracle() -> list[sp.Expr]:
        return [div(grad(f))]

    return oracle


@dataclass
class Case:
    id: str
    title: str
    group: str
    request: VectorInput
    status: Status = Status.OK
    oracle: Oracle | None = None
    needs: str | None = None  # expected `blocked[0].needs`
    restriction: str | None = None  # a restriction the result must state
    check: Callable[[VectorResult], bool] | None = None  # extra requirement (steps, evidence)


def make(
    expression: str,
    system: str = "cartesian",
    fields: tuple[FieldDecl, ...] = (),
    **kw: Any,
) -> VectorInput:
    return VectorInput(expression=expression, coordinates=system, fields=list(fields), **kw)


def _cases() -> list[Case]:
    c: list[Case] = []

    def add(case: Case) -> None:
        c.append(case)

    cf, ca = scalar("f", TXT["cart_f"]), vector("A", *TXT["cart_a"])
    yf, ya = scalar("f", TXT["cyl_f"]), vector("A", *TXT["cyl_a"])
    sf, sa = scalar("f", TXT["sph_f"]), vector("A", *TXT["sph_a"])
    for system, f, a, F, A, grad, div, curl in (
        ("cartesian", cf, ca, CART_F, CART_A, cart_grad, cart_div, cart_curl),
        ("cylindrical", yf, ya, CYL_F, CYL_A, cyl_grad, cyl_div, cyl_curl),
        ("spherical", sf, sa, SPH_F, SPH_A, sph_grad, sph_div, sph_curl),
    ):
        add(
            Case(
                f"{system}-gradient",
                f"Gradient, {system}",
                "valid",
                make(r"\nabla f", system, (f,)),
                oracle=_apply(grad, F),
            )
        )
        add(
            Case(
                f"{system}-divergence",
                f"Divergence, {system}",
                "valid",
                make(r"\nabla\cdot\mathbf{A}", system, (a,)),
                oracle=_apply(div, A, True),
            )
        )
        add(
            Case(
                f"{system}-curl",
                f"Curl, {system}",
                "valid",
                make(r"\nabla\times\mathbf{A}", system, (a,)),
                oracle=_apply(curl, A),
            )
        )
        add(
            Case(
                f"{system}-laplacian",
                f"Scalar Laplacian, {system}",
                "valid",
                make(r"\nabla^2 f", system, (f,)),
                oracle=_laplacian(grad, div, F),
            )
        )
    add(
        Case(
            "cylindrical-vector-laplacian",
            "Vector Laplacian, cylindrical (not componentwise)",
            "valid",
            make(r"\nabla^2\mathbf{A}", "cylindrical", (ya,)),
            oracle=lambda: cyl_vector_laplacian(CYL_A),
        )
    )
    add(
        Case(
            "spherical-curl-curl",
            "Curl of curl, spherical",
            "valid",
            make(r"\nabla\times(\nabla\times\mathbf{A})", "spherical", (sa,)),
            oracle=lambda: sph_curl(sph_curl(SPH_A)),
        )
    )
    add(
        Case(
            "cartesian-cross",
            "Cross product",
            "valid",
            make(
                r"\mathbf{A}\times\mathbf{B}",
                "cartesian",
                (vector("A", "1", "2", "3"), vector("B", "0", "1", "-1")),
            ),
            oracle=lambda: [sp.Integer(-5), sp.Integer(1), sp.Integer(1)],
        )
    )
    add(
        Case(
            "cartesian-triple",
            "Scalar triple product",
            "valid",
            make(
                r"\mathbf{A}\cdot(\mathbf{B}\times\mathbf{C})",
                "cartesian",
                (
                    vector("A", "1", "2", "3"),
                    vector("B", "0", "1", "-1"),
                    vector("C", "2", "0", "1"),
                ),
            ),
            oracle=lambda: [sp.Integer(-9)],
        )
    )
    add(
        Case(
            "cartesian-directional",
            "Directional derivative along a unit vector",
            "valid",
            make(r"(\hat{x}\cdot\nabla) f", "cartesian", (cf,)),
            oracle=lambda: [sp.diff(CART_F, x)],
        )
    )
    add(
        Case(
            "position-divergence",
            "Divergence of the position vector",
            "valid",
            make(r"\nabla\cdot\mathbf{r}", "spherical"),
            oracle=lambda: [sp.Integer(3)],
        )
    )
    add(
        Case(
            "position-curl",
            "Curl of the position vector",
            "valid",
            make(r"\nabla\times\mathbf{r}", "cylindrical"),
            oracle=lambda: [sp.Integer(0)] * 3,
        )
    )
    add(
        Case(
            "partial-vector-phi",
            "Partial derivative of a vector field in phi (frame rotates)",
            "valid",
            make(r"\partial_\phi \mathbf{A}", "cylindrical", (vector("A", "0", "rho", "0"),)),
            oracle=lambda: [-rho, sp.Integer(0), sp.Integer(0)],
        )
    )
    add(
        Case(
            "jacobian-cylindrical",
            "Jacobian determinant, cylindrical",
            "valid",
            VectorInput(action="jacobian", coordinates="cylindrical"),
            oracle=lambda: [rho],
        )
    )
    add(
        Case(
            "jacobian-spherical",
            "Jacobian determinant, spherical",
            "valid",
            VectorInput(action="jacobian", coordinates="spherical"),
            oracle=lambda: [r**2 * sp.sin(th)],
        )
    )
    add(
        Case(
            "convert-radial",
            "Convert x, y, z to spherical components",
            "valid",
            make(
                r"\mathbf{F}",
                "cartesian",
                (vector("F", "x", "y", "z"),),
                action="convert",
                to="spherical",
            ),
            oracle=lambda: [r, sp.Integer(0), sp.Integer(0)],
        )
    )
    add(
        Case(
            "convert-scalar",
            "Convert r^2 to cylindrical",
            "valid",
            make("f", "spherical", (scalar("f", "r**2"),), action="convert", to="cylindrical"),
            oracle=lambda: [rho**2 + z**2],
        )
    )
    # abstract fields: two engine routes must agree
    f2, a2, b2 = (
        abstract("f", smoothness="c2"),
        abstract("A", "vector", "c2"),
        abstract("B", "vector", "c2"),
    )
    for system in ("cartesian", "cylindrical", "spherical"):
        add(
            Case(
                f"{system}-curl-grad",
                f"curl grad f = 0, {system}",
                "consistency",
                make(r"\nabla\times(\nabla f)", system, (f2,)),
                oracle=lambda: [sp.Integer(0)] * 3,
            )
        )
        add(
            Case(
                f"{system}-div-curl",
                f"div curl A = 0, {system}",
                "consistency",
                make(r"\nabla\cdot(\nabla\times\mathbf{A})", system, (a2,)),
                oracle=lambda: [sp.Integer(0)],
            )
        )
    add(
        Case(
            "curl-curl-identity",
            "curl curl A - (grad div A - lap A) = 0, spherical",
            "consistency",
            make(
                r"\nabla\times(\nabla\times\mathbf{A}) - \nabla(\nabla\cdot\mathbf{A}) + \nabla^2\mathbf{A}",
                "spherical",
                (a2,),
            ),
            oracle=lambda: [sp.Integer(0)] * 3,
        )
    )
    add(
        Case(
            "div-cross-identity",
            "div(A x B) - (B.curl A - A.curl B) = 0, cylindrical",
            "consistency",
            make(
                r"\nabla\cdot(\mathbf{A}\times\mathbf{B}) - \mathbf{B}\cdot(\nabla\times\mathbf{A}) + \mathbf{A}\cdot(\nabla\times\mathbf{B})",
                "cylindrical",
                (a2, b2),
            ),
            oracle=lambda: [sp.Integer(0)],
        )
    )
    # conditional: nothing is assumed smooth
    add(
        Case(
            "curl-grad-no-smoothness",
            "curl grad f without declared smoothness",
            "conditional",
            make(r"\nabla\times(\nabla f)", "cartesian", (abstract("f"),)),
            Status.CONDITIONS_NOT_SATISFIED,
            needs="c2",
        )
    )
    add(
        Case(
            "div-curl-no-smoothness",
            "div curl A without declared smoothness",
            "conditional",
            make(r"\nabla\cdot(\nabla\times\mathbf{A})", "spherical", (abstract("A", "vector"),)),
            Status.CONDITIONS_NOT_SATISFIED,
            needs="c2",
        )
    )
    add(
        Case(
            "time-swap-no-smoothness",
            "d/dt curl E - curl dE/dt without declared smoothness",
            "conditional",
            make(
                r"\frac{\partial}{\partial t}\nabla\times\mathbf{E} - \nabla\times\frac{\partial \mathbf{E}}{\partial t}",
                "cartesian",
                (FieldDecl(name="E", kind="vector", depends_on=["x", "y", "z", "t"]),),
            ),
            Status.CONDITIONS_NOT_SATISFIED,
            needs="c2",
        )
    )
    # singular fields and coordinate singularities
    add(
        Case(
            "point-charge-field",
            "div of r_vec / r^3 away from the origin",
            "singular",
            make(r"\nabla\cdot\left(\frac{\mathbf{r}}{r^{3}}\right)", "spherical"),
            oracle=lambda: [sp.Integer(0)],
            restriction=r"r \neq 0",
        )
    )
    add(
        Case(
            "line-charge-field",
            "div of rho_hat / rho away from the axis",
            "singular",
            make(r"\nabla\cdot\left(\frac{\hat{\rho}}{\rho}\right)", "cylindrical"),
            oracle=lambda: [sp.Integer(0)],
            restriction=r"\rho \neq 0",
        )
    )
    add(
        Case(
            "polar-axis-gradient",
            "gradient in spherical coordinates states sin(theta) != 0",
            "singular",
            make(r"\nabla f", "spherical", (sf,)),
            oracle=lambda: sph_grad(SPH_F),
            restriction=r"\sin{\left(\theta \right)} \neq 0",
        )
    )
    # invalid input
    for cid, title, expr, flds, extra in (
        (
            "two-vectors-side-by-side",
            "product of two vectors",
            r"\mathbf{A}\mathbf{B}",
            (abstract("A", "vector"), abstract("B", "vector")),
            {},
        ),
        (
            "scalar-plus-vector",
            "scalar plus vector",
            r"\mathbf{A} + f",
            (abstract("A", "vector"), abstract("f")),
            {},
        ),
        ("div-of-scalar", "divergence of a scalar", r"\nabla\cdot f", (abstract("f"),), {}),
        (
            "grad-of-vector",
            "gradient of a vector",
            r"\nabla\mathbf{A}",
            (abstract("A", "vector"),),
            {},
        ),
        (
            "cross-of-scalars",
            "cross product of two scalars",
            r"a \times b",
            (abstract("a"), abstract("b")),
            {},
        ),
        (
            "undeclared-symbol",
            "an undeclared symbol is not a constant",
            r"\nabla (k f)",
            (abstract("f"),),
            {},
        ),
        (
            "ambiguous-operator-scope",
            "nabla f g is ambiguous",
            r"\nabla f g",
            (abstract("f"), abstract("g")),
            {},
        ),
        (
            "undeclared-vector",
            "bold letter that is not a field",
            r"\mathbf{C}",
            (abstract("f"),),
            {},
        ),
    ):
        add(Case(cid, title, "invalid", make(expr, "cartesian", flds, **extra), Status.PARSE_ERROR))
    # unsupported
    add(
        Case(
            "convert-abstract",
            "converting an abstract field has no formula to re-express",
            "unsupported",
            make("A", "cartesian", (abstract("A", "vector"),), action="convert", to="spherical"),
            Status.UNSUPPORTED,
        )
    )
    add(
        Case(
            "operator-depth",
            "five nested operators",
            "unsupported",
            make(
                r"\nabla\times(" * 5 + r"\mathbf{A}" + ")" * 5,
                "cartesian",
                (abstract("A", "vector"),),
            ),
            Status.RESOURCE_LIMIT,
        )
    )
    return c


# -- identities and controlled derivations (the handbook proposes, the engine verifies) --------------
def _rules() -> list[Any]:
    """The handbook's vector identities. The engine never reads the handbook: this benchmark asks the
    knowledge layer for them (an import at run time, since the benchmark is a script, not the engine)."""
    from minerva_knowledge import KnowledgeBase
    from minerva_knowledge.identities import vector_identity_rules

    return vector_identity_rules(KnowledgeBase.load())


def _wrong_rule() -> Any:
    from .models import VectorRule

    return VectorRule(
        id="wrong.curl_curl",
        lhs="curl(curl(A))",
        rhs="grad(div(A)) + lap(A)",
        variables={"A": "vector"},
        requires=["c2"],
    )


def _step(
    expression: str,
    action: str,
    fields: tuple[FieldDecl, ...],
    system: str = "cartesian",
    rules: list[Any] | None = None,
    **kw: Any,
) -> VectorInput:
    return VectorInput(
        expression=expression,
        format="text",
        coordinates=system,
        fields=list(fields),
        rules=_rules() if rules is None else rules,
        action=action,
        **kw,
    )


def _timed(name: str, smoothness: str | None = "smooth") -> FieldDecl:
    return FieldDecl(
        name=name,
        kind="vector",
        smoothness=smoothness,
        depends_on=["x", "y", "z", "t"],
    )


def _identity_step(r: VectorResult) -> bool:
    return (
        r.evidence == "identity"
        and len(r.steps) >= 1
        and all(s.evidence == "identity" and s.symbolic_zero is True for s in r.steps)
        and r.given == []
    )


def _conditional_on(*texts: str) -> Callable[[VectorResult], bool]:
    def check(r: VectorResult) -> bool:
        return r.evidence == "conditional" and set(texts) <= set(r.given)

    return check


FARADAY = Premise(equation="curl(E) = -dt(B)", label="Faraday")
AMPERE = Premise(equation="curl(B) = mu0*eps0*dt(E)", label="Ampere-Maxwell")
GAUSS = Premise(equation="div(E) = 0", label="Gauss")


def _step_cases() -> list[Case]:
    c2a, c2b = abstract("A", "vector", "c2"), abstract("B", "vector", "c2")
    c2f = abstract("f", "scalar", "c2")
    em = (_timed("E"), _timed("B"))
    out = [
        Case(
            "apply-curl-curl",
            "Rewrite curl curl A with the divergence and the Laplacian",
            "identities",
            _step("curl(curl(A))", "apply_identity", (c2a,), rule_id="vector.curl_curl", path=[]),
            check=lambda r: _identity_step(r) and "Lap(A)" in (r.result_text or ""),
        ),
        Case(
            "apply-curl-grad",
            "curl grad f becomes zero",
            "identities",
            _step("curl(grad(f))", "apply_identity", (c2f,), rule_id="vector.curl_grad", path=[]),
            check=lambda r: _identity_step(r) and r.result_text == "0",
        ),
        Case(
            "apply-div-cross-spherical",
            "div(A x B) in spherical coordinates",
            "identities",
            _step(
                "div(cross(A, B))",
                "apply_identity",
                (abstract("A", "vector"), abstract("B", "vector")),
                "spherical",
                rule_id="vector.div_cross",
                path=[],
            ),
            check=_identity_step,
        ),
        Case(
            "apply-curl-cross-cylindrical",
            "curl(A x B) in cylindrical coordinates",
            "identities",
            _step(
                "curl(cross(A, B))",
                "apply_identity",
                (abstract("A", "vector"), abstract("B", "vector")),
                "cylindrical",
                rule_id="vector.curl_cross",
                path=[],
            ),
            check=_identity_step,
        ),
        Case(
            "apply-reverse",
            "grad div A - lap A read right to left",
            "identities",
            _step(
                "grad(div(A)) - lap(A)",
                "apply_identity",
                (c2a,),
                rule_id="vector.curl_curl",
                reverse=True,
            ),
            check=lambda r: _identity_step(r) and r.steps[0].reverse,
        ),
        Case(
            "apply-to-a-part",
            "An identity on one part of a larger expression",
            "identities",
            _step(
                "2*curl(curl(A)) + B",
                "apply_identity",
                (c2a, abstract("B", "vector")),
                rule_id="vector.curl_curl",
            ),
            check=lambda r: _identity_step(r) and r.steps[0].path != [],
        ),
        Case(
            "list-identities",
            "List the identities that match, apply none",
            "identities",
            _step("curl(curl(A))", "identities", (c2a,)),
            check=lambda r: bool(r.applicable) and r.steps == [] and r.result_latex is None,
        ),
        Case(
            "substitute-faraday",
            "Substitute a Maxwell equation the user gave",
            "derivations",
            _step("curl(curl(E))", "substitute", em, premises=[FARADAY], constants=["mu0", "eps0"]),
            check=_conditional_on("curl(E) = -dt(B)"),
        ),
        Case(
            "apply-curl-to-faraday",
            "Apply curl to both sides of Faraday's law",
            "derivations",
            _step("curl(E) = -dt(B)", "apply", em, operator="curl"),
            check=_conditional_on("curl(E) = -dt(B)"),
        ),
        Case(
            "derive-wave-equation-lhs",
            "curl curl E = -mu0 eps0 d2E/dt2 from Faraday and Ampere-Maxwell",
            "derivations",
            _step(
                "curl(curl(E))",
                "derive",
                em,
                premises=[FARADAY, AMPERE],
                target="-mu0*eps0*dt(dt(E))",
                constants=["mu0", "eps0"],
            ),
            check=_conditional_on("curl(E) = -dt(B)", "curl(B) = mu0*eps0*dt(E)"),
        ),
        Case(
            "derive-wave-equation-rhs",
            "curl curl E = -lap E from Gauss's law",
            "derivations",
            _step("curl(curl(E))", "derive", em, premises=[GAUSS], target="-lap(E)"),
            check=_conditional_on("div(E) = 0"),
        ),
        Case(
            "derive-one-identity",
            "curl curl A to grad div A - lap A",
            "derivations",
            _step("curl(curl(A))", "derive", (c2a,), target="grad(div(A)) - lap(A)"),
            check=lambda r: _identity_step(r) and len(r.steps) == 1,
        ),
        # refusals
        Case(
            "identity-without-smoothness",
            "curl curl A without declared smoothness",
            "conditional",
            _step(
                "curl(curl(A))",
                "apply_identity",
                (abstract("A", "vector"),),
                rule_id="vector.curl_curl",
                path=[],
            ),
            Status.CONDITIONS_NOT_SATISFIED,
            needs="c2",
        ),
        Case(
            "derive-respects-smoothness",
            "derive does not use an identity the fields are not smooth enough for",
            "conditional",
            _step(
                "curl(curl(A))",
                "derive",
                (abstract("A", "vector"),),
                target="grad(div(A)) - lap(A)",
            ),
            Status.CONDITIONS_NOT_SATISFIED,
            needs="c2",
        ),
        Case(
            "wrong-identity",
            "A rule with a sign error is not verified",
            "invalid",
            _step(
                "curl(curl(A))",
                "apply_identity",
                (c2a,),
                rules=[_wrong_rule()],
                rule_id="wrong.curl_curl",
                path=[],
            ),
            Status.NOT_VERIFIED,
        ),
        Case(
            "ambiguous-part",
            "Two places match: nothing is picked",
            "invalid",
            _step(
                "curl(curl(A)) + curl(curl(B))",
                "apply_identity",
                (c2a, c2b),
                rule_id="vector.curl_curl",
            ),
            Status.UNCHANGED,
            check=lambda r: len(r.applicable) == 2,
        ),
        Case(
            "premise-does-not-occur",
            "A premise that is not in the expression changes nothing",
            "invalid",
            _step("curl(B)", "substitute", em, premises=[FARADAY]),
            Status.UNCHANGED,
        ),
        Case(
            "several-premises-no-order",
            "Several premises: the user names one",
            "invalid",
            _step("curl(E) + curl(B)", "substitute", em, premises=[FARADAY, AMPERE]),
            Status.PARSE_ERROR,
        ),
        Case(
            "derive-without-premise",
            "The target needs a premise that was not given",
            "derivations",
            _step("curl(curl(E))", "derive", em, target="-lap(E)"),
            Status.UNCHANGED,
            check=lambda r: r.search_complete is not None,
        ),
        Case(
            "operator-of-the-wrong-kind",
            "div of a scalar equation",
            "invalid",
            _step("div(E) = 0", "apply", em, operator="curl"),
            Status.PARSE_ERROR,
        ),
    ]
    return out


def _em(
    template_id: str, premises: list[str] | None = None, smoothness: str | None = "c2"
) -> VectorInput:
    from .em import TEMPLATES

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
        rules=_rules(),
        action="em_derive",
        template=template_id,
    )


def _em_ok(conditional: bool) -> Callable[[VectorResult], bool]:
    def check(r: VectorResult) -> bool:
        return (
            r.evidence == ("conditional" if conditional else "identity")
            and bool(r.steps)
            and r.model_check is not None
            and r.model_check.status == "passed"
            and bool(r.equations_latex)
        )

    return check


def _em_cases() -> list[Case]:
    from .em import TEMPLATES

    out = [
        Case(
            f"em-{tid.replace('_', '-')}",
            TEMPLATES[tid].title,
            "electromagnetism",
            _em(tid),
            check=_em_ok(bool(TEMPLATES[tid].needs)),
        )
        for tid in TEMPLATES
    ]
    faraday_flipped = ["div(E) = 0", "curl(E) = dt(B)", "curl(B) = mu0*epsilon0*dt(E)"]
    out += [
        Case(
            "em-missing-premise",
            "The wave equation without Ampere-Maxwell",
            "electromagnetism",
            _em("wave_equation_e", ["div(E) = 0", "curl(E) = -dt(B)"]),
            Status.CONDITIONS_NOT_SATISFIED,
            check=lambda r: len(r.needs_premises) == 1 and r.steps == [],
        ),
        Case(
            "em-premise-with-a-flipped-sign",
            "A premise that is not the one needed counts as missing",
            "electromagnetism",
            _em("wave_equation_e", faraday_flipped),
            Status.CONDITIONS_NOT_SATISFIED,
            check=lambda r: r.needs_premises == ["curl(E) = -dt(B)"],
        ),
        Case(
            "em-gauge-needs-smoothness",
            "Gauge invariance without a declared twice differentiable chi",
            "electromagnetism",
            _em("gauge_invariance", [], smoothness=None),
            Status.CONDITIONS_NOT_SATISFIED,
        ),
        Case(
            "em-continuity-needs-smoothness",
            "Continuity without declared smoothness",
            "electromagnetism",
            _em("continuity", smoothness=None),
            Status.CONDITIONS_NOT_SATISFIED,
        ),
    ]
    return out


def _int(
    expression: str,
    kind: str,
    fields: tuple[FieldDecl, ...],
    system: str = "cartesian",
    **kw: Any,
) -> VectorInput:
    return VectorInput(
        expression=expression,
        format="text",
        coordinates=system,
        fields=list(fields),
        action="integrate",
        integral=kind,
        **kw,
    )


def _value(*expected: str) -> Callable[[VectorResult], bool]:
    """The integral's value(s) equal the hand-computed closed form(s), as expressions."""

    def check(r: VectorResult) -> bool:
        if r.integral is None or len(r.integral.values) != len(expected):
            return False
        syms = {n: sp.Symbol(n, positive=True) for n in ("R", "rho0", "epsilon0")}
        return all(
            v.text is not None
            and sp.simplify(sp.sympify(v.text, locals=syms) - sp.sympify(e, locals=syms)) == 0
            for v, e in zip(r.integral.values, expected, strict=True)
        )

    return check


def _holds(r: VectorResult) -> bool:
    return bool(r.integral and r.integral.holds and r.integral.hypotheses)


def _estat_cases() -> list[Case]:
    from .models import (
        BoundaryProblem,
        CheckProblem,
        FaceCondition,
        FieldProblem,
        MultipoleProblem,
        PointCharge,
        Region,
    )

    unit = Region(bounds=[("0", "1")] * 3)

    def req(problem: Any, constants: list[str] | None = None) -> VectorInput:
        return VectorInput(
            expression="",
            format="text",
            action="em_compute",
            em=problem,
            constants=constants or [],
        )

    def central(label: str, value: float, tol: float) -> Callable[[VectorResult], bool]:
        def check(r: VectorResult) -> bool:
            if r.em is None:
                return False
            item = next((i for i in r.em.items if i.label.startswith(label)), None)
            return item is not None and abs(float(item.numeric or "nan") - value) < tol

        return check

    return [
        Case(
            "es-field-dipole",
            "Field of a dipole: Coulomb sum, gradient of the potential, central difference",
            "electrostatics",
            req(
                FieldProblem(
                    kind="field",
                    charges=[
                        PointCharge(q="q", position=("0", "0", "0")),
                        PointCharge(q="-q", position=("0", "0", "d")),
                    ],
                    point=("1", "2", "3"),
                ),
                ["q", "d"],
            ),
            check=lambda r: (
                r.em is not None and r.em.agreement is not None and r.em.agreement < 1e-6
            ),
        ),
        Case(
            "es-field-at-charge",
            "The field at a point charge is refused",
            "electrostatics",
            req(
                FieldProblem(
                    kind="field",
                    charges=[PointCharge(q="1", position=("0", "0", "0"))],
                    point=("0", "0", "0"),
                )
            ),
            Status.UNSUPPORTED,
        ),
        Case(
            "es-field-inside-density",
            "A field point inside the charge density is refused",
            "electrostatics",
            req(FieldProblem(kind="field", charges=[], density="1", box=unit, point=("0.5",) * 3)),
            Status.UNSUPPORTED,
        ),
        Case(
            "es-multipole-three-charges",
            "Moments of three charges and the measured next-multipole decay",
            "electrostatics",
            req(
                MultipoleProblem(
                    kind="multipole",
                    charges=[
                        PointCharge(q="1", position=("0", "0", "1")),
                        PointCharge(q="-1", position=("0", "0", "-1")),
                        PointCharge(q="1", position=("1", "0", "0")),
                    ],
                )
            ),
            check=lambda r: r.em is not None and r.em.items[0].latex == "1",
        ),
        Case(
            "es-laplace-sine-edge",
            "Laplace on the unit square with a sine on the top edge (phi = sin(pi x) sinh(pi y) / sinh(pi))",
            "electrostatics",
            req(BoundaryProblem(kind="boundary_value", a="1", b="1", top="sin(pi*x)")),
            check=central("phi(0.5, 0.5)", 0.1992684077, 1e-6),
        ),
        Case(
            "es-poisson-unit-square",
            "Poisson with source 1 on the unit square (centre value 0.0736713532)",
            "electrostatics",
            req(BoundaryProblem(kind="boundary_value", a="1", b="1", source="1")),
            check=central("phi(0.5, 0.5)", 0.0736713532, 3e-5),
        ),
        Case(
            "es-poisson-nonzero-edge",
            "Poisson with a nonzero edge is refused, not half solved",
            "electrostatics",
            req(BoundaryProblem(kind="boundary_value", a="1", b="1", top="1", source="1")),
            Status.UNSUPPORTED,
        ),
        Case(
            "es-check-harmonic",
            "A harmonic candidate with its face value",
            "electrostatics",
            req(
                CheckProblem(
                    kind="check_solution",
                    potential="sin(pi*x)*sinh(pi*y)",
                    box=unit,
                    conditions=[FaceCondition(face="y=lo", value="0")],
                )
            ),
            check=lambda r: r.em is not None and r.em.holds is True,
        ),
        Case(
            "es-check-wrong-candidate",
            "x^2 does not solve Laplace's equation: reported, not accepted",
            "electrostatics",
            req(CheckProblem(kind="check_solution", potential="x**2", box=unit)),
            check=lambda r: r.em is not None and r.em.holds is False,
        ),
    ]


def _integral_cases() -> list[Case]:
    from .models import Curve, Param, Region, Surface

    def p(name: str, lo: str, hi: str) -> Param:
        return Param(name=name, lower=lo, upper=hi)

    def vec(name: str, *c: str) -> FieldDecl:
        return FieldDecl(name=name, kind="vector", components=list(c))

    def sca(name: str, c: str) -> FieldDecl:
        return FieldDecl(name=name, kind="scalar", components=[c])

    R = {"constants": ["R"], "assumptions": {"R": ["positive"]}}
    sphere_u = Surface(q=["R", "u", "v"], u=p("u", "0", "pi"), v=p("v", "0", "2*pi"))
    sphere_in = Surface(q=["R", "v", "u"], u=p("u", "0", "2*pi"), v=p("v", "0", "pi"))
    ball = Region(bounds=[("0", "R"), ("0", "pi"), ("0", "2*pi")])
    field = vec("F", "1/r**2", "0", "0")
    gauss = {
        "constants": ["R", "rho0", "epsilon0"],
        "assumptions": {"R": ["positive"], "rho0": ["positive"], "epsilon0": ["positive"]},
    }
    e = vec("E", "rho0*r/(3*epsilon0)", "0", "0")
    return [
        Case(
            "int-line-work",
            "Work of (y, x, 0) along a parabola",
            "integrals",
            _int(
                "F",
                "line",
                (vec("F", "y", "x", "0"),),
                curve=Curve(q=["s", "s**2", "0"], parameter=p("s", "0", "1")),
            ),
            check=_value("1"),
        ),
        Case(
            "int-line-circulation-spherical",
            "Circulation of e_phi around the equator of a sphere",
            "integrals",
            _int(
                "F",
                "line",
                (vec("F", "0", "0", "1"),),
                "spherical",
                curve=Curve(q=["2", "pi/2", "s"], parameter=p("s", "0", "2*pi")),
            ),
            check=_value("4*pi"),
        ),
        Case(
            "int-line-helix-length",
            "Length of a helix (scalar line integral)",
            "integrals",
            _int(
                "f",
                "line_scalar",
                (sca("f", "1"),),
                curve=Curve(q=["cos(s)", "sin(s)", "s"], parameter=p("s", "0", "2*pi")),
            ),
            check=_value("2*sqrt(2)*pi"),
        ),
        Case(
            "int-flux-point-charge",
            "Flux of r_hat / r^2 through a sphere is 4 pi",
            "integrals",
            _int("F", "flux", (field,), "spherical", surface=sphere_u, **R),
            check=_value("4*pi"),
        ),
        Case(
            "int-flux-orientation",
            "The same flux with the parameters swapped is -4 pi",
            "integrals",
            _int("F", "flux", (field,), "spherical", surface=sphere_in, **R),
            check=_value("-4*pi"),
        ),
        Case(
            "int-volume-ball",
            "Volume of a ball",
            "integrals",
            _int("f", "volume", (sca("f", "1"),), "spherical", region=ball, **R),
            check=_value("4*pi*R**3/3"),
        ),
        Case(
            "int-volume-cylinder",
            "Integral of z over a cylinder",
            "integrals",
            _int(
                "f",
                "volume",
                (sca("f", "z"),),
                "cylindrical",
                region=Region(bounds=[("0", "2"), ("0", "2*pi"), ("0", "3")]),
            ),
            check=_value("18*pi"),
        ),
        Case(
            "int-gauss-law-ball",
            "Gauss's law, integral form: flux of E in a uniformly charged ball = rho V / epsilon0",
            "integrals",
            _int("E", "flux", (e,), "spherical", surface=sphere_u, **gauss),
            check=_value("4*pi*rho0*R**3/(3*epsilon0)"),
        ),
        Case(
            "int-divergence-theorem-cartesian",
            "Divergence theorem on a box, Cartesian",
            "integrals",
            _int(
                "F",
                "divergence_theorem",
                (vec("F", "x**2", "y*z", "z**3"),),
                region=Region(bounds=[("0", "1"), ("0", "2"), ("0", "1")]),
            ),
            check=lambda r: _holds(r) and _value("5", "5")(r),
        ),
        Case(
            "int-divergence-theorem-cylindrical",
            "Divergence theorem on a cylindrical box",
            "integrals",
            _int(
                "F",
                "divergence_theorem",
                (vec("F", "rho*z", "rho**2", "z**2"),),
                "cylindrical",
                region=Region(bounds=[("1", "2"), ("0", "pi"), ("0", "1")]),
            ),
            check=_holds,
        ),
        Case(
            "int-divergence-theorem-ball",
            "Divergence theorem over a ball (F = r e_r): 32 pi",
            "integrals",
            _int(
                "F",
                "divergence_theorem",
                (vec("F", "r", "0", "0"),),
                "spherical",
                region=Region(bounds=[("0", "2"), ("0", "pi"), ("0", "2*pi")]),
            ),
            check=lambda r: _holds(r) and _value("32*pi", "32*pi")(r),
        ),
        Case(
            "int-stokes-surface",
            "Stokes' theorem on a parametrised surface",
            "integrals",
            _int(
                "F",
                "stokes_theorem",
                (vec("F", "-y*z", "x*z", "x*y"),),
                surface=Surface(q=["u", "v", "u*v"], u=p("u", "0", "1"), v=p("v", "0", "1")),
            ),
            check=lambda r: _holds(r) and _value("1", "1")(r),
        ),
        Case(
            "int-gradient-theorem",
            "Gradient theorem along a curve",
            "integrals",
            _int(
                "f",
                "gradient_theorem",
                (sca("f", "x**2*y + z*x"),),
                curve=Curve(q=["s", "s**2", "1-s"], parameter=p("s", "0", "1")),
            ),
            check=lambda r: _holds(r) and _value("1", "1")(r),
        ),
        # refusals
        Case(
            "int-point-charge-ball-refused",
            "The divergence theorem over a ball with a pole at the centre",
            "singular",
            _int(
                "F",
                "divergence_theorem",
                (field,),
                "spherical",
                region=Region(bounds=[("0", "2"), ("0", "pi"), ("0", "2*pi")]),
            ),
            Status.UNSUPPORTED,
            check=lambda r: r.diagnostics.reason == "singular_in_domain",
        ),
        Case(
            "int-pole-on-the-curve-refused",
            "A line integral through a pole",
            "singular",
            _int(
                "F",
                "line",
                (vec("F", "1/x", "0", "0"),),
                curve=Curve(q=["s", "0", "0"], parameter=p("s", "-1", "1")),
            ),
            Status.UNSUPPORTED,
        ),
        Case(
            "int-abstract-field-refused",
            "An integral of an abstract field",
            "unsupported",
            _int(
                "F",
                "line",
                (abstract("F", "vector"),),
                curve=Curve(q=["s", "0", "0"], parameter=p("s", "0", "1")),
            ),
            Status.UNSUPPORTED,
        ),
        Case(
            "int-missing-domain",
            "A flux without a surface: the domain is never invented",
            "invalid",
            _int("F", "flux", (vec("F", "1", "0", "0"),)),
            Status.PARSE_ERROR,
        ),
        Case(
            "int-wrong-kind",
            "A line integral of a scalar expression",
            "invalid",
            _int(
                "f",
                "line",
                (sca("f", "1"),),
                curve=Curve(q=["s", "0", "0"], parameter=p("s", "0", "1")),
            ),
            Status.PARSE_ERROR,
        ),
    ]


def _distributional_cases() -> list[Case]:
    """The distributional divergence and Laplacian of a point-source Coulomb kernel. The expected strengths and
    sources are the textbook values typed here (div r/r^3 = 4 pi delta, lap 1/r = -4 pi delta), not produced by
    the code under test; wrong claims and unsupported fields are cases too."""
    from ..distributions.point_delta import PointDelta

    a1, a2, a3, big_c, q, e0 = sp.symbols("a_1 a_2 a_3 C q epsilon_0", real=True)

    def delta(strength: sp.Expr, source: tuple[Any, Any, Any] = (0, 0, 0)) -> Any:
        want = strength * PointDelta(*source)
        local = {"PointDelta": PointDelta, **{s.name: s for s in (a1, a2, a3, big_c, q, e0)}}

        def check(result: VectorResult) -> bool:
            sing = result.singular
            if sing is None or sing.proof_status != "derived" or result.result_text is None:
                return False
            if not all(w.passed for w in sing.weak_checks + sing.flux_checks):
                return False
            if not all(step.status == "established" for step in sing.derivation):
                return False
            got = sp.sympify(result.result_text, locals=local)
            return bool(sp.simplify(got - want) == 0)

        return check

    def kernel(strength: str, source: tuple[str, str, str], power: str = "3") -> FieldDecl:
        q2 = " + ".join(f"({v} - {a})**2" for v, a in zip("xyz", source, strict=True))
        return vector(
            "F",
            *[
                f"{strength}*({v} - {a})/({q2})**({power}/2)"
                for v, a in zip("xyz", source, strict=True)
            ],
        )

    real = {"a_1": ["real"], "a_2": ["real"], "a_3": ["real"], "C": ["real"]}
    cases: list[Case] = []

    def add(
        case_id: str, title: str, request: VectorInput, status: Status = Status.OK, **kw: Any
    ) -> None:
        cases.append(
            Case(
                case_id,
                title,
                "distributional",
                request.model_copy(update={"action": "distributional"}),
                status,
                **kw,
            )
        )

    div = r"\nabla\cdot\left(\frac{\mathbf{r}}{r^{3}}\right)"
    add(
        "dist-coulomb",
        "div of r/r^3 is 4 pi delta (spherical)",
        make(div, "spherical"),
        check=delta(4 * sp.pi),
    )
    add(
        "dist-rhat",
        "div of rhat/r^2 is 4 pi delta (spherical)",
        make(r"\nabla\cdot\left(\frac{\hat{r}}{r^{2}}\right)", "spherical"),
        check=delta(4 * sp.pi),
    )
    add(
        "dist-cartesian-norm",
        "div of r/|r|^3 is 4 pi delta (Cartesian)",
        make(r"\nabla\cdot\left(\frac{\mathbf{r}}{|\mathbf{r}|^{3}}\right)"),
        check=delta(4 * sp.pi),
    )
    add(
        "dist-cartesian-components",
        "(x,y,z)/(x^2+y^2+z^2)^(3/2) (explicit components)",
        make(r"\nabla\cdot F", fields=(kernel("1", ("0", "0", "0")),)),
        check=delta(4 * sp.pi),
    )
    add(
        "dist-cylindrical",
        "the Coulomb field in cylindrical components",
        make(
            r"\nabla\cdot F",
            "cylindrical",
            (vector("F", "rho/(rho**2 + z**2)**(3/2)", "0", "z/(rho**2 + z**2)**(3/2)"),),
        ),
        check=delta(4 * sp.pi),
    )
    add(
        "dist-laplacian",
        "Laplacian of 1/r is -4 pi delta",
        make(r"\nabla^{2}\frac{1}{r}", "spherical"),
        check=delta(-4 * sp.pi),
    )
    add(
        "dist-laplacian-cartesian",
        "div grad of 1/|r| is -4 pi delta (Cartesian)",
        make(r"\nabla\cdot\left(\nabla\frac{1}{|\mathbf{r}|}\right)"),
        check=delta(-4 * sp.pi),
    )
    add(
        "dist-translated",
        "translated and scaled source: 4 pi C delta(r - a)",
        make(
            r"\nabla\cdot F",
            fields=(kernel("C", ("a_1", "a_2", "a_3")),),
            constants=["C", "a_1", "a_2", "a_3"],
            assumptions=real,
        ),
        check=delta(4 * sp.pi * big_c, (a1, a2, a3)),
    )
    add(
        "dist-negative-strength",
        "negative strength: -8 pi delta(r - (1, 0, 1/2))",
        make(r"\nabla\cdot F", fields=(kernel("-2", ("1", "0", "1/2")),)),
        check=delta(-8 * sp.pi, (1, 0, sp.Rational(1, 2))),
    )
    add(
        "dist-zero-strength",
        "zero strength gives the zero distribution",
        make(r"\nabla\cdot F", fields=(kernel("0", ("0", "0", "0")),)),
        check=lambda r: r.singular is not None and r.result_latex == "0",
    )
    add(
        "dist-point-charge",
        "Gauss's law for a point charge: q/epsilon_0 delta",
        make(
            r"\nabla\cdot\left(\frac{q}{4\pi\epsilon_0}\frac{\mathbf{r}}{r^{3}}\right)",
            "spherical",
            constants=["q", "epsilon_0"],
            assumptions={"epsilon_0": ["positive"]},
        ),
        check=delta(q / e0),
    )
    add(
        "dist-claim-sign",
        "the +4 pi claim for the Laplacian of 1/r is rejected",
        make(r"\nabla^{2}\frac{1}{r}", "spherical", target=r"4\pi\delta^{(3)}(\mathbf r)"),
        check=lambda r: (
            r.singular is not None
            and r.singular.claim is not None
            and not r.singular.claim.holds
            and r.singular.claim.reason == "different_strength"
        ),
    )
    add(
        "dist-claim-normalisation",
        "the claim div(r/r^3) = delta (not 4 pi delta) is rejected",
        make(div, "spherical", target=r"\delta^{(3)}(\mathbf r)"),
        check=lambda r: (
            r.singular is not None and r.singular.claim is not None and not r.singular.claim.holds
        ),
    )
    add(
        "dist-claim-pointwise",
        "the claim div(r/r^3) = 0 as a distributional identity is rejected",
        make(div, "spherical", target="0"),
        check=lambda r: (
            r.singular is not None
            and r.singular.claim is not None
            and r.singular.claim.reason == "no_delta_term"
        ),
    )
    add(
        "dist-r-over-r4",
        "r/r^4 is not the Coulomb kernel",
        make(r"\nabla\cdot F", fields=(kernel("1", ("0", "0", "0"), "4"),)),
        Status.UNSUPPORTED,
    )
    add(
        "dist-two-sources",
        "two point sources are not supported",
        make(
            r"\nabla\cdot (F + G)",
            fields=(
                kernel("1", ("0", "0", "0")),
                kernel("1", ("1", "0", "0")).model_copy(update={"name": "G"}),
            ),
        ),
        Status.UNSUPPORTED,
    )
    add(
        "dist-abstract-field",
        "an abstract field has no source",
        make(r"\nabla\cdot F", fields=(FieldDecl(name="F", kind="vector"),)),
        Status.UNSUPPORTED,
    )
    add(
        "dist-source-not-real",
        "an undeclared source position is not assumed real",
        make(
            r"\nabla\cdot F",
            fields=(kernel("1", ("a_1", "a_2", "a_3")),),
            constants=["a_1", "a_2", "a_3"],
        ),
        Status.UNSUPPORTED,
    )
    add(
        "dist-no-delta-through-evaluate",
        "a Dirac delta in a component is refused by the pointwise actions",
        VectorInput(expression=r"\nabla\cdot F", fields=[vector("F", "DiracDelta(x)", "0", "0")]),
        Status.UNSUPPORTED,
        check=lambda r: r.diagnostics.reason == "unsupported_distribution_operation",
    )
    return cases


def _all_cases() -> list[Case]:
    return [
        *_cases(),
        *_step_cases(),
        *_em_cases(),
        *_integral_cases(),
        *_estat_cases(),
        *_distributional_cases(),
    ]


CASES = _all_cases()


def _read(text: str) -> sp.Expr:
    return sp.sympify(text, locals=LOCALS)


def _correct(case: Case, result: VectorResult) -> bool:
    if result.status is not case.status:
        return False
    if case.check is not None and not case.check(result):
        return False
    if case.status is not Status.OK:
        if case.needs is not None:
            return bool(result.blocked) and result.blocked[0].needs == case.needs
        return result.result_latex is None
    if case.request.action not in NO_NUMERIC_CHECK and (
        result.verification is None or not result.verification.numeric_passed
    ):
        return False
    if case.restriction is not None and case.restriction not in result.restrictions_latex:
        return False
    if case.oracle is None:
        return True
    if result.result_kind == "matrix":
        if result.determinant_latex is None:
            return False
        det = parse_latex(result.determinant_latex)
        det = det.subs({s: LOCALS[s.name] for s in det.free_symbols if s.name in LOCALS})
        return all(sp.simplify(det - w) == 0 for w in case.oracle())
    texts = result.components_text or ([result.result_text] if result.result_text else [])
    if not texts:
        return False
    want = case.oracle()
    got = [_read(t) for t in texts]
    return len(got) == len(want) and all(
        sp.simplify(g - w) == 0 for g, w in zip(got, want, strict=True)
    )


def run(repeats: int = 3) -> dict[str, Any]:
    deterministic: dict[str, Any] = {"cases": {}, "groups": {}}
    timings: dict[str, float] = {}
    for case in CASES:
        times: list[float] = []
        result = vector_calculus(case.request, LIMITS)
        for _ in range(repeats):
            t0 = time.perf_counter()
            vector_calculus(case.request, LIMITS)
            times.append((time.perf_counter() - t0) * 1000)
        v = result.verification
        deterministic["cases"][case.id] = {
            "group": case.group,
            "status": result.status.value,
            "expected": case.status.value,
            "correct": _correct(case, result),
            "points_checked": v.points_checked if v else 0,
            "points_agreed": v.points_agreed if v else 0,
            "operator_depth": result.diagnostics.operator_depth,
        }
        timings[case.id] = round(statistics.median(times), 1)
    for group in GROUPS:
        members = [e for e in deterministic["cases"].values() if e["group"] == group]
        deterministic["groups"][group] = {
            "cases": len(members),
            "correct": sum(1 for e in members if e["correct"]),
        }
    total = len(CASES)
    deterministic["coverage"] = {
        "corpus": total,
        "correct": sum(1 for e in deterministic["cases"].values() if e["correct"]),
        "shown_results": sum(1 for e in deterministic["cases"].values() if e["status"] == "ok"),
        "refused_as_expected": sum(
            1 for e in deterministic["cases"].values() if e["status"] != "ok" and e["correct"]
        ),
    }
    return {
        "deterministic": deterministic,
        "measurements": {
            "median_ms": timings,
            "slowest_ms": max(timings.values()),
            "python": platform.python_version(),
            "repeats": repeats,
        },
    }


def render_report(data: dict[str, Any]) -> str:
    d, m = data["deterministic"], data["measurements"]
    lines = [
        "# Vector calculus benchmark",
        "",
        "Generated by `python -m minerva_engine.vector.benchmark` (do not edit). References are textbook "
        "del-operator formulas typed from their definitions, closed forms, and refusals the corpus expects.",
        "",
        f"**{d['coverage']['correct']} of {d['coverage']['corpus']} problems answered correctly** "
        f"({d['coverage']['shown_results']} results shown and checked against a reference, "
        f"{d['coverage']['refused_as_expected']} refused with the expected status).",
        "",
        "| Group | Problems | Correct |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {g} | {e['cases']} | {e['correct']} |" for g, e in d["groups"].items()]
    lines += [
        "",
        "| Problem | Group | Status | Correct | Points | Median ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in CASES:
        e = d["cases"][case.id]
        lines.append(
            f"| {case.title} | {e['group']} | {e['status']} | {'yes' if e['correct'] else 'NO'} | "
            f"{e['points_agreed']}/{e['points_checked']} | {m['median_ms'][case.id]} |"
        )
    lines += [
        "",
        "## Limits",
        "",
        "- Results are numerically checked against an independent Cartesian finite-difference evaluation; "
        "no symbolic proof is claimed for an evaluated result.",
        "- A result that needs mixed partial derivatives to commute is withheld unless the field's "
        "smoothness is declared.",
        "- The axis and origin of curvilinear systems are reported as restrictions. A delta at a point is "
        "represented only as the result of the `distributional` action for the Coulomb-kernel family "
        "(one point source, constant strength); every other distribution of several variables is refused.",
        "- At most four nested operators; three coordinate systems; no tensor fields, no integrals "
        "(later phases).",
        "",
    ]
    return "\n".join(lines)


def compare(stored: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    old, new = stored["deterministic"], fresh["deterministic"]
    return [
        f"{cid}: stored {old['cases'].get(cid)} != fresh {e}"
        for cid, e in new["cases"].items()
        if old["cases"].get(cid) != e
    ] + [f"missing from fresh run: {cid}" for cid in old["cases"] if cid not in new["cases"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the vector calculus engine.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path, default=Path("docs/benchmarks/vector.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/vector-benchmark.md"))
    parser.add_argument("--check", action="store_true", help="compare the deterministic part")
    args = parser.parse_args(argv)
    data = run(1 if args.check else args.repeats)
    wrong = [cid for cid, e in data["deterministic"]["cases"].items() if not e["correct"]]
    if wrong:
        print("incorrect:", ", ".join(wrong))
        return 1
    if args.check:
        stored = json.loads(args.json.read_text())
        problems = compare(stored, data)
        for p in problems:
            print(p)
        return 1 if problems else 0
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    args.report.write_text(render_report(data))
    print(f"{data['deterministic']['coverage']['correct']} of {len(CASES)} correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
