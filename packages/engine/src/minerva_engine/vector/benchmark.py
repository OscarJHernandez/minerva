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
NO_NUMERIC_CHECK = ("identities", "substitute")
GROUPS = (
    "valid",
    "consistency",
    "conditional",
    "singular",
    "invalid",
    "unsupported",
    "identities",
    "derivations",
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
            Status.UNCHANGED,
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


def _all_cases() -> list[Case]:
    return [*_cases(), *_step_cases()]


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
        "- The axis and origin of curvilinear systems are reported as restrictions; distributions "
        "(a delta at the origin) are not represented.",
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
