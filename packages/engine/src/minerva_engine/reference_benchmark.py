"""Multistep problems with answers established independently of Minerva.

Each problem names the textbook or table its expected outcome comes from and is judged by a route that
does not use Minerva's own result path: `mpmath.expm` on explicit matrices, a regularised delta sequence
integrated by quadrature, the quarter-period integral of an oscillator, plain arithmetic for a
substitution. A problem expects either a **correct** answer (the engine's result must agree with the
oracle) or a **refusal** (the engine must say `unsupported`, `conditions_not_satisfied` or a blocked step,
never an `ok` result: the oracle shows that any such result would be wrong or unjustified).

`python -m minerva_engine.reference_benchmark [--check]` runs them. The outcome of every problem is
deterministic and is written to `docs/benchmarks/reference.json` (`--check` compares against it and
fails on any difference, which is what CI does); latencies are recorded but never gated. Geometry has its
own closed-form corpus in `geometry/benchmark.py`; this file adds the other domains.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

from .distributions.models import DistributionInput
from .latexparse import parse_latex
from .matrix import MatrixCalculusInput, MatrixDecl
from .models import ExpressionInput, Limits
from .service import Engine
from .textparse import parse_text
from .vector import FieldDecl, VectorInput

mp.mp.dps = 30

Judge = Callable[[Engine], tuple[str, str]]  # -> (outcome, detail)


@dataclass(frozen=True)
class Problem:
    id: str
    category: str
    source: str  # where the expected outcome comes from, by locator only
    expected: str  # "correct" or "refused"
    judge: Judge


def _expr(latex: str, **assumptions: list[str]) -> ExpressionInput:
    return ExpressionInput(input=latex, assumptions=dict(assumptions))


def _verdict(expected: str, ok: bool, agrees: bool | None, detail: str) -> tuple[str, str]:
    """`ok`: the engine returned a result; `agrees`: that result matches the oracle (None: not tested)."""
    if expected == "refused":
        return (
            ("wrong", f"returned a result where a refusal was due: {detail}")
            if ok
            else (
                "refused_as_expected",
                detail,
            )
        )
    if not ok:
        return "unsupported_regression", detail
    return ("correct", detail) if agrees else ("wrong", detail)


def _matrix(entries: list[list[int]]) -> mp.matrix:
    return mp.matrix(entries)


def _close(a: mp.matrix, b: mp.matrix, tol: float = 1e-12) -> bool:
    return bool(mp.norm(a - b) <= tol * max(1, mp.norm(b)))


_A = [[0, 1], [0, 0]]
_B = [[0, 0], [1, 0]]


def _numeric_entries(entries: list[list[str]], **values: float) -> mp.matrix:
    syms = {k: sp.Symbol(k) for k in values}
    rows = []
    for row in entries:
        out = []
        for cell in row:
            e = parse_text(cell) if isinstance(cell, str) else sp.sympify(cell)
            out.append(
                mp.mpmathify(complex(e.subs({syms[k]: v for k, v in values.items()}).evalf(30)))
            )
        rows.append(out)
    return mp.matrix(rows)


def _exp_product(engine: Engine) -> tuple[str, str]:
    """exp(A) exp(B) for noncommuting nilpotent matrices: entries against mpmath.expm."""
    result = engine.matrix_calculus(
        MatrixCalculusInput(
            expression=_expr(r"\exp(A)\exp(B)"),
            matrices=[
                MatrixDecl(name="A", entries=[["0", "1"], ["0", "0"]]),
                MatrixDecl(name="B", entries=[["0", "0"], ["1", "0"]]),
            ],
            action="evaluate",
        )
    )
    if result.status.value != "ok" or result.result_entries is None:
        return _verdict("correct", False, None, f"status {result.status.value}")
    truth = mp.expm(_matrix(_A)) * mp.expm(_matrix(_B))
    return _verdict(
        "correct",
        True,
        _close(_numeric_entries(result.result_entries), truth),
        "entries vs mpmath.expm",
    )


def _exp_merge_refused(engine: Engine) -> tuple[str, str]:
    """exp(A) exp(B) = exp(A+B) is false for these matrices, so an abstract merge must be refused."""
    a, b = _matrix(_A), _matrix(_B)
    if _close(mp.expm(a) * mp.expm(b), mp.expm(a + b), 1e-6):  # the oracle: the merge is false
        raise RuntimeError("oracle: these matrices commute, the problem is ill-posed")
    result = engine.matrix_calculus(
        MatrixCalculusInput(
            expression=_expr(r"\exp(A)\exp(B)"),
            matrices=[MatrixDecl(name="A", rows=2), MatrixDecl(name="B", rows=2)],
            action="simplify",
        )
    )
    return _verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _frechet(engine: Engine) -> tuple[str, str]:
    """d/dt exp(A + tB) for noncommuting A, B at t = 0.7: entries against a central difference of expm."""
    result = engine.matrix_calculus(
        MatrixCalculusInput(
            expression=_expr(r"\exp(A+tB)"),
            matrices=[
                MatrixDecl(name="A", entries=[["0", "1"], ["0", "0"]]),
                MatrixDecl(name="B", entries=[["0", "0"], ["1", "0"]]),
            ],
            action="differentiate",
            variable="t",
        )
    )
    if result.status.value != "ok" or result.result_entries is None:
        return _verdict("correct", False, None, f"status {result.status.value}")
    t0, h = mp.mpf("0.7"), mp.mpf("1e-8")
    a, b = _matrix(_A), _matrix(_B)
    truth = (mp.expm(a + (t0 + h) * b) - mp.expm(a + (t0 - h) * b)) / (2 * h)
    return _verdict(
        "correct",
        True,
        _close(_numeric_entries(result.result_entries, t=0.7), truth, 1e-7),
        "entries vs central difference of expm",
    )


def _delta(
    latex: str, lower: str, upper: str, endpoint: str | None = None
) -> Callable[[Engine], Any]:
    def run(engine: Engine) -> Any:
        return engine.distribution(
            DistributionInput(
                expression=_expr(latex, x=["real"]),
                action="integrate",
                variable="x",
                lower=lower,
                upper=upper,
                endpoint=endpoint,
            )
        )

    return run


def _delta_composition(engine: Engine) -> tuple[str, str]:
    """integral of (x^2+1) delta(x^2-4): sum over the roots of f(r)/|g'(r)| = 5/4 + 5/4, by a Gaussian sequence."""
    result = _delta(r"(x^2+1)\operatorname{DiracDelta}(x^2-4)", r"-\infty", r"\infty")(engine)
    if result.status.value != "ok" or not result.result_latex:
        return _verdict("correct", False, None, f"status {result.status.value}")
    eps = mp.mpf("1e-3")

    def seq(x: mp.mpf) -> mp.mpf:
        return (x**2 + 1) * mp.exp(-(((x**2 - 4) / eps) ** 2)) / (eps * mp.sqrt(mp.pi))

    truth = mp.quad(seq, [-2.2, -2, -1.8]) + mp.quad(seq, [1.8, 2, 2.2])
    got = complex(parse_latex(result.result_latex).evalf(20))
    return _verdict(
        "correct", True, abs(got - complex(truth)) < 1e-4, "result vs regularised delta sequence"
    )


def _delta_repeated_root(engine: Engine) -> tuple[str, str]:
    result = _delta(r"\operatorname{DiracDelta}(x^2)", "-1", "1")(engine)
    return _verdict(
        "refused",
        result.status.value == "ok",
        None,
        f"status {result.status.value}, reason {result.reason}",
    )


def _delta_on_the_limit(engine: Engine) -> tuple[str, str]:
    result = _delta(r"\operatorname{DiracDelta}(x)", "0", "1")(
        engine
    )  # half, all or nothing: only the user decides
    return _verdict(
        "refused",
        result.status.value == "ok",
        None,
        f"status {result.status.value}, reason {result.reason}",
    )


def _substitution(engine: Engine) -> tuple[str, str]:
    """x^2 + y with the premise y = 3x: the result at x = 2 is 4 + 6, by plain arithmetic."""
    step = engine.substitute(_expr("x^2+y"), ["y = 3x"])
    if step.status.value != "ok" or not step.result_latex:
        return _verdict("correct", False, None, f"status {step.status.value}")
    x = sp.Symbol("x")
    value = parse_latex(step.result_latex).subs(x, 2)
    return _verdict(
        "correct",
        True,
        value == 10 and step.evidence == "conditional",
        "value at x = 2 and conditional evidence",
    )


def _substitution_order(engine: Engine) -> tuple[str, str]:
    """x = y and y = 2 applied to x + y: the order changes the result (4 versus y + 2), so Minerva must ask."""
    x, y = sp.symbols("x y")
    in_order = (x + y).subs(x, y).subs(y, 2)
    at_once = (x + y).subs({x: y, y: 2}, simultaneous=True)
    if in_order == at_once:  # the oracle: the two orders disagree
        raise RuntimeError("oracle: the order does not matter, the problem is ill-posed")
    step = engine.substitute(_expr("x+y"), ["x = y", "y = 2"])
    return _verdict("refused", step.status.value == "ok", None, f"status {step.status.value}")


def _duffing(engine: Engine) -> tuple[str, str]:
    """Frequency of x'' + x + eps x^3 = 0 against the exact period from the quarter-period integral.

    The first correction 3 eps A^2 / 8 is Landau and Lifshitz, Mechanics, section 28; the exact period
    comes from energy conservation: T / 4 = integral from 0 to A of dx / sqrt(A^2 - x^2 + eps (A^4 - x^4) / 2).
    """
    report = engine.perturbation_solve(
        r"\frac{1}{2}\dot x^2 - \frac{1}{2}x^2 - \frac{\epsilon}{4}x^4",
        ["x"],
        assumptions={"epsilon": ["positive"]},
        small_parameter="epsilon",
        order=1,
    )
    if report.status.value != "ok" or not report.frequency_latex:
        return _verdict("correct", False, None, f"status {report.status.value}")
    eps, amp = mp.mpf("0.01"), mp.mpf(1)
    quarter = mp.quad(lambda s: 1 / mp.sqrt(amp**2 - s**2 + eps * (amp**4 - s**4) / 2), [0, amp])
    exact = 2 * mp.pi / (4 * quarter)
    rhs = report.frequency_latex.split("=", 1)[1]
    value = parse_latex(rhs).subs({sp.Symbol("epsilon"): sp.Rational(1, 100), sp.Symbol("A"): 1})
    # first order: the remaining error is of order eps^2
    return _verdict(
        "correct",
        True,
        abs(complex(value.evalf(20)) - complex(exact)) < 1e-4,
        "frequency vs exact period",
    )


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
        return _verdict("correct", False, None, f"conversion: {to_cyl.status.value}")
    lap = engine.vector_calculus(
        VectorInput(expression=r"\nabla^2 \mathbf{G}", coordinates="cylindrical",
                    fields=[_vector_field("G", to_cyl.components_text)])
    )  # fmt: skip
    if lap.status.value != "ok":
        return _verdict("correct", False, None, f"laplacian: {lap.status.value}")
    back = engine.vector_calculus(
        VectorInput(expression="H", coordinates="cylindrical", to="cartesian", action="convert",
                    fields=[_vector_field("H", lap.components_text)])
    )  # fmt: skip
    if back.status.value != "ok":
        return _verdict("correct", False, None, f"back-conversion: {back.status.value}")
    got = [sp.sympify(t, locals=sym) for t in back.components_text]
    agrees = all(sp.simplify(g - t) == 0 for g, t in zip(got, truth, strict=True))
    return _verdict(
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
            return _verdict("correct", False, None, f"{system}: {result.status.value}")
        values.append(result.result_text)
    return _verdict("correct", True, values == ["3", "3", "3"], "div r = 3")


def _point_charge_field(engine: Engine) -> tuple[str, str]:
    """div(r_vec/r^3) = 0 away from the origin, with r != 0 stated (the origin carries the delta)."""
    result = engine.vector_calculus(
        VectorInput(expression=r"\nabla\cdot\left(\frac{\mathbf{r}}{r^{3}}\right)",
                    coordinates="spherical")
    )  # fmt: skip
    if result.status.value != "ok":
        return _verdict("correct", False, None, f"status {result.status.value}")
    agrees = result.result_text == "0" and r"r \neq 0" in result.restrictions_latex
    return _verdict("correct", True, agrees, "zero away from the origin; origin excluded")


def _div_curl_without_smoothness(engine: Engine) -> tuple[str, str]:
    """div curl A = 0 needs equal mixed partials: with no smoothness declared nothing may be shown."""
    result = engine.vector_calculus(
        VectorInput(expression=r"\nabla\cdot(\nabla\times\mathbf{A})", coordinates="spherical",
                    fields=[FieldDecl(name="A", kind="vector")])
    )  # fmt: skip
    return _verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _cross_of_scalars(engine: Engine) -> tuple[str, str]:
    """a x b is not defined for scalars: it must not be read as a product."""
    result = engine.vector_calculus(
        VectorInput(expression=r"a \times b", fields=[FieldDecl(name="a", kind="scalar"),
                                                         FieldDecl(name="b", kind="scalar")])
    )  # fmt: skip
    return _verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _em_request(
    template_id: str, premises: list[str] | None = None, smoothness: str | None = "c2"
) -> Any:
    """The request the `electromagnetism` tool makes, with the handbook's identities."""
    from minerva_knowledge import KnowledgeBase
    from minerva_knowledge.identities import vector_identity_rules

    from .vector import Premise
    from .vector.em import TEMPLATES

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
        return _verdict("correct", False, None, f"status {result.status.value}")
    E, _B, eps, mu, _omega, v = _plane_wave()
    x, y, z, t = v["x"], v["y"], v["z"], v["t"]
    lap = [sum(sp.diff(e, a, 2) for a in (x, y, z)) for e in E]
    residual = [sp.simplify(lap[i] - mu * eps * sp.diff(E[i], t, 2)) for i in range(3)]
    expected = r"\nabla^{2} \mathbf{E} = \epsilon_{0} \mu_{0} \frac{\partial^{2} \mathbf{E}}{\partial t^{2}}"
    agrees = result.equations_latex == [expected] and all(r == 0 for r in residual)
    return _verdict("correct", True, agrees, "textbook form; plane wave satisfies it")


def _em_poynting(engine: Engine) -> tuple[str, str]:
    """div S + du/dt = 0 for a plane wave (J = 0), by plain SymPy, and the engine derives the theorem."""
    result = engine.vector_calculus(_em_request("poynting"))
    if result.status.value != "ok":
        return _verdict("correct", False, None, f"status {result.status.value}")
    E, B, eps, mu, _omega, v = _plane_wave()
    x, y, z, t = v["x"], v["y"], v["z"], v["t"]
    cross = (
        E[1] * B[2] - E[2] * B[1],
        E[2] * B[0] - E[0] * B[2],
        E[0] * B[1] - E[1] * B[0],
    )
    div_s = sum(sp.diff(c / mu, a) for c, a in zip(cross, (x, y, z), strict=True))
    u = eps * sum(e**2 for e in E) / 2 + sum(b**2 for b in B) / (2 * mu)
    agrees = sp.simplify(div_s + sp.diff(u, t)) == 0 and result.evidence == "conditional"
    return _verdict(
        "correct", True, agrees, "plane wave conserves energy; derivation is conditional"
    )


def _em_gauge(engine: Engine) -> tuple[str, str]:
    """E and B are invariant under A -> A + grad chi, phi -> phi - dchi/dt (plain SymPy on explicit
    potentials), and Minerva derives it as an identity with no premise."""
    result = engine.vector_calculus(_em_request("gauge_invariance", []))
    if result.status.value != "ok":
        return _verdict("correct", False, None, f"status {result.status.value}")
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
    agrees = all(sp.simplify(p - q) == 0 for p, q in zip(e, e2, strict=True)) and (
        result.evidence == "identity" and not result.given
    )
    return _verdict("correct", True, agrees, "E invariant on explicit potentials; an identity")


def _em_missing_premise(engine: Engine) -> tuple[str, str]:
    """Without Gauss's law as a premise, continuity cannot follow: refuse and ask, never assume."""
    from .vector.em import TEMPLATES

    needs = TEMPLATES["continuity"].needs
    result = engine.vector_calculus(_em_request("continuity", [needs[0]]))
    return _verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _em_flipped_sign(engine: Engine) -> tuple[str, str]:
    """A premise with the wrong sign is not the one the derivation needs (it would give the wrong
    wave equation): refused as a missing premise."""
    result = engine.vector_calculus(
        _em_request(
            "wave_equation_e",
            ["div(E) = 0", "curl(E) = dt(B)", "curl(B) = mu0*epsilon0*dt(E)"],
        )
    )
    return _verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _integral_request(**kw: Any) -> Any:
    from .vector import Curve, Param, Region, Surface

    del Curve, Param, Region, Surface
    return VectorInput(format="text", action="integrate", **kw)


def _int_flux(engine: Engine) -> tuple[str, str]:
    """The flux of r-hat / r^2 through a sphere is 4 pi (Gauss, textbook), whatever the radius."""
    from .vector import Param, Surface

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
        return _verdict("correct", False, None, f"status {result.status.value}")
    agrees = result.integral.values[0].text == "4*pi"
    return _verdict("correct", True, agrees, "4 pi, as Gauss's law gives")


def _int_line_against_mpmath(engine: Engine) -> tuple[str, str]:
    """A line integral along a helix, against nested mpmath quadrature written by hand in Cartesian form
    (`F . r'(s)` with r' written out: none of Minerva's integral code is used)."""
    from .vector import Curve, Param

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
        return _verdict("correct", False, None, f"status {result.status.value}")

    def integrand(s: Any) -> Any:
        x, y, z = mp.cos(s), mp.sin(s), s / 3
        dx, dy, dz = -mp.sin(s), mp.cos(s), mp.mpf(1) / 3
        return (-(y**2)) * dx + (x * z) * dy + (x * y + z) * dz

    truth = mp.quad(integrand, [0, 2 * mp.pi])
    value = result.integral.values[0]
    got = mp.mpf(value.numeric) if value.numeric else mp.mpf(sp.N(sp.sympify(value.text), 20))
    return _verdict("correct", True, abs(got - truth) < 1e-9, "helix, nested mpmath quadrature")


def _int_pole_refused(engine: Engine) -> tuple[str, str]:
    """The divergence theorem needs F defined on the whole ball; 1/r^2 has a pole at its centre (the
    true flux is 4 pi while div F = 0 elsewhere): refuse rather than report two unequal sides."""
    from .vector import Region

    result = engine.vector_calculus(
        _integral_request(
            expression="F",
            integral="divergence_theorem",
            coordinates="spherical",
            fields=[FieldDecl(name="F", kind="vector", components=["1/r**2", "0", "0"])],
            region=Region(bounds=[("0", "2"), ("0", "pi"), ("0", "2*pi")]),
        )
    )
    return _verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _estat_request(problem: Any) -> Any:
    from .vector import VectorInput

    return VectorInput(expression="", format="text", action="em_compute", em=problem)


def _estat_centre(result: Any) -> Any:
    if result.status.value != "ok" or result.em is None:
        return None
    item = next((i for i in result.em.items if i.label.startswith("phi(0.5, 0.5)")), None)
    return mp.mpf(item.numeric) if item is not None and item.numeric else None


def _estat_laplace(engine: Engine) -> tuple[str, str]:
    """phi(x, y) = sin(pi x) sinh(pi y) / sinh(pi) solves Laplace's equation with phi = sin(pi x) on the top
    edge and 0 on the others: the oracle is that closed form at the centre, 1 / (2 cosh(pi / 2))."""
    from .vector import BoundaryProblem

    result = engine.vector_calculus(
        _estat_request(BoundaryProblem(kind="boundary_value", a="1", b="1", top="sin(pi*x)"))
    )
    got = _estat_centre(result)
    truth = 1 / (2 * mp.cosh(mp.pi / 2))
    return _verdict(
        "correct", got is not None, got is not None and abs(got - truth) < 1e-6, "centre value"
    )


def _estat_poisson(engine: Engine) -> tuple[str, str]:
    """laplacian(phi) = -1 on the unit square with zero edges: the double sine series summed by mpmath
    (odd p, q only), written by hand, not the engine's coefficients."""
    from .vector import BoundaryProblem

    result = engine.vector_calculus(
        _estat_request(BoundaryProblem(kind="boundary_value", a="1", b="1", source="1"))
    )
    got = _estat_centre(result)
    truth = sum(
        16 * (-1) ** ((p - 1) // 2 + (q - 1) // 2) / (mp.pi**4 * p * q * (p * p + q * q))
        for p in range(1, 400, 2)
        for q in range(1, 400, 2)
    )
    return _verdict(
        "correct", got is not None, got is not None and abs(got - truth) < 3e-5, "centre value"
    )


def _estat_point_charge_refused(engine: Engine) -> tuple[str, str]:
    from .vector import FieldProblem, PointCharge

    result = engine.vector_calculus(
        _estat_request(
            FieldProblem(
                kind="field",
                charges=[PointCharge(q="1", position=("0", "0", "0"))],
                point=("0", "0", "0"),
            )
        )
    )
    return _verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _field_tensor_request(quantity: str, components: list[str]) -> Any:
    from .geometry import GeometryInput
    from .geometry.models import TensorField

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
    from .geometry import geometry

    ex, ey, ez, bx, by, bz = 4, 5, 6, 1, 2, 3
    f = [
        [0, -ex, -ey, -ez],
        [ex, 0, bz, -by],
        [ey, -bz, 0, bx],
        [ez, by, -bx, 0],
    ]
    result = geometry(
        _field_tensor_request("field_invariant", [str(v) for row in f for v in row]),
        Limits(timeout_seconds=60),
    )
    q = result.quantities[0]
    ok = q.status.value == "ok" and q.scalar_text is not None
    return _verdict("correct", ok, ok and sp.sympify(q.scalar_text) == -126, "invariant")


def _field_not_antisymmetric_refused(engine: Engine) -> tuple[str, str]:
    from .geometry import geometry

    result = geometry(
        _field_tensor_request("field_invariant", ["0", "1", "0", "0", "1", *["0"] * 11]),
        Limits(timeout_seconds=60),
    )
    return _verdict("refused", result.quantities[0].status.value == "ok", None, "symmetric input")


def _statmech(case_id: str) -> Judge:
    """A problem of the statistical-mechanics corpus (its references are plain arithmetic)."""

    def judge(engine: Engine) -> tuple[str, str]:
        from .statmech.benchmark import CASES, judge_case

        return judge_case(next(c for c in CASES if c.id == case_id), engine)

    return judge


PROBLEMS: list[Problem] = [
    Problem(
        "matrix-exp-product-noncommuting",
        "matrix",
        "Higham, Functions of Matrices (matrix exponential)",
        "correct",
        _exp_product,
    ),
    Problem(
        "matrix-exp-merge-refused",
        "matrix",
        "Higham, Functions of Matrices (matrix exponential: exp(A)exp(B) = exp(A+B) needs commuting A, B in general)",
        "refused",
        _exp_merge_refused,
    ),
    Problem(
        "matrix-frechet-derivative",
        "matrix",
        "Higham, Functions of Matrices (Frechet derivative of the exponential)",
        "correct",
        _frechet,
    ),
    Problem(
        "delta-composition-two-roots",
        "distribution",
        "Gel'fand and Shilov, Generalized Functions vol. 1 (delta of a function)",
        "correct",
        _delta_composition,
    ),
    Problem(
        "delta-repeated-root-refused",
        "distribution",
        "Gel'fand and Shilov, Generalized Functions vol. 1 (delta of a function with a repeated root is not defined)",
        "refused",
        _delta_repeated_root,
    ),
    Problem(
        "delta-on-the-limit-refused",
        "distribution",
        "convention of the integral at an endpoint (ambiguous without a stated one)",
        "refused",
        _delta_on_the_limit,
    ),
    Problem("substitution-with-premise", "substitution", "arithmetic", "correct", _substitution),
    Problem(
        "substitution-order-refused",
        "substitution",
        "arithmetic (the two orders differ)",
        "refused",
        _substitution_order,
    ),
    Problem(
        "duffing-frequency-shift",
        "mechanics",
        "Landau and Lifshitz, Mechanics, section 28 (anharmonic oscillations)",
        "correct",
        _duffing,
    ),
    Problem(
        "vector-laplacian-across-coordinates",
        "vector",
        "Arfken and Weber, Mathematical Methods for Physicists (vector Laplacian; Cartesian components)",
        "correct",
        _laplacian_across_coordinates,
    ),
    Problem(
        "divergence-of-position-vector",
        "vector",
        "Griffiths, Introduction to Electrodynamics, chapter 1 (div r = 3)",
        "correct",
        _div_position,
    ),
    Problem(
        "point-charge-field-away-from-origin",
        "vector",
        "Griffiths, Introduction to Electrodynamics, section 1.5 (divergence of r-hat / r^2)",
        "correct",
        _point_charge_field,
    ),
    Problem(
        "div-curl-without-smoothness-refused",
        "vector",
        "Marsden and Tromba, Vector Calculus (div curl = 0 for C^2 fields)",
        "refused",
        _div_curl_without_smoothness,
    ),
    Problem(
        "cross-product-of-scalars-refused",
        "vector",
        "definition of the cross product (two vectors)",
        "refused",
        _cross_of_scalars,
    ),
    Problem(
        "em-wave-equation-from-maxwell",
        "electromagnetism",
        "Griffiths, Introduction to Electrodynamics, chapter 9 (wave equation in vacuum; plane waves)",
        "correct",
        _em_wave,
    ),
    Problem(
        "em-poynting-theorem-plane-wave",
        "electromagnetism",
        "Griffiths, Introduction to Electrodynamics, chapter 8 (Poynting's theorem)",
        "correct",
        _em_poynting,
    ),
    Problem(
        "em-gauge-invariance-of-the-fields",
        "electromagnetism",
        "Griffiths, Introduction to Electrodynamics, chapter 10 (gauge transformations)",
        "correct",
        _em_gauge,
    ),
    Problem(
        "em-continuity-without-gauss-refused",
        "electromagnetism",
        "Griffiths, Introduction to Electrodynamics, chapter 7 (continuity needs Gauss's law and Ampere-Maxwell)",
        "refused",
        _em_missing_premise,
    ),
    Problem(
        "em-wave-equation-flipped-faraday-refused",
        "electromagnetism",
        "Griffiths, Introduction to Electrodynamics, chapter 7 (the sign of Faraday's law)",
        "refused",
        _em_flipped_sign,
    ),
    Problem(
        "integral-flux-inverse-square",
        "integral",
        "Griffiths, Introduction to Electrodynamics, section 1.3 (flux of r-hat over r squared)",
        "correct",
        _int_flux,
    ),
    Problem(
        "integral-line-against-mpmath",
        "integral",
        "direct quadrature of F . dr along the helix (mpmath), written by hand",
        "correct",
        _int_line_against_mpmath,
    ),
    Problem(
        "integral-divergence-theorem-pole-refused",
        "integral",
        "Griffiths, Introduction to Electrodynamics, section 1.5 (the delta function at the origin)",
        "refused",
        _int_pole_refused,
    ),
    Problem(
        "estat-laplace-closed-form",
        "electrostatics",
        "separation of variables on a rectangle, Griffiths, Introduction to Electrodynamics, section 3.3",
        "correct",
        _estat_laplace,
    ),
    Problem(
        "estat-poisson-double-series",
        "electrostatics",
        "double sine series, summed by mpmath written by hand (Griffiths, section 3.3)",
        "correct",
        _estat_poisson,
    ),
    Problem(
        "estat-field-at-a-point-charge-refused",
        "electrostatics",
        "Coulomb's law is singular at the charge (Griffiths, section 2.1)",
        "refused",
        _estat_point_charge_refused,
    ),
    Problem(
        "field-tensor-invariant",
        "electromagnetic tensor",
        "Jackson, Classical Electrodynamics, section 11.9 (F_{mu nu}F^{mu nu} = 2(B^2 - E^2/c^2))",
        "correct",
        _field_invariant_numbers,
    ),
    Problem(
        "field-tensor-symmetric-refused",
        "electromagnetic tensor",
        "a field strength is antisymmetric (Jackson, section 11.9)",
        "refused",
        _field_not_antisymmetric_refused,
    ),
    Problem(
        "statmech-two-level-thermodynamics",
        "statmech",
        "closed form Z = 1 + e^(-beta eps), plain arithmetic",
        "correct",
        _statmech("two-level-thermodynamics"),
    ),
    Problem(
        "statmech-quantum-oscillator",
        "statmech",
        "geometric series: Z = 1/(2 sinh(x/2)), Einstein heat capacity",
        "correct",
        _statmech("quantum-harmonic-oscillator"),
    ),
    Problem(
        "statmech-ideal-gas-pressure-and-entropy",
        "statmech",
        "Z = Z_1^N/N! with the exact ln N!: pV = NkT, entropy via lgamma",
        "correct",
        _statmech("classical-ideal-gas"),
    ),
    Problem(
        "statmech-divergent-series",
        "statmech",
        "E_n = -n: the terms grow like e^(beta n)",
        "refused",
        _statmech("divergent-partition-function"),
    ),
    Problem(
        "statmech-indistinguishability-is-a-choice",
        "statmech",
        "N! is a convention only the user chooses",
        "refused",
        _statmech("indistinguishability-is-the-users-choice"),
    ),
]


def run_problems(engine: Engine | None = None) -> list[dict[str, Any]]:
    engine = engine or Engine(Limits(timeout_seconds=60, max_concurrency=2))
    rows: list[dict[str, Any]] = []
    for p in PROBLEMS:
        start = time.monotonic()
        try:
            outcome, detail = p.judge(engine)
        except Exception as exc:  # a crash is a result: it must never be read as a pass
            outcome, detail = "wrong", f"{type(exc).__name__} while judging"
        rows.append(
            {
                "id": p.id,
                "category": p.category,
                "source": p.source,
                "expected": p.expected,
                "outcome": outcome,
                "detail": detail,
                "latency_ms": int((time.monotonic() - start) * 1000),
            }
        )
    return rows


def deterministic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: r[k] for k in ("id", "category", "source", "expected", "outcome")} for r in rows]


def markdown(rows: list[dict[str, Any]]) -> str:
    good = {"correct", "refused_as_expected"}
    lines = [
        "# Reference problems",
        "",
        "Generated by `python -m minerva_engine.reference_benchmark`; outcomes are deterministic, latencies are not gated.",
        "",
        f"{sum(r['outcome'] in good for r in rows)} of {len(rows)} as expected; "
        f"wrong: {sum(r['outcome'] == 'wrong' for r in rows)}; "
        f"unsupported where an answer is expected: {sum(r['outcome'] == 'unsupported_regression' for r in rows)}.",
        "",
        "| Problem | Category | Expected | Outcome | Source | ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{r['id']}` | {r['category']} | {r['expected']} | {r['outcome']} | {r['source']} | {r['latency_ms']} |"
        for r in rows
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[4]
    json_path, md_path = (
        root / "docs/benchmarks/reference.json",
        root / "docs/reference-benchmark.md",
    )
    rows = run_problems()
    bad = [r for r in rows if r["outcome"] in ("wrong", "unsupported_regression")]
    if "--check" in argv:
        expected = json.loads(json_path.read_text())["problems"] if json_path.exists() else None
        if expected != deterministic(rows):
            print("reference outcomes differ from docs/benchmarks/reference.json", file=sys.stderr)
            return 1
    else:
        json_path.write_text(json.dumps({"problems": deterministic(rows)}, indent=2) + "\n")
        md_path.write_text(markdown(rows))
    for r in bad:
        print(f"{r['id']}: {r['outcome']} ({r['detail']})", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
