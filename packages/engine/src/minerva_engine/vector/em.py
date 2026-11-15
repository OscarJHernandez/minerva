"""Named electromagnetic derivations, built from general steps and the premises the user supplies.

A derivation here is not a stored answer. A template names the *task*: which equations it needs as
premises, which expressions to start from and reach, and how the final equation follows from the routes
found (`L - R = sum of coefficient * (start - target)` over the routes, an algebraic fact checked by
expansion). Each route is a chain found by the general search in `steps.py` over the vector identities
of the handbook and the supplied premises, and every identity step is verified on its own. A premise is
matched exactly: the user supplies the field equations, and if a needed one is missing the derivation is
refused with the list of what is needed.

The final equation is then evaluated on exact solutions of the premises (`models.py` style, Cartesian
only): plane waves in vacuum, fields built from random potentials together with the sources they imply, a
harmonic potential. Agreement is "consistent with N exact solutions", not a verification; disagreement
withholds the result. A result that used a premise is conditional on it and is never called an identity;
only a derivation that needs no premise (gauge invariance of the fields) is an identity.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import mpmath as mp
import sympy as sp

from ..geometry.numeric import close
from ..models import Status
from .context import Context, Refused
from .expressions import kind_of
from .models import ModelCheck, VectorDiagnostics, VectorResult, VectorStep
from .numeric import Evaluator, Setup, Skip, random_field
from .rules import _dot_sorted
from .steps import ParsedPremise, _premises, _rules, find_chain, split_equation

MODEL_COUNT = 2
POINTS_PER_MODEL = 4
TOL = 1e-8


@dataclass(frozen=True)
class Result:
    lhs: str
    rhs: str
    combination: tuple[
        tuple[str, int], ...
    ]  # (coefficient, route): lhs - rhs = sum c * (start - target)


@dataclass(frozen=True)
class Template:
    id: str
    title: str
    summary: str
    fields: tuple[
        tuple[str, str], ...
    ]  # (name, kind): every field is declared as a function of t too
    constants: tuple[str, ...]
    smooth: tuple[str, ...]  # fields the user must declare twice differentiable
    needs: tuple[str, ...]  # premise equations, text syntax
    routes: tuple[tuple[str, str], ...]  # (start, target)
    results: tuple[Result, ...]
    model: str


TEMPLATES: dict[str, Template] = {
    t.id: t
    for t in [
        Template(
            "wave_equation_e",
            "Wave equation for the electric field in vacuum",
            "curl curl E two ways: through Faraday and Ampere-Maxwell, and through the curl-curl "
            "identity and Gauss's law",
            (("E", "vector"), ("B", "vector")),
            ("mu0", "epsilon0"),
            ("E", "B"),
            ("div(E) = 0", "curl(E) = -dt(B)", "curl(B) = mu0*epsilon0*dt(E)"),
            (
                ("curl(curl(E))", "-mu0*epsilon0*dt(dt(E))"),
                ("curl(curl(E))", "-lap(E)"),
            ),
            (Result("lap(E)", "mu0*epsilon0*dt(dt(E))", (("1", 1), ("-1", 0))),),
            "vacuum_waves",
        ),
        Template(
            "wave_equation_b",
            "Wave equation for the magnetic field in vacuum",
            "curl curl B two ways: through Ampere-Maxwell and Faraday, and through the curl-curl "
            "identity and Gauss's law for magnetism",
            (("E", "vector"), ("B", "vector")),
            ("mu0", "epsilon0"),
            ("E", "B"),
            ("div(B) = 0", "curl(B) = mu0*epsilon0*dt(E)", "curl(E) = -dt(B)"),
            (
                ("curl(curl(B))", "-mu0*epsilon0*dt(dt(B))"),
                ("curl(curl(B))", "-lap(B)"),
            ),
            (Result("lap(B)", "mu0*epsilon0*dt(dt(B))", (("1", 1), ("-1", 0))),),
            "vacuum_waves",
        ),
        Template(
            "continuity",
            "Charge conservation from Gauss's law and the Ampere-Maxwell law",
            "div curl B two ways: it is zero, and through Ampere-Maxwell and Gauss's law it is "
            "mu0 (div J + d rho / dt)",
            (("E", "vector"), ("B", "vector"), ("J", "vector"), ("rho", "scalar")),
            ("mu0", "epsilon0"),
            ("E", "B"),
            ("curl(B) = mu0*J + mu0*epsilon0*dt(E)", "div(E) = rho/epsilon0"),
            (
                ("div(curl(B))", "0"),
                ("div(curl(B))", "mu0*(div(J) + dt(rho))"),
            ),
            (Result("div(J) + dt(rho)", "0", (("1/mu0", 0), ("-1/mu0", 1))),),
            "sourced",
        ),
        Template(
            "poynting",
            "Poynting's theorem from Faraday's and the Ampere-Maxwell law",
            "the divergence of E x B / mu0 through the product rule and the two laws, and the time "
            "derivative of the energy density through the product rule; the two are assembled",
            (("E", "vector"), ("B", "vector"), ("J", "vector")),
            ("mu0", "epsilon0"),
            ("E", "B"),
            ("curl(E) = -dt(B)", "curl(B) = mu0*J + mu0*epsilon0*dt(E)"),
            (
                (
                    "div(cross(E, B))/mu0",
                    "-dot(J, E) - epsilon0*dot(E, dt(E)) - dot(B, dt(B))/mu0",
                ),
                (
                    "dt(epsilon0*dot(E, E)/2 + dot(B, B)/(2*mu0))",
                    "epsilon0*dot(E, dt(E)) + dot(B, dt(B))/mu0",
                ),
            ),
            (
                Result(
                    "div(cross(E, B))/mu0 + dt(epsilon0*dot(E, E)/2 + dot(B, B)/(2*mu0)) + dot(J, E)",
                    "0",
                    (("1", 0), ("1", 1)),
                ),
            ),
            "sourced",
        ),
        Template(
            "gauge_invariance",
            "Gauge invariance of E and B",
            "the fields built from the transformed potentials equal those built from the original "
            "ones: an identity (no premise), for a twice differentiable gauge function",
            (("phi", "scalar"), ("A", "vector"), ("chi", "scalar")),
            (),
            ("phi", "A", "chi"),
            (),
            (
                ("curl(A + grad(chi))", "curl(A)"),
                ("-grad(phi - dt(chi)) - dt(A + grad(chi))", "-grad(phi) - dt(A)"),
            ),
            (
                Result("curl(A + grad(chi))", "curl(A)", (("1", 0),)),
                Result(
                    "-grad(phi - dt(chi)) - dt(A + grad(chi))", "-grad(phi) - dt(A)", (("1", 1),)
                ),
            ),
            "potentials",
        ),
        Template(
            "poisson",
            "Poisson's equation from Gauss's law and the electrostatic potential",
            "the Laplacian of the potential as the divergence of its gradient, with E = -grad phi "
            "and Gauss's law",
            (("phi", "scalar"), ("E", "vector"), ("rho", "scalar")),
            ("epsilon0",),
            ("phi", "E"),
            ("div(E) = rho/epsilon0", "grad(phi) = -E"),
            (("lap(phi)", "-rho/epsilon0"),),
            (Result("lap(phi)", "-rho/epsilon0", (("1", 0),)),),
            "poisson",
        ),
        Template(
            "laplace",
            "Laplace's equation from Gauss's law and the electrostatic potential",
            "as Poisson's equation in a charge-free region",
            (("phi", "scalar"), ("E", "vector")),
            (),
            ("phi", "E"),
            ("div(E) = 0", "grad(phi) = -E"),
            (("lap(phi)", "0"),),
            (Result("lap(phi)", "0", (("1", 0),)),),
            "laplace",
        ),
    ]
}

# What Minerva cannot derive yet, so that no suggestion promises it.
PLANNED = (
    "wave equations for the potentials in the Lorenz gauge",
    "interface (boundary) conditions by a pillbox argument (needs the integral theorems)",
    "Maxwell's equations in matter with a general constitutive relation",
)


def describe(template: Template) -> dict[str, Any]:
    return {
        "id": template.id,
        "title": template.title,
        "summary": template.summary,
        "fields": [{"name": n, "kind": k, "depends_on": "x, y, z, t"} for n, k in template.fields],
        "constants": list(template.constants),
        "smooth": list(template.smooth),
        "needs": list(template.needs),
        "premise_free": not template.needs,
    }


# -- exact solutions -----------------------------------------------------------------------------
Model = tuple[dict[str, sp.Expr], dict[sp.Symbol, sp.Rational]]  # (components by name, constants)


def _rat(rng: random.Random, lo: int, hi: int, den: int = 4) -> sp.Rational:
    return sp.Rational(rng.randint(lo, hi), den)


def _components(name: str, vector: tuple[sp.Expr, sp.Expr, sp.Expr]) -> dict[str, sp.Expr]:
    return {f"{name}_{c}": e for c, e in zip(("x", "y", "z"), vector, strict=True)}


def _curl(a: tuple[sp.Expr, ...], v: tuple[sp.Symbol, ...]) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    return (
        sp.diff(a[2], v[1]) - sp.diff(a[1], v[2]),
        sp.diff(a[0], v[2]) - sp.diff(a[2], v[0]),
        sp.diff(a[1], v[0]) - sp.diff(a[0], v[1]),
    )


def _build_model(kind: str, ctx: Context, template: Template, rng: random.Random) -> Model:
    x, y, z = ctx.low.q
    t = ctx.low.t
    consts = ctx.low.constants
    fields: dict[str, sp.Expr] = {}
    params: dict[sp.Symbol, sp.Rational] = {}
    if kind == "vacuum_waves":
        eps = _rat(rng, 4, 12, 4)
        c = _rat(rng, 4, 12, 4)
        params[consts["epsilon0"]] = eps
        params[consts["mu0"]] = 1 / (eps * c**2)
        k = (_rat(rng, 1, 6), _rat(rng, -6, -1), _rat(rng, 1, 6))
        u = (_rat(rng, 1, 5), _rat(rng, 1, 5), _rat(rng, -5, -1))
        e0 = (
            k[1] * u[2] - k[2] * u[1],
            k[2] * u[0] - k[0] * u[2],
            k[0] * u[1] - k[1] * u[0],
        )
        knorm = sp.sqrt(sum(a**2 for a in k))
        omega = c * knorm
        phase = k[0] * x + k[1] * y + k[2] * z - omega * t + _rat(rng, 0, 7)
        kxe = (
            k[1] * e0[2] - k[2] * e0[1],
            k[2] * e0[0] - k[0] * e0[2],
            k[0] * e0[1] - k[1] * e0[0],
        )
        fields |= _components("E", tuple(a * sp.cos(phase) for a in e0))
        fields |= _components("B", tuple(a / omega * sp.cos(phase) for a in kxe))
        for name in ("J", "rho"):
            if any(n == name for n, _ in template.fields):
                fields |= (
                    _components(name, (sp.Integer(0),) * 3)
                    if name == "J"
                    else {name: sp.Integer(0)}
                )
        return fields, params
    spatial = (x, y, z)
    if kind in ("sourced", "potentials"):
        phi = random_field([x, y, z, t], rng)
        a = tuple(random_field([x, y, z, t], rng) for _ in range(3))
        if kind == "potentials":
            fields["phi"] = phi
            fields |= _components("A", a)
            fields["chi"] = random_field([x, y, z, t], rng)
            return fields, params
        eps, mu = _rat(rng, 2, 9, 3), _rat(rng, 2, 9, 3)
        params[consts["epsilon0"]] = eps
        params[consts["mu0"]] = mu
        e = tuple(-sp.diff(phi, v) - sp.diff(ai, t) for v, ai in zip(spatial, a, strict=True))
        b = _curl(a, spatial)
        curl_b = _curl(b, spatial)
        fields |= _components("E", e)
        fields |= _components("B", b)
        fields["rho"] = eps * sum(sp.diff(ei, v) for ei, v in zip(e, spatial, strict=True))
        fields |= _components(
            "J",
            tuple(curl_b[i] / mu - eps * sp.diff(e[i], t) for i in range(3)),
        )
        return fields, params
    # electrostatics
    if "epsilon0" in consts:
        params[consts["epsilon0"]] = _rat(rng, 2, 9, 3)
    if kind == "poisson":
        phi = random_field([x, y, z], rng)
        fields["phi"] = phi
        e = tuple(-sp.diff(phi, v) for v in spatial)
        fields |= _components("E", e)
        fields["rho"] = -params[consts["epsilon0"]] * sum(sp.diff(phi, v, 2) for v in spatial)
        return fields, params
    harmonic = [x**2 - y**2, x * y, z, x**2 + y**2 - 2 * z**2, x * y * z, x * z, y * z]
    phi = sum((_rat(rng, -5, 5) * h for h in rng.sample(harmonic, 3)), sp.Integer(0))
    fields["phi"] = phi
    fields |= _components("E", tuple(-sp.diff(phi, v) for v in spatial))
    return fields, params


def _evaluate_models(
    ctx: Context, template: Template, equations: list[tuple[sp.Expr, sp.Expr]], seed: int
) -> ModelCheck:
    if ctx.system.name != "cartesian":
        return ModelCheck(
            status="not_run",
            kind=template.model,
            note="the exact solutions are built in Cartesian coordinates; none was run here",
        )
    rng = random.Random(seed)  # noqa: S311  (sampling, not security)
    checked = worst_points = 0
    worst = 0.0
    with mp.workdps(60):
        for _ in range(MODEL_COUNT):
            try:
                fields, params = _build_model(template.model, ctx, template, rng)
            except KeyError as exc:
                return ModelCheck(
                    status="not_run",
                    kind=template.model,
                    note=f"a constant the model needs was not declared: {exc}",
                )
            # fields the request declared but the model does not set are never read by these equations
            for d in ctx.low.declared.values():
                for comp in d.comps:
                    fields.setdefault(comp, sp.Integer(0))
            setup = Setup(ctx.low, fields, params)
            attempts = got = 0
            while got < POINTS_PER_MODEL and attempts < 30:
                attempts += 1
                q = ctx.system.sample(rng)
                t_value = mp.mpf(rng.uniform(0.1, 1.0))
                ev = Evaluator(setup, ctx.system, t_value)
                X = mp.matrix([ev._x(*q)[i] for i in range(3)])
                try:
                    for left, right in equations:
                        a = ev.value(left.subs(params), X)
                        b = ev.value(right.subs(params), X) if right != 0 else (a * 0)
                        pairs = (
                            [(a[i], b[i]) for i in range(3)]
                            if isinstance(a, mp.matrix)
                            else [(a, b)]
                        )
                        scale = max([1, *[abs(p) for p, _ in pairs], *[abs(r) for _, r in pairs]])
                        for p, r in pairs:
                            worst = max(worst, float(abs(p - r) / scale))
                            if not close(p, r, TOL, scale):
                                return ModelCheck(
                                    status="failed",
                                    kind=template.model,
                                    models=checked,
                                    points=worst_points,
                                    max_residual=worst,
                                    note="the derived equation disagrees with an exact solution "
                                    "of the premises",
                                )
                except (Skip, ValueError, ZeroDivisionError, TypeError, OverflowError):
                    continue
                got += 1
                worst_points += 1
            checked += 1
    if worst_points < MODEL_COUNT * POINTS_PER_MODEL // 2:
        return ModelCheck(
            status="not_run",
            kind=template.model,
            note="too few usable sample points to say anything",
        )
    return ModelCheck(
        status="passed",
        kind=template.model,
        models=checked,
        points=worst_points,
        max_residual=worst,
        note=f"consistent with {checked} exact solutions of the premises at {worst_points} points; "
        "this is not a verification",
    )


# -- the derivation ------------------------------------------------------------------------------
def _missing_premises(ctx: Context, template: Template, supplied: list[ParsedPremise]) -> list[str]:
    missing = []
    for need in template.needs:
        sides = split_equation(need)
        assert sides is not None  # noqa: S101  (templates are well formed)
        lhs, rhs = ctx.parse(sides[0]), ctx.parse(sides[1])
        if not any(p.lhs == lhs and p.rhs == rhs for p in supplied):
            missing.append(need)
    return missing


def _used(supplied: list[ParsedPremise], ctx: Context, template: Template) -> list[ParsedPremise]:
    out: list[ParsedPremise] = []
    for need in template.needs:
        sides = split_equation(need)
        assert sides is not None  # noqa: S101
        lhs, rhs = ctx.parse(sides[0]), ctx.parse(sides[1])
        out.extend(p for p in supplied if p.lhs == lhs and p.rhs == rhs)
    return out


def run_em(ctx: Context, result: VectorResult, diag: VectorDiagnostics) -> None:
    template = TEMPLATES.get(ctx.inp.template or "")
    if template is None:
        raise Refused(
            Status.UNSUPPORTED,
            f"unknown derivation {ctx.inp.template!r}; available: {', '.join(sorted(TEMPLATES))}",
            "unsupported_expression",
        )
    result.derivation = template.id
    declared = {n: d.decl.kind for n, d in ctx.low.declared.items()}
    absent = [n for n, k in template.fields if declared.get(n) != k]
    if absent:
        raise Refused(
            Status.PARSE_ERROR,
            f"this derivation needs the fields {', '.join(f'{n} ({k})' for n, k in template.fields)} "
            f"declared (missing or of the wrong kind: {', '.join(absent)})",
            "bad_declaration",
        )
    supplied = _premises(ctx)
    missing = _missing_premises(ctx, template, supplied)
    if missing:
        result.needs_premises = missing
        result.status = Status.CONDITIONS_NOT_SATISFIED
        result.message = (
            f"{template.title} needs these field equations as premises, and they were not supplied "
            "exactly as written: " + "; ".join(missing) + ". They are yours to adopt: Minerva "
            "does not assume them."
        )
        diag.reason = "needs_premises"
        return
    premises = _used(supplied, ctx, template)
    rules = _rules(ctx)
    chains = []
    all_steps: list[VectorStep] = []
    given: list[str] = []
    complete = True
    for i, (start_text, target_text) in enumerate(template.routes):
        start, target = ctx.parse(start_text), ctx.parse(target_text)
        chain = find_chain(
            ctx, start, target, rules, premises, action="em_derive", prefix=f"[route {i + 1}] "
        )
        complete = complete and chain.complete
        if chain.status is not Status.OK or chain.final is None:
            result.blocked = chain.blocked
            result.search_complete = complete
            result.steps = all_steps
            result.status = chain.status
            result.message = (
                f"route {i + 1} ({start_text} to {target_text}) failed: {chain.message}"
            )
            diag.reason = chain.reason or "not_found"
            return
        for step in chain.steps:
            step.chain = i
        all_steps.extend(chain.steps)
        for g in chain.given:
            if g not in given:
                given.append(g)
        chains.append((start, target))
    # the final equations follow from the routes by algebra: lhs - rhs = sum c_i (start_i - target_i)
    equations: list[tuple[sp.Expr, sp.Expr]] = []
    shown: list[str] = []
    for res in template.results:
        left, right = ctx.parse(res.lhs), ctx.parse(res.rhs)
        combo = sp.Integer(0)
        for coefficient, route in res.combination:
            start, target = chains[route]
            combo += ctx.parse(coefficient) * sp.Add(start, sp.Mul(-1, target))
        difference = sp.Add(left, sp.Mul(-1, right), sp.Mul(-1, combo))
        if _dot_sorted(sp.expand(difference)) != 0:
            result.status = Status.NOT_VERIFIED
            result.message = (
                "the routes do not combine into the stated equation (algebra check failed)"
            )
            diag.reason = "verification_failed"
            return
        equations.append((left, right))
        shown.append(f"{ctx.latex(left)} = {ctx.latex(right)}")
    model = _evaluate_models(ctx, template, equations, seed=20260514)
    result.model_check = model
    if model.status == "failed":
        result.status = Status.NOT_VERIFIED
        result.message = (
            "the derived equation disagrees with exact solutions of its own premises, so it is not "
            "shown: " + model.note
        )
        diag.reason = "model_check_failed"
        return
    result.steps = all_steps
    result.given = given
    result.evidence = "conditional" if given else "identity"
    result.equations_latex = shown
    result.result_kind = "scalar" if kind_of(equations[0][0]) == "scalar" else "vector"
    result.result_latex = shown[0] if len(shown) == 1 else r" \quad ".join(shown)
    result.result_text = "; ".join(f"{a} = {b}" for a, b in equations)
    result.search_complete = complete
    result.status = Status.OK
    result.message = (
        f"{template.title}: "
        + (
            "an identity (no premise was needed), "
            if not given
            else "conditional on "
            + "; ".join(given)
            + " (supplied by you, not checked by Minerva), "
        )
        + f"{len(all_steps)} verified step(s) over {len(template.routes)} route(s). "
        + (
            model.note
            if model.status == "passed"
            else "the check on exact solutions was not run: " + model.note
        )
    )


__all__ = ["PLANNED", "TEMPLATES", "Template", "describe", "run_em"]
