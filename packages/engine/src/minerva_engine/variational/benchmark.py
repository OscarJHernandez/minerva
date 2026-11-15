"""Reference corpus for the variational calculus engine, with closed forms from outside Minerva.

Every case states what a correct answer is *before* Minerva is asked: the Euler-Lagrange equation and
boundary coefficients written by hand, or an action value, a stationary path and a classification
that follow from an exact solution (checked here against `mpmath`). A case passes only if every one of
its checks does. The deterministic part (what each check expected, what was found to six digits, and
whether it agreed) is compared with the stored report by `--check`; times are measurements and are
never compared.

    uv run python -m minerva_engine.variational.benchmark [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

from ..models import Limits, Status
from .firstvar import first_variation
from .models import End, VariationInput, VariationReport
from .operation import variation
from .setup import Problem, read_problem

LIMITS = Limits(timeout_seconds=120)
TOL = mp.mpf("1e-9")


@dataclass
class Check:
    name: str
    expected: str
    got: str
    ok: bool


def num(value: Any) -> str:
    return str(mp.nstr(mp.mpmathify(value), 6))


def fixed(end: str, value: str, order: int = 0, function: str = "x") -> dict[str, Any]:
    return {"function": function, "order": order, "end": end, "kind": "fixed", "value": value}


def free(end: str, order: int = 0, function: str = "x") -> dict[str, Any]:
    return {"function": function, "order": order, "end": end, "kind": "free"}


def problem_of(**kw: Any) -> Problem:
    kw.setdefault("functions", ["x"])
    kw.setdefault("lower", "0")
    kw.setdefault("upper", "1")
    return read_problem(VariationInput(action="first", **kw), LIMITS)


def run(action: str, **kw: Any) -> VariationReport:
    kw.setdefault("functions", ["x"])
    kw.setdefault("lower", "0")
    kw.setdefault("upper", "1")
    return variation(VariationInput(action=action, **kw), LIMITS)


def reference(problem: Problem, text: str) -> sp.Expr:
    """A hand-written expression in x, x_dot, x_ddot, x_d3, x_d4, y..., t and the parameters."""
    jets = problem.jets
    names: dict[str, Any] = {"t": jets.variable}
    for i, fn in enumerate(jets.names):
        for k in range(jets.depth + 1):
            from ..build import spelling

            names[spelling(fn, k)] = jets.u[i][k]
    for p in problem.parameters:
        names[p.name] = p
    names["V"] = sp.Function("V", real=True)
    return sp.sympify(text, locals=names)


def derivation_check(name: str, problem: Problem, got: sp.Expr, text: str) -> Check:
    want = reference(problem, text)
    ok = bool(sp.simplify(got - want) == 0)
    return Check(name, text, str(sp.simplify(got)) if not ok else text, ok)


def value_check(name: str, got: Any, want: Any, tol: Any = TOL) -> Check:
    ok = abs(mp.mpmathify(got) - mp.mpmathify(want)) <= tol * max(1, abs(mp.mpmathify(want)))
    return Check(name, num(want), num(got), bool(ok))


def status_check(report: VariationReport, want: Status) -> Check:
    return Check("status", want.value, report.status.value, report.status is want)


# -- cases -------------------------------------------------------------------------------------------
def euler_and_boundary(
    integrand: str, euler: list[str], pi: dict[tuple[End, int, int], str], **kw: Any
) -> list[Check]:
    p = problem_of(integrand=integrand, **kw)
    d = first_variation(p)
    checks = [
        derivation_check(f"EL[{n}]", p, d.euler[i], euler[i]) for i, n in enumerate(p.jets.names)
    ]
    for (end, i, j), text in pi.items():
        checks.append(
            derivation_check(
                f"coefficient {end} {p.jets.names[i]}^({j})", p, d.coefficients[(end, i, j)], text
            )
        )
    return checks


def solved_action(want: Any, report: VariationReport) -> list[Check]:
    checks = [status_check(report, Status.OK)]
    if report.solution and report.solution.paths:
        checks.append(value_check("action", report.solution.paths[0].action, want))
        checks.append(
            Check(
                "stationarity",
                "within tolerance",
                report.solution.paths[0].stationarity.verdict,
                report.solution.paths[0].stationarity.verdict == "stationary_within_tolerance",
            )
        )
    else:
        checks.append(Check("a stationary path", "found", "none", False))
    return checks


def case_free_particle() -> list[Check]:
    checks = euler_and_boundary(
        r"\frac{m}{2}\dot{x}^2",
        ["-m*x_ddot"],
        {("upper", 0, 0): "m*x_dot", ("lower", 0, 0): "-m*x_dot"},
    )
    r = run(
        "solve",
        integrand=r"\frac{m}{2}\dot{x}^2",
        values={"m": "2"},
        conditions=[fixed("lower", "0"), fixed("upper", "3")],
    )
    return checks + solved_action(mp.mpf(2) * 9 / 2, r)  # m X^2 / (2 T)


def case_harmonic() -> list[Check]:
    checks = euler_and_boundary(
        r"\frac12\dot{x}^2-\frac12 x^2", ["-x - x_ddot"], {("upper", 0, 0): "x_dot"}
    )
    r = run(
        "solve",
        integrand=r"\frac12\dot{x}^2-\frac12 x^2",
        conditions=[fixed("lower", "0"), fixed("upper", "1")],
    )
    return checks + solved_action(mp.cot(1) / 2, r)


def case_harmonic_long() -> list[Check]:
    r = run(
        "solve",
        integrand=r"\frac12\dot{x}^2-\frac12 x^2",
        upper="4",
        conditions=[fixed("lower", "0"), fixed("upper", "1")],
        with_second=True,
    )
    facts = r.solution.second.facts if r.solution and r.solution.second else []
    negative = Check(
        "a negative direction", "not_a_weak_minimum", ",".join(facts), "not_a_weak_minimum" in facts
    )
    return [*solved_action(mp.cot(4) / 2, r), negative]


def case_resonant() -> list[Check]:
    r = run(
        "solve",
        integrand=r"\frac12\dot{x}^2-\frac12 x^2",
        upper=r"\pi",
        conditions=[fixed("lower", "0"), fixed("upper", "1")],
    )
    paths = len(r.solution.paths) if r.solution else -1
    return [status_check(r, Status.NOT_VERIFIED), Check("paths shown", "0", str(paths), paths == 0)]


def case_coupled() -> list[Check]:
    r = run(
        "solve",
        integrand=r"\frac12(\dot{x}^2+\dot{y}^2)-\frac12(x^2+y^2)-\frac34(x-y)^2",
        functions=["x", "y"],
        conditions=[
            fixed("lower", "0"),
            fixed("upper", "1"),
            fixed("lower", "0", function="y"),
            fixed("upper", "0", function="y"),
        ],
    )
    return solved_action(mp.cot(1) / 4 + mp.cot(2) / 2, r)


def case_time_dependent() -> list[Check]:
    checks = euler_and_boundary(
        r"\frac{t}{2}\dot{x}^2",
        ["-(t*x_ddot + x_dot)"],
        {("upper", 0, 0): "2*x_dot", ("lower", 0, 0): "-x_dot"},
        lower="1",
        upper="2",
    )
    r = run(
        "solve",
        integrand=r"\frac{t}{2}\dot{x}^2",
        lower="1",
        upper="2",
        conditions=[fixed("lower", "0"), fixed("upper", "1")],
    )
    return checks + solved_action(1 / (2 * mp.log(2)), r)


def case_forced() -> list[Check]:
    r = run(
        "solve",
        integrand=r"\frac12\dot{x}^2+tx",
        conditions=[fixed("lower", "0"), fixed("upper", "0")],
    )
    return solved_action(-mp.mpf(1) / 90, r)


def case_higher_derivative() -> list[Check]:
    return euler_and_boundary(
        r"\frac{m}{2}\dot{x}^2+\frac{\alpha}{2}\ddot{x}^2-V(x)",
        ["alpha*x_d4 - m*x_ddot - Derivative(V(x), x)"],
        {
            ("upper", 0, 0): "m*x_dot - alpha*x_d3",
            ("upper", 0, 1): "alpha*x_ddot",
            ("lower", 0, 0): "-(m*x_dot - alpha*x_d3)",
            ("lower", 0, 1): "-alpha*x_ddot",
        },
        given_functions=[{"name": "V", "args": ["x"]}],
    )


def case_cantilever() -> list[Check]:
    p = problem_of(
        integrand=r"\frac12\ddot{x}^2-qx",
        conditions=[fixed("lower", "0"), fixed("lower", "0", 1), free("upper"), free("upper", 1)],
    )
    from .boundary import classify

    d = first_variation(p)
    effects = {(e.end, e.order): e.effect for e in classify(p, d)}
    natural = effects[("upper", 0)] == effects[("upper", 1)] == "natural_condition"
    r = run(
        "solve",
        integrand=r"\frac12\ddot{x}^2-x",
        conditions=[fixed("lower", "0"), fixed("lower", "0", 1), free("upper"), free("upper", 1)],
    )
    return [
        derivation_check(
            "natural x'''(1)=0 coefficient", p, d.coefficients[("upper", 0, 0)], "-x_d3"
        ),
        derivation_check(
            "natural x''(1)=0 coefficient", p, d.coefficients[("upper", 0, 1)], "x_ddot"
        ),
        Check("upper end terms stay", "natural_condition", str(effects[("upper", 0)]), natural),
        *solved_action(-mp.mpf(1) / 40, r),
    ]


def case_simply_supported() -> list[Check]:
    r = run(
        "solve",
        integrand=r"\frac12\ddot{x}^2-x",
        conditions=[fixed("lower", "0"), fixed("upper", "0"), free("lower", 1), free("upper", 1)],
    )
    return solved_action(-mp.mpf(1) / 240, r)


def case_stiff() -> list[Check]:
    r = run(
        "solve",
        integrand=r"\frac12\dot{x}^2+\frac{\alpha}{2}\ddot{x}^2",
        values={"alpha": "0.0001"},
        conditions=[fixed("lower", "0"), fixed("upper", "1"), free("lower", 1), free("upper", 1)],
    )
    return solved_action(mp.mpf(1) / 2, r)


def case_free_end() -> list[Check]:
    r = run(
        "solve", integrand=r"\frac12\dot{x}^2-x", conditions=[fixed("lower", "0"), free("upper")]
    )
    return solved_action(-mp.mpf(1) / 6, r)


def case_endpoint_term() -> list[Check]:
    checks = euler_and_boundary(
        r"\frac12\dot{x}^2",
        ["-x_ddot"],
        {("upper", 0, 0): "x_dot + kappa*x - c"},
        upper_term=r"\frac{\kappa}{2}x^2-c x",
    )
    r = run(
        "solve",
        integrand=r"\frac12\dot{x}^2",
        upper_term=r"\frac{k}{2}x^2-c x",
        values={"k": "2", "c": "3"},
        conditions=[fixed("lower", "0"), free("upper")],
    )
    return checks + solved_action(-mp.mpf(3) / 2, r)


def case_null_lagrangians() -> list[Check]:
    checks = []
    for source in (r"x\dot{x}", r"x\ddot{x}+\dot{x}^2"):
        p = problem_of(integrand=source)
        d = first_variation(p)
        checks.append(Check(f"EL of {source}", "0", str(d.euler[0]), d.euler[0] == 0 and d.null))
    return checks


def case_phase_space() -> list[Check]:
    return euler_and_boundary(
        r"p\dot{x}-\frac12(p^2+x^2)",
        ["-p_dot - x", "x_dot - p"],
        {("upper", 0, 0): "p"},
        functions=["x", "p"],
    )


def case_mixed_orders() -> list[Check]:
    return euler_and_boundary(
        r"\frac12\ddot{x}^2+\frac12\dot{y}^2+xy",
        ["x_d4 + y", "-y_ddot + x"],
        {("upper", 0, 0): "-x_d3", ("upper", 0, 1): "x_ddot", ("upper", 1, 0): "y_dot"},
        functions=["x", "y"],
    )


def case_cubic() -> list[Check]:
    r = run(
        "second",
        integrand=r"\dot{x}^3",
        conditions=[fixed("lower", "0"), fixed("upper", "-1")],
        paths=[{"analytic": {"x": "-t"}}],
    )
    n = r.second.numeric if r.second else None
    facts = n.facts if n else []
    return [
        status_check(r, Status.OK),
        Check(
            "not a weak minimum",
            "not_a_weak_minimum",
            ",".join(facts),
            "not_a_weak_minimum" in facts,
        ),
        Check(
            "Legendre",
            "legendre_violated_for_minimum",
            ",".join(facts),
            "legendre_violated_for_minimum" in facts,
        ),
    ]


def case_arc_length() -> list[Check]:
    checks = euler_and_boundary(
        r"\sqrt{1+\dot{x}^2}",
        ["-x_ddot/(1+x_dot**2)**(3/2)"],
        {("upper", 0, 0): "x_dot/sqrt(1+x_dot**2)"},
    )
    r = run(
        "solve",
        integrand=r"\sqrt{1+\dot{x}^2}",
        conditions=[fixed("lower", "0"), fixed("upper", "1")],
    )
    return checks + solved_action(mp.sqrt(2), r)


def case_catenary() -> list[Check]:
    checks = []
    values = []
    for guess in (1.7, 0.47):
        c = mp.findroot(lambda c: c * mp.cosh(1 / c) - 2, guess)
        r = run(
            "solve",
            integrand=r"x\sqrt{1+\dot{x}^2}",
            lower="-1",
            upper="1",
            conditions=[fixed("lower", "2"), fixed("upper", "2")],
            paths=[{"analytic": {"x": f"{float(c)}*\\cosh(t/{float(c)})*(1+0.05*t)"}}],
        )
        want = c + c**2 / 2 * mp.sinh(2 / c)
        checks += solved_action(want, r)
        values.append(want)
    checks.append(
        Check(
            "two different stationary paths",
            "different",
            num(abs(values[0] - values[1])),
            abs(values[0] - values[1]) > 0.1,
        )
    )
    return checks


BRACHISTOCHRONE = r"\frac{\sqrt{1 + \left(\frac{dx}{dy}\right)^2}}{\sqrt{2 g y}}"


def case_brachistochrone() -> list[Check]:
    """Time of descent with x as a function of y, written with the Leibniz derivative and no bounds.

    Reference (by hand): P = x' / (sqrt(2 g y) sqrt(1 + x'^2)) and EL = -(D P), so the first integral
    P = const gives the cycloid. The same functional on an interval across the singularity at y = 0 must
    never be shown as verified: it is reported as not checked, naming the way out.
    """
    assume = {"g": ["positive"]}
    p = problem_of(
        integrand=BRACHISTOCHRONE, variable="y", lower=None, upper=None, assumptions=assume
    )
    prime = problem_of(
        integrand=BRACHISTOCHRONE.replace(r"\frac{dx}{dy}", "x'"),
        variable="y",
        lower=None,
        upper=None,
        assumptions=assume,
    )
    d = first_variation(p)
    checks = [
        derivation_check(
            "EL[x]",
            p,
            d.euler[0],
            "x_dot/(2*t*sqrt(2*g*t)*sqrt(1+x_dot**2)) - x_ddot/(sqrt(2*g*t)*(1+x_dot**2)**(3/2))",
        ),
        Check(
            "x' reads as the same functional",
            "equal",
            "equal" if prime.lagrangian == p.lagrangian else "different",
            prime.lagrangian == p.lagrangian,
        ),
    ]
    r = run(
        "first", integrand=BRACHISTOCHRONE, variable="y", lower=None, upper=None, assumptions=assume
    )
    checks.append(status_check(r, Status.OK))
    checks.append(
        Check(
            "deciding check without bounds",
            "passed",
            r.first.checks.deciding if r.first else "none",
            bool(r.first and r.first.checks.deciding == "passed"),
        )
    )
    across = run(
        "first", integrand=BRACHISTOCHRONE, variable="y", lower="-1", upper="1", assumptions=assume
    )
    checks.append(status_check(across, Status.NOT_VERIFIED))
    checks.append(
        Check(
            "across y = 0: not checked, never ok",
            "could not be checked",
            across.message[:40],
            "could not be checked" in across.message and across.first is None,
        )
    )
    return checks


def case_singular() -> list[Check]:
    r = run("analyze", integrand=r"t\dot{x}^2")
    a = r.analysis
    return [
        status_check(r, Status.OK),
        Check(
            "Legendre determinant",
            "2 t",
            a.hessian_det_latex if a else "",
            bool(a and a.hessian_det_latex == "2 t"),
        ),
        Check(
            "regularity",
            "conditional",
            a.regularity if a else "",
            bool(a and a.regularity == "conditional"),
        ),
    ]


def case_ritz() -> list[Check]:
    r = run(
        "evaluate",
        integrand=r"\frac12\dot{x}^2-x",
        paths=[{"analytic": {"x": "c*t*(1-t)"}}],
        sweep={"name": "c", "values": ["0", "1/4", "1/2", "1"]},
    )
    values = [t.action for t in r.evaluation.trials] if r.evaluation else []
    want = [0, -mp.mpf(1) / 32, -mp.mpf(1) / 24, 0]
    return [status_check(r, Status.OK)] + [
        value_check(f"S(c={c})", v, w)
        for c, v, w in zip(["0", "1/4", "1/2", "1"], values, want, strict=False)
    ]


CASES: list[tuple[str, str, Callable[[], list[Check]]]] = [
    ("free-particle", "Free particle: EL, boundary term, action m X^2 / 2T", case_free_particle),
    ("harmonic", "Harmonic oscillator on [0, 1]: action cot(1)/2", case_harmonic),
    (
        "harmonic-long",
        "Harmonic oscillator on [0, 4]: stationary, not a minimum",
        case_harmonic_long,
    ),
    (
        "harmonic-resonant",
        "Harmonic oscillator on [0, pi] with x(pi)=1: no solution",
        case_resonant,
    ),
    ("coupled", "Two coupled oscillators: action cot(1)/4 + cot(2)/2", case_coupled),
    ("time-dependent", "Explicit time dependence: d/dt versus partial", case_time_dependent),
    ("forced", "Forced oscillator-free problem: action -1/90", case_forced),
    (
        "higher-derivative",
        "m/2 x'^2 + alpha/2 x''^2 - V(x): EL and boundary form",
        case_higher_derivative,
    ),
    ("cantilever", "Cantilever beam: natural conditions, action -1/40", case_cantilever),
    (
        "simply-supported",
        "Simply supported beam: slopes free, action -1/240",
        case_simply_supported,
    ),
    ("stiff", "Stiff higher-derivative problem (alpha = 1e-4): action 1/2", case_stiff),
    ("free-end", "Free end: natural condition x'(1)=0, action -1/6", case_free_end),
    ("endpoint-term", "Endpoint term: action -c^2 / 2(1+k)", case_endpoint_term),
    ("null-lagrangians", "Total derivatives have no interior equation", case_null_lagrangians),
    ("phase-space", "p x' - H(x, p): orders (1, 0)", case_phase_space),
    ("mixed-orders", "Mixed orders (2, 1)", case_mixed_orders),
    ("cubic", "x'^3 with x(1) = -1: a stationary non-minimum", case_cubic),
    ("arc-length", "Arc length: EL and the straight line", case_arc_length),
    ("catenary", "Catenary: two distinct stationary paths", case_catenary),
    ("singular", "t x'^2: the Legendre determinant 2t vanishes at an end", case_singular),
    (
        "brachistochrone",
        "Brachistochrone dx/dy form, no bounds: EL, first integral, singular interval refused",
        case_brachistochrone,
    ),
    ("ritz", "Trial family c t (1 - t): action c^2/6 - c/6", case_ritz),
]


def execute() -> dict[str, Any]:
    results = []
    times: dict[str, float] = {}
    for case_id, title, fn in CASES:
        started = time.monotonic()
        checks = fn()
        times[case_id] = round(time.monotonic() - started, 2)
        results.append(
            {
                "id": case_id,
                "title": title,
                "ok": all(c.ok for c in checks),
                "checks": [c.__dict__ for c in checks],
            }
        )
    return {
        "deterministic": {
            "cases": results,
            "passed": sum(1 for r in results if r["ok"]),
            "total": len(results),
        },
        "measurements": {"seconds": times},
    }


def render(data: dict[str, Any]) -> str:
    det = data["deterministic"]
    lines = [
        "# Variational calculus benchmark",
        "",
        "Generated by `python -m minerva_engine.variational.benchmark`; do not edit. Each case states the",
        "expected result from a closed form or a hand derivation, independently of Minerva, before the",
        "engine is asked. Convention: `EL_i = sum_k (-1)^k D^k dL/du_i^(k)` and",
        "`dS = integral EL.eta dt + [sum Pi eta^(j)]_a^b`.",
        "",
        f"**{det['passed']} of {det['total']} cases pass.**",
        "",
        "| case | result | checks |",
        "| --- | --- | --- |",
    ]
    for case in det["cases"]:
        lines.append(
            f"| {case['title']} | {'pass' if case['ok'] else 'FAIL'} | {len(case['checks'])} |"
        )
    lines += ["", "## Checks", ""]
    for case in det["cases"]:
        lines.append(f"### {case['title']}")
        lines.append("")
        lines.append("| check | expected | found | agrees |")
        lines.append("| --- | --- | --- | --- |")
        for c in case["checks"]:
            lines.append(
                f"| {c['name']} | `{c['expected']}` | `{c['got']}` | {'yes' if c['ok'] else 'NO'} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the variational calculus engine.")
    parser.add_argument("--json", type=Path, default=Path("docs/benchmarks/variational.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/variational-benchmark.md"))
    parser.add_argument("--check", action="store_true", help="compare with the stored report")
    args = parser.parse_args(argv)
    data = execute()
    det = data["deterministic"]
    if args.check:
        stored = json.loads(args.json.read_text())["deterministic"]
        problems = []
        if stored != det:
            problems.append("the deterministic section differs from the stored one")
        if args.report.read_text() != render(data):
            problems.append("the Markdown report differs")
        if det["passed"] != det["total"]:
            problems.append(f"{det['total'] - det['passed']} case(s) fail")
        if problems:
            print("variational benchmark is out of date:", *problems, sep="\n  ", file=sys.stderr)
            return 1
        print("variational benchmark is up to date")
        return 0
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    args.report.write_text(render(data))
    print(f"{det['passed']}/{det['total']} cases pass; wrote {args.json} and {args.report}")
    return 0 if det["passed"] == det["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
