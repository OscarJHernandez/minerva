"""Reference corpus for the approximation engine, with answers from outside Minerva.

Every case states what a correct answer is *before* Minerva is asked: closed forms worked out by
hand, Bessel and Hurwitz-zeta values from `mpmath`, the partial quotients of pi from the literature,
or a plain numerical evaluation of the original function. A refusal case passes only if the engine
refused with the stated reason and showed no result. A case passes only if every check does. The
deterministic part (what each check expected, what was found to six digits, and whether it agreed) is
compared with the stored report by `--check`; times are measurements and are never compared.

    uv run python -m minerva_engine.approx.benchmark [--check]
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
from .models import ApproximationInput, ApproximationReport
from .operation import approximation

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


def run(method: str, expression: str, **kw: Any) -> ApproximationReport:
    kw.setdefault("variable", "x")
    return approximation(
        ApproximationInput(input=expression, format="text", method=method, **kw), LIMITS
    )


def close(name: str, got: Any, want: Any, tol: Any = TOL) -> Check:
    got, want = mp.mpmathify(got), mp.mpmathify(want)
    return Check(name, num(want), num(got), bool(abs(got - want) <= tol * (1 + abs(want))))


def equal(name: str, got: Any, want: Any) -> Check:
    return Check(name, str(want), str(got), got == want)


def refused(report: ApproximationReport, reason: str) -> list[Check]:
    return [
        Check("refused", reason, str(report.reason), report.reason == reason),
        Check(
            "no result shown",
            "none",
            str(report.result_latex or report.statement_latex),
            report.result_latex is None and report.statement_latex is None,
        ),
    ]


def worst(report: ApproximationReport) -> str:
    """The largest sampled error of a report, as the engine printed it."""
    return report.error.max_abs if report.error and report.error.max_abs else "-"


def coefficient(report: ApproximationReport, order: int) -> sp.Expr:
    from .. import latexprint  # noqa: F401  (keeps the import graph honest)

    for o in report.orders:
        if o.order == order:
            text = o.coefficient_latex.replace("\\", "")
            import re

            text = re.sub(r"frac\{(-?\d+)\}\{(\d+)\}", r"(\1)/(\2)", text).replace(" ", "")
            return sp.sympify(text)
    return sp.Integer(0)


# --- local expansions ---------------------------------------------------------------------------
def case_taylor() -> list[Check]:
    r = run("taylor", "sin(x)", point="0", order=5)
    return [
        equal(f"coefficient of x^{k}", coefficient(r, k), w)
        for k, w in ((1, 1), (3, sp.Rational(-1, 6)), (5, sp.Rational(1, 120)))
    ]


def case_relativistic() -> list[Check]:
    r = run(
        "perturbation",
        "m*c**2/sqrt(1-v**2/c**2)",
        small="v/c",
        order=4,
        assumptions={"m": ["positive"], "c": ["positive"], "v": ["real"]},
    )
    q = {o.order: o.term_latex for o in r.orders}
    return [
        equal("terms of order 0, 2, 4", sorted(q), [0, 2, 4]),
        equal("order 2", q[2], r"\frac{m v^{2}}{2}"),
        equal("order 4", q[4], r"\frac{3 m v^{4}}{8 c^{2}}"),
    ]


def case_laurent() -> list[Check]:
    r = run("laurent", "1/(x*(x-3))", point="3", order=2)
    extras = {e.label: e.latex for e in r.extras}
    with mp.workdps(30):  # the residue as a contour integral, from outside the series machinery

        def f(z: Any) -> Any:
            return 1 / (z * (z - 3))

        residue = mp.quad(
            lambda t: f(3 + mp.mpf("0.5") * mp.expjpi(2 * t)) * mp.mpf("0.5") * mp.expjpi(2 * t),
            [0, 1],
        )
    return [
        equal("pole order", extras["Pole order"], "1"),
        close("residue", mp.mpf(1) / 3, mp.re(residue)),
        equal("residue shown", extras["Residue"], r"\frac{1}{3}"),
    ]


def case_asymptotic() -> list[Check]:
    r = run("asymptotic", "sqrt(x**2+x)", point="oo", order=3)
    X = mp.mpf(10) ** 6
    with mp.workdps(40):
        want = mp.sqrt(X**2 + X)
        series = X + mp.mpf(1) / 2 - 1 / (8 * X) + 1 / (16 * X**2) - 5 / (128 * X**3)
    return [
        Check(
            "four-term expansion minus direct evaluation at x = 1e6",
            "< 1e-24",
            num(series - want),
            bool(abs(series - want) < mp.mpf("1e-24")),
        ),
        equal("status", r.status.value, "ok"),
        equal("labelled asymptotic", "asymptotic_not_convergent" in r.warnings, True),
    ]


def case_multivariable() -> list[Check]:
    r = run("multivariable", "exp(x+y)", variables=["x", "y"], points=["0", "0"], order=2)
    x, y = sp.symbols("x y")
    return [
        equal(
            "polynomial",
            sp.expand(sp.sympify(r.result_text) - (1 + x + y + x**2 / 2 + x * y + y**2 / 2)),
            0,
        )
    ]


def case_limits() -> list[Check]:
    both = run("limit", "1/x", point="0")
    sin = run("limit", "sin(x)/x", point="0")
    gap = run("limit", "sqrt(x**2+x) - x", point="oo")
    return [
        equal("sin x / x at 0", sin.result_latex, "1"),
        equal("sqrt(x^2+x) - x at infinity", gap.result_latex, r"\frac{1}{2}"),
        equal("one-sided limits differ", "one_sided_limits_differ" in both.warnings, True),
    ]


# --- rational and polynomial approximants -------------------------------------------------------
def case_pade() -> list[Check]:
    r = run("pade", "exp(x)", point="0", degree=2, denominator_degree=2)
    x = sp.Symbol("x")
    table = (1 + x / 2 + x**2 / 12) / (
        1 - x / 2 + x**2 / 12
    )  # the classical [2/2] approximant of e^x
    one = run("pade", "exp(x)", point="0", degree=1, denominator_degree=1)
    return [
        equal(
            "[2/2] against the classical table", sp.simplify(sp.sympify(r.result_text) - table), 0
        ),
        equal("[1/1] has a pole the function does not", "spurious_pole" in one.warnings, True),
        equal("[1/1] shown as", one.result_latex, r"\frac{1 + \frac{x}{2}}{1 - \frac{x}{2}}"),
    ]


def case_chebyshev() -> list[Check]:
    r = run("chebyshev", "exp(x)", lower="-1", upper="1", degree=6)
    by = {o.order: mp.mpf(o.coefficient_latex) for o in r.orders}
    checks = [
        close(
            f"coefficient {k} against 2 I_k(1)",
            by[k],
            mp.besseli(k, 1) * (1 if k == 0 else 2),
            mp.mpf("1e-9"),
        )
        for k in range(4)
    ]
    bound = mp.e / (mp.factorial(7) * 2**6)  # the classical interpolation error bound
    worst = mp.mpf(r.error.max_abs) if r.error and r.error.max_abs else mp.inf
    checks.append(
        Check(
            "sampled error below the interpolation bound",
            f"< {num(bound)}",
            num(worst),
            bool(worst < bound),
        )
    )
    return checks


def case_chebyshev_tolerance() -> list[Check]:
    r = run(
        "chebyshev", "sin(x)", lower="0", upper="3", tolerance="1e-6", tolerance_kind="absolute"
    )
    chosen = next(c for c in r.candidates if c.chosen)
    import numpy as np
    from numpy.polynomial import Chebyshev

    def sampled(degree: int) -> float:  # numpy's own interpolation, measured on its own grid
        p = Chebyshev.interpolate(np.sin, degree, domain=[0, 3])
        t = np.linspace(0, 3, 4001)
        return float(np.max(np.abs(np.sin(t) - p(t))))

    return [
        equal("degree chosen", chosen.label, "degree 8"),
        Check("numpy agrees degree 8 meets 1e-6", "< 1e-06", num(sampled(8)), sampled(8) < 1e-6),
        Check("numpy agrees degree 6 does not", ">= 1e-06", num(sampled(6)), sampled(6) >= 1e-6),
        equal("tolerance reported as met", r.tolerance_met, True),
    ]


def case_fourier() -> list[Check]:
    r = run("fourier", "x", lower="-pi", upper="pi", modes=6)
    b = {o.order: mp.mpf(o.coefficient_latex) for o in r.orders if o.term_latex.startswith("b")}
    return [close(f"b_{n}", b[n], mp.mpf(2) * (-1) ** (n + 1) / n) for n in range(1, 7)] + [
        equal("Gibbs warned at the jump", "gibbs" in r.warnings, True)
    ]


def case_legendre() -> list[Check]:
    r = run(
        "legendre", "1/sqrt(5/4 - x)", lower="-1", upper="1", degree=6
    )  # sum h^n P_n(x), h = 1/2
    by = {o.order: mp.mpf(o.coefficient_latex) for o in r.orders}
    return [close(f"a_{n}", by[n], mp.mpf(2) ** -n, mp.mpf("1e-10")) for n in range(7)]


def case_piecewise() -> list[Check]:
    exact = run("piecewise", "abs(x)", lower="-1", upper="1", segments=4, kind="linear")
    off = run("piecewise", "abs(x)", lower="-1", upper="1", segments=3, kind="linear")
    return [
        Check(
            "kink on a knot is exact",
            "< 1e-12",
            worst(exact),
            worst(exact) != "-" and mp.mpf(worst(exact)) < mp.mpf("1e-12"),
        ),
        Check(
            "kink inside a segment is not",
            "> 0.05",
            worst(off),
            worst(off) != "-" and mp.mpf(worst(off)) > mp.mpf("0.05"),
        ),
    ]


# --- numbers and sums ---------------------------------------------------------------------------
def case_continued_fractions() -> list[Check]:
    pi = run("continued_fraction", "pi", order=13)
    e = run("continued_fraction", "E", order=12)

    def quotients(report: ApproximationReport) -> list[int]:
        inside = (report.statement_latex or "").split("[")[1].split("]")[0]
        return [
            int(p) for p in inside.replace(";", ",").split(",") if p.strip().lstrip("-").isdigit()
        ]

    return [
        equal(
            "quotients of pi (OEIS A001203)",
            quotients(pi),
            [3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14],
        ),
        equal("quotients of e (OEIS A003417)", quotients(e), [2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8]),
        equal("355/113 is the fourth convergent", pi.orders[3].coefficient_latex, "355/113"),
    ]


def case_euler_maclaurin() -> list[Check]:
    r = run("euler_maclaurin", "1/k**2", variable="k", lower="10", upper="1000", order=3)
    with mp.workdps(40):
        exact = mp.zeta(2, 10) - mp.zeta(2, 1001)  # Hurwitz zeta: sum_{k=10}^{1000} 1/k^2
    value = mp.mpf(r.result_latex or "0")
    errors = [float(c.error.max_abs) for c in r.candidates if c.error and c.error.max_abs]
    return [
        close("against Hurwitz zeta", value, exact, mp.mpf("1e-9")),
        equal(
            "each Bernoulli correction helps",
            errors == sorted(errors, reverse=True),
            True,
        ),
    ]


def case_acceleration() -> list[Check]:
    r = run("accelerate", "1/k**2", variable="k", lower="1", max_terms=40)
    alt = run("accelerate", "(-1)**k/k", variable="k", lower="1", max_terms=40)
    return [
        close("zeta(2)", mp.mpf(r.result_latex or "0"), mp.zeta(2), mp.mpf("1e-12")),
        close(
            "alternating harmonic series is -ln 2",
            mp.mpf(alt.result_latex or "0"),
            -mp.log(2),
            mp.mpf("1e-12"),
        ),
    ]


# --- composed expansions ------------------------------------------------------------------------
def case_integral_expansion() -> list[Check]:
    r = run(
        "integral_expansion",
        "sqrt(1+eps*x**2)",
        small="eps",
        lower="0",
        upper="1",
        order=3,
        assumptions={"eps": ["positive"], "x": ["positive"]},
    )
    eps = sp.Symbol("eps", positive=True)
    return [
        equal(
            "coefficients 1, 1/6, -1/40, 1/112",
            sp.simplify(
                sp.sympify(r.result_text, locals={"eps": eps})
                - (1 + eps / 6 - eps**2 / 40 + eps**3 / 112)
            ),
            0,
        )
    ]


def case_root_and_inverse() -> list[Check]:
    root = run("root_expansion", "x**2 - 1 - eps*x", small="eps", point="1", order=5)
    eps = sp.Symbol("eps")
    exact = sp.series((eps + sp.sqrt(eps**2 + 4)) / 2, eps, 0, 6).removeO()
    inverse = run("reversion", "sin(x)", small="y", point="0", order=5)
    y = sp.Symbol("eps_y")
    return [
        equal(
            "root of x^2 - 1 - eps x",
            sp.simplify(sp.sympify(root.result_text, locals={"eps": eps}) - exact),
            0,
        ),
        equal(
            "arcsin as the inverse of sin",
            sp.simplify(
                sp.sympify(inverse.result_text, locals={"eps_y": y})
                - sp.series(sp.asin(y), y, 0, 6).removeO()
            ),
            0,
        ),
    ]


def case_laplace() -> list[Check]:
    r = run(
        "laplace", "exp(M*(log(x)-x))", variable="x", small="M", lower="1/10", upper="3", order=2
    )
    a = [
        sp.nsimplify(
            o.coefficient_latex.replace("\\", "")
            .replace("frac{1}{12}", "1/12")
            .replace("frac{1}{288}", "1/288"),
            rational=True,
        )
        for o in r.orders
    ]
    M = mp.mpf(400)
    with mp.workdps(
        40
    ):  # Stirling: Gamma(M+1) = M^(M+1) e^-M sqrt(2 pi / M) (1 + 1/(12M) + 1/(288 M^2) + ...)
        ratio = mp.gamma(M + 1) / (M ** (M + 1) * mp.exp(-M) * mp.sqrt(2 * mp.pi / M))
        series = 1 + 1 / (12 * M) + 1 / (288 * M**2)
    return [
        equal("coefficients 1, 1/12, 1/288", a, [1, sp.Rational(1, 12), sp.Rational(1, 288)]),
        Check(
            "Stirling series minus Gamma(M+1) ratio at M = 400",
            "< 1e-9",
            num(series - ratio),
            bool(abs(series - ratio) < mp.mpf("1e-9")),
        ),
    ]


# --- strategy and refusals ----------------------------------------------------------------------
def case_strategy() -> list[Check]:
    shapes: list[tuple[dict[str, Any], str, str]] = [
        ({"point": "0", "order": 4}, "sin(x)", "taylor"),
        ({"point": "oo", "order": 3}, "1/(x+1)", "asymptotic"),
        ({"small": "eps", "order": 3}, "1/sqrt(1-eps**2)", "perturbation"),
        ({"period": "2*pi", "modes": 3, "lower": "-pi", "upper": "pi"}, "x", "fourier"),
        ({"order": 5}, "pi", "continued_fraction"),
        ({"lower": "0", "upper": "2", "degree": 5}, "log(1+x)", "compare"),
    ]
    out = [
        equal(f"{expr} {sorted(args)}", run("auto", expr, **args).method, want)
        for args, expr, want in shapes
    ]
    out.append(
        equal(
            "a pole moves Taylor to Laurent",
            run("auto", "1/x", point="0", order=3).method,
            "laurent",
        )
    )
    return out


def case_refusals() -> list[Check]:
    return [
        *refused(run("laurent", "exp(1/x)", point="0", order=3), "not_laurent"),
        *refused(run("chebyshev", "1/(x-1)", lower="0", upper="2", degree=8), "domain_invalid"),
        *refused(run("chebyshev", "sin(x)", degree=6), "domain_unspecified"),
        *refused(
            run("pade", "cos(x)", point="0", degree=1, denominator_degree=1), "degenerate_system"
        ),
        *refused(
            run("root_expansion", "x**2 - eps", small="eps", point="0", order=2),
            "degenerate_system",
        ),
        *refused(run("asymptotic", "exp(x)", point="oo", order=2), "essential_singularity"),
        # A list of variables sent to a method of one variable, and a variable the expression
        # does not contain: both used to give a checked "series" of the unchanged expression.
        *refused(
            run("taylor", "cos(a - b)", variables=["a", "b"], point="0", order=2),
            "fields_not_for_method",
        ),
        *refused(run("taylor", "cos(a - b)", point="0", order=2), "variable_missing"),
        Check(
            "oscillating sum is not shown",
            Status.NOT_VERIFIED.value,
            run(
                "euler_maclaurin", "cos(1000*k)", variable="k", lower="1", upper="200", order=2
            ).status.value,
            run(
                "euler_maclaurin", "cos(1000*k)", variable="k", lower="1", upper="200", order=2
            ).status
            is Status.NOT_VERIFIED,
        ),
    ]


CASES: list[tuple[str, str, Callable[[], list[Check]]]] = [
    ("taylor", "Taylor series of sin x", case_taylor),
    ("relativistic", "Non-relativistic expansion in v/c", case_relativistic),
    ("laurent", "Laurent series: pole order and residue", case_laurent),
    ("asymptotic", "Asymptotic expansion of sqrt(x^2 + x)", case_asymptotic),
    ("multivariable", "Two-variable Taylor polynomial", case_multivariable),
    ("limits", "Limits, removable and one-sided", case_limits),
    ("pade", "Padé approximants of exp", case_pade),
    ("chebyshev", "Chebyshev coefficients of exp", case_chebyshev),
    ("chebyshev-tolerance", "Lowest Chebyshev degree for a tolerance", case_chebyshev_tolerance),
    ("fourier", "Fourier series of a sawtooth", case_fourier),
    ("legendre", "Legendre generating function", case_legendre),
    ("piecewise", "Piecewise linear and a kink", case_piecewise),
    ("continued-fractions", "Continued fractions of pi and e", case_continued_fractions),
    ("euler-maclaurin", "Euler-Maclaurin against Hurwitz zeta", case_euler_maclaurin),
    ("acceleration", "Accelerated series", case_acceleration),
    ("integral-expansion", "An integral expanded in a parameter", case_integral_expansion),
    ("root-and-inverse", "Root expansion and function reversion", case_root_and_inverse),
    ("laplace", "Laplace's method gives Stirling's series", case_laplace),
    ("strategy", "The strategy picks the expected method", case_strategy),
    ("refusals", "What must be refused, with the stated reason", case_refusals),
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
        "# Approximation benchmark",
        "",
        "Generated by `python -m minerva_engine.approx.benchmark`; do not edit. Each case states the",
        "expected result from a closed form, a literature value or an independent `mpmath` / `numpy`",
        "computation before the engine is asked. A refusal case passes only if the engine refused with",
        "the stated reason and showed no result.",
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
        lines += [
            f"### {case['title']}",
            "",
            "| check | expected | found | agrees |",
            "| --- | --- | --- | --- |",
        ]
        for c in case["checks"]:
            lines.append(
                f"| {c['name']} | `{c['expected']}` | `{c['got']}` | {'yes' if c['ok'] else 'NO'} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the approximation engine.")
    parser.add_argument("--json", type=Path, default=Path("docs/benchmarks/approximation.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/approximation-benchmark.md"))
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
            print("approximation benchmark is out of date:", *problems, sep="\n  ", file=sys.stderr)
            return 1
        print("approximation benchmark is up to date")
        return 0
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    args.report.write_text(render(data))
    print(f"{det['passed']}/{det['total']} cases pass; wrote {args.json} and {args.report}")
    return 0 if det["passed"] == det["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
