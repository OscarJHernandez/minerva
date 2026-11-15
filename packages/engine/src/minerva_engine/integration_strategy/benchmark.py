"""Integrals solved with and without the strategy engine, against independent references.

Every case names the technique a request must be able to use and the reference it is judged by:
mpmath quadrature for a definite value, a central difference of the result for an antiderivative.
A case also records what the *default* call (no strategy requested) did, so the report shows which
integrals only the strategies solve. A ``refused`` case must come back without a result.

    python -m minerva_engine.integration_strategy.benchmark [--check]

writes ``docs/benchmarks/integration_strategies.json`` and ``docs/integration-strategies-benchmark.md``;
``--check`` compares the deterministic part with the committed JSON and fails on a wrong answer.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

from ..errors import MinervaMathError
from ..models import ExpressionInput, Limits, Status
from ..service import Engine
from ..textparse import parse_text

REAL = {"x": ["real"]}
POS = {"x": ["real"], "a": ["positive"], "b": ["positive"], "n": ["positive"]}


@dataclass(frozen=True)
class Case:
    id: str
    technique: str
    source: str  # how the reference is obtained
    expression: str
    lower: str | None = None
    upper: str | None = None
    assumptions: dict[str, list[str]] = field(default_factory=lambda: dict(REAL))
    values: dict[str, float] = field(default_factory=dict)
    expect: str = "result"  # result | refused


CASES: list[Case] = [
    Case("u-substitution", "substitution", "derivative check", "x/(1+x**2)**2"),
    Case("u-substitution-exp", "substitution", "derivative check", "cos(x)*exp(sin(x))"),
    Case("parts-linear", "parts", "derivative check", "x*exp(x)"),
    Case("parts-quadratic", "parts", "derivative check", "x**2*sin(x)"),
    Case("trig-power-reduction", "trigonometric", "derivative check", "sin(x)**2*cos(x)**2"),
    Case("trig-fourth-power", "trigonometric", "derivative check", "cos(x)**4"),
    Case("partial-fractions", "partial_fractions", "derivative check", "1/(x**2-1)"),
    Case(
        "partial-fractions-repeated",
        "partial_fractions",
        "derivative check",
        "1/((x+1)**2*(x**2+1))",
    ),
    Case("parity-odd", "parity", "mpmath quadrature", "x**3*exp(-x**2)", "-oo", "oo"),
    Case("parity-even", "parity", "mpmath quadrature", "x**2*exp(-x**2)", "-1", "1"),
    Case(
        "king-rule",
        "reflection",
        "closed form pi/4",
        "sin(x)**n/(sin(x)**n+cos(x)**n)",
        "0",
        "pi/2",
        POS,
        {"n": 1.7},
    ),
    Case(
        "frullani-exp",
        "frullani",
        "mpmath quadrature",
        "(exp(-a*x)-exp(-b*x))/x",
        "0",
        "oo",
        POS,
        {"a": 0.7, "b": 2.1},
    ),
    Case(
        "frullani-atan",
        "frullani",
        "mpmath quadrature",
        "(atan(a*x)-atan(b*x))/x",
        "0",
        "oo",
        POS,
        {"a": 0.7, "b": 2.1},
    ),
    Case(
        "schwinger-power",
        "schwinger",
        "mpmath quadrature",
        "1/(x**2+a**2)**2",
        "0",
        "oo",
        POS,
        {"a": 1.3},
    ),
    Case(
        "parameter-gaussian",
        "parameter_differentiation",
        "mpmath quadrature",
        "exp(-a*x**2)",
        "-oo",
        "oo",
        POS,
        {"a": 1.3},
    ),
    Case("numerical-no-closed-form", "numerical", "mpmath quadrature", "exp(x**x)", "0", "1"),
    Case(
        "refuse-frullani-undeclared",
        "frullani",
        "scales must be declared positive",
        "(exp(-a*x)-exp(-b*x))/x",
        "0",
        "oo",
        REAL,
        expect="refused",
    ),
    Case(
        "refuse-frullani-cosine",
        "frullani",
        "g(infinity) does not exist",
        "(cos(a*x)-cos(b*x))/x",
        "0",
        "oo",
        POS,
        expect="refused",
    ),
    Case(
        "refuse-odd-divergent",
        "parity",
        "the half-range integral diverges",
        "1/x",
        "-1",
        "1",
        expect="refused",
    ),
    Case(
        "refuse-numerical-divergent",
        "numerical",
        "the integral diverges",
        "1/x",
        "0",
        "1",
        expect="refused",
    ),
    Case(
        "refuse-parts-on-rational",
        "parts",
        "no integration by parts applies",
        "1/(1+x**2)",
        expect="result",
    ),
]


def _fn(text: str, assumptions: dict[str, list[str]], names: list[str]) -> Any:
    expr = parse_text(text, assumptions=assumptions)
    return sp.lambdify(["x", *names], expr, "mpmath")


def _bound(text: str) -> Any:
    return {"oo": mp.inf, "-oo": -mp.inf}.get(text) or mp.mpf(str(sp.N(parse_text(text), 25)))


def judge(case: Case, result: Any) -> tuple[str, str]:
    """('correct' | 'wrong' | 'refused_as_expected', detail) for the result of a requested strategy."""
    mp.mp.dps = 30
    shown = result.status is Status.OK and (result.result_text or result.numeric)
    if case.expect == "refused":
        if shown and result.strategy and result.strategy.technique == case.technique:
            return "wrong", "a result where a refusal was due"
        return "refused_as_expected", ""
    if not shown:
        return "wrong", f"no result ({result.status.value})"
    if result.strategy is None or (result.strategy.technique not in (case.technique, "direct")):
        return "wrong", "the technique is not the one reported"
    names = sorted(case.values)
    args = [mp.mpf(case.values[k]) for k in names]
    original = _fn(case.expression, case.assumptions, names)
    try:
        got_expr = result.result_text and parse_text(
            result.result_text, assumptions=case.assumptions
        )
    except MinervaMathError:
        got_expr = None
    if case.lower is None:
        if got_expr is None:
            return "wrong", "the antiderivative could not be read back"
        F = sp.lambdify("x", got_expr, "mpmath")
        h = mp.mpf("1e-10")
        for t in (mp.mpf("0.31"), mp.mpf("0.77"), mp.mpf("1.9")):
            if abs((F(t + h) - F(t - h)) / (2 * h) - original(t, *args)) > 1e-7 * max(
                1, abs(original(t, *args))
            ):
                return "wrong", "derivative of the result differs from the integrand"
        return "correct", ""
    lo, hi = _bound(case.lower or ""), _bound(case.upper or "")
    want = mp.quad(lambda t: original(t, *args), [lo, 0, hi] if lo < 0 < hi else [lo, hi])
    if result.result_kind == "numeric" or got_expr is None:
        got = mp.mpf(result.numeric) if result.numeric else None
    else:
        got = sp.lambdify(names, got_expr, "mpmath")(*args)
    if got is None or abs(got - want) > 1e-8 * max(1, abs(want)):
        return "wrong", "value differs from quadrature"
    return "correct", ""


def run_cases(engine: Engine | None = None) -> list[dict[str, Any]]:
    engine = engine or Engine(Limits(timeout_seconds=60, max_concurrency=2))
    rows = []
    for case in CASES:
        start = time.monotonic()
        data = ExpressionInput(input=case.expression, format="text", assumptions=case.assumptions)
        try:
            default = engine.integrate(data, "x", case.lower, case.upper)
            requested = engine.integrate(data, "x", case.lower, case.upper, case.technique)
            outcome, detail = judge(case, requested)
        except Exception as exc:  # a crash is a result: it must never be read as a pass
            default, requested = None, None
            outcome, detail = "wrong", f"{type(exc).__name__} while judging"
        report = requested.strategy if requested else None
        rows.append(
            {
                "id": case.id,
                "technique": case.technique,
                "source": case.source,
                "expected": case.expect,
                "outcome": outcome,
                "default": (
                    "-"
                    if default is None
                    else (default.strategy.technique or default.status.value)
                    if default.status is Status.OK and default.strategy
                    else default.status.value
                ),
                "selected": (report.technique if report and report.technique else "-"),
                "verification": report.verification if report else "none",
                "detail": detail,
                "latency_ms": int((time.monotonic() - start) * 1000),
            }
        )
    return rows


def deterministic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "id",
        "technique",
        "source",
        "expected",
        "outcome",
        "default",
        "selected",
        "verification",
    )
    return [{k: r[k] for k in keys} for r in rows]


def markdown(rows: list[dict[str, Any]]) -> str:
    good = {"correct", "refused_as_expected"}
    lines = [
        "# Integration strategies benchmark",
        "",
        "Generated by `python -m minerva_engine.integration_strategy.benchmark`; outcomes are "
        "deterministic, latencies are not gated. Each result is judged by mpmath quadrature or a "
        "central difference, never by the technique that produced it. *Default* is what a call "
        "without a requested strategy returns (`direct` = SymPy's route solved it).",
        "",
        f"{sum(r['outcome'] in good for r in rows)} of {len(rows)} as expected; "
        f"wrong: {sum(r['outcome'] == 'wrong' for r in rows)}.",
        "",
        "| Case | Technique requested | Expected | Outcome | Default call | Selected | Verification | ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{r['id']}` | {r['technique']} | {r['expected']} | {r['outcome']} | {r['default']} | "
        f"{r['selected']} | {r['verification']} | {r['latency_ms']} |"
        for r in rows
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[5]
    json_path = root / "docs/benchmarks/integration_strategies.json"
    md_path = root / "docs/integration-strategies-benchmark.md"
    rows = run_cases()
    bad = [r for r in rows if r["outcome"] == "wrong"]
    if "--check" in argv:
        expected = json.loads(json_path.read_text())["cases"] if json_path.exists() else None
        if expected != deterministic(rows):
            print(
                "outcomes differ from docs/benchmarks/integration_strategies.json", file=sys.stderr
            )
            return 1
    else:
        json_path.write_text(json.dumps({"cases": deterministic(rows)}, indent=2) + "\n")
        md_path.write_text(markdown(rows))
    for r in bad:
        print(f"{r['id']}: {r['outcome']} ({r['detail']})", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
