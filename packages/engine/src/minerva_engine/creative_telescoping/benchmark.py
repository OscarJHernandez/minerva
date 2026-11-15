"""Creative-telescoping problems whose answers come from outside Minerva.

Every reference is independent of the discovery route: a closed form written here, or mpmath
quadrature of the integrand itself (and finite-difference derivatives of that quadrature for the
differential equations). A case expects one of

* ``evaluated``  the engine returns a value, which must match the reference at sampled parameters,
  and an equation that the quadrature satisfies;
* ``equation``   only the equation (and a proved certificate) may be returned; the equation must be
  satisfied by the quadrature and no value may appear;
* ``certificate`` only the proved certificate may be returned;
* ``refused``    no result, with one of the enumerated reasons the case names;
* ``rejected``   a hand-made, incorrect certificate that verification must reject (or, for the one
  control, a correct one that it must accept).

    python -m minerva_engine.creative_telescoping.benchmark [--check]

writes ``docs/benchmarks/creative_telescoping.json`` and ``docs/creative-telescoping-benchmark.md``;
``--check`` compares the deterministic part with the committed JSON and fails on any difference or
any wrong answer (CI runs it through ``tests/test_creative_telescoping_benchmark.py``). Latencies are
recorded and never gated.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

from ..errors import ParseError
from ..models import ExpressionInput, Limits, Status
from ..ops import parse_expression
from ..service import Engine
from . import certificate
from .models import TelescopingInput, TelescopingResult
from .operators import DifferentialOperator

POSITIVE = {"a": ["positive"], "x": ["real"]}
SAMPLES = (0.7, 1.3, 2.1)

Closed = Callable[..., float]


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    source: str  # how the reference is obtained, by description only
    request: dict[str, Any]
    expect: str  # evaluated | equation | certificate | refused | rejected | accepted
    reasons: tuple[str, ...] = ()  # acceptable reasons of a refusal
    closed: Closed | None = None  # a closed form of the value, as an extra reference
    params: dict[str, float] = field(default_factory=dict)  # values of the other symbols
    forged: dict[str, Any] | None = None  # for rejected / accepted: f, G, coefficients


def _req(
    expr: str,
    lower: str | None,
    upper: str | None,
    assumptions: dict[str, list[str]] | None = None,
    parameter: str = "a",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "input": expr,
        "format": "latex",
        "variable": "x",
        "parameter": parameter,
        "lower": lower,
        "upper": upper,
        "assumptions": assumptions if assumptions is not None else POSITIVE,
        **extra,
    }


CASES: list[Case] = [
    # 1. Gaussian integrals
    Case(
        "gaussian-a",
        "gaussian",
        "closed form sqrt(pi/a)",
        _req(r"\exp(-a x^2)", "-oo", "oo"),
        "evaluated",
        closed=lambda a: mp.sqrt(mp.pi / a),
    ),
    Case(
        "gaussian-shifted",
        "gaussian",
        "closed form sqrt(pi) exp(a^2/4)",
        _req(r"\exp(-x^2 + a x)", "-oo", "oo"),
        "evaluated",
        closed=lambda a: mp.sqrt(mp.pi) * mp.exp(a**2 / 4),
    ),
    Case(
        "gaussian-two-parameters",
        "gaussian",
        "closed form sqrt(pi/a) exp(b^2/(4a)); the other symbol is a constant",
        _req(r"\exp(-a x^2 - b x)", "-oo", "oo", {"a": ["positive"], "b": ["real"], "x": ["real"]}),
        "evaluated",
        closed=lambda a, b: mp.sqrt(mp.pi / a) * mp.exp(b**2 / (4 * a)),
        params={"b": 0.8},
    ),
    # 2. Laplace-type integrals
    Case(
        "laplace-power",
        "laplace",
        "closed form Gamma(n+1)/a^(n+1)",
        _req(r"x^n \exp(-a x)", "0", "oo", {"a": ["positive"], "n": ["positive"], "x": ["real"]}),
        "evaluated",
        closed=lambda a, n: mp.gamma(n + 1) / a ** (n + 1),
        params={"n": 2.5},
    ),
    Case(
        "laplace-linear",
        "laplace",
        "closed form 1/a^2",
        _req(r"x \exp(-a x)", "0", "oo"),
        "evaluated",
        closed=lambda a: 1 / a**2,
    ),
    # 3. rational and exponential factors
    Case(
        "lorentzian",
        "rational_exponential",
        "closed form pi/a",
        _req(r"\frac{1}{x^2+a^2}", "-oo", "oo"),
        "evaluated",
        closed=lambda a: mp.pi / a,
    ),
    Case(
        "exponential-integral",
        "rational_exponential",
        "closed form exp(a) E1(a) (exponential integral) and the equation I' - I = -1/a",
        _req(r"\frac{\exp(-a x)}{1+x}", "0", "oo"),
        "evaluated",
        closed=lambda a: mp.exp(a) * mp.e1(a),
    ),
    # 4. higher-order operators
    Case(
        "reciprocal-gaussian",
        "higher_order",
        "closed form sqrt(pi) exp(-2a) / 2 (order two)",
        _req(r"\exp(-x^2 - a^2/x^2)", "0", "oo"),
        "evaluated",
        closed=lambda a: mp.sqrt(mp.pi) * mp.exp(-2 * a) / 2,
    ),
    # 5. special functions with known differential equations
    Case(
        "bessel-j0",
        "special_function",
        "quadrature; the Bessel equation a I'' + I' + a I = 0 is checked against it",
        _req(r"\frac{\exp(\imaginaryI a x)}{\sqrt{1-x^2}}", "-1", "1"),
        "equation",
    ),
    Case(
        "airy-type",
        "special_function",
        "quadrature; the inhomogeneous Airy equation I'' + a I = 1 is checked against it",
        _req(r"\exp(-x^3/3 - a x)", "0", "oo"),
        "equation",
    ),
    # 6. nonzero boundary contributions
    Case(
        "finite-interval",
        "nonzero_boundary",
        "closed form (1 - exp(-a))/a",
        _req(r"\exp(-a x)", "0", "1"),
        "evaluated",
        closed=lambda a: (1 - mp.exp(-a)) / a,
    ),
    Case(
        "inhomogeneous-boundary",
        "nonzero_boundary",
        "quadrature; the equation, with its nonzero boundary term at x = 1, is checked against it",
        _req(r"\frac{\exp(-a x)}{1+x}", "0", "1"),
        "equation",
    ),
    # 7. parameter-dependent convergence
    Case(
        "convergence-needs-sign",
        "convergence_restriction",
        "no sign declared for the parameter: nothing may be assumed",
        _req(r"\exp(-a x)", "0", "oo", {"x": ["real"]}),
        "certificate",
        reasons=("convergence_not_established",),
    ),
    Case(
        "convergence-fails-for-negative-parameter",
        "convergence_restriction",
        "exp(-a x) with a < 0 grows: the integral diverges",
        _req(r"\exp(-a x)", "0", "oo", {"a": ["negative"], "x": ["real"]}),
        "refused",
        reasons=("divergent",),
    ),
    # 8. singularities and divergence
    Case(
        "divergent-at-zero",
        "singular_divergent",
        "int_0^1 dx/(a x) diverges",
        _req(r"\frac{1}{a x}", "0", "1"),
        "refused",
        reasons=("divergent",),
    ),
    Case(
        "divergent-power-at-infinity",
        "singular_divergent",
        "x^-2 exp(-a x) is not integrable at 0",
        _req(r"x^{-2} \exp(-a x)", "0", "oo"),
        "refused",
        reasons=("divergent",),
    ),
    Case(
        "exponential-growth",
        "singular_divergent",
        "exp(a x) grows on [0, oo) for a > 0",
        _req(r"\exp(a x)", "0", "oo"),
        "refused",
        reasons=("divergent",),
    ),
    Case(
        "pole-inside-interval",
        "singular_divergent",
        "1/(x^2-1) has a pole at x = 1, inside (0, oo)",
        _req(r"\frac{\exp(-a x)}{x^2-1}", "0", "oo"),
        "certificate",
        reasons=("singularity_not_excluded",),
    ),
    # 9. beyond the supported class
    Case(
        "trigonometric",
        "beyond_class",
        "sin(a x) is not hyperexponential",
        _req(r"\sin(a x)", "0", "1"),
        "refused",
        reasons=("not_hyperexponential",),
    ),
    Case(
        "sum-of-exponentials",
        "beyond_class",
        "a sum of two exponentials has a non-rational logarithmic derivative",
        _req(r"\exp(-a x) + \exp(-2 a x)", "0", "oo"),
        "refused",
        reasons=("not_hyperexponential",),
    ),
    Case(
        "parameter-in-exponent",
        "beyond_class",
        "x^a: f_a / f = log x is not rational",
        _req(r"x^a \exp(-x)", "0", "oo"),
        "refused",
        reasons=("not_hyperexponential",),
    ),
    Case(
        "parameter-missing",
        "beyond_class",
        "the integrand does not depend on the parameter",
        _req(r"\exp(-x)", "0", "oo"),
        "refused",
        reasons=("parameter_missing",),
    ),
]


# 10. intentionally incorrect certificates (the control must be accepted)
def _forged_cases() -> list[Case]:
    x = sp.Symbol("x", real=True)
    a = sp.Symbol("a", positive=True)
    f = sp.exp(-a * x**2)
    good = (sp.Integer(1), 2 * a)
    one_ppm = sp.Rational(1, 1_000_000)

    def forged(coefficients: tuple[sp.Expr, ...], g: sp.Expr) -> dict[str, Any]:
        return {"f": f, "x": x, "a": a, "coefficients": coefficients, "G": g}

    return [
        Case(
            "forged-control",
            "forged_certificate",
            "the correct pair (1 + 2a d/da, x f) is accepted",
            {},
            "accepted",
            forged=forged(good, x * f),
        ),
        Case(
            "forged-sign",
            "forged_certificate",
            "G with the wrong sign",
            {},
            "rejected",
            forged=forged(good, -x * f),
        ),
        Case(
            "forged-coefficient-1e-6",
            "forged_certificate",
            "a coefficient off by one part in a million",
            {},
            "rejected",
            forged=forged((1 + one_ppm, 2 * a), x * f),
        ),
        Case(
            "forged-certificate-1e-6",
            "forged_certificate",
            "G off by one part in a million",
            {},
            "rejected",
            forged=forged(good, x * f * (1 + one_ppm)),
        ),
        Case(
            "forged-dropped-term",
            "forged_certificate",
            "the order-zero term of the operator dropped",
            {},
            "rejected",
            forged=forged((sp.Integer(0), 2 * a), x * f),
        ),
        Case(
            "forged-zero-operator",
            "forged_certificate",
            "the zero operator with G = 0 is a trivial relation",
            {},
            "rejected",
            forged=forged((sp.Integer(0), sp.Integer(0)), sp.Integer(0)),
        ),
        Case(
            "forged-wrong-exponent",
            "forged_certificate",
            "G built from exp(-2 a x^2)",
            {},
            "rejected",
            forged=forged(good, x * sp.exp(-2 * a * x**2)),
        ),
    ]


CASES += _forged_cases()


# -- independent references --------------------------------------------------------------------
def _function(case: Case) -> tuple[Callable[..., Any], list[str]]:
    r = case.request
    expr = parse_expression(
        ExpressionInput(input=r["input"], format="latex", assumptions=r["assumptions"]),
        Limits(),
    )
    names = sorted(s.name for s in expr.free_symbols if s.name != "x")
    return sp.lambdify(["x", *names], expr, "mpmath"), names


def _bound(text: str) -> Any:
    return {"oo": mp.inf, "-oo": -mp.inf}.get(text) or mp.mpf(text)


def _integral(case: Case, fn: Callable[..., Any], names: list[str]) -> Callable[[Any], Any]:
    lo, hi = _bound(case.request["lower"]), _bound(case.request["upper"])
    parameter = case.request["parameter"]

    def at(a: Any) -> Any:
        values = {parameter: a, **{k: mp.mpf(v) for k, v in case.params.items()}}
        args = [values[n] for n in names]
        return mp.quad(lambda t: fn(t, *args), [lo, 0, hi] if lo < 0 < hi else [lo, hi])

    return at


def _samples(case: Case) -> list[float]:
    return (
        [-s for s in SAMPLES]
        if "negative" in case.request["assumptions"].get("a", [])
        else list(SAMPLES)
    )


def _text_function(text: str, case: Case, symbols: list[str]) -> Callable[..., Any]:
    expr = parse_expression(
        ExpressionInput(input=text, format="text", assumptions=case.request["assumptions"]),
        Limits(),
    )
    function: Callable[..., Any] = sp.lambdify(symbols, expr, "mpmath")
    return function


def equation_residual(case: Case, result: TelescopingResult, a0: float) -> float:
    """sum p_k I^(k)(a0) - rhs, relative to the largest term, with I from quadrature alone."""
    if result.operator is None or result.equation is None:
        return float("inf")
    fn, names = _function(case)
    integral = _integral(case, fn, names)
    parameter = case.request["parameter"]
    symbols = sorted({parameter, *case.params})
    values = {parameter: mp.mpf(a0), **{k: mp.mpf(v) for k, v in case.params.items()}}
    args = [values[s] for s in symbols]
    terms = []
    for k, text in enumerate(result.operator.coefficients_text):
        p = _text_function(text, case, symbols)(*args)
        terms.append(p * (mp.diff(integral, mp.mpf(a0), k) if k else integral(mp.mpf(a0))))
    rhs = _text_function(result.equation.rhs_text, case, symbols)(*args)
    scale = max([abs(t) for t in terms] + [abs(rhs), mp.mpf(1)])
    return float(abs(mp.fsum(terms) - rhs) / scale)


def _judge_forged(case: Case) -> tuple[str, str]:
    data = case.forged or {}
    op = DifferentialOperator(data["a"], tuple(data["coefficients"]))
    check = certificate.verify(data["f"], data["x"], op, data["G"])
    if case.expect == "accepted":
        return ("correct", "accepted") if check.passed else ("wrong", f"rejected: {check.note}")
    if check.passed:
        return "wrong", "a forged certificate was accepted"
    return "rejected_as_expected", check.note


def judge_case(case: Case, engine: Engine) -> tuple[str, dict[str, Any]]:
    """(outcome, measurements). Measurements never decide; the outcome does."""
    if case.forged is not None:
        outcome, note = _judge_forged(case)
        return outcome, {"note": note}
    result = engine.creative_telescoping(TelescopingInput(**case.request))
    facts: dict[str, Any] = {
        "status": result.status.value,
        "result_outcome": result.outcome,
        "reason": result.reason,
        "order": result.operator.order if result.operator else None,
        "symbolic_zero": bool(result.certificate_check and result.certificate_check.symbolic_zero),
    }
    if case.expect == "refused":
        if result.status is Status.OK or (result.operator is not None and result.outcome != "none"):
            return "wrong", {**facts, "note": "a result where a refusal was due"}
        if result.reason not in case.reasons:
            return "wrong", {**facts, "note": f"refused with {result.reason}"}
        return "refused_as_expected", facts
    if case.expect == "certificate":
        if result.outcome != "certificate" or result.reason not in case.reasons:
            return "wrong", {**facts, "note": f"outcome {result.outcome} / {result.reason}"}
        if result.result_latex or not facts["symbolic_zero"]:
            return "wrong", {**facts, "note": "a value, or an unproved certificate, was shown"}
        return "correct", facts
    if result.status is not Status.OK or result.outcome not in ("evaluated", "equation"):
        return "unsupported_regression", {**facts, "note": result.message[:120]}
    if case.expect == "equation" and result.outcome != "equation":
        return "wrong", {**facts, "note": "a value where only the equation is established"}
    if case.expect == "evaluated" and result.outcome != "evaluated":
        return "unsupported_regression", {**facts, "note": result.message[:120]}
    mp.mp.dps = 30
    fn, names = _function(case)
    integral = _integral(case, fn, names)
    parameter = case.request["parameter"]
    for a0 in _samples(case):
        residual = equation_residual(case, result, a0)
        if residual > 1e-7:
            return "wrong", {**facts, "note": f"equation residual {residual:.2e} at a={a0}"}
    if result.outcome == "evaluated":
        if result.result_text is None or result.value_verification is None:
            return "wrong", {**facts, "note": "an evaluated outcome without a value"}
        symbols = sorted({parameter, *case.params})
        try:
            value: Callable[..., Any] | None = _text_function(result.result_text, case, symbols)
        except ParseError:  # a special function Minerva's parser does not read back
            value = None
        sampled = {float(p.parameter): mp.mpf(p.value) for p in result.value_verification.samples}
        for a0 in _samples(case):
            values = {parameter: mp.mpf(a0), **{k: mp.mpf(v) for k, v in case.params.items()}}
            if value is not None:
                got = value(*[values[s] for s in symbols])
            elif a0 in sampled:
                got = sampled[a0]
            else:
                return "wrong", {**facts, "note": "the value could not be re-read or sampled"}
            want = integral(mp.mpf(a0))
            if abs(got - want) > 1e-8 * max(1, abs(want)):
                return "wrong", {**facts, "note": f"value off at a={a0}"}
            if case.closed is not None:
                wanted = inspect.signature(case.closed).parameters
                exact = case.closed(**{n: values[n] for n in wanted})
                if abs(got - exact) > 1e-8 * max(1, abs(exact)):
                    return "wrong", {**facts, "note": f"closed form disagrees at a={a0}"}
    return "correct", facts


# -- running -----------------------------------------------------------------------------------
GOOD = {"correct", "refused_as_expected", "rejected_as_expected"}


def run_cases(engine: Engine | None = None) -> list[dict[str, Any]]:
    engine = engine or Engine(Limits(timeout_seconds=120, max_concurrency=2))
    rows: list[dict[str, Any]] = []
    for case in CASES:
        start = time.monotonic()
        try:
            outcome, facts = judge_case(case, engine)
        except Exception as exc:  # a crash is a result: it must never be read as a pass
            outcome, facts = "wrong", {"note": f"{type(exc).__name__} while judging"}
        rows.append(
            {
                "id": case.id,
                "category": case.category,
                "source": case.source,
                "expected": case.expect,
                "reasons": list(case.reasons),
                "outcome": outcome,
                "order": facts.get("order"),
                "symbolic_zero": facts.get("symbolic_zero"),
                "failure_class": facts.get("reason"),
                "detail": facts.get("note", ""),
                "latency_ms": int((time.monotonic() - start) * 1000),
            }
        )
    return rows


def deterministic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "id",
        "category",
        "source",
        "expected",
        "reasons",
        "outcome",
        "order",
        "symbolic_zero",
        "failure_class",
    )
    return [{k: r[k] for k in keys} for r in rows]


def markdown(rows: list[dict[str, Any]]) -> str:
    discovered = [r for r in rows if r["order"] is not None]
    lines = [
        "# Creative telescoping benchmark",
        "",
        "Generated by `python -m minerva_engine.creative_telescoping.benchmark`; outcomes are "
        "deterministic, latencies are not gated. Every reference is a closed form written in the "
        "benchmark or mpmath quadrature of the integrand itself (with finite-difference derivatives "
        "of that quadrature for the equations), never Minerva's discovery route.",
        "",
        f"{sum(r['outcome'] in GOOD for r in rows)} of {len(rows)} as expected; "
        f"wrong: {sum(r['outcome'] == 'wrong' for r in rows)}; "
        f"unsupported where an answer is expected: "
        f"{sum(r['outcome'] == 'unsupported_regression' for r in rows)}.",
        "",
        f"Discovery: a verified telescoping relation was found for {len(discovered)} problems "
        f"(operator orders {sorted({r['order'] for r in discovered})}); "
        f"the certificate was proved exactly in {sum(bool(r['symbolic_zero']) for r in discovered)}.",
        "",
        "| Problem | Category | Expected | Outcome | Order | Exact proof | Failure class | Reference | ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{r['id']}` | {r['category']} | {r['expected']} | {r['outcome']} | "
        f"{r['order'] if r['order'] is not None else '-'} | "
        f"{'yes' if r['symbolic_zero'] else '-'} | {r['failure_class'] or '-'} | "
        f"{r['source']} | {r['latency_ms']} |"
        for r in rows
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[5]
    json_path = root / "docs/benchmarks/creative_telescoping.json"
    md_path = root / "docs/creative-telescoping-benchmark.md"
    rows = run_cases()
    bad = [r for r in rows if r["outcome"] in ("wrong", "unsupported_regression")]
    if "--check" in argv:
        expected = json.loads(json_path.read_text())["problems"] if json_path.exists() else None
        if expected != deterministic(rows):
            print("outcomes differ from docs/benchmarks/creative_telescoping.json", file=sys.stderr)
            return 1
    else:
        json_path.write_text(json.dumps({"problems": deterministic(rows)}, indent=2) + "\n")
        md_path.write_text(markdown(rows))
    for r in bad:
        print(f"{r['id']}: {r['outcome']} ({r['detail']})", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
