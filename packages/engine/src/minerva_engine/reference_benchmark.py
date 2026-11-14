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
