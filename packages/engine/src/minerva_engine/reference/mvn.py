"""Bivariate Normal problems with answers worked by hand from the textbook formulas (issue #229).

The formulas: marginal and affine image, Anderson, An Introduction to Multivariate Statistical Analysis
(3rd ed.), Theorems 2.4.3 and 2.4.4; conditional distribution, the same book, Theorem 2.5.1. The numbers
below were worked out by hand for these inputs; they are not regenerated from Minerva.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

from ..service import Engine
from ..stats.mvn_models import MvnInput
from ..textparse import parse_text
from .core import Problem, verdict

MEAN = ["1", "-2"]
COV = [["4", "1"], ["1", "2"]]
XY = ["x", "y"]


def _equal(texts: list[str], expected: list[sp.Rational]) -> bool:
    return len(texts) == len(expected) and all(
        sp.simplify(parse_text(t) - e) == 0 for t, e in zip(texts, expected, strict=True)
    )


def _case(
    request: dict[str, object], mean: list[sp.Rational], cov: list[list[sp.Rational]]
) -> Callable[[Engine], tuple[str, str]]:
    def judge(engine: Engine) -> tuple[str, str]:
        r = engine.multivariate_normal(MvnInput(**request))
        ok = r.status.value == "ok"
        agrees = (
            ok
            and _equal(r.mean_text, mean)
            and all(_equal(row, exp) for row, exp in zip(r.covariance_text, cov, strict=True))
            and len(r.covariance_text) == len(cov)
        )
        return verdict("correct", ok, agrees if ok else None, f"status {r.status.value}")

    return judge


def _refused(request: dict[str, object], reason: str) -> Callable[[Engine], tuple[str, str]]:
    def judge(engine: Engine) -> tuple[str, str]:
        r = engine.multivariate_normal(MvnInput(**request))
        return verdict("refused", r.status.value == "ok", None, f"reason {r.reason} (due {reason})")

    return judge


R = sp.Rational
BASE: dict[str, object] = {"mean": MEAN, "covariance": COV, "variables": XY}

PROBLEMS: list[Problem] = [
    Problem(
        "mvn-marginal-of-y",
        "statistics",
        "Anderson, Multivariate Statistical Analysis, Thm 2.4.3 (marginal of a normal vector)",
        "correct",
        _case({**BASE, "action": "marginal", "keep": "y"}, [R(-2)], [[R(2)]]),
    ),
    Problem(
        "mvn-conditional-x-given-y",
        "statistics",
        "Anderson, Multivariate Statistical Analysis, Thm 2.5.1 (mean 1 + (1/2)(1 + 2), variance 4 - 1/2)",
        "correct",
        _case({**BASE, "action": "conditional", "given": {"y": "1"}}, [R(5, 2)], [[R(7, 2)]]),
    ),
    Problem(
        "mvn-affine-sum-and-difference",
        "statistics",
        "Anderson, Multivariate Statistical Analysis, Thm 2.4.4 (A mu + b, A Sigma A^T by hand)",
        "correct",
        _case(
            {
                **BASE,
                "action": "affine",
                "matrix": [["1", "1"], ["1", "-1"]],
                "offset": ["0", "3"],
            },
            [R(-1), R(6)],
            [[R(8), R(2)], [R(2), R(4)]],
        ),
    ),
    Problem(
        "mvn-affine-one-row",
        "statistics",
        "Anderson, Multivariate Statistical Analysis, Thm 2.4.4 (2 X - Y + 5: mean 9, variance 16 - 4 + 2)",
        "correct",
        _case(
            {**BASE, "action": "affine", "matrix": [["2", "-1"]], "offset": ["5"]},
            [R(9)],
            [[R(14)]],
        ),
    ),
    Problem(
        "mvn-singular-covariance-refused",
        "statistics",
        "a covariance of determinant zero has no density (Anderson, section 2.4, singular case)",
        "refused",
        _refused({**BASE, "action": "density", "covariance": [["1", "1"], ["1", "1"]]}, "singular"),
        evidence="refusal",
    ),
    Problem(
        "mvn-indefinite-covariance-refused",
        "statistics",
        "a covariance matrix is positive semidefinite: det [[1, 2], [2, 1]] = -3",
        "refused",
        _refused(
            {**BASE, "action": "density", "covariance": [["1", "2"], ["2", "1"]]}, "indefinite"
        ),
        evidence="refusal",
    ),
]
