"""Statistical mechanics: the corpus of `minerva_engine.statmech.benchmark`, whose references are closed forms
and plain arithmetic written in that module, not computed by the operation under test."""

from __future__ import annotations

from ..service import Engine
from .core import Problem


def _statmech(case_id: str):  # type: ignore[no-untyped-def]
    def judge(engine: Engine) -> tuple[str, str]:
        from ..statmech.benchmark import CASES, judge_case

        return judge_case(next(c for c in CASES if c.id == case_id), engine)

    return judge


PROBLEMS: list[Problem] = [
    Problem(
        "statmech-two-level-thermodynamics",
        "statmech",
        "closed form Z = 1 + e^(-beta eps), plain arithmetic",
        "correct",
        _statmech("two-level-thermodynamics"),
        oracle="arithmetic",
    ),
    Problem(
        "statmech-quantum-oscillator",
        "statmech",
        "geometric series: Z = 1/(2 sinh(x/2)), Einstein heat capacity",
        "correct",
        _statmech("quantum-harmonic-oscillator"),
        oracle="closed_form",
    ),
    Problem(
        "statmech-ideal-gas-pressure-and-entropy",
        "statmech",
        "Z = Z_1^N/N! with the exact ln N!: pV = NkT, entropy via lgamma",
        "correct",
        _statmech("classical-ideal-gas"),
        oracle="closed_form",
    ),
    Problem(
        "statmech-divergent-series",
        "statmech",
        "E_n = -n: the terms grow like e^(beta n)",
        "refused",
        _statmech("divergent-partition-function"),
        oracle="closed_form",
        evidence="refusal",
    ),
    Problem(
        "statmech-indistinguishability-is-a-choice",
        "statmech",
        "N! is a convention only the user chooses",
        "refused",
        _statmech("indistinguishability-is-the-users-choice"),
        oracle="arithmetic",
        evidence="refusal",
    ),
]
