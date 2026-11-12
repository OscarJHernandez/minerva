"""Typed public entry point of the engine."""

from __future__ import annotations

import time
from typing import Any

from .claims import ClaimCheck
from .differentiate import DifferentiateResult, DiffVariable
from .lagrangian import LagrangianReport
from .models import (
    ExpressionInput,
    FormulaReading,
    IdentitiesResult,
    Limits,
    Rule,
    SimplifyResult,
    Status,
    TensorInput,
    TensorResult,
)
from .motion import MotionReport, MotionRequest
from .pi_groups import (
    ConstantCandidate,
    DimensionalReport,
    QuantitySpec,
    UnitsCheckReport,
)
from .sandbox import Outcome, Sandbox, SandboxResult
from .series import SeriesResult
from .solve import SolveReport


class Engine:
    """Runs operations in the sandbox and converts failures into explicit statuses."""

    def __init__(self, limits: Limits | None = None, sandbox: Sandbox | None = None) -> None:
        self.limits = limits or Limits()
        self.sandbox = sandbox or Sandbox(self.limits)

    def _run(self, op: str, payload: dict[str, Any]) -> tuple[SandboxResult, int]:
        start = time.monotonic()
        result = self.sandbox.run(op, {**payload, "limits": self.limits.model_dump()})
        return result, int((time.monotonic() - start) * 1000)

    @staticmethod
    def _failure_status(outcome: Outcome) -> tuple[Status, str]:
        if outcome is Outcome.TIMEOUT:
            return Status.RESOURCE_LIMIT, "The computation exceeded its time limit and was stopped."
        if outcome is Outcome.KILLED:
            return Status.RESOURCE_LIMIT, "The computation exceeded its resource limits."
        if outcome is Outcome.BUSY:
            return Status.RESOURCE_LIMIT, "The server is busy; try again shortly."
        return Status.INTERNAL_ERROR, "The computation failed unexpectedly."

    def warm_up(self) -> None:
        """Start the sandbox's fork server and import SymPy so the first request is fast."""
        self.features(ExpressionInput(input="x", format="text"))

    def simplify(self, expression: ExpressionInput) -> SimplifyResult:
        outcome, ms = self._run("simplify", {"expression": expression.model_dump()})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return SimplifyResult(status=status, message=message, duration_ms=ms)
        result = SimplifyResult(**outcome.value)
        result.duration_ms = ms
        return result

    def differentiate(
        self, expression: ExpressionInput, variables: list[DiffVariable]
    ) -> DifferentiateResult:
        """Derivative of `expression` (any mix of variables, total order up to 10), each step verified."""
        payload = {
            "expression": expression.model_dump(),
            "variables": [v.model_dump() for v in variables],
        }
        outcome, ms = self._run("differentiate", payload)
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return DifferentiateResult(status=status, message=message, duration_ms=ms)
        result = DifferentiateResult(**outcome.value)
        result.duration_ms = ms
        return result

    def series(
        self, expression: ExpressionInput, variable: str, point: str = "0", order: int = 6
    ) -> SeriesResult:
        """Taylor polynomial about a finite real point; coefficients and remainder order verified."""
        payload = {
            "expression": expression.model_dump(),
            "variable": variable,
            "point": point,
            "order": order,
        }
        outcome, ms = self._run("series", payload)
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return SeriesResult(status=status, message=message, duration_ms=ms)
        result = SeriesResult(**outcome.value)
        result.duration_ms = ms
        return result

    def apply_rules(self, expression: ExpressionInput, rules: list[Rule]) -> IdentitiesResult:
        payload = {
            "expression": expression.model_dump(),
            "rules": [r.model_dump() for r in rules],
        }
        outcome, ms = self._run("identities", payload)
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return IdentitiesResult(status=status, message=message, duration_ms=ms)
        result = IdentitiesResult(**outcome.value)
        result.duration_ms = ms
        return result

    def tensor_canonicalize(self, tensor: TensorInput) -> TensorResult:
        outcome, ms = self._run("tensor_canonicalize", {"tensor": tensor.model_dump()})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return TensorResult(status=status, message=message, duration_ms=ms)
        result = TensorResult(**outcome.value)
        result.duration_ms = ms
        return result

    def features(self, expression: ExpressionInput) -> tuple[Status, str, list[str], str | None]:
        """Structural features of the expression (parsed inside the sandbox)."""
        outcome, _ = self._run("features", {"expression": expression.model_dump()})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return status, message, [], None
        v = outcome.value
        return Status(v["status"]), v["message"], list(v["features"]), v["input_latex"]

    def check_claims(
        self, formulas: list[str], assumptions: dict[str, list[str]] | None = None
    ) -> list[ClaimCheck]:
        """Numerically test equations written by the model. Failures leave claims unchecked."""
        if not formulas:
            return []
        outcome, _ = self._run(
            "check_claims", {"formulas": formulas, "assumptions": assumptions or {}}
        )
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            return [
                ClaimCheck(status="unverified", detail="check did not complete") for _ in formulas
            ]
        return [ClaimCheck(**r) for r in outcome.value["results"]]

    def read_formulas(self, formulas: list[str]) -> list[FormulaReading]:
        """Show how the strict parser reads formulas transcribed from an image (sandboxed)."""
        if not formulas:
            return []
        outcome, _ = self._run("read_formulas", {"formulas": formulas})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return [FormulaReading(status=status, message=message) for _ in formulas]
        return [FormulaReading(**r) for r in outcome.value["results"]]

    def solve(
        self,
        equation: str,
        variable: str | None = None,
        assumptions: dict[str, list[str]] | None = None,
    ) -> SolveReport:
        """Solve a polynomial equation `lhs = rhs` (LaTeX) in the sandbox."""
        outcome, ms = self._run(
            "solve", {"equation": equation, "variable": variable, "assumptions": assumptions or {}}
        )
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return SolveReport(status=status, message=message, duration_ms=ms)
        report = SolveReport(**outcome.value)
        report.duration_ms = ms
        return report

    def analyze_lagrangian(
        self,
        lagrangian: str,
        coordinates: list[str],
        time: str = "t",
        assumptions: dict[str, list[str]] | None = None,
    ) -> LagrangianReport:
        """Equations of motion and verified constants of motion of a Lagrangian (LaTeX), sandboxed."""
        outcome, ms = self._run(
            "lagrangian",
            {
                "lagrangian": lagrangian,
                "coordinates": coordinates,
                "time": time,
                "assumptions": assumptions or {},
            },
        )
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return LagrangianReport(status=status, message=message, duration_ms=ms)
        report = LagrangianReport(**outcome.value)
        report.duration_ms = ms
        return report

    def dimensional_analysis(
        self,
        quantities: list[QuantitySpec],
        target: str | None = None,
        repeating: list[str] | None = None,
        candidates: list[ConstantCandidate] | None = None,
    ) -> DimensionalReport:
        """Dimensionless groups, and what is missing for the unknown, in the sandbox."""
        outcome, ms = self._run(
            "dimensional_analysis",
            {
                "quantities": [q.model_dump() for q in quantities],
                "target": target,
                "repeating": repeating or [],
                "candidates": [c.model_dump() for c in candidates or []],
            },
        )
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return DimensionalReport(status=status, message=message, duration_ms=ms)
        report = DimensionalReport(**outcome.value)
        report.duration_ms = ms
        return report

    def check_units(self, equation: str, quantities: list[QuantitySpec]) -> UnitsCheckReport:
        """Dimensional consistency of an equation (LaTeX) given units of its symbols, sandboxed."""
        outcome, ms = self._run(
            "check_units",
            {"equation": equation, "quantities": [q.model_dump() for q in quantities]},
        )
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return UnitsCheckReport(status=status, message=message, duration_ms=ms)
        report = UnitsCheckReport(**outcome.value)
        report.duration_ms = ms
        return report

    def integrate_motion(self, request: MotionRequest) -> MotionReport:
        """Numerical solution of a Lagrangian system (optionally a one-parameter sweep), verified, sandboxed."""
        outcome, ms = self._run("integrate_motion", request.model_dump())
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return MotionReport(status=status, message=message, duration_ms=ms)
        report = MotionReport(**outcome.value)
        report.duration_ms = ms
        return report
