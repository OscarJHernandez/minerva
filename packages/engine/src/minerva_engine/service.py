"""Typed public entry point of the engine."""

from __future__ import annotations

import contextlib
import re
import time
from typing import Any, Literal

from .claims import ClaimCheck
from .differentiate import DifferentiateResult, DiffVariable
from .discovery import BUDGETS, DiscoveryReport, Pins
from .distributions.models import DistributionInput, DistributionResult
from .expansion import ExpansionReport
from .geometry import MAX_DIMENSION, GeometryInput, GeometryResult
from .integrate import IntegrateResult
from .invariants import InvariantReport
from .lagrangian import LagrangianReport
from .matrix import MatrixCalculusInput, MatrixCalculusResult, MatrixDiagnostics, MatrixStage
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
    TransformResult,
)
from .motion import MotionReport, MotionRequest
from .observe import Observer, SandboxCall, current_trace_id
from .perturbation import CompareRequest, PerturbationReport
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
        # Told about every sandbox call (see observe.py); None by default, and never allowed to fail a call.
        self.observer: Observer | None = None

    def _run(
        self, op: str, payload: dict[str, Any], timeout: float | None = None
    ) -> tuple[SandboxResult, int]:
        start = time.monotonic()
        limits = self.limits.model_copy(
            update={"timeout_seconds": timeout} if timeout is not None else {}
        )
        result = self.sandbox.run(op, {**payload, "limits": limits.model_dump()}, timeout=timeout)
        ms = int((time.monotonic() - start) * 1000)
        observer = self.observer
        if observer is not None:
            # Observing must never change a computation: whatever the observer does, the result stands.
            with contextlib.suppress(Exception):
                observer(SandboxCall(op, result.outcome.value, ms, current_trace_id()))
        return result, ms

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

    def integrate(
        self,
        expression: ExpressionInput,
        variable: str,
        lower: str | None = None,
        upper: str | None = None,
    ) -> IntegrateResult:
        """An antiderivative (verified by a numerical derivative) or, with both bounds, a definite
        integral (verified by independent quadrature)."""
        payload = {
            "expression": expression.model_dump(),
            "variable": variable,
            "lower": lower,
            "upper": upper,
        }
        outcome, ms = self._run("integrate", payload)
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return IntegrateResult(status=status, message=message, duration_ms=ms)
        result = IntegrateResult(**outcome.value)
        result.duration_ms = ms
        return result

    def distribution(self, request: DistributionInput) -> DistributionResult:
        """Simplify, integrate or differentiate an expression with Dirac deltas and step
        functions; shown only if the exact conditions hold and a test-function pairing agrees."""
        outcome, ms = self._run("distribution", {"input": request.model_dump()})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return DistributionResult(
                status=status,
                action=request.action,
                reason="resource_limit" if status is Status.RESOURCE_LIMIT else None,
                message=message,
                duration_ms=ms,
            )
        result = DistributionResult(**outcome.value)
        result.duration_ms = ms
        return result

    def matrix_calculus(self, request: MatrixCalculusInput) -> MatrixCalculusResult:
        """Evaluate, simplify, differentiate or integrate an expression in declared, noncommuting
        matrices and their exponentials; the method is chosen from the structure and the result is
        shown only if its own verification passed. The active trace id (see `observe.trace_scope`) is
        the diagnostics' trace id unless the request carries one."""
        active = current_trace_id()
        if (
            request.trace_id is None
            and active is not None
            and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", active)
        ):
            request = request.model_copy(update={"trace_id": active})
        outcome, ms = self._run("matrix_calculus", {"input": request.model_dump()})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return MatrixCalculusResult(
                status=status,
                action=request.action,
                message=message,
                duration_ms=ms,
                diagnostics=MatrixDiagnostics(
                    trace_id=request.trace_id,
                    stages=[MatrixStage(name="total", ms=float(ms))],
                    failure_reason=f"sandbox_{outcome.outcome.value}",
                ),
            )
        result = MatrixCalculusResult(**outcome.value)
        result.duration_ms = ms
        return result

    def geometry(
        self, request: GeometryInput, *, max_dimension: int = MAX_DIMENSION
    ) -> GeometryResult:
        """Metric, connection, curvature, geodesics and tensor operations in explicit coordinates.

        Every quantity is shown only if its independent numerical check passed; conventions,
        assumptions, domain restrictions and per-stage diagnostics are part of the result."""
        outcome, ms = self._run(
            "geometry", {"input": request.model_dump(), "max_dimension": max_dimension}
        )
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return GeometryResult(status=status, message=message, duration_ms=ms)
        result = GeometryResult(**outcome.value)
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

    def transform(
        self,
        expression: ExpressionInput,
        rules: list[Rule],
        goal: str = "simplify",
        target: list[str] | None = None,
        width: int = 4,
        depth: int = 4,
    ) -> TransformResult:
        """A bounded search for a chain of verified rewrites (handbook rules, SymPy moves) toward `goal`."""
        payload = {
            "expression": expression.model_dump(),
            "rules": [r.model_dump() for r in rules],
            "goal": goal,
            "target": target or [],
            "width": width,
            "depth": depth,
        }
        outcome, ms = self._run("transform", payload)
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return TransformResult(status=status, message=message, duration_ms=ms)
        result = TransformResult(**outcome.value)
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
        self,
        formulas: list[str],
        assumptions: dict[str, list[str]] | None = None,
        relations: list[str] | None = None,
        matrices: list[str] | None = None,
    ) -> list[ClaimCheck]:
        """Numerically test equations written by the model. Failures leave claims unchecked.

        `relations` are formulas stated earlier in the conversation; definitions among them are used
        by substitution (see `claims.py`). `matrices` are names declared as matrices in the
        conversation: equations that mention them are never tested by sampling numbers."""
        if not formulas:
            return []
        outcome, _ = self._run(
            "check_claims",
            {
                "formulas": formulas,
                "assumptions": assumptions or {},
                "relations": relations or [],
                "matrices": matrices or [],
            },
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
        format: Literal["latex", "text"] = "latex",
    ) -> LagrangianReport:
        """Equations of motion and verified constants of motion of a Lagrangian, sandboxed.

        LaTeX writes velocities `\\dot{q}`; in text they are `q_dot`."""
        outcome, ms = self._run(
            "lagrangian",
            {
                "lagrangian": lagrangian,
                "coordinates": coordinates,
                "time": time,
                "assumptions": assumptions or {},
                "format": format,
            },
        )
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return LagrangianReport(status=status, message=message, duration_ms=ms)
        report = LagrangianReport(**outcome.value)
        report.duration_ms = ms
        return report

    def expand_motion(
        self,
        lagrangian: str,
        coordinates: list[str],
        time: str = "t",
        assumptions: dict[str, list[str]] | None = None,
        equilibrium: dict[str, str] | None = None,
        order: int = 3,
    ) -> ExpansionReport:
        """Expand the accelerations about an equilibrium and give the linear modes (sandboxed)."""
        payload = {
            "lagrangian": lagrangian,
            "coordinates": coordinates,
            "time": time,
            "assumptions": assumptions or {},
            "equilibrium": equilibrium or {},
            "order": order,
        }
        outcome, ms = self._run("expand_motion", payload)
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return ExpansionReport(status=status, message=message, duration_ms=ms)
        report = ExpansionReport(**outcome.value)
        report.duration_ms = ms
        return report

    def perturbation_solve(
        self,
        lagrangian: str,
        coordinates: list[str],
        time: str = "t",
        assumptions: dict[str, list[str]] | None = None,
        small_parameter: str | None = None,
        equilibrium: str = "0",
        order: int = 2,
        compare: CompareRequest | None = None,
    ) -> PerturbationReport:
        """Poincare-Lindstedt solution of a one-degree-of-freedom oscillator (sandboxed)."""
        payload = {
            "lagrangian": lagrangian,
            "coordinates": coordinates,
            "time": time,
            "assumptions": assumptions or {},
            "small_parameter": small_parameter,
            "equilibrium": equilibrium,
            "order": order,
            "compare": compare.model_dump() if compare else None,
        }
        outcome, ms = self._run("perturbation_solve", payload)
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return PerturbationReport(status=status, message=message, duration_ms=ms)
        report = PerturbationReport(**outcome.value)
        report.duration_ms = ms
        return report

    def search_invariants(
        self,
        lagrangian: str,
        coordinates: list[str],
        angles: list[str] | None = None,
        degree: int = 2,
        power_bound: int = 3,
        harmonics: int = 1,
        time: str = "t",
        assumptions: dict[str, list[str]] | None = None,
        format: Literal["latex", "text"] = "latex",
    ) -> InvariantReport:
        """Search a declared family for constants of motion beyond Noether's easy cases, sandboxed."""
        outcome, ms = self._run(
            "search_invariants",
            {
                "lagrangian": lagrangian,
                "coordinates": coordinates,
                "angles": angles or [],
                "degree": degree,
                "power_bound": power_bound,
                "harmonics": harmonics,
                "time": time,
                "assumptions": assumptions or {},
                "format": format,
            },
        )
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return InvariantReport(status=status, message=message, duration_ms=ms)
        report = InvariantReport(**outcome.value)
        report.duration_ms = ms
        return report

    def discover_invariants(
        self,
        function: str,
        coordinates: list[str],
        system: str = "lagrangian",
        momenta: list[str] | None = None,
        angles: list[str] | None = None,
        budget: str = "standard",
        pins: Pins | None = None,
        time: str = "t",
        assumptions: dict[str, list[str]] | None = None,
    ) -> DiscoveryReport:
        """Adaptive search for conserved quantities of a Lagrangian or Hamiltonian, sandboxed."""
        seconds = BUDGETS[budget].seconds if budget in BUDGETS else self.limits.timeout_seconds
        outcome, ms = self._run(
            "discover_invariants",
            {
                "system": system,
                "function": function,
                "coordinates": coordinates,
                "momenta": momenta,
                "angles": angles or [],
                "budget": budget,
                "pins": pins.model_dump() if pins else None,
                "time": time,
                "assumptions": assumptions or {},
            },
            timeout=seconds,
        )
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return DiscoveryReport(status=status, message=message, duration_ms=ms)
        report = DiscoveryReport(**outcome.value)
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
