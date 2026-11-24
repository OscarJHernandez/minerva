"""Typed public entry point of the engine."""

from __future__ import annotations

import contextlib
import re
import time
from typing import Any, Literal

from .approx.models import ApproximationInput, ApproximationReport
from .claim_evidence import ClaimEvidence, SeriesEvidence
from .claims import ClaimCheck, clean_assumptions
from .creative_telescoping.models import TelescopingInput, TelescopingResult
from .differentiate import DifferentiateResult, DiffVariable
from .discovery import BUDGETS, DiscoveryReport, Pins
from .distributions.models import DistributionInput, DistributionResult
from .expansion import ExpansionReport
from .geometry import MAX_DIMENSION, GeometryInput, GeometryResult
from .identity_check import IdentityResult
from .indexed.models import SumsInput, SumsResult
from .integrate import IntegrateResult
from .invariants import InvariantReport
from .lagrangian import LagrangianReport
from .matrix import MatrixCalculusInput, MatrixCalculusResult, MatrixDiagnostics, MatrixStage
from .models import (
    ExpressionInput,
    ExpressionInspection,
    FormulaReading,
    IdentitiesResult,
    IsolateResult,
    Limits,
    Rule,
    SimplifyResult,
    Status,
    StepResult,
    TensorInput,
    TensorResult,
    TransformResult,
)
from .motion import MotionReport, MotionRequest
from .observe import Observer, SandboxCall, current_trace_id
from .ode import (
    OdeAnalysis,
    OdeAnalyzeInput,
    OdeDiagnostics,
    OdeInput,
    OdeSolution,
    OdeSolveInput,
    OdeStage,
)
from .ode.models import OdeNumericInput, OdeNumericReport
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
from .statmech.models import StatMechInput, StatMechResult
from .stats.models import ExpectationInput, ExpectationResult, StatModelInput, StatModelResult
from .variational import VariationInput, VariationReport
from .vector import VectorDiagnostics, VectorInput, VectorResult

# Operations that read `noncommutative` declarations (polynomial expressions only; see ncverify.py).
NONCOMMUTATIVE_OPERATIONS = frozenset({"substitute", "rewrite_at", "subterms"})


def _declares_noncommutative(value: Any) -> bool:
    """True if any `assumptions` mapping anywhere in the payload declares a noncommutative symbol."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "assumptions" and isinstance(item, dict):
                if any(isinstance(v, list) and "noncommutative" in v for v in item.values()):
                    return True
            elif _declares_noncommutative(item):
                return True
    elif isinstance(value, list):
        return any(_declares_noncommutative(item) for item in value)
    return False


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
        # One place decides which operations accept noncommuting symbols. Everything else reports
        # `unsupported` rather than silently treating them as commuting (which would be wrong).
        if op not in NONCOMMUTATIVE_OPERATIONS and _declares_noncommutative(payload):
            return SandboxResult(Outcome.UNSUPPORTED), 0
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
                observer(
                    SandboxCall(
                        op,
                        result.outcome.value,
                        ms,
                        current_trace_id(),
                        result.error_type,
                        result.error_module,
                        result.error_line,
                    )
                )
        return result, ms

    @staticmethod
    def _failure_status(outcome: Outcome) -> tuple[Status, str]:
        if outcome is Outcome.TIMEOUT:
            return Status.RESOURCE_LIMIT, "The computation exceeded its time limit and was stopped."
        if outcome is Outcome.KILLED:
            return Status.RESOURCE_LIMIT, "The computation exceeded its resource limits."
        if outcome is Outcome.BUSY:
            return Status.RESOURCE_LIMIT, "The server is busy; try again shortly."
        if outcome is Outcome.UNSUPPORTED:
            return Status.UNSUPPORTED, (
                "Noncommutative symbols are supported only by substitute, collect and rewriting "
                "one part of an expression; this operation does not accept them."
            )
        return Status.INTERNAL_ERROR, "The computation failed unexpectedly."

    def _step(self, op: str, payload: dict[str, Any], operation: str) -> StepResult:
        outcome, ms = self._run(op, payload)
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return StepResult(
                status=status,
                operation=operation,
                message=message,
                duration_ms=ms,
            )
        result = StepResult(**outcome.value)
        result.duration_ms = ms
        return result

    def subterms(
        self,
        expression: ExpressionInput,
        *,
        find: str | None = None,
        rule: Rule | None = None,
        reverse: bool = False,
        debug: bool = False,
    ) -> StepResult:
        """The parts of an expression that can be chosen: all, where `find` occurs, or where `rule` matches."""
        payload = {
            "expression": expression.model_dump(),
            "find": find,
            "rule": rule.model_dump() if rule else None,
            "reverse": reverse,
            "debug": debug,
        }
        return self._step("subterms", payload, "subterms")

    def rewrite_at(
        self,
        expression: ExpressionInput,
        action: str,
        *,
        selection: dict[str, Any] | None = None,
        part: str | None = None,
        occurrence: int | None = None,
        rule: Rule | None = None,
        reverse: bool = False,
        by: list[str] | None = None,
        target: str | None = None,
        debug: bool = False,
    ) -> StepResult:
        """Apply one move (`rule`, `replace`, `simplify`, `expand`, `factor`, `cancel`, `collect`) to one part.

        `replace` rewrites the part as the form in `target` (the user's); the whole expression is verified."""
        payload = {
            "expression": expression.model_dump(),
            "action": action,
            "selection": selection,
            "part": part,
            "occurrence": occurrence,
            "rule": rule.model_dump() if rule else None,
            "reverse": reverse,
            "by": by or [],
            "target": target,
            "debug": debug,
        }
        return self._step("rewrite_at", payload, "rewrite")

    def substitute(
        self,
        expression: ExpressionInput,
        relations: list[str],
        *,
        order: str | None = None,
        debug: bool = False,
        inherited_restrictions: list[str] | None = None,
    ) -> StepResult:
        """Replace the left side of each relation `l = r` by its right side, one relation at a time.

        The relations are premises (not verified by Minerva). If their order changes the result and
        `order` (`given` or `simultaneous`) is not stated, the request is refused.
        """
        payload = {
            "expression": expression.model_dump(),
            "relations": relations,
            "order": order,
            "debug": debug,
            "inherited_restrictions": inherited_restrictions or [],
        }
        return self._step("substitute", payload, "substitute")

    def isolate(
        self, equation: ExpressionInput, variable: str, *, debug: bool = False
    ) -> IsolateResult:
        """Solve `lhs = rhs` for one variable, listing every solution and the conditions it needs."""
        outcome, ms = self._run(
            "isolate", {"expression": equation.model_dump(), "variable": variable}
        )
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return IsolateResult(status=status, message=message, duration_ms=ms)
        result = IsolateResult(**outcome.value)
        result.duration_ms = ms
        return result

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
        strategy: str | None = None,
    ) -> IntegrateResult:
        """An antiderivative (verified by a numerical derivative) or, with both bounds, a definite
        integral (verified by independent quadrature). Where SymPy's route fails, or `strategy` asks
        for one (`auto` or a technique name), deterministic strategies are tried and the result says
        which one produced it."""
        payload = {
            "expression": expression.model_dump(),
            "variable": variable,
            "lower": lower,
            "upper": upper,
            "strategy": strategy,
        }
        outcome, ms = self._run("integrate", payload)
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return IntegrateResult(status=status, message=message, duration_ms=ms)
        result = IntegrateResult(**outcome.value)
        result.duration_ms = ms
        return result

    def statistical_model(self, request: StatModelInput) -> StatModelResult:
        """Moments, likelihoods, estimators, Fisher information and posteriors of a probability
        family, derived from its density; each result is shown only if an independent check passed."""
        outcome, ms = self._run("statistical_model", {"input": request.model_dump()})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return StatModelResult(
                status=status,
                action=request.action,
                reason="resource_limit" if status is Status.RESOURCE_LIMIT else None,
                message=message,
                duration_ms=ms,
            )
        result = StatModelResult(**outcome.value)
        result.duration_ms = ms
        return result

    def statmech(self, request: StatMechInput) -> StatMechResult:
        """Canonical-ensemble statistical mechanics of an explicit spectrum: partition function,
        thermodynamic derivations with a held-fixed set, and thermal averages. Every formula is
        shown only if an independent enumeration of the levels agrees."""
        outcome, ms = self._run("statmech", {"input": request.model_dump()})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return StatMechResult(
                status=status,
                action=request.action,
                reason="resource_limit" if status is Status.RESOURCE_LIMIT else None,
                message=message,
                duration_ms=ms,
            )
        result = StatMechResult(**outcome.value)
        result.duration_ms = ms
        return result

    def approximation(self, request: ApproximationInput) -> ApproximationReport:
        """Approximate an expression by the chosen method; shown only if an independent check agrees."""
        outcome, ms = self._run("approximation", {"input": request.model_dump()})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return ApproximationReport(
                status=status,
                reason="resource_limit" if status is Status.RESOURCE_LIMIT else None,
                method=request.method,
                message=message,
                duration_ms=ms,
            )
        report = ApproximationReport(**outcome.value)
        report.duration_ms = ms
        return report

    def creative_telescoping(self, request: TelescopingInput) -> TelescopingResult:
        """Find and prove a telescoping relation for a parameterised integral, derive the ODE of
        the integral, and evaluate it only if every condition and an independent check agree."""
        outcome, ms = self._run("creative_telescoping", {"input": request.model_dump()})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return TelescopingResult(
                status=status,
                reason="resource_limit" if status is Status.RESOURCE_LIMIT else None,
                message=message,
                variable=request.variable,
                parameter=request.parameter,
                duration_ms=ms,
            )
        result = TelescopingResult(**outcome.value)
        result.duration_ms = ms
        return result

    def expectation(self, request: ExpectationInput) -> ExpectationResult:
        """Expand E, Var and Cov of declared random variables by linearity; independence only where
        declared (the result is then conditional on it). Shown only if it agrees with the original on
        explicit finite joint distributions."""
        outcome, ms = self._run("expectation", {"input": request.model_dump()})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return ExpectationResult(
                status=status,
                action=request.action,
                reason="resource_limit" if status is Status.RESOURCE_LIMIT else None,
                message=message,
                duration_ms=ms,
            )
        result = ExpectationResult(**outcome.value)
        result.duration_ms = ms
        return result

    def sums(self, request: SumsInput) -> SumsResult:
        """Linearity, closed forms, factoring, log of a product and derivatives of finite sums; the
        result is shown only if it agrees with the original on explicit instances at several sizes."""
        outcome, ms = self._run("sums", {"input": request.model_dump()})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return SumsResult(
                status=status,
                action=request.action,
                reason="resource_limit" if status is Status.RESOURCE_LIMIT else None,
                message=message,
                duration_ms=ms,
            )
        result = SumsResult(**outcome.value)
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

    def ode_analyze(self, request: OdeInput | OdeAnalyzeInput) -> OdeAnalysis:
        """Read a differential equation or system, describe its structure and rank the approaches that
        can be tried on it. Nothing is solved; the active trace id becomes the diagnostics' trace id."""
        active = current_trace_id()
        trace_id = active if active and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", active) else None
        outcome, ms = self._run(
            "ode_analyze", {"input": request.model_dump(), "trace_id": trace_id}
        )
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return OdeAnalysis(
                status=status,
                message=message,
                duration_ms=ms,
                diagnostics=OdeDiagnostics(
                    trace_id=trace_id,
                    stages=[OdeStage(name="total", ms=float(ms))],
                    failure_reason=f"sandbox_{outcome.outcome.value}",
                ),
            )
        result = OdeAnalysis(**outcome.value)
        result.duration_ms = ms
        return result

    def ode_solve_symbolic(self, request: OdeInput | OdeSolveInput) -> OdeSolution:
        """Closed-form solutions through SymPy, each shown only after the equation as written holds
        for it at sampled points (derivatives taken numerically). Conditions are applied afterwards and
        checked separately. SymPy failing is `unsupported`, never "no solution exists"."""
        active = current_trace_id()
        trace_id = active if active and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", active) else None
        outcome, ms = self._run(
            "ode_solve_symbolic", {"input": request.model_dump(), "trace_id": trace_id}
        )
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return OdeSolution(
                status=status,
                message=message,
                duration_ms=ms,
                diagnostics=OdeDiagnostics(
                    trace_id=trace_id,
                    stages=[OdeStage(name="total", ms=float(ms))],
                    failure_reason=f"sandbox_{outcome.outcome.value}",
                ),
            )
        result = OdeSolution(**outcome.value)
        result.duration_ms = ms
        return result

    def ode_solve_numerical(self, request: OdeNumericInput) -> OdeNumericReport:
        """Integrate an initial-value problem whose every number the user supplied. A trajectory is shown
        only if an independent integrator agrees, refining the tolerance converges, the integral form
        of the equation holds over the steps and any verified first integral is conserved; it is an
        approximation on the interval given, never exact. The sandbox limit applies to the whole run."""
        active = current_trace_id()
        trace_id = active if active and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", active) else None
        outcome, ms = self._run(
            "ode_solve_numerical", {"input": request.model_dump(), "trace_id": trace_id}
        )
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return OdeNumericReport(
                status=status,
                message=message,
                method_requested=request.method,
                duration_ms=ms,
                diagnostics=OdeDiagnostics(
                    trace_id=trace_id,
                    stages=[OdeStage(name="total", ms=float(ms))],
                    failure_reason=f"sandbox_{outcome.outcome.value}",
                ),
            )
        result = OdeNumericReport(**outcome.value)
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

    def vector_calculus(self, request: VectorInput) -> VectorResult:
        """Gradient, divergence, curl, Laplacian and products of declared scalar and vector fields in
        Cartesian, cylindrical or spherical coordinates; Jacobians and conversions between systems.

        A result is shown only if the independent Cartesian finite-difference reading agrees; one
        that needs mixed partial derivatives to commute is withheld unless the field was declared
        smooth. The active trace id is the diagnostics' trace id unless the request carries one."""
        active = current_trace_id()
        if (
            request.trace_id is None
            and active is not None
            and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", active)
        ):
            request = request.model_copy(update={"trace_id": active})
        outcome, ms = self._run("vector_calculus", {"input": request.model_dump()})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return VectorResult(
                status=status,
                action=request.action,
                coordinates=request.coordinates,
                message=message,
                duration_ms=ms,
                diagnostics=VectorDiagnostics(
                    action=request.action,
                    coordinates=request.coordinates,
                    trace_id=request.trace_id,
                    reason="resource_limit"
                    if outcome.outcome is Outcome.TIMEOUT
                    else "sandbox_failure",
                ),
            )
        result = VectorResult(**outcome.value)
        result.duration_ms = ms
        return result

    def variation(self, request: VariationInput) -> VariationReport:
        """Functionals S[x] = ∫ L(t, x, x', ..., x^(n)) dt: variations, Euler-Lagrange equations,
        boundary terms, numerical action, stationary paths and second variations.

        A quantity is shown only if its independent numerical check passed."""
        outcome, ms = self._run("variation", {"input": request.model_dump()})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return VariationReport(
                status=status, message=message, action=request.action, duration_ms=ms
            )
        result = VariationReport(**outcome.value)
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

    def time_derivative(
        self,
        expression: ExpressionInput,
        coordinates: list[str],
        time: str = "t",
        lagrangian: str | None = None,
        definitions: dict[str, str] | None = None,
    ) -> DifferentiateResult:
        """Total time derivative along a motion (chain rule over coordinates, velocities, time).
        `definitions` (name -> expression) are inserted for the names of the expression, exactly."""
        payload = {
            "expression": expression.model_dump(),
            "coordinates": coordinates,
            "time": time,
            "lagrangian": lagrangian,
            "definitions": definitions or {},
        }
        outcome, ms = self._run("time_derivative", payload)
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

    def inspect_expression(
        self, expression: ExpressionInput, target: Literal["latex", "text"] | None = None
    ) -> ExpressionInspection:
        outcome, _ = self._run("inspect", {"expression": expression.model_dump(), "target": target})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return ExpressionInspection(status=status, message=message)
        return ExpressionInspection.model_validate(outcome.value)

    def features(self, expression: ExpressionInput) -> tuple[Status, str, list[str], str | None]:
        """Structural features of the expression (parsed inside the sandbox)."""
        outcome, _ = self._run("features", {"expression": expression.model_dump()})
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return status, message, [], None
        v = outcome.value
        return Status(v["status"]), v["message"], list(v["features"]), v["input_latex"]

    def check_identity(
        self, latex: str, assumptions: dict[str, list[str]] | None = None
    ) -> IdentityResult:
        """Is the one stated equation an identity? A verdict with a witness when it is not (see `identity_check.py`)."""
        clean, noncommutative = clean_assumptions(assumptions or {})
        outcome, ms = self._run(
            "check_identity",
            {"latex": latex, "assumptions": clean, "matrices": sorted(noncommutative)},
        )
        if outcome.outcome is not Outcome.OK or outcome.value is None:
            status, message = self._failure_status(outcome.outcome)
            return IdentityResult(status=status, message=message, duration_ms=ms)
        result = IdentityResult(**outcome.value)
        result.duration_ms = ms
        return result

    def check_claims(
        self,
        formulas: list[str],
        assumptions: dict[str, list[str]] | None = None,
        relations: list[str] | None = None,
        matrices: list[str] | None = None,
        vectors: list[str] | None = None,
        evidence: list[ClaimEvidence] | None = None,
        series: list[SeriesEvidence] | None = None,
    ) -> list[ClaimCheck]:
        """Numerically test equations written by the model. Failures leave claims unchecked.

        `relations` are formulas stated earlier in the conversation; definitions among them are used
        by substitution (see `claims.py`). `matrices` are names declared as matrices in the
        conversation: equations that mention them are never tested by sampling numbers. `vectors`
        are names declared as vector fields; vector notation is never sampled either way."""
        if not formulas:
            return []
        # A `noncommutative` declaration means "matrix" here; it must not reach the engine as an
        # assumption, where it would stop the whole check.
        clean, noncommutative = clean_assumptions(assumptions or {})
        outcome, _ = self._run(
            "check_claims",
            {
                "formulas": formulas,
                "assumptions": clean,
                "relations": relations or [],
                "matrices": sorted({*(matrices or []), *noncommutative}),
                "vectors": sorted(vectors or []),
                "evidence": [e.model_dump() for e in evidence or []],
                "series": [s.model_dump() for s in series or []],
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
        format: Literal["latex", "text"] = "latex",
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
                "format": format,
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
