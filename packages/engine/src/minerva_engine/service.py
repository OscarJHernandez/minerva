"""Typed public entry point of the engine."""

from __future__ import annotations

import time
from typing import Any

from .models import (
    ExpressionInput,
    IdentitiesResult,
    Limits,
    Rule,
    SimplifyResult,
    Status,
    TensorInput,
    TensorResult,
)
from .sandbox import Outcome, Sandbox, SandboxResult


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
