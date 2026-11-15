"""Models shared by the sum, expectation and statistics operations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..differentiate import DerivativeVerification
from ..distributions.models import Condition
from ..models import ExpressionInput, Status


class InstanceVerification(BaseModel):
    """Evidence for an identity between expressions with sums and indexed data.

    Both sides are expanded explicitly at several sizes (a sum over 1..n becomes n terms, each x[k] a
    symbol of its own) and the ordinary checks run on every instance. Agreement at several sizes with
    independent data is evidence, not a proof for every n: `general_symbolic_zero` stays None unless a
    symbolic zero for symbolic n was established.
    """

    method: str = (
        "explicit expansion at several sizes with independent data per element, then the sampled "
        "numeric check and the symbolic zero test on every instance"
    )
    sizes: list[dict[str, int]] = Field(default_factory=list)
    instances_checked: int = 0
    instances_agreed: int = 0
    exact_zero_instances: int = 0
    general_symbolic_zero: bool | None = None
    passed: bool = False
    note: str = ""

    def describe(self) -> str:
        if not self.sizes or not self.sizes[0]:
            return "checked on explicit instances"
        names = sorted(self.sizes[0])
        shown = "; ".join(", ".join(f"{k} = {s[k]}" for k in names) for s in self.sizes)
        return f"checked for {shown} (explicit expansion), not proved for every size"


SumAction = Literal[
    "expand", "evaluate_constant", "factor", "log_of_product", "differentiate", "check"
]
SumReason = Literal[
    "parse_error",
    "range_not_provable",
    "conditions_unmet",
    "unsupported_form",
    "verification_failed",
    "missing_candidate",
    "resource_limit",
]


class SumsInput(BaseModel):
    expression: ExpressionInput
    action: SumAction = "expand"
    variable: str | None = None  # differentiate
    # `check`: the answer to be checked, keyed like the benchmark outputs ({"result": "..."})
    candidate: dict[str, str] | None = None


class SumStep(BaseModel):
    title: str
    before_latex: str
    after_latex: str
    after_text: str
    evidence: Literal["identity", "conditional"] = "identity"


class SumsResult(BaseModel):
    status: Status
    action: SumAction
    reason: SumReason | None = None
    input_latex: str | None = None
    result_latex: str | None = None
    result_text: str | None = None
    steps: list[SumStep] = Field(default_factory=list)
    verification: InstanceVerification | None = None
    derivative_verification: DerivativeVerification | None = None
    conditions: list[Condition] = Field(default_factory=list)
    message: str = ""
    duration_ms: float = 0.0
