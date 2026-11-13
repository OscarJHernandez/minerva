"""Typed request/response models shared by the engine, knowledge base and API."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ALLOWED_ASSUMPTIONS: frozenset[str] = frozenset(
    {
        "real",
        "positive",
        "negative",
        "nonnegative",
        "nonpositive",
        "nonzero",
        "integer",
        "rational",
    }
)

# What a symbol may be declared as. `noncommutative` is not an assumption a handbook rule can require
# (it is not in ALLOWED_ASSUMPTIONS): it changes what multiplication means, so only operations that
# say they support it accept it (see `Engine._run`).
SYMBOL_DECLARATIONS: frozenset[str] = ALLOWED_ASSUMPTIONS | {"noncommutative"}

SYMBOL_NAME_MAX = 32


class Status(StrEnum):
    """Outcome of a computation. Anything other than OK/UNCHANGED is never a result."""

    OK = "ok"  # a different expression was found and verified
    UNCHANGED = "unchanged"  # no simpler form was found; the input is returned as is
    NOT_VERIFIED = "not_verified"  # a candidate was produced but could not be verified
    CONDITIONS_NOT_SATISFIED = "conditions_not_satisfied"  # needs assumptions not declared
    UNSUPPORTED = "unsupported"
    PARSE_ERROR = "parse_error"
    RESOURCE_LIMIT = "resource_limit"
    INTERNAL_ERROR = "internal_error"


class ExpressionInput(BaseModel):
    """A user expression plus declared assumptions about its symbols."""

    input: str = Field(min_length=1, max_length=20_000)
    format: Literal["latex", "text"] = "latex"
    assumptions: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("assumptions")
    @classmethod
    def _check_assumptions(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        if len(value) > 32:
            raise ValueError("too many symbols with assumptions")
        for name, items in value.items():
            if not name or len(name) > SYMBOL_NAME_MAX:
                raise ValueError("invalid symbol name in assumptions")
            unknown = set(items) - SYMBOL_DECLARATIONS
            if unknown:
                raise ValueError(f"unknown assumption(s): {sorted(unknown)}")
            if "noncommutative" in items and len(items) > 1:
                raise ValueError("a noncommutative symbol cannot carry other assumptions")
        return value


class Verification(BaseModel):
    """How a transformation was checked. `passed` is only true if a check actually ran."""

    symbolic_zero: bool = False  # simplify(original - result) == 0
    numeric_points_checked: int = 0
    numeric_points_agreed: int = 0
    numeric_points_skipped: int = 0  # sample points where both sides were undefined or failed
    passed: bool = False
    note: str = ""


class SimplifyResult(BaseModel):
    status: Status
    input_latex: str | None = None
    result_latex: str | None = None
    result_text: str | None = None
    verification: Verification | None = None
    # Points the result excludes that the input, as written, did not (and vice versa): `x \\neq 1`.
    restrictions_latex: list[str] = Field(default_factory=list)
    restrictions_complete: bool = True  # False: they could not be determined, see `message`
    message: str = ""
    duration_ms: int = 0


class RuleCondition(BaseModel):
    """Requirement on what a pattern variable is bound to, e.g. x positive."""

    variable: str
    assumption: str

    @field_validator("assumption")
    @classmethod
    def _known(cls, value: str) -> str:
        if value not in ALLOWED_ASSUMPTIONS:
            raise ValueError(f"unknown assumption {value!r}")
        return value


class Rule(BaseModel):
    """A rewrite rule `lhs -> rhs` (text syntax) over declared pattern variables."""

    id: str
    lhs: str
    rhs: str
    variables: list[str]
    conditions: list[RuleCondition] = Field(default_factory=list)

    @model_validator(mode="after")
    def _conditions_use_declared_variables(self) -> Rule:
        unknown = {c.variable for c in self.conditions} - set(self.variables)
        if unknown:
            raise ValueError(f"conditions refer to undeclared variable(s) {sorted(unknown)}")
        return self


class Applicability(StrEnum):
    APPLICABLE = "applicable"  # matched, conditions proven, rewrite verified
    CONDITIONS_NOT_SATISFIED = "conditions_not_satisfied"  # matched, but not provably valid
    NOT_VERIFIED = "not_verified"  # matched and conditions hold, rewrite failed verification
    NO_MATCH = "no_match"


class RuleApplication(BaseModel):
    rule_id: str
    applicability: Applicability
    matched_subexpression_latex: str | None = None
    rewritten_latex: str | None = None
    # Text form of the rewrite (re-readable by the text parser); set only when applicable.
    rewritten_text: str | None = None
    unmet_conditions: list[RuleCondition] = Field(default_factory=list)
    verification: Verification | None = None
    restrictions_latex: list[str] = Field(default_factory=list)
    restrictions_complete: bool = True
    message: str = ""


class IdentitiesResult(BaseModel):
    status: Status
    input_latex: str | None = None
    applications: list[RuleApplication] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0


TransformGoal = Literal["simplify", "expand", "factor", "rewrite"]


class TransformStep(BaseModel):
    """One rewrite in a chain. `name` is a handbook rule id, or a SymPy move (`sympy.expand`)."""

    kind: Literal["rule", "builtin"]
    name: str
    direction: Literal["forward", "reverse"] | None = None  # rules only
    matched_latex: str | None = None
    before_latex: str
    after_latex: str
    verification: Verification


class TransformChain(BaseModel):
    result_latex: str
    result_text: str
    steps: list[TransformStep]
    verification: Verification  # input -> result, end to end


class BlockedMove(BaseModel):
    """A rule that would improve the objective but whose conditions are not provable. Not applied."""

    rule_id: str
    direction: Literal["forward", "reverse"]
    matched_latex: str
    unmet_conditions: list[RuleCondition]


class TransformResult(BaseModel):
    status: Status
    goal: TransformGoal = "simplify"
    input_latex: str | None = None
    result_latex: str | None = None
    result_text: str | None = None
    steps: list[TransformStep] = Field(default_factory=list)
    verification: Verification | None = None
    alternatives: list[TransformChain] = Field(default_factory=list)
    blocked: list[BlockedMove] = Field(default_factory=list)
    candidates_tried: int = 0
    discarded_not_verified: int = 0
    search_complete: bool = True
    # Points the result excludes that the input, as written, did not: `x \\neq 1`.
    restrictions_latex: list[str] = Field(default_factory=list)
    restrictions_complete: bool = True
    message: str = ""
    duration_ms: int = 0


class FormulaReading(BaseModel):
    """How Minerva's strict parser reads one transcribed formula. Nothing is simplified or evaluated."""

    status: Status
    message: str = ""
    read_as: str | None = None  # the parsed sides re-rendered as LaTeX, joined by " = "


class Limits(BaseModel):
    """Static (pre-parse) and dynamic (sandbox) resource limits."""

    max_input_chars: int = 2_000
    max_tree_nodes: int = 2_000
    max_tree_depth: int = 60
    max_int_digits: int = 40
    max_pow_exponent: int = 10_000
    timeout_seconds: float = 10.0
    memory_mb: int = 1_024
    max_concurrency: int = 4


class TensorSpec(BaseModel):
    """Declared index structure and symmetry of a tensor head."""

    rank: int = Field(ge=1, le=6)
    symmetry: Literal["none", "symmetric", "antisymmetric", "riemann"] = "none"


class TensorInput(BaseModel):
    """A tensor polynomial such as `S(a,b)*A(-a,-b)`; a leading `-` marks a lowered index."""

    input: str = Field(min_length=1, max_length=2_000)
    tensors: dict[str, TensorSpec]
    dimension: int = Field(default=4, ge=1, le=64)

    @field_validator("tensors")
    @classmethod
    def _names(cls, value: dict[str, TensorSpec]) -> dict[str, TensorSpec]:
        if not value or len(value) > 16:
            raise ValueError("declare between 1 and 16 tensors")
        for name, spec in value.items():
            if not name.isidentifier() or len(name) > SYMBOL_NAME_MAX:
                raise ValueError("invalid tensor name")
            if spec.symmetry == "riemann" and spec.rank != 4:
                raise ValueError("riemann symmetry requires rank 4")
        return value


class TensorResult(BaseModel):
    status: Status
    input_text: str | None = None
    result_text: str | None = None
    result_latex: str | None = None
    independent_verification: bool = False
    message: str = ""
    duration_ms: int = 0


# --- controlled transformation steps ------------------------------------------------------------
class Reason(StrEnum):
    """Why a step was refused or a candidate dropped. A code, never user text: it is safe to log."""

    PATTERN_NOT_FOUND = "pattern_not_found"  # the relation's left side does not occur
    NO_MATCH = "no_match"  # the chosen rule or move does not apply to the chosen part
    STALE_SELECTION = "stale_selection"  # the chosen part is no longer in the expression
    AMBIGUOUS_SELECTION = "ambiguous_selection"  # the part occurs more than once
    CONDITIONS_UNMET = (
        "conditions_unmet"  # a rule's conditions are not provable from the assumptions
    )
    INVENTS_STRUCTURE = "invents_structure"
    NOT_VERIFIED = "not_verified"
    ORDER_DEPENDENT = "order_dependent"  # relations give different results in different orders
    CYCLIC = "cyclic"  # relations feed each other
    PREMISE_FALSE = "premise_false"  # a relation that can never hold, or relates only numbers
    NOT_AN_EQUATION = "not_an_equation"
    NONCOMMUTATIVE = "noncommutative"  # outside the supported noncommutative polynomial subset
    NOT_POLYNOMIAL = "not_polynomial"
    DEGREE = "degree"  # no closed form for this degree
    FORM = "form"  # the variable occurs in a form this version cannot invert
    EXTRANEOUS = "extraneous"  # a candidate that does not satisfy the original equation
    SIZE_LIMIT = "size_limit"
    DEADLINE = "deadline"


VerificationMethod = Literal[
    "none", "numeric", "congruence", "matrix", "back_substitution", "forward_probe"
]


class StepDiagnostics(BaseModel):
    """What a transformation did and why it ended as it did. Durations, counts and codes only: nothing
    here can hold an expression, so it is safe to log by default. Shown under the result and carried in
    the saved step and the chat debug report."""

    parse_ms: int = 0
    candidates_ms: int = 0
    transform_ms: int = 0
    verify_ms: int = 0
    candidates: int = 0
    rejected: dict[str, int] = Field(default_factory=dict)  # Reason value -> count
    verification_method: VerificationMethod = "none"
    verification_outcome: Literal["passed", "failed", "not_run"] = "not_run"
    symbolic_zero: bool = False
    restrictions_outcome: Literal["not_checked", "none", "some", "incomplete"] = "not_checked"
    assumptions_declared: int = 0
    premises: int = 0
    # Opt-in local debugging only (never set by default): the candidates that were rejected, as text.
    # `collect_detail` is not part of the model's data; `detail` stays empty unless it was requested.
    collect_detail: bool = Field(default=False, exclude=True)
    detail: list[str] = Field(default_factory=list)

    def note(self, text: str) -> None:
        """Record a line of detail (it may contain expressions) if debugging was asked for."""
        if self.collect_detail and len(self.detail) < 40:
            self.detail.append(text[:500])

    @contextmanager
    def timed(
        self, field: Literal["parse_ms", "candidates_ms", "transform_ms", "verify_ms"]
    ) -> Iterator[None]:
        """Add the time spent in the block to one of the phase durations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            setattr(self, field, getattr(self, field) + int((time.perf_counter() - start) * 1000))

    def reject(self, reason: Reason, count: int = 1) -> None:
        self.rejected[reason.value] = self.rejected.get(reason.value, 0) + count

    def log_fields(self) -> dict[str, Any]:
        """The flat, allow-listed fields for the server log line."""
        return {
            "parse_ms": self.parse_ms,
            "candidates_ms": self.candidates_ms,
            "transform_ms": self.transform_ms,
            "verify_ms": self.verify_ms,
            "candidates": self.candidates,
            "rejections": ",".join(f"{k}:{v}" for k, v in sorted(self.rejected.items())),
            "verification_method": self.verification_method,
            "verification_outcome": self.verification_outcome,
            "restrictions_outcome": self.restrictions_outcome,
        }


Evidence = Literal["identity", "conditional"]


class Premise(BaseModel):
    """A relation the user supplied. Minerva does not verify it: results that use it are conditional."""

    latex: str
    text: str


class Step(BaseModel):
    """One recorded transformation with its justification."""

    action: Literal["substitute", "builtin", "rule", "collect", "peel", "polynomial"]
    name: str = ""  # a rule id, a SymPy move (`sympy.expand`), or what was peeled
    direction: Literal["forward", "reverse"] | None = None
    matched_latex: str | None = None  # the part acted on
    before_latex: str
    after_latex: str
    # What a substitution looks like before the arithmetic SymPy does on construction: `x^2 + (1 - x^2)`.
    display_latex: str | None = None
    evidence: Evidence = "identity"
    given: list[str] = Field(default_factory=list)  # premises (LaTeX) this step relies on
    conditions_latex: list[str] = Field(default_factory=list)
    verification: Verification | None = None
    note: str = ""


class Subterm(BaseModel):
    """A part of an expression the user can choose. `path` indexes into the expression tree; `args`
    selects some of the terms or factors of a sum or product."""

    index: int
    path: list[int]
    args: list[int] | None = None
    latex: str
    text: str
    highlight_latex: str  # the whole expression with this part boxed
    occurrence: int = 1
    occurrences: int = 1
    blocked: bool = False  # a rule matches here but its conditions are not provable


class StepResult(BaseModel):
    status: Status
    operation: Literal["rewrite", "substitute", "subterms"]
    input_latex: str | None = None
    result_latex: str | None = None
    result_text: str | None = None
    steps: list[Step] = Field(default_factory=list)
    evidence: Evidence = "identity"
    given: list[Premise] = Field(default_factory=list)
    verification: Verification | None = None
    restrictions_latex: list[str] = Field(default_factory=list)
    restrictions_complete: bool = True
    conditions_latex: list[str] = Field(default_factory=list)
    unmet_conditions: list[RuleCondition] = Field(
        default_factory=list
    )  # a rule that was not applied
    reason: Reason | None = None
    # substitute: True if other orders of the relations give another result, None if not examined.
    order_dependent: bool | None = None
    subterms: list[Subterm] = Field(default_factory=list)
    truncated: bool = False
    diagnostics: StepDiagnostics = Field(default_factory=StepDiagnostics)
    message: str = ""
    duration_ms: int = 0


class Solution(BaseModel):
    """One value of the unknown. `text` is the right side in text syntax, `equation_text` is `x = ...`
    (what a derivation continues from). A solution that depends on an integer parameter (`k_1`) is a
    family: `branch` says so."""

    latex: str
    text: str
    equation_text: str
    conditions_latex: list[str] = Field(
        default_factory=list
    )  # conditions specific to this solution
    branch: str = ""
    verification: Verification | None = None


class IsolateResult(BaseModel):
    """The solutions of an equation for one variable, with the conditions under which each holds.

    Solutions are over the complex numbers: declared assumptions on the unknown are not used to select
    among them (`assumption_note`). An empty list with status ok means the equation has no solution.
    """

    status: Status
    equation_latex: str | None = None
    variable: str | None = None
    method: Literal["inversion", "polynomial"] | None = None
    steps: list[Step] = Field(default_factory=list)
    solutions: list[Solution] = Field(default_factory=list)
    conditions_latex: list[str] = Field(default_factory=list)  # conditions on every solution
    excluded_latex: list[str] = Field(default_factory=list)  # values the equation itself excludes
    # Parameter values (LaTeX equations like `a = 0`) at which the equation degenerates: the solutions
    # do not describe it there and it must be examined separately.
    exceptional: list[str] = Field(default_factory=list)
    verification: Verification | None = None
    complete: bool = False  # the check that no solution was dropped passed
    completeness_note: str = ""
    assumption_note: str = ""
    reason: Reason | None = None
    diagnostics: StepDiagnostics = Field(default_factory=StepDiagnostics)
    message: str = ""
    duration_ms: int = 0
