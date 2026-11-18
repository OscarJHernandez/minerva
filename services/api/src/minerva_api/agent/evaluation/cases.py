"""Benchmark cases: data, not code, so they can be checked, hashed and compared between runs.

Ground truth comes from the hand-written expectations of the engine's own tests and handbook benchmark
(`source` says which), never from what an agent produced. A case may carry a `reference` call: a tool
call that is correct for the prompt. The test suite runs every reference against the real engine and
requires that it meets `expect_card`, so the expectations are themselves checked.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

CASES_FILE = Path(__file__).with_name("cases.yaml")
RELIABILITY_FILE = Path(__file__).with_name("cases_reliability.yaml")
HELDOUT_FILE = Path(__file__).with_name("cases_heldout.yaml")
PROBE_FILE = Path(__file__).with_name("cases_probe.yaml")
# `heldout` and `probe` are not part of `all`: see cases_heldout.yaml and cases_probe.yaml.
SUITES = ("core", "reliability", "all", "heldout", "probe")

Category = Literal[
    "symbolic",  # simple symbolic calculations
    "multistep",  # transformations and tasks that need several tools in a row
    "calculus",  # integration, differentiation, series
    "retrieval",  # handbook and identity lookup
    "tensor_matrix",  # tensor and matrix expressions
    "physics",  # Lagrangians, units, dimensions; assumptions and definitions
    "ambiguous",  # the right move is a question, or no tool at all
    "unsupported",  # outside what the tools can do: the answer is a clear refusal
    "false_claim",  # the prompt or the temptation is an unjustified or false statement
    "followup",  # needs context from earlier turns
]
CATEGORIES: tuple[str, ...] = Category.__args__  # type: ignore[attr-defined]

# Not tools: the right behaviour is to answer with no tool call, or to ask a question first.
NO_TOOL = "none"
ASK = "ask"


class ArgCheck(BaseModel):
    """Declarative checks on the arguments of a call to the expected tool."""

    equals: dict[str, Any] = Field(default_factory=dict)  # argument == value
    present: list[str] = Field(default_factory=list)  # argument given and not empty
    absent: list[str] = Field(default_factory=list)  # argument missing or empty
    # argument is a list that contains each item (an item that is an object matches by subset)
    includes: dict[str, list[Any]] = Field(default_factory=dict)
    assumes: dict[str, str] = Field(default_factory=dict)  # symbol -> assumption it was given

    def holds(self, arguments: dict[str, Any]) -> bool:
        try:
            return self._holds(arguments)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False  # arguments of the wrong shape fail the case, they do not crash the run

    def _holds(self, arguments: dict[str, Any]) -> bool:
        if any(arguments.get(k) != v for k, v in self.equals.items()):
            return False
        if any(arguments.get(k) in (None, "", [], {}) for k in self.present):
            return False
        if any(arguments.get(k) not in (None, "", [], {}) for k in self.absent):
            return False
        for key, wanted in self.includes.items():
            have = arguments.get(key)
            if not isinstance(have, list):
                return False
            for item in wanted:
                if isinstance(item, dict):
                    if not any(isinstance(h, dict) and item.items() <= h.items() for h in have):
                        return False
                elif str(item) not in {str(h) for h in have}:
                    return False
        assumptions = arguments.get("assumptions") or {}
        for sym, assumption in self.assumes.items():
            given = assumptions.get(sym)
            if not isinstance(given, list) or assumption not in given:
                return False
        return True

    @property
    def about_assumptions(self) -> bool:
        return bool(self.assumes) or "assumptions" in self.absent or "assumptions" in self.present


class ExpectCard(BaseModel):
    """What the engine must return for the reference call (and what a correct run shows)."""

    status: str = "ok"
    # Text syntax. Compared as an expression (equal after parsing), so a different but equal form passes.
    result_text: str | None = None


class Reference(BaseModel):
    tool: str
    arguments: dict[str, Any]


class Followup(BaseModel):
    """A later turn of a live conversation: the agent has really answered the earlier ones.

    Its state (earlier results, restrictions, declared matrices, assumptions) is rebuilt from the earlier
    answers exactly as the chat route does, so `@r1` and inherited restrictions work as they do in production.
    The fields mean what they mean on `EvalCase`.
    """

    prompt: str
    expected_tool: str
    also_ok: list[str] = Field(default_factory=list)
    forbid: list[str] = Field(default_factory=list)
    tools_optional: bool = False
    sequence: list[str] = Field(default_factory=list)
    args: ArgCheck | None = None
    expect_card: ExpectCard | None = None
    reference: Reference | None = None  # a correct call for this turn, run by --check-cases
    reply_matches: str | None = None
    reply_forbids: str | None = None
    max_unchecked: int | None = None
    max_calls: int = 1
    wrong_formulas: list[str] = Field(default_factory=list)
    right_formulas: list[str] = Field(default_factory=list)
    must_reuse: bool = False
    restrictions_reported: str | None = None
    stops_after_failure: bool = False
    forbid_assumptions: bool = False

    def as_case(self, parent: EvalCase, number: int) -> EvalCase:
        return EvalCase(
            id=f"{parent.id}#{number}",
            category=parent.category,
            source=parent.source,
            informational=parent.informational,
            **self.model_dump(),
        )


class EvalCase(BaseModel):
    id: str
    category: Category
    prompt: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    # The tool a correct run reaches for first, or NO_TOOL / ASK.
    expected_tool: str
    also_ok: list[str] = Field(default_factory=list)  # other tools that are acceptable instead
    forbid: list[str] = Field(default_factory=list)
    # Calling no tool is also acceptable (a correct reply without one still has to pass every check).
    tools_optional: bool = False
    sequence: list[str] = Field(default_factory=list)  # tools that must all appear, in this order
    args: ArgCheck | None = None
    expect_card: ExpectCard | None = None
    reference: Reference | None = None
    reply_matches: str | None = None  # regular expression the reply must match (case-insensitive)
    reply_forbids: str | None = None  # ... and one it must not
    max_unchecked: int | None = None  # formulas the model wrote that Minerva could not check
    max_calls: int = 1  # calls beyond this count as unnecessary
    # Formulas (LaTeX) that are false or unjustified here: marking one as verified is the worst failure.
    wrong_formulas: list[str] = Field(default_factory=list)
    # Formulas (LaTeX) that are right here: marking one as contradicted is a checker false positive.
    right_formulas: list[str] = Field(default_factory=list)
    informational: bool = False  # reported, but not counted in the score
    # Words allowed in the reply (default: score.MAX_REPLY_WORDS). A prompt that asks for a derivation and its
    # verification needs more than a one-line answer; the cap is there to catch rambling, not explanation.
    max_reply_words: int | None = None
    source: str = ""  # where the expected outcome comes from (a test or benchmark file)
    # --- reliability checks (cases_reliability.yaml). All off by default, and left out of the case-set digest
    # while off, so the digest of the original cases does not change.
    # A later tool call must refer to an earlier verified result (`@r1`), not retype it.
    must_reuse: bool = False
    # Regular expression: the reply must state the domain restriction a result carries (x != 1 ...).
    restrictions_reported: str | None = None
    # After a tool result that is not verified (not_verified, error, ...) no further tool is called.
    stops_after_failure: bool = False
    # The user stated no assumptions, so none may be passed to a tool (no invented positivity, realness, ...).
    forbid_assumptions: bool = False
    followups: list[Followup] = Field(default_factory=list)  # later turns, answered live

    @field_validator("reply_matches", "reply_forbids", "restrictions_reported")
    @classmethod
    def _regex(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"not a regular expression: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _consistent(self) -> EvalCase:
        if self.expected_tool in (NO_TOOL, ASK) and (self.args or self.reference or self.sequence):
            raise ValueError(
                f"{self.id}: no tool is expected, so args/reference/sequence make no sense"
            )
        if self.expected_tool == ASK and not self.reply_matches:
            raise ValueError(
                f"{self.id}: a case that expects a question must say how to recognise it"
            )
        if self.reference and self.reference.tool != self.expected_tool:
            raise ValueError(f"{self.id}: the reference call must use the expected tool")
        if not self.source:
            raise ValueError(f"{self.id}: say where the expected outcome comes from")
        for number, turn in enumerate(self.followups, start=2):
            turn.as_case(self, number)  # validates the later turns the same way
        return self

    def turns(self) -> list[EvalCase]:
        """This case and each follow-up turn, as the cases that are scored one by one."""
        return [self, *(t.as_case(self, n) for n, t in enumerate(self.followups, start=2))]

    @property
    def expects_tool(self) -> bool:
        return self.expected_tool not in (NO_TOOL, ASK)

    @property
    def acceptable_tools(self) -> set[str]:
        return {self.expected_tool, *self.also_ok} if self.expects_tool else set()


def load_cases(path: Path = CASES_FILE) -> list[EvalCase]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{path.name} must contain a list of cases")
    cases = [EvalCase(**item) for item in raw]
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids")
    return cases


def load_suite(suite: str = "core") -> list[EvalCase]:
    """`core` (cases.yaml), `reliability` (cases_reliability.yaml: live multi-turn and assumption checks), both
    (`all`), `heldout` (cases_heldout.yaml: the same workflows in other words, run on their own) or `probe`
    (cases_probe.yaml: written before the baseline traces were read, frozen by a pinned digest)."""
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; choose from {', '.join(SUITES)}")
    cases: list[EvalCase] = []
    if suite in ("core", "all"):
        cases += load_cases(CASES_FILE)
    if suite in ("reliability", "all"):
        cases += load_cases(RELIABILITY_FILE)
    if suite == "heldout":
        cases += load_cases(HELDOUT_FILE)
    if suite == "probe":
        cases += load_cases(PROBE_FILE)
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids across suites")
    return cases


_RELIABILITY_FIELDS = (
    "max_reply_words",
    "must_reuse",
    "restrictions_reported",
    "stops_after_failure",
    "forbid_assumptions",
    "followups",
)


def _digest_view(case: EvalCase) -> str:
    """The case as hashed: reliability fields still at their default are left out, so adding them to the
    schema did not change the digest of any earlier case."""
    unused = {
        name
        for name in _RELIABILITY_FIELDS
        if getattr(case, name) == EvalCase.model_fields[name].get_default(call_default_factory=True)
    }
    return case.model_dump_json(exclude=unused)


def cases_digest(cases: list[EvalCase]) -> str:
    """Fingerprint of the case set, so two reports can tell whether they ran the same questions."""
    blob = "\n".join(_digest_view(c) for c in cases)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
