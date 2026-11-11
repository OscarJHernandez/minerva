"""Schema of the knowledge base.

Entries live in YAML files under `data/entries/`; chapters in `data/chapters.yaml`.
Adding content means adding YAML, not code. Every entry declares how it can be verified
(`verification`) and the test suite runs the matching checker, so an entry cannot claim
more than it has been checked for.
"""

from __future__ import annotations

from typing import Annotated, Literal

from minerva_engine import Rule, RuleCondition
from pydantic import BaseModel, Field, field_validator, model_validator

EntryKind = Literal[
    "identity",
    "derivative",
    "integral",
    "series",
    "symmetry",
    "equation",
    "constant",
    "theorem",
]

# How an entry is checked by the test suite and the build:
#   symbolic    - SymPy proves/checks it (rule verification, differentiation)
#   numeric     - checked by numerical evaluation at sample points (integrals, series)
#   structural  - checked by the tensor canonicalizer
#   dimensional - dimensional consistency only (physics equations)
#   stated      - recorded from the literature, NOT machine-checked; never auto-applied
Verification = Literal["symbolic", "numeric", "structural", "dimensional", "stated"]


class Reference(BaseModel):
    """Pointer to a source. Citation only: no text from the source is reproduced."""

    source: str
    locator: str | None = None
    url: str | None = None


class SymbolDoc(BaseModel):
    name: str
    meaning: str
    unit: str | None = None  # SI base-unit expression such as "kg*m/s**2"; "1" if dimensionless


class DerivativeCheck(BaseModel):
    kind: Literal["derivative"] = "derivative"
    function: str
    variable: str
    result: str


class AntiderivativeCheck(BaseModel):
    kind: Literal["antiderivative"] = "antiderivative"
    integrand: str
    antiderivative: str
    variable: str


class ParamRange(BaseModel):
    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> ParamRange:
        if self.min >= self.max:
            raise ValueError("min must be below max")
        return self


class IntegralCheck(BaseModel):
    kind: Literal["integral"] = "integral"
    integrand: str
    variable: str
    lower: str
    upper: str
    result: str
    params: dict[str, ParamRange] = Field(default_factory=dict)


class SeriesCheck(BaseModel):
    kind: Literal["series"] = "series"
    term: str
    index: str
    lower: int
    upper: str = "oo"
    result: str
    params: dict[str, ParamRange] = Field(default_factory=dict)


class TensorCheck(BaseModel):
    kind: Literal["tensor"] = "tensor"
    input: str
    tensors: dict[str, dict[str, int | str]]
    expected: str


class EquationCheck(BaseModel):
    kind: Literal["equation"] = "equation"
    lhs: str
    rhs: str
    units: dict[str, str]


Check = Annotated[
    DerivativeCheck
    | AntiderivativeCheck
    | IntegralCheck
    | SeriesCheck
    | TensorCheck
    | EquationCheck,
    Field(discriminator="kind"),
]


class ConstantValue(BaseModel):
    value: str  # decimal string, e.g. "299792458" or "6.67430e-11"
    uncertainty: str | None = None  # None means exact (defined) value
    unit: str


class Entry(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
    title: str
    kind: EntryKind
    chapter: str  # chapter id from chapters.yaml, e.g. "math.trigonometry"
    section: str | None = None
    statement_latex: str | None = None  # defaults to lhs = rhs of `rule`
    validity: str = ""
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    reference: Reference | None = None
    provenance: Literal["original", "public-domain-fact"] = "original"
    verification: Verification
    rule: Rule | None = None
    check: Check | None = None
    symbols: list[SymbolDoc] = Field(default_factory=list)
    constant: ConstantValue | None = None
    notes: str = ""

    @field_validator("tags", "keywords")
    @classmethod
    def _short(cls, value: list[str]) -> list[str]:
        if len(value) > 30:
            raise ValueError("too many tags/keywords")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Entry:
        v = self.verification
        if v == "stated":
            if self.rule is not None or self.check is not None:
                raise ValueError("a `stated` entry must not carry machine-checkable content")
            if not self.statement_latex:
                raise ValueError("a `stated` entry needs statement_latex")
            return self
        if self.rule is None and self.check is None and self.constant is None:
            raise ValueError("a checked entry needs a `rule`, a `check` or a `constant`")
        if self.rule is None and not self.statement_latex:
            raise ValueError("entries without a rule need statement_latex")
        expected = {
            "symbolic": ("rule", "derivative", "antiderivative"),
            "numeric": ("rule", "derivative", "antiderivative", "integral", "series"),
            "structural": ("tensor",),
            "dimensional": ("equation",),
        }[v]
        source = "rule" if self.rule is not None else (self.check.kind if self.check else "")
        if self.constant is not None and self.rule is None and self.check is None:
            return self
        if source not in expected:
            raise ValueError(f"verification {v!r} does not match content {source!r}")
        return self


class Section(BaseModel):
    id: str
    title: str


class Chapter(BaseModel):
    id: str
    book: Literal["mathematics", "physics"]
    title: str
    summary: str = ""
    planned: bool = False  # in the outline but without entries yet
    sections: list[Section] = Field(default_factory=list)


__all__ = [
    "AntiderivativeCheck",
    "Chapter",
    "Check",
    "ConstantValue",
    "DerivativeCheck",
    "Entry",
    "EntryKind",
    "EquationCheck",
    "IntegralCheck",
    "ParamRange",
    "Reference",
    "Rule",
    "RuleCondition",
    "Section",
    "SeriesCheck",
    "SymbolDoc",
    "TensorCheck",
    "Verification",
]
