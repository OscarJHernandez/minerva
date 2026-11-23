"""How an input was read, and why it could not be: typed records shared by every operation.

Nothing here holds a mathematical result. A `Normalization` says that a spelling was read as another
(`Xdot` as the velocity of `X`), so the reading is shown instead of silent; an `InputDiagnostic` says what
could not be read in a closed vocabulary (`InputCode`) plus the user's own names and the spellings that would
work, so code can act on it without parsing prose. `symbols` and `suggestions` are the user's names: they
belong in the user's own card and stored message, never in a log line (only the `code` is logged).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InputCode(StrEnum):
    """Why an input could not be interpreted. Lower snake case: the code is also a log `reason_code`."""

    UNRECOGNIZED_DERIVATIVE = "unrecognized_derivative"
    AMBIGUOUS_DERIVATIVE = "ambiguous_derivative"
    MIXED_DERIVATIVE_SPELLINGS = "mixed_derivative_spellings"
    UNDECLARED_DEPENDENT = "undeclared_dependent"
    DECLARED_NAME_ABSENT = "declared_name_absent"
    ACCELERATION_IN_INPUT = "acceleration_in_input"
    TEXT_SYNTAX_IN_LATEX = "text_syntax_in_latex"
    LATEX_IN_TEXT = "latex_in_text"
    EQUATION_NOT_EXPRESSION = "equation_not_expression"
    UNKNOWN_COMMAND = "unknown_command"
    RESERVED_NAME = "reserved_name"
    UNBALANCED = "unbalanced"
    AMBIGUOUS_NOTATION = "ambiguous_notation"
    INPUT_TOO_LARGE = "input_too_large"
    SYNTAX = "syntax"
    DEPENDENT_ARGUMENT_NOT_VARIABLE = "dependent_argument_not_variable"
    DERIVATIVE_OF_EXPRESSION = "derivative_of_expression"
    TOTAL_DERIVATIVE_REQUESTED = "total_derivative_requested"
    # `time_derivative` of an expression that is not a function of the motion as written (bare names such
    # as constants of motion a tool found earlier), and the refusals of a definition for such a name.
    NO_DYNAMIC_DEPENDENCE = "no_dynamic_dependence"
    UNSUPPORTED_DEFINITION = "unsupported_definition"
    AMBIGUOUS_CONSTANT = "ambiguous_constant"
    CONSTANT_NOT_VERIFIED = "constant_not_verified"


class NormalizationKind(StrEnum):
    """Fixed vocabulary for what was changed in reading an input; counts of these may be logged."""

    DERIVATIVE_SPELLING = "derivative_spelling"
    FUNCTION_APPLICATION = "function_application"
    DERIVATIVE_CALL = "derivative_call"
    TRANSPORT = "transport"
    ASSIGNMENT = "assignment"
    SHAPE = "shape"
    FORMAT = "format"


class Normalization(BaseModel):
    """One change made to read an input, shown to the user and the model (never silent)."""

    model_config = ConfigDict(frozen=True)

    kind: NormalizationKind
    source: str
    target: str
    note: str = ""

    def describe(self) -> str:
        text = f"read `{self.source}` as `{self.target}`"
        return f"{text} ({self.note})" if self.note else text


class InputDiagnostic(BaseModel):
    """Why an input could not be read: a closed code, the field, the user's names and the fixes."""

    code: InputCode
    field: str = ""
    symbols: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    # True when the spelling can be fixed and the call repeated; False when the input is understood
    # but needs a decision (ambiguous) or belongs to another operation.
    recoverable: bool = False


class InputReading(BaseModel):
    """How an input that was accepted was read."""

    format: Literal["latex", "text"] = "latex"
    normalizations: list[Normalization] = Field(default_factory=list)


class InputAware(BaseModel):
    """Mixin for reports of operations that read user mathematics."""

    diagnostic: InputDiagnostic | None = None
    reading: InputReading | None = None
