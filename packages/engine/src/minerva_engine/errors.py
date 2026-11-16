"""Exceptions raised while parsing or guarding user-supplied mathematics."""

from __future__ import annotations

from collections.abc import Iterable

from .notation.models import InputCode, InputDiagnostic


class MinervaMathError(Exception):
    """Base class. Messages are safe to show to users (they never embed secrets)."""


class ParseError(MinervaMathError):
    """The input is not valid or not supported syntax.

    The message is the whole story for a person. A reader that can say *what kind* of problem it is also
    passes a `code` (a closed `InputCode`), the user's own `symbols` involved and the `suggestions` that
    would work; callers that know the argument add its name with `diagnostic(field)`. All are optional, so
    an existing `raise ParseError("...")` is unchanged.
    """

    def __init__(
        self,
        message: str,
        *,
        code: InputCode | None = None,
        symbols: Iterable[str] = (),
        suggestions: Iterable[str] = (),
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.symbols = tuple(symbols)
        self.suggestions = tuple(suggestions)
        self.recoverable = recoverable

    def diagnostic(self, field: str = "") -> InputDiagnostic | None:
        if self.code is None:
            return None
        return InputDiagnostic(
            code=self.code,
            field=field,
            symbols=list(self.symbols),
            suggestions=list(self.suggestions),
            recoverable=self.recoverable,
        )


def diagnostic_of(exc: Exception, field: str = "latex") -> InputDiagnostic | None:
    """The typed reason carried by a parse error (None for any other exception or an uncoded one)."""
    return exc.diagnostic(field) if isinstance(exc, ParseError) else None


class LimitError(MinervaMathError):
    """The input exceeds a static safety limit (size, depth, magnitude)."""


class UnsupportedInput(MinervaMathError):
    """The input parses but belongs to a different operation (reported as `unsupported`)."""


class RangeNotProvable(MinervaMathError):
    """A sum or product has symbolic bounds whose range is not provably non-empty."""
