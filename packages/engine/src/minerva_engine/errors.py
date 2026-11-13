"""Exceptions raised while parsing or guarding user-supplied mathematics."""


class MinervaMathError(Exception):
    """Base class. Messages are safe to show to users (they never embed secrets)."""


class ParseError(MinervaMathError):
    """The input is not valid or not supported syntax."""


class LimitError(MinervaMathError):
    """The input exceeds a static safety limit (size, depth, magnitude)."""


class UnsupportedInput(MinervaMathError):
    """The input parses but belongs to a different operation (reported as `unsupported`)."""


class RangeNotProvable(MinervaMathError):
    """A sum or product has symbolic bounds whose range is not provably non-empty."""
