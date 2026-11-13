"""Exceptions raised while parsing or guarding user-supplied mathematics."""


class MinervaMathError(Exception):
    """Base class. Messages are safe to show to users (they never embed secrets)."""


class ParseError(MinervaMathError):
    """The input is not valid or not supported syntax."""


class UnsupportedError(MinervaMathError):
    """Valid input that this operation does not support (for example noncommuting symbols)."""


class LimitError(MinervaMathError):
    """The input exceeds a static safety limit (size, depth, magnitude)."""
