"""Which of the two strict parsers reads an input written without saying (`format: auto`).

The input is never rewritten: this only picks LaTeX or text. A backslash means LaTeX; text markers (`*`, a
derivative name such as `theta_dot`, a name followed by `(`) with no backslash mean text; anything else is LaTeX.
"""

from __future__ import annotations

import re
from typing import Literal

from .derivatives import DERIVATIVE_NAME

Syntax = Literal["latex", "text"]
TEXT_MARKERS = re.compile(rf"\*|{DERIVATIVE_NAME}|\b[A-Za-z]\w*\s*\(")


def choose_syntax(source: str) -> Syntax:
    """LaTeX or text for `source`, by the rule above."""
    return "text" if "\\" not in source and TEXT_MARKERS.search(source) else "latex"
