"""Did the user say which method to use, or hand the numbers over? (Ask unless they did.)

`ode_analyze` lists the methods that apply. A message that is only an equation, or "solve this", names
none: the assistant then lists them and asks, instead of running the first. The reading is a fixed set of
words about the user's own message; it never reads the model's text, and a doubt means "not named" (the
assistant asks), which costs one question and never a wrong answer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from fractions import Fraction

_METHOD_WORDS = re.compile(
    r"\bclosed[- ]form\b|\bexact(?:ly)?\b|\banalytic(?:al(?:ly)?)?\b|\bsymbolic(?:ally)?\b|"
    r"\bgeneral solution\b|\bnumeric(?:al(?:ly)?)?\b|\brunge\b|\bsimulat\w+|\bpower series\b|"
    r"\btaylor\b|\bseries (?:solution|expansion)\b|\bperturb\w*|\bpoincar\w*|\blindstedt\b|"
    r"\bsmall parameter\b|\bphase (?:plane|portrait)\b|\bequilibri\w*|\bstability\b|\bstable\b|"
    r"\bfixed points?\b",
    re.IGNORECASE,
)
# A reply to the assistant's list of methods: "the first one", "2", "option 3", "the last".
_REPLY_WORDS = re.compile(
    r"^\s*(?:option\s*|method\s*|number\s*|no\.?\s*|#)?[1-9]\s*[.)]?\s*$|"
    r"\b(?:first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|last)\b(?:\s+(?:one|option|method))?|"
    r"\b(?:option|method|number)\s*[1-9]\b",
    re.IGNORECASE,
)


def names_a_method(text: str, *, replying: bool = False) -> bool:
    """True if the user's message names a solution method; with `replying` (the assistant has just
    listed methods) a pick by position also counts."""
    if _METHOD_WORDS.search(text):
        return True
    return replying and bool(_REPLY_WORDS.search(text))


# --- numbers for a numerical run ---------------------------------------------------------------------
# Initial values, parameter values and the end of the interval are the user's. The one exception is a
# user who says, in words, that the assistant should choose: then the assistant may, and the result says
# so. The phrases are strict (a doubt means "not delegated": the assistant asks) and only a *user*
# message counts.
_DELEGATION = re.compile(
    r"\b(?:you|minerva)\s+(?:can\s+|may\s+|should\s+|please\s+)?(?:choose|pick|decide|select|set)\b|"
    r"\b(?:choose|pick|select)\s+(?:some(?:thing|\s+\w+){0,2}|any(?:thing)?|whatever|reasonable|sensible|"
    r"sane|typical|default|good|suitable|the\s+values|them|these|values|numbers)\b|"
    r"\b(?:something|anything|whatever|values|numbers)\s+(?:that(?:'s|\s+is)\s+)?"
    r"(?:reasonable|sensible|sane|typical|suitable)\b|"
    r"\b(?:use|go\s+with)\s+(?:your|the|those|these)\s+(?:own\s+)?(?:example|suggest\w*|values|numbers|defaults?|"
    r"parameters|settings)\b|\bwhatever\s+(?:you|works|is\s+fine)\b|\bup\s+to\s+you\b",
    re.IGNORECASE,
)
# A question about what would be reasonable is not a hand-over.
_QUESTION = re.compile(
    r"\b(?:which|what|how)\b[^.?!]*\b(?:reasonable|sensible|values|numbers)\b[^.?!]*\?", re.I
)
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.^])[-+]?(?:\d+\s*/\s*\d+|\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"
)
DELEGATION_WINDOW = 3  # the message that delegates and the two after it ("try again", Retry)


def delegates_numbers(text: str) -> bool:
    """True if a user message hands the choice of numbers to the assistant."""
    if _QUESTION.search(text):
        return False
    return bool(_DELEGATION.search(text))


def numbers_in(text: str) -> list[float]:
    """The numbers written in a text (`0.5`, `1e-3`, `3/4`); a digit inside a name (`y1`, `r_0`) is not one."""
    out: list[float] = []
    for match in _NUMBER.finditer(text):
        token = match.group().replace(" ", "")
        try:
            out.append(float(Fraction(token)) if "/" in token else float(token))
        except (ValueError, ZeroDivisionError):
            continue
    return out


def delegation_active(user_messages: Sequence[str]) -> bool:
    """Whether the user handed the numbers over in one of the last few messages and has not since given
    numbers of their own. Oldest message first."""
    recent = list(user_messages)[-DELEGATION_WINDOW:]
    for index in range(len(recent) - 1, -1, -1):
        if delegates_numbers(recent[index]):
            return not any(numbers_in(later) for later in recent[index + 1 :])
    return False
