"""Did the user say which method to use? (Differential equations: ask unless they did.)

`ode_analyze` lists the methods that apply. A message that is only an equation, or "solve this", names
none: the assistant then lists them and asks, instead of running the first. The reading is a fixed set of
words about the user's own message; it never reads the model's text, and a doubt means "not named" (the
assistant asks), which costs one question and never a wrong answer.
"""

from __future__ import annotations

import re

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
