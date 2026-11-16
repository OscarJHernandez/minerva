"""Reasoning that a model writes into its reply, and how it is kept out of what the user reads.

Some models (a Gemma-style template served by Ollama, with thinking switched off) put a "thought"
channel into the visible text: `thought\\n ...reasoning... <channel|>the answer`, sometimes with nothing
between the two (`thought\\n<channel|>`), and sometimes it never reaches the closing marker (the call
ends at the length limit). Shown, that reasoning is a column of half-sentences and of formulas nobody
checked; stored, it is replayed into every later turn.

Two tools, both pure text handling that never reads a formula:

- `ReasoningGate` filters one model call's stream. A call whose text *starts* with a `thought` line is
  held back until the closing marker and dropped with it; a call that ends without the marker shows
  nothing. Ordinary replies are held for at most a few characters (until it is clear the text does not
  start that way), so they still stream live.
- `clean_reply` cleans a whole text (a stored reply, the text of a finished call): a leak at the start,
  a `thought` line with a closing marker later in the text, a `thought` line with no marker (everything
  from it on) and stray markers. A word such as "thoughtful" or a line that only mentions "thought" in
  a sentence is left alone.
"""

from __future__ import annotations

import re

MARKERS = ("<channel|>", "<|channel>", "<|channel|>")
_MARKER = re.compile(r"<\|?channel\|?>")
_THOUGHT_LINE = re.compile(r"[ \t]*thought[ \t]*\r?\n")
_LEAK_AT_START = re.compile(r"\s*thought[ \t]*\r?\n")
_SPAN = re.compile(r"(?:^|\n)[ \t]*thought[ \t]*\r?\n.*?<\|?channel\|?>[ \t]*", re.DOTALL)
_LONGEST_MARKER = max(len(m) for m in MARKERS)


def _partial_marker_at_end(text: str) -> int:
    """How many trailing characters could be the start of a marker split across two chunks."""
    for k in range(min(len(text), _LONGEST_MARKER - 1), 0, -1):
        tail = text[-k:]
        if any(marker.startswith(tail) for marker in MARKERS):
            return k
    return 0


def clean_reply(text: str) -> str:
    """The text without reasoning that leaked into it (see the module docstring)."""
    start = _LEAK_AT_START.match(text)
    if start:
        marker = _MARKER.search(text, start.end())
        text = text[marker.end() :].lstrip() if marker else ""
    text = _SPAN.sub("\n", text)
    # A `thought` line that never reached its marker: the call ended inside the reasoning.
    for match in re.finditer(r"(?:^|\n)(?=[ \t]*thought[ \t]*\r?\n)", text):
        if _THOUGHT_LINE.match(text, match.end()):
            text = text[: match.start()].rstrip()
            break
    return _MARKER.sub("", text)


class ReasoningGate:
    """Filter for the text of one model call, chunk by chunk: `feed` returns what may be shown now and
    `finish` what is left at the end of the call. Use a new gate per call."""

    def __init__(self) -> None:
        self._state = "start"  # start: undecided; dropping: inside a leak; pass: ordinary text
        self._buffer = ""
        self.leaked = False  # a leading reasoning block was seen (and dropped)

    def feed(self, text: str) -> str:
        if not text:
            return ""
        if self._state == "pass":
            return self._passing(text)
        self._buffer += text
        if self._state == "start":
            self._decide()
        if self._state == "dropping":
            return self._dropping()
        if self._state == "pass":
            text, self._buffer = self._buffer, ""
            return self._passing(text)
        return ""

    def finish(self) -> str:
        buffer, self._buffer = self._buffer, ""
        if self._state == "dropping":
            return ""
        if self._state == "pass":
            return buffer  # the end of a text that looked like a marker but was not one
        self._state = "pass"
        return clean_reply(buffer) if buffer.strip() else buffer

    # -- states ------------------------------------------------------------------------------------
    def _decide(self) -> None:
        stripped = self._buffer.lstrip()
        if not stripped or "thought".startswith(stripped):
            return  # not enough text yet to tell
        if not stripped.startswith("thought"):
            self._state = "pass"
            return
        rest = stripped[len("thought") :]
        if re.match(r"[ \t]*\r?\n", rest):
            self._state, self.leaked = "dropping", True
        elif rest.strip(" \t") == "":
            return  # "thought" and spaces so far: wait for the line to end
        else:
            self._state = "pass"  # "thoughtful ...", "thought about ..."

    def _dropping(self) -> str:
        marker = _MARKER.search(self._buffer)
        if marker is None:
            self._buffer = self._buffer[-_LONGEST_MARKER:]  # only a split marker can still matter
            return ""
        rest = self._buffer[marker.end() :].lstrip()
        self._buffer = ""
        self._state = "pass"
        return self._passing(rest) if rest else ""

    def _passing(self, text: str) -> str:
        text = self._buffer + text
        self._buffer = ""
        text = _MARKER.sub("", text)
        keep = _partial_marker_at_end(text)
        if keep:
            text, self._buffer = text[:-keep], text[-keep:]
        return text
