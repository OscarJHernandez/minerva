"""Check equations the language model wrote in its prose.

A claim like `\\frac{x^2-1}{x-1} = x+1 \\quad \\text{for } x \\neq 1` is parsed with the same strict parser
as user input and tested numerically at assumption-respecting sample points (complex by default).
`holds` means every valid sample agreed. `fails` is reserved for an equation that is false for every
value (the two sides differ by a nonzero constant). An equation that is merely false at generic points
is `not_identity`: Minerva cannot tell a false identity from an equation to solve, a definition or an
equation of motion (`x^2 = 4x`, `F = ma`, `m\\ddot x = -kx`), so it does not call it contradicted.
Anything else could not be decided. Conditions written next to the equation (`for x \\neq 1`) are ignored: the check is
"equal as functions at generic points", which is what simplification claims mean.

A *relation* (a symbol on one side only, or a time derivative: `\\theta = \\phi + \\omega t`, `F = ma`,
`m\\ddot x = -kx`) is not tested as an identity at all. It is checked only against definitions stated earlier in the
conversation, by substitution, and the result names them (`given`); Minerva has not verified those definitions.
A relation that disagrees with them after substitution `fails`; one they cannot decide stays `unverified`.

Two kinds of notation are never sampled, because the permissive parser reads them as something else: Big-O
(`O(t^3)` is the product of a symbol O and t^3) and an applied function that nobody declared (`g(t)` is g times t).
A claim written with either is `unverified`, never learned as a definition, and never "holds given" or "fails";
the one exception is `f(x) = P(x) + O((x-a)^{n+1})` that restates a series a tool verified, which is linked to
that result only if every part agrees (`claim_evidence.SeriesIndex`). The expansion point and the places where the
series tool measured its error are parameters of that call, not definitions (`x = 0` in "about x = 0"): they are
neither learned nor tested against.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

import sympy as sp
from pydantic import BaseModel

from .claim_evidence import (
    ClaimEvidence,
    EvidenceIndex,
    IndexLabel,
    SeriesEvidence,
    SeriesIndex,
    function_label,
    is_index_label,
    read_index_label,
)
from .distributions.guard import contains_distribution
from .errors import MinervaMathError
from .indexed.guard import contains_indexed
from .latexparse import _GREEK, parse_latex
from .models import ALLOWED_ASSUMPTIONS, Limits
from .restrictions import domain_restrictions
from .verify import verify_equal

ClaimStatus = Literal[
    "holds",
    "supported",
    "conditional",
    "fails",
    "not_identity",
    "unverified",
    "not_an_equation",
    "unparsed",
]

_TRAILER = re.compile(r"\\q?quad\b.*$|\\text\{[^}]*\}.*$|,\s*(?:for|if|where|when)\b.*$", re.DOTALL)


ClaimKind = Literal["identity", "relation"]

MAX_DEFINITIONS = 100
_SUBSTITUTION_ROUNDS = 20


class ClaimCheck(BaseModel):
    """`kind`: "identity" was tested for all values of its symbols; "relation" was not (see the module
    docstring). `given`: definitions stated earlier that the result depends on (not verified by Minerva)."""

    status: ClaimStatus
    detail: str = ""
    kind: ClaimKind | None = None
    given: list[str] = []
    # `supported`: the same statement as a result a tool verified (see `claim_evidence.py`): which result,
    # how the link was established, and where that result is valid.
    evidence_source: str | None = None
    evidence_rule: str | None = None
    evidence_basis: Literal["symbolic", "numeric"] | None = None
    restrictions: list[str] = []


def split_equation(latex: str) -> list[str] | None:
    """Split on top-level `=`; None if there is no equation."""
    text = _TRAILER.sub("", latex).strip().rstrip(" ,;.")
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        elif ch == "=" and depth == 0:
            parts.append(text[start:i])
            start = i + 1
        i += 1
    parts.append(text[start:])
    parts = [p.strip() for p in parts]
    return parts if len(parts) >= 2 and all(parts) else None


def check_claim(latex: str, assumptions: dict[str, list[str]], limits: Limits) -> ClaimCheck:
    """Check one equation on its own (no definitions from earlier in the conversation)."""
    return ClaimChecker(assumptions, limits).check(latex)


def check_claims(
    formulas: list[str],
    assumptions: dict[str, list[str]],
    relations: list[str],
    limits: Limits,
    matrices: Sequence[str] = (),
    vectors: Sequence[str] = (),
    evidence: Sequence[ClaimEvidence] = (),
    series: Sequence[SeriesEvidence] = (),
) -> list[ClaimCheck]:
    """Check equations in order. `relations` are formulas stated earlier in the conversation: the
    ones that are definitions (`\\theta = \\phi + \\omega t`) are used by substitution. Each checked
    formula that is itself a definition is remembered for the ones after it. `matrices` are names the
    conversation declared as matrices: an equation that mentions one is never tested by sampling
    numbers (that would treat them as commuting scalars). `vectors` are names the conversation
    declared as vector fields: the same rule applies, and so does the notation (`\\nabla`, bold or
    arrowed letters, `\\times` between non-numerals), which is never sampled whatever the names. `series` are
    the Taylor series tools verified in the conversation: a claim that restates one is linked to it, and the
    centre and error locations of one are parameters, not definitions (see the module docstring)."""
    clean, noncommutative = clean_assumptions(assumptions)
    checker = ClaimChecker(
        clean, limits, frozenset(matrices) | noncommutative, frozenset(vectors), evidence, series
    )
    for relation in relations:
        checker.learn(relation)
    results: list[ClaimCheck] = []
    for latex in formulas:
        try:
            result = checker.check(latex)
        except Exception:
            result = ClaimCheck(status="unparsed", detail="could not be checked")
        results.append(result)
        checker.learn(latex)
    return results


def clean_assumptions(
    assumptions: dict[str, list[str]],
) -> tuple[dict[str, list[str]], frozenset[str]]:
    """Split declared assumptions into the ones the parser can use and the names declared
    `noncommutative`. A noncommutative symbol is a matrix or operator for this check (never sampled
    as a number), and an assumption the parser does not know is dropped for that symbol instead of
    making every claim in the conversation unparseable."""
    clean: dict[str, list[str]] = {}
    noncommutative: set[str] = set()
    for name, items in assumptions.items():
        if not isinstance(items, list):
            continue
        if "noncommutative" in items:
            noncommutative.add(name)
            continue
        kept = [i for i in items if i in ALLOWED_ASSUMPTIONS]
        if kept:
            clean[name] = kept
    return clean, frozenset(noncommutative)


# A symbol in LaTeX source: a command (`\Omega`) or one letter with an optional subscript.
_SYMBOL_TOKEN = re.compile(r"\\([A-Za-z]+)|([A-Za-z])(?:_\{?([A-Za-z0-9,]+)\}?)?")
# Transpose and adjoint notation: the parser would read `^T` as a power with a symbol T.
_SINGLE_SYMBOL = re.compile(r"[A-Za-z](?:_\{?[A-Za-z0-9,]+\}?)?")
_TRANSPOSE = re.compile(
    r"\^\s*\{?\s*(?:\\(?:mathrm|mathsf|text|operatorname)\s*\{\s*)?"
    r"(?:[TH](?![A-Za-z])|\\(?:top|dagger|ast|intercal)\b|\*)"
)
_TRANSPOSE_DETAIL = (
    "this uses transpose or adjoint notation; Minerva reads a superscript T as a symbol, not as a "
    "transpose, so it does not test it"
)
_MATRIX_DETAIL = (
    "this mentions a matrix declared in the conversation; Minerva does not test "
    "matrix equations by sampling numbers (that would treat the matrices as "
    "commuting scalars), so it is not checked unless a matrix calculation produced it"
)


# Vector notation. The strict parser reads `\times` and `\cdot` as scalar products and does not
# know `\nabla`, so a vector statement must be recognised from its text before any sampling:
# `a \times b = b \times a` is true of commuting scalars and false of vectors.
_VECTOR_COMMAND = re.compile(
    r"\\(?:nabla|mathbf|boldsymbol|bm|vec|hat|overrightarrow|curl|grad|div)\b|\\partial\b"
)
_TIMES = re.compile(r"\\times\b")
_NUMERAL_BEFORE = re.compile(r"(?:\d|\d\s*\^\s*\{?\s*[-+]?\d+\s*\}?)\s*$")
_NUMERAL_AFTER = re.compile(r"^\s*\d")
_VECTOR_DETAIL = (
    "this uses vector notation (\u2207, bold or arrowed vectors, a cross product); Minerva does not "
    "test vector equations by sampling numbers (that would treat vectors as commuting scalars), so "
    "it is not checked unless a vector calculation produced it"
)


def uses_vector_notation(latex: str) -> bool:
    """True if `latex` is written with vector operators or a cross product. `\\times` between two
    numerals (`3 \\times 10^{8}`) or with a numeral on one side (`2 \\times x`) is ordinary
    multiplication; between anything else it may be a cross product."""
    if _VECTOR_COMMAND.search(latex):
        return True
    for match in _TIMES.finditer(latex):
        before, after = latex[: match.start()], latex[match.end() :]
        if not (_NUMERAL_BEFORE.search(before) or _NUMERAL_AFTER.search(after)):
            return True
    return False


def _norm(name: str) -> str:
    return name.replace("{", "").replace("}", "").replace(",", "")


def mentions_name(latex: str, names: frozenset[str] | set[str]) -> bool:
    """True if the source text of `latex` contains one of `names` as a symbol. Read from the text,
    before the parser can cancel anything: `AB - BA` is zero as commuting symbols, and the matrices
    would no longer be visible in the parsed expression. A declared `A` covers `A_1`, `A_{ij}`."""
    if not names:
        return False
    declared = {_norm(n) for n in names}
    for match in _SYMBOL_TOKEN.finditer(latex):
        command, letter, subscript = match.groups()
        if command is not None:
            token, base = _norm(command), _norm(command)
        else:
            base = letter
            token = f"{letter}_{_norm(subscript)}" if subscript else letter
        if token in declared or base in declared:
            return True
    return False


def _text(latex: str) -> str:
    return _TRAILER.sub("", latex).strip().rstrip(" ,;.")


def _definition(parsed: list[sp.Expr]) -> tuple[sp.Symbol, sp.Expr] | None:
    """`symbol = expression` where the expression does not contain the symbol."""
    if len(parsed) != 2:
        return None
    a, b = parsed
    if isinstance(a, sp.Symbol) and a not in b.free_symbols:
        return a, b
    if isinstance(b, sp.Symbol) and b not in a.free_symbols:
        return b, a
    return None


_DISTRIBUTION_NOTATION = re.compile(
    r"\\(?:delta(?:\s*\^\s*(?:\{\s*(?:\(\s*3\s*\)|3)\s*\}|3))?|theta|Theta|vartheta)\s*(?:\\left)?\s*\("
)
_DISTRIBUTION_DETAIL = (
    "could not check: `\\delta(...)` or `\\theta(...)` may be a Dirac delta or step function, or "
    "only a function of its argument; it is never tested by sampling (the distribution tool reads a "
    "delta only for a reading the user confirmed)"
)
_IMPLICATION = re.compile(
    r"\\(?:implies|Rightarrow|Longrightarrow|iff|Leftrightarrow|Longleftrightarrow)\b"
)
_IMPLICATION_DETAIL = "an implication between statements is not tested as an equation; each side may be checked on its own"
# E and I are Euler's number and the imaginary unit in Minerva's input: `E = ...` as a definition of an
# energy cannot be read, and is not a claim about the constant.
_RESERVED_LONE = re.compile(r"(?:E|I)")
_RESERVED_NAME = r"Q_{c}"
_RESERVED_DETAIL = (
    "a definition of a quantity named E or I; Minerva reads those letters as Euler's number and the "
    "imaginary unit, so it is not tested (a name such as E_0 or W would be)"
)
_SPACES = re.compile(r"\s+")
# A component of a tensor or connection written by its label (`\Gamma^{r}{}_{t t} = ...`, `claim_evidence`): it is
# linked to a verified row with the same label, or not tested at all; never read as powers or products.
_LABEL_NO_ROW_DETAIL = (
    "an indexed component (a tensor or connection component, read as a label, not as powers or products); "
    "no verified result in this conversation lists a component with this label, so it is not tested"
)
_LABEL_RELATION_DETAIL = (
    "relates indexed components (read as labels, not as powers or products); not tested"
)
_LABEL_VALUE_DETAIL = "an indexed component; this value could not be read, so it is not compared with the verified component"
_INDEXED_DETAIL = (
    "could not check: sums, indexed data and expectations cannot be tested by sampling numbers "
    "(use the sums or expectation operation)"
)

# Leibniz notation: \frac{dy}{dt}, \frac{d^2 y}{dx^2}, \frac{\mathrm{d}}{\mathrm{d}x}, \frac{\partial f}{\partial x}.
# The general parser cancels the d's (it reads \frac{dy}{dt} as y/t), which would let a differential
# equation be "checked" as an identity about ratios.
_DERIVATIVE = r"(?:\\mathrm\{d\}|\\operatorname\{d\}|\\partial|d)"
_LEIBNIZ = re.compile(
    rf"\\[dt]?frac\s*\{{\s*{_DERIVATIVE}[^{{}}]*\}}\s*\{{\s*{_DERIVATIVE}", re.DOTALL
)
_LEIBNIZ_DETAIL = (
    "this uses derivative notation; an equation involving a derivative is a differential equation, "
    "not an identity, and is not tested by sampling (use the ODE tools to analyse or solve it)"
)

# A bare differential (`dx`, `du`, `d\\theta`) in a relation: after a change of variable (`u = \\pi/2 - x`, so
# `dx = -du`) it is notation, not the product of a symbol d and a symbol x. Read as that product, a correct
# relation was "contradicted" by the definition of u. The `d` must not follow a letter or a backslash
# (`ad`, `\\delta`).
_DIFFERENTIAL = re.compile(r"(?<![A-Za-z\\])d(?:\\[A-Za-z]+|[A-Za-z])(?![A-Za-z])")
_DIFFERENTIAL_DETAIL = (
    "this relates differentials (dx, du, ...); a differential is notation for a change of variable, not "
    "a product of symbols, so it is not tested by sampling"
)


# Big-O and little-o: `O(t^3)`, `O\left(t^3\right)`, `\mathcal{O}(...)`, `\operatorname{O}(...)`, `o(t)`. The parser reads
# `O(t^3)` as the product O * t^3, and substituting a point into that tests another statement: pointwise agreement
# never establishes an asymptotic remainder.
_BIG_O = re.compile(
    r"\\(?:mathcal|mathscr|mathrm|operatorname)\s*\{\s*[Oo]\s*\}"
    r"|(?<![A-Za-z\\_])[Oo]\s*(?:\\left\s*)?\("
)
_BIG_O_DETAIL = (
    "this uses Big-O notation (an asymptotic remainder); pointwise agreement cannot establish it, so it "
    "is not tested unless it restates a Taylor series the series tool computed term for term"
)
# An applied function nobody declared: a letter (or a Greek command) directly followed by a parenthesis. The
# permissive parser reads `g(t)` as g times t. It is a product only if the name is also used as a plain symbol in
# the same formula (`x(x+1)`, `a(b+c) = ab + ac`, `n(n+1)`); otherwise the notation is not read at all.
_GREEK_NAMES = "|".join(sorted(_GREEK, key=len, reverse=True))
_APPLIED = re.compile(
    rf"(?<![A-Za-z\\])(?P<name>\\(?:{_GREEK_NAMES})(?![A-Za-z])|[A-Za-z](?:_\{{[^{{}}]*\}}|_[A-Za-z0-9])?)"
    r"(?:\\left\s*)?\("
)
_FUNCTION_DETAIL = (
    "this applies a name to an argument (`g(t)`) that the conversation did not declare as a function; "
    "Minerva would have to read it as a product, so it is not tested"
)
_LOCATION_DETAIL = (
    "the expansion point or a place where the series tool measured its error: a parameter of that call, "
    "not a definition; not tested"
)
_REMAINDER_HEAD = re.compile(
    r"\+\s*(?:\\(?:mathcal|mathscr|mathrm|operatorname)\s*\{\s*O\s*\}|O)\s*(?:\\left\s*)?\("
)


def applied_function_notation(latex: str) -> bool:
    """True if a name is applied to an argument without a space (`g(t)`) and is not also used as a plain
    symbol elsewhere in the formula, so that reading it as a product would be a guess."""
    text = _text(latex)
    for match in _APPLIED.finditer(text):
        name = match.group("name")
        if name.startswith("\\"):
            wanted = name[1:]
        else:
            letter, _, subscript = name.partition("_")
            wanted = f"{letter}_{_norm(subscript)}" if subscript else letter
        used_plain = False
        for token in _SYMBOL_TOKEN.finditer(text):
            if token.start() == match.start("name"):
                continue
            command, letter_, subscript_ = token.groups()
            if command is not None:
                key = command
            else:
                key = f"{letter_}_{_norm(subscript_)}" if subscript_ else letter_
            if key != wanted:
                continue
            after = text[token.end() :]
            if after.startswith(("(", "\\left(")):
                continue
            used_plain = True
            break
        if not used_plain:
            return True
    return False


def split_remainder(side: str) -> tuple[str, str] | None:
    """`P + O(arg)` -> (`P`, `arg`) when the side ends with one Big-O term; None otherwise."""
    head = _REMAINDER_HEAD.search(side)
    if head is None:
        return None
    body = side[head.end() :]
    depth = 1
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                if body[i + 1 :].strip():
                    return None
                argument = re.sub(r"\\right\s*$", "", body[:i]).strip()
                polynomial = side[: head.start()].strip()
                return (polynomial, argument) if polynomial and argument else None
    return None


class ClaimChecker:
    """Checks equations one after another against the definitions stated so far."""

    def __init__(
        self,
        assumptions: dict[str, list[str]],
        limits: Limits,
        matrices: frozenset[str] = frozenset(),
        vectors: frozenset[str] = frozenset(),
        evidence: Sequence[ClaimEvidence] = (),
        series: Sequence[SeriesEvidence] = (),
    ) -> None:
        self.assumptions = assumptions
        self.limits = limits
        self.matrices = matrices
        self.vectors = vectors
        self.evidence = EvidenceIndex(list(evidence), assumptions, limits) if evidence else None
        self.series = SeriesIndex(list(series), limits) if series else None
        # `g(x) = expression` as stated in the conversation (latest wins): only a series claim written `g(x) = ...` reads it
        self.function_definitions: dict[tuple[str, str], str] = {}
        # symbol -> (right side with earlier definitions substituted, its text, texts it was built on)
        self.definitions: dict[sp.Symbol, tuple[sp.Expr, str, list[str]]] = {}
        # the definitions that were accepted, in order: a redefinition rebuilds the others without the old one
        self._log: list[tuple[str, sp.Symbol]] = []
        self._replaying = False

    def _parse(self, latex: str) -> list[sp.Expr] | None:
        sides = split_equation(latex)
        if sides is None:
            return None
        try:
            return [parse_latex(side, self.assumptions, self.limits) for side in sides]
        except MinervaMathError:
            return None

    def learn(self, latex: str) -> None:
        """Remember `latex` if it is a definition. A later definition of a symbol replaces the earlier
        one; one that would define a symbol in terms of itself is ignored."""
        if self._is_vector(latex):
            return  # a vector statement is never a scalar definition
        self._remember_function(latex)
        if _BIG_O.search(latex) or applied_function_notation(latex):
            return  # read as a product, it would define a symbol by a statement about something else
        if self.series is not None and self.series.location(split_equation(latex) or []):
            return  # the centre or a measurement point of a series result: a parameter, not a definition
        if any(self._labels(split_equation(latex) or [])):
            return  # a component of a tensor or connection, written by its label: not a scalar definition
        if mentions_name(_text(latex), self.matrices) or _TRANSPOSE.search(latex):
            self._learn_alias(latex)
            return  # a definition of or by a matrix is not a scalar substitution
        parsed = self._parse(latex)
        if parsed and self._mentions_matrix(parsed):
            return
        if parsed and any(contains_distribution(p) or contains_indexed(p) for p in parsed):
            return
        found = _definition(parsed) if parsed else None
        if found is None:
            return
        symbol, raw = found
        (rhs,), used = self._substitute([raw])
        if symbol in rhs.free_symbols:
            return
        if symbol not in self.definitions and len(self.definitions) >= MAX_DEFINITIONS:
            return
        if (
            not self._replaying
            and symbol in self.definitions
            and self.definitions[symbol][0] != rhs
        ):
            # A redefinition. A definition made in terms of the old value (`K = 48 M^2 / r^6` while `r = 0` held)
            # has that value baked in, and the new one (`r = 2M`) would never reach it: it was then compared
            # with a number that came from a statement the conversation had since replaced. Rebuild the others
            # without the old definition of `symbol`, so what was written in terms of it follows the new value.
            kept = [text for text, defined in self._log if defined != symbol]
            self.definitions, self._log = {}, []
            self._replaying = True
            try:
                for text in kept:
                    self.learn(text)
            finally:
                self._replaying = False
            (rhs,), used = self._substitute([raw])
        self.definitions[symbol] = (rhs, _text(latex), self._given(used))
        self._log.append((latex, symbol))

    def _remember_function(self, latex: str) -> None:
        """`g(x) = \\sin x`: what the conversation said `g` is. Read only to link a series claim written `g(x) = ...`."""
        sides = split_equation(latex)
        if sides is None or len(sides) != 2 or _BIG_O.search(latex):
            return
        label = function_label(sides[0])
        if label is not None:
            self.function_definitions[label] = sides[1]

    def _is_vector(self, latex: str) -> bool:
        text = _text(latex)
        return uses_vector_notation(text) or mentions_name(text, self.vectors)

    def _learn_alias(self, latex: str) -> None:
        """`C = AB` with A, B matrices: C is a matrix too, so `CD = DC` is never sampled either."""
        sides = split_equation(latex)
        if sides is None or len(sides) != 2:
            return
        for one, other in (sides, sides[::-1]):
            if _SINGLE_SYMBOL.fullmatch(one) and mentions_name(other, self.matrices):
                self.matrices = self.matrices | {_norm(one)}

    def _mentions_matrix(self, parsed: list[sp.Expr]) -> bool:
        return bool(self.matrices) and any(
            s.name in self.matrices for p in parsed for s in p.free_symbols
        )

    def _given(self, used: list[sp.Symbol]) -> list[str]:
        """Texts of the definitions in `used` and of those they were built on, each once."""
        given: list[str] = []
        for s in used:
            for text in [*self.definitions[s][2], self.definitions[s][1]]:
                if text not in given:
                    given.append(text)
        return given

    def _substitute(self, exprs: list[sp.Expr]) -> tuple[list[sp.Expr], list[sp.Symbol]]:
        used: list[sp.Symbol] = []
        for _ in range(_SUBSTITUTION_ROUNDS):
            present = {s for e in exprs for s in e.free_symbols if s in self.definitions}
            if not present:
                break
            used.extend(s for s in sorted(present, key=lambda s: s.name) if s not in used)
            mapping = {s: self.definitions[s][0] for s in present}
            exprs = [e.xreplace(mapping) for e in exprs]
        return exprs, used

    def check(self, latex: str) -> ClaimCheck:
        sides = split_equation(latex)
        if sides is None:
            return ClaimCheck(status="not_an_equation")
        if self._is_vector(latex):
            if _LEIBNIZ.search(latex) and not self._is_vector(re.sub(r"\\partial\b", "", latex)):
                # a plain derivative (`\partial` is also a vector-calculus command): say which it is
                return ClaimCheck(status="unverified", kind="relation", detail=_LEIBNIZ_DETAIL)
            return ClaimCheck(status="unverified", kind="relation", detail=_VECTOR_DETAIL)
        if mentions_name(_text(latex), self.matrices):
            return ClaimCheck(status="unverified", kind="relation", detail=_MATRIX_DETAIL)
        if _TRANSPOSE.search(latex):
            return ClaimCheck(status="unverified", kind="relation", detail=_TRANSPOSE_DETAIL)
        if _LEIBNIZ.search(latex):
            return ClaimCheck(status="unverified", kind="relation", detail=_LEIBNIZ_DETAIL)
        if (
            _DIFFERENTIAL.search(latex) and "\\int" not in latex
        ):  # an integral keeps its own reading
            return ClaimCheck(status="unverified", kind="relation", detail=_DIFFERENTIAL_DETAIL)
        if _IMPLICATION.search(latex):
            return ClaimCheck(status="unverified", kind="relation", detail=_IMPLICATION_DETAIL)
        if self.series is not None:
            if self.series.location(sides) is not None:
                return ClaimCheck(status="unverified", kind="relation", detail=_LOCATION_DETAIL)
            linked_series = self._match_series(sides)
            if linked_series is not None:
                return linked_series
        if _BIG_O.search(latex):
            return ClaimCheck(status="unverified", kind="relation", detail=_BIG_O_DETAIL)
        if not _DISTRIBUTION_NOTATION.search(latex):
            indexed = self._indexed_claim(sides)
            if indexed is not None:
                return indexed
        if self.evidence is not None and not _DISTRIBUTION_NOTATION.search(latex):
            linked = self._match_evidence(sides)
            if linked is not None:
                return linked
        if any(_RESERVED_LONE.fullmatch(side.strip()) for side in sides):
            return ClaimCheck(status="unverified", kind="relation", detail=_RESERVED_DETAIL)
        if _DISTRIBUTION_NOTATION.search(latex):
            # \delta(x) and \theta(x) read as products here, which would make a true statement about
            # a delta or step function look false; a delta cannot be sampled at all.
            return ClaimCheck(status="unverified", detail=_DISTRIBUTION_DETAIL)
        if applied_function_notation(latex):
            return ClaimCheck(status="unverified", kind="relation", detail=_FUNCTION_DETAIL)
        denominators: list[sp.Expr] = []
        try:
            parsed = [
                parse_latex(side, self.assumptions, self.limits, denominators) for side in sides
            ]
        except MinervaMathError as exc:
            return ClaimCheck(status="unparsed", detail=str(exc))
        if self._mentions_matrix(parsed):
            return ClaimCheck(status="unverified", kind="relation", detail=_MATRIX_DETAIL)
        if any(contains_distribution(p) for p in parsed):
            return ClaimCheck(status="unverified", detail=_DISTRIBUTION_DETAIL)
        if any(contains_indexed(p) for p in parsed):
            # sampling numbers cannot stand in for data vectors or random variables
            return ClaimCheck(status="unverified", detail=_INDEXED_DETAIL)
        outcome, detail = _compare(parsed)
        if outcome == "holds":
            return _holds_with_domain(parsed, denominators, detail).model_copy(
                update={"kind": "identity"}
            )
        if outcome != "fails":
            return ClaimCheck(status="unverified", detail=detail)
        # Prose uses e and i for Euler's number and the imaginary unit; the parser keeps them as plain
        # symbols (see AGENTS.md). Give the claim the benefit of that reading, and say so.
        used = {s.name for p in parsed for s in p.free_symbols}
        if used & {"e", "i"}:
            swap = {
                s: (sp.E if s.name == "e" else sp.I)
                for p in parsed
                for s in p.free_symbols
                if s.name in ("e", "i")
            }
            if _compare([p.subs(swap) for p in parsed])[0] == "holds":
                return ClaimCheck(
                    status="holds",
                    detail="holds reading e as Euler's number and i as the imaginary unit",
                    kind="identity",
                )
        # The sides differ by the same nonzero number for every value: false whatever it means.
        if any(_nonzero_constant(parsed[0] - other) for other in parsed[1:]):
            return ClaimCheck(status="fails", detail=detail, kind="identity")
        # Definitions, coordinate changes and equations of motion are not identities: do not test them
        # as one. Use what the conversation stated about the symbols instead.
        kind = _not_an_identity_kind(parsed)
        if kind is None and len({frozenset(p.free_symbols) for p in parsed}) > 1:
            kind = "a relation between quantities"
        if kind is not None:
            return self._check_relation(parsed, kind, latex)
        # A claim that is false for complex symbols may be the usual real-variable statement whose
        # condition is written in the prose, not in the formula. Try the weakest natural assumptions.
        conditional = _relaxed(sides, self.assumptions, self.limits)
        if conditional is not None:
            return conditional.model_copy(update={"kind": "identity"})
        return ClaimCheck(
            status="not_identity",
            detail=f"not an identity ({detail}); it may still be an equation to solve or a definition",
            kind="identity",
        )

    def _match_series(self, sides: list[str]) -> ClaimCheck | None:
        """`supported` if the claim is `f = P + O(...)` and every part is the series a tool verified."""
        if self.series is None or len(sides) != 2:
            return None
        for function_side, remainder_side in (sides, sides[::-1]):
            if _BIG_O.search(function_side):
                continue
            split = split_remainder(remainder_side)
            if split is None:
                continue
            found = self.series.match(function_side, *split, self.function_definitions)
            if found is not None:
                where = f" ({found.source})" if found.source else ""
                return ClaimCheck(
                    status="supported",
                    detail=f"{found.detail}{where}",
                    kind="relation",
                    evidence_source=found.source or None,
                    evidence_rule=found.rule,
                    evidence_basis=found.basis,
                )
        return None

    def _labels(self, sides: list[str]) -> list[IndexLabel | None]:
        """The index label of each side, or None for a side that is not read as one (`claim_evidence`)."""
        kernels = self.evidence.kernels if self.evidence is not None else frozenset()
        out: list[IndexLabel | None] = []
        for side in sides:
            label = read_index_label(side)
            if label is not None and not is_index_label(
                label, side, self.assumptions, self.limits, kernels
            ):
                label = None
            out.append(label)
        return out

    def _indexed_claim(self, sides: list[str]) -> ClaimCheck | None:
        """`label = value`: `supported` if a verified row has exactly this label and an equivalent value, else
        unchecked with the reason. None when no side is an index label (the claim is checked as before)."""
        labels = self._labels(sides)
        if not any(labels):
            return None
        if len(sides) != 2 or all(labels):
            return ClaimCheck(status="unverified", kind="relation", detail=_LABEL_RELATION_DETAIL)
        label, text = labels[0], sides[1]
        if label is None:
            label, text = labels[1], sides[0]
        if label is None:
            return None
        rows = self.evidence.indexed_rows(label) if self.evidence is not None else []
        if self.evidence is None or not rows:
            return ClaimCheck(status="unverified", kind="relation", detail=_LABEL_NO_ROW_DETAIL)
        try:
            value = parse_latex(text, self.assumptions, self.limits)
        except MinervaMathError:
            return ClaimCheck(status="unverified", kind="relation", detail=_LABEL_VALUE_DETAIL)
        if (
            self._mentions_matrix([value])
            or contains_distribution(value)
            or contains_indexed(value)
        ):
            return ClaimCheck(status="unverified", kind="relation", detail=_LABEL_VALUE_DETAIL)
        found = self.evidence.match_indexed(label, value)
        if found is None:
            row = rows[0]
            where = f" ({row.source})" if row.source else ""
            # Not shown equivalent is not "false": the claim stays unchecked and names the verified row.
            return ClaimCheck(
                status="unverified",
                kind="relation",
                detail=f"an indexed component; the verified result{where} lists ${row.latex}$ for this label, "
                "and this value was not shown to be equivalent to it (index symmetries are not used), so it "
                "is not tested further",
            )
        where = f" ({found.source})" if found.source else ""
        return ClaimCheck(
            status="supported",
            detail=f"{found.detail}{where}",
            kind="relation",
            evidence_source=found.source or None,
            evidence_rule=found.rule,
            evidence_basis=found.basis,
            restrictions=list(found.restrictions),
        )

    def _match_evidence(self, sides: list[str]) -> ClaimCheck | None:
        """`supported` if the claim is the same statement as a result a tool verified (never otherwise)."""
        if self.evidence is None or len(sides) != 2:
            return None
        # `E = ...` names an energy, but E is Euler's number to the parser: the name is only a label here
        # (matching it needs the right side to equal a verified conserved quantity), so it is replaced.
        named = [
            _RESERVED_NAME if _RESERVED_LONE.fullmatch(side.strip()) else side for side in sides
        ]
        try:
            parsed = [parse_latex(side, self.assumptions, self.limits) for side in named]
        except MinervaMathError:
            return None
        if (
            self._mentions_matrix(parsed)
            or any(contains_distribution(p) or contains_indexed(p) for p in parsed)
            or not self.evidence.items
        ):
            return None
        found = self.evidence.match(parsed)
        if found is None:
            differs = self.evidence.mismatch(parsed)
            if differs is None:
                return None
            where = f" ({differs.source})" if differs.source else ""
            # Still unchecked, never "false": the name is the tool's, the expression is not.
            return ClaimCheck(
                status="unverified",
                kind="relation",
                detail=f"This uses the name of a verified result but a different expression: the verified "
                f"result{where} is ${differs.latex}$. Compare them before relying on this equation.",
                evidence_source=differs.source or None,
                evidence_rule="mismatch",
            )
        where = f" ({found.source})" if found.source else ""
        return ClaimCheck(
            status="supported",
            detail=f"{found.detail}{where}",
            kind="relation",
            evidence_source=found.source or None,
            evidence_rule=found.rule,
            evidence_basis=found.basis,
            restrictions=list(found.restrictions),
        )

    def _check_relation(self, parsed: list[sp.Expr], kind: str, latex: str = "") -> ClaimCheck:
        """Not an identity, so sampling the symbols independently proves nothing. Use the definitions
        stated earlier; without them there is nothing to test."""
        unchecked = ClaimCheck(
            status="unverified", detail=f"{kind}; not tested as an identity", kind="relation"
        )
        substituted, used = self._substitute(parsed)
        if not used:
            return unchecked
        if any(e.has(sp.zoo, sp.nan, sp.oo, -sp.oo) for e in substituted):
            # A definition that is undefined where it was substituted (`r = 0` into `1/r^6`) says nothing about the
            # claim: comparing the claim with "undefined" at sample points is not a disagreement.
            return ClaimCheck(
                status="unverified",
                detail="a definition stated earlier gives an undefined value here (a division by zero); "
                "not tested",
                kind="relation",
            )
        given = self._given(used)
        if latex and _SPACES.sub("", _text(latex)) in {_SPACES.sub("", g) for g in given}:
            # The claim is one of the definitions it would be "given": a statement cannot justify itself.
            return ClaimCheck(
                status="unverified",
                detail="a definition stated earlier in the conversation; not verified by Minerva",
                kind="relation",
            )
        sets = [frozenset(p.free_symbols) for p in substituted]
        if any(s != sets[0] for s in sets[1:]):
            return unchecked
        outcome, detail = _compare(substituted)
        if outcome == "holds":
            return ClaimCheck(
                status="conditional",
                detail="holds given " + "; ".join(given),
                kind="relation",
                given=given,
            )
        if outcome == "fails" and _definition(parsed) is None:
            return ClaimCheck(
                status="fails",
                detail="does not agree with " + "; ".join(given) + f" ({detail})",
                kind="relation",
                given=given,
            )
        return unchecked


INCOMPLETE_DETAIL = (
    "check incomplete: the targeted probes were cut short by the time limit (not a disagreement)"
)


def _compare(parsed: list[sp.Expr]) -> tuple[str, str]:
    checked = agreed = 0
    incomplete = False
    for other in parsed[1:]:
        verification = verify_equal(parsed[0], other, symbolic=False)
        checked += verification.numeric_points_checked
        agreed += verification.numeric_points_agreed
        if verification.numeric_points_checked < 4:
            return "unverified", "too few valid sample points"
        incomplete = incomplete or verification.probes_cut_short
    if agreed != checked:  # a disagreement found is one whatever the clock did
        return "fails", f"{checked - agreed} of {checked} sample points disagree"
    if incomplete:  # V-017: points that agree before the probes were cut short are not `holds`
        return "unverified", INCOMPLETE_DETAIL
    return "holds", f"{agreed}/{checked} sample points agree"


def _nonzero_constant(difference: sp.Expr) -> bool:
    """True if the two sides differ by the same nonzero number everywhere (false for every value)."""
    try:
        value = sp.simplify(difference)
    except Exception:  # undecided is not "constant"
        return False
    return bool(value.is_number and value.is_finite and value != 0)


def _not_an_identity_kind(parsed: list[sp.Expr]) -> str | None:
    """Name the kind of equation that is not meant as an identity, or None.

    Time derivatives (`\\dot x` parses as the symbol `x_dot`) mean an equation of motion; a bare symbol
    on one side that does not occur on the other (`F = m a`, `L = T - V`, `x = 2`) is a definition or
    a value being stated.
    """
    names = {s.name for p in parsed for s in p.free_symbols}
    if any(name.endswith(("_dot", "_ddot")) for name in names):
        return "an equation of motion"
    if len(parsed) == 2:
        for one, other in (parsed, parsed[::-1]):
            if one.is_Symbol and one not in other.free_symbols:
                return "a definition or a value for one symbol"
        if bool(parsed[0].free_symbols) != bool(parsed[1].free_symbols):
            return "an equation to solve"
    return None


def _relaxed(
    sides: list[str], assumptions: dict[str, list[str]], limits: Limits
) -> ClaimCheck | None:
    """`conditional` if the claim holds once symbols are assumed real, then positive.

    A symbol the user declared only `real` is relaxed to `positive` too; other declared
    assumptions are never overridden.
    """
    names = sorted(
        {s.name for side in sides for s in parse_latex(side, assumptions, limits).free_symbols}
    )
    for label in ("real", "positive"):
        relaxed: dict[str, list[str]] = {}
        changed: list[str] = []
        for name in names:
            declared = assumptions.get(name)
            if declared is None:
                relaxed[name] = [label]
                changed.append(name)
            elif label == "positive" and set(declared) == {"real"}:
                relaxed[name] = ["positive"]
                changed.append(name)
            else:
                relaxed[name] = declared
        if not changed:
            continue
        try:
            retry = [parse_latex(side, relaxed, limits) for side in sides]
        except MinervaMathError:
            return None
        if _compare(retry)[0] == "holds":
            verb = "is" if len(changed) == 1 else "are"
            return ClaimCheck(
                status="conditional",
                detail=f"false in general, but holds if {', '.join(changed)} {verb} {label}",
            )
    return None


def _holds_with_domain(
    parsed: list[sp.Expr], denominators: list[sp.Expr], detail: str
) -> ClaimCheck:
    """`holds`, unless the two sides are defined at different points (a cancelled singularity)."""
    notes: list[str] = []
    for other in parsed[1:]:
        found, _ = domain_restrictions(parsed[0], other, denominators)
        notes.extend(n for n in found if n not in notes)
    if notes:
        return ClaimCheck(
            status="conditional",
            detail="equal as functions, but not defined at the same points: valid only where "
            + ", ".join(notes),
        )
    return ClaimCheck(status="holds", detail=detail)
