"""Agent tool `multivariate_normal`: a bivariate Normal given by the USER's mean and covariance.

The model chooses the action and passes what the user said; the engine derives the result and shows it only if
quadrature of the printed answer agrees (`minerva_engine.stats.mvn`). Three rules are repeated in the schema
because the model is the weak point: the mean, the covariance, the value conditioned on and the map are the
user's, never invented; a zero off-diagonal entry is a statement of uncorrelatedness the user must make (it is
never inferred from "independent-looking" names or from a missing number); a result that holds given a premise
is said to hold *given* it.

Provenance (`assumed_values`): an entry of the mean or the covariance that no message of the user wrote is
labelled "assumed by the assistant" on the card and in the tool text. A label, never a refusal and never an
input to the computation; it errs towards "assumed" (a number the user wrote another way is listed too).
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from fractions import Fraction
from typing import Literal

from minerva_engine import MvnInput, MvnResult
from pydantic import BaseModel, Field

from .choice import delegation_active, numbers_in
from .events import Card
from .tools import ToolContext, ToolOutput

CONDITIONAL = "holds only given a premise that Minerva does not check"
ASSUMED_LABEL = "mean and covariance assumed by the assistant"
DELEGATED_LABEL = "mean and covariance chosen by the assistant at the user's request"
# The line a later turn re-sends: the model as read, so a follow-up asks about exactly it.
MODEL_LINE = "model (send exactly this again for a follow-up)"


class MvnArgs(BaseModel):
    """Arguments of `multivariate_normal`. Everything not given by the user stays empty."""

    mean: list[str] = Field(
        min_length=1,
        max_length=8,
        description="The mean vector (mu_x, mu_y): exactly two entries in text syntax ('1', '-2', 'mu_1'), "
        "ONLY as the USER stated them. Never invent a mean, and never write 0 because none was given: ask.",
    )
    covariance: list[list[str]] = Field(
        min_length=1,
        max_length=8,
        description="The covariance matrix as two rows of two entries in text syntax, e.g. "
        '[["4", "1"], ["1", "2"]], ONLY as the USER stated it. Never invent an entry. A zero '
        "off-diagonal entry means the components are uncorrelated: write it only if the user said so; "
        "never infer it (from the names, from 'independent-looking' variables, or because a number is "
        "missing), and ask for a missing entry. It must be symmetric and positive definite, else the "
        "tool refuses.",
    )
    variables: list[str] | None = Field(
        default=None,
        max_length=8,
        description="The names of the two components, e.g. ['x', 'y'], as the user named them. Needed for "
        "density, marginal and conditional; ask if they were not given.",
    )
    action: Literal["density", "moments", "marginal", "conditional", "affine"] = Field(
        description="density: the joint density, from the mean and covariance. moments: the mean, "
        "covariance and correlation as read. marginal: the distribution of one component (`keep`). "
        "conditional: the distribution of the other component given the value of one (`given`). "
        "affine: the image under a linear map (`matrix`, `offset`).",
    )
    keep: str | None = Field(
        default=None,
        max_length=32,
        description="marginal only: the name of the component to keep, one of `variables`. Ask the user.",
    )
    given: dict[str, str] | None = Field(
        default=None,
        max_length=4,
        description='conditional only: {name of the component fixed: its value}, e.g. {"x": "3"}, '
        "ONLY the value the USER gave. Never choose a value to condition on.",
    )
    matrix: list[list[str]] | None = Field(
        default=None,
        max_length=4,
        description="affine only: the matrix A, one or two rows of two entries, ONLY as the USER gave it. "
        "A singular map is refused, never replaced.",
    )
    offset: list[str] | None = Field(
        default=None,
        max_length=4,
        description="affine only: the offset b, one entry per row of `matrix`, ONLY as the USER gave it "
        "(ask if they did not say; do not assume 0).",
    )
    assumptions: dict[str, list[str]] = Field(
        default_factory=dict,
        description='Properties of symbols in the entries, e.g. {"s": ["positive"], "r": ["real"]}. With '
        "symbolic entries, a positive definite covariance that the assumptions cannot decide makes the "
        "result HOLD GIVEN that condition.",
    )


# ---------------------------------------------------------------------------------- provenance


_PLAIN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?")


def _as_number(entry: str) -> float | None:
    """The value of an entry that is one plain number (`4`, `-2`, `0.5`, `1/2`), else None."""
    text = entry.replace(" ", "")
    if not _PLAIN.fullmatch(text):
        return None
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None


_COVARIANCE_WORDS = re.compile(
    r"covarian|varian|correlat|independen|uncorrelat|sigma|matri|\u03a3|\u03c3", re.I
)
_SEPARATORS = re.compile(r"[&,;\n]|\\\\|\\(?:begin|end)\{[A-Za-z*]+\}|\band\b")


def _flatten(text: str) -> str:
    """A piece of text in the spelling shared by a user's `\\sigma_{x}^{2}` and a call's `sigma_x^2`."""
    for noise in ("\\left", "\\right", "\\,", "\\;", "\\!"):
        text = text.replace(noise, "")
    # A product stays a product: `a*b` is not the name `ab`. In LaTeX a command starts a new factor
    # (`\rho\sigma_x` is rho times sigma_x), so a backslash reads as a product sign.
    text = text.replace("\\cdot", "*")
    text = re.sub(r"\*?\s*\\", "*", text)
    return re.sub(r"[\s${}]", "", text).lstrip("*")


def _segments(messages: Sequence[str]) -> list[str]:
    """What the user wrote, cut where one value ends and the next begins (a comma, `&`, a row break), each
    piece flattened: an entry must be a whole piece or a whole word of one, never a part of a longer name."""
    return [
        flat
        for message in messages
        for piece in _SEPARATORS.split(message)
        if (flat := _flatten(piece))
    ]


def _written(entry: str, segments: list[str], numbers: list[float]) -> bool:
    value = _as_number(entry)
    if value is not None:
        return any(abs(value - n) <= 1e-12 * max(1.0, abs(value)) for n in numbers)
    needle = _flatten(entry)
    if not needle:
        return True
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(needle)}(?![A-Za-z0-9_])")
    return any(pattern.search(segment) for segment in segments)


def _model_entries(args: MvnArgs) -> list[str]:
    return [*args.mean, *(e for row in args.covariance for e in row)]


def provenance(args: MvnArgs, ctx: ToolContext) -> tuple[str, list[str]]:
    """`user`, `assistant` or `assistant_delegated`, and the entries of the mean and covariance that no
    message of the user wrote. Outside a chat turn (`user_messages is None`) the values are the caller's own."""
    messages: Sequence[str] | None = ctx.user_messages
    if messages is None:
        return "user", []
    entries = _model_entries(args)
    if delegation_active(messages):
        return "assistant_delegated", entries
    # A covariance entry counts as the user's only if it is in a message about the covariance: a 0 or a 1
    # written for the mean is not a statement that the components are uncorrelated.
    about_covariance = [m for m in messages if _COVARIANCE_WORDS.search(m)]
    segments = _segments(messages)
    cov_segments = _segments(about_covariance)
    numbers = [n for m in messages for n in numbers_in(m)]
    cov_numbers = [n for m in about_covariance for n in numbers_in(m)]
    cov_entries = {e for row in args.covariance for e in row}
    mine = [
        e
        for e in dict.fromkeys(entries)
        if not (
            _written(e, cov_segments, cov_numbers)
            if e in cov_entries
            else _written(e, segments, numbers)
        )
    ]
    return ("assistant" if mine else "user"), mine


# ---------------------------------------------------------------------------------- the tool


def _grid(rows: list[list[str]]) -> str:
    return json.dumps(rows, ensure_ascii=False)


def _pmatrix(rows: list[list[str]]) -> str:
    return "\\begin{pmatrix}" + " \\\\ ".join(" & ".join(r) for r in rows) + "\\end{pmatrix}"


def _distribution_lines(result: MvnResult) -> list[str]:
    names = ", ".join(result.variables)
    lines = [f"distribution of ({names}):"]
    lines.append("- mean = [" + ", ".join(f"${m}$" for m in result.mean_latex) + "]")
    lines.append(
        "- covariance = ["
        + "; ".join("[" + ", ".join(f"${c}$" for c in row) + "]" for row in result.covariance_latex)
        + "]"
        + (" (a variance)" if len(result.covariance_latex) == 1 else "")
    )
    if result.density_latex:
        lines.append(f"- density = ${result.density_latex}$")
    if result.correlation_latex:
        lines.append(f"- correlation = ${result.correlation_latex}$")
    return lines


def run_multivariate_normal(args: MvnArgs, ctx: ToolContext) -> ToolOutput:
    request = MvnInput(**args.model_dump())
    result = ctx.engine.multivariate_normal(request)
    who, values = provenance(args, ctx)
    label = {"assistant": ASSUMED_LABEL, "assistant_delegated": DELEGATED_LABEL}.get(who)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="multivariate_normal",
        type="multivariate_normal",
        title="Bivariate Normal",
        data={
            "result": result.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
            "provenance": {"who": who, "values": values, "label": label},
        },
    )
    status = result.status.value
    conditional = result.evidence == "conditional" or bool(result.premises)
    head = f"status={status}; action={result.action}"
    lines: list[str] = []
    # What a follow-up must send: the model as read (the result's own echo), else as the model sent it.
    mean_text = result.input_mean_text or args.mean
    cov_text = result.input_covariance_text or args.covariance
    names = result.variables if result.action in ("density", "moments") else args.variables
    model_line = f"{MODEL_LINE}: mean={json.dumps(mean_text)}; covariance={_grid(cov_text)}" + (
        f"; variables={json.dumps(names or args.variables)}" if (names or args.variables) else ""
    )
    formulas: list[str] = []
    if status == "ok":
        head += f"; evidence={'conditional' if conditional else result.evidence}"
        lines.extend(_distribution_lines(result))
        # Whole formulas only: a bare entry (`4`, `-2`) would mark every such number in the reply as a
        # tool result, whatever it was about.
        formulas = [
            _pmatrix([[m] for m in result.mean_latex]),
            _pmatrix(result.covariance_latex),
            result.density_latex,
            result.correlation_latex,
        ]
        lines.extend(
            f"- check ({c.kind}): {c.method}: {c.points_agreed}/{c.points_checked}"
            + ("" if c.passed else " FAILED")
            for c in result.checks
        )
        lines.extend(
            f"- condition ${c.latex}$: {'established' if c.established else 'NOT established (' + c.how + ')'}"
            for c in result.conditions
        )
        if conditional:
            head += (
                "; the result HOLDS GIVEN: " + "; ".join(result.premises) + ". Minerva did not "
                "check these: say 'given', never 'proved'"
            )
        lines.append(
            "numerically checked (quadrature of the printed result), not proved: say 'numerically checked'"
        )
        lines.append(result.message)
    else:
        head += (
            f"; result=none (reason={result.reason or status}); {result.message}. No formula is "
            "given for this call"
        )
        if result.reason == "variables_missing":
            head += ". Ask the user for the names of the two components"
        if result.reason in ("not_symmetric", "not_positive_definite", "singular_covariance"):
            head += ". Do not repair the covariance: tell the user what the tool said and ask"
    if label is not None:
        lines.append(
            f"{label}: {', '.join(values[:12])}. The card says so. Tell the user these were not "
            "theirs, that the result holds given them, and ask them to confirm or change them."
        )
    known: list[str] = []
    quoted: list[tuple[str, str]] = []
    shown = [f for f in formulas if f]
    if status == "ok" and result.evidence == "identity" and not conditional:
        known = shown
    elif status == "ok":
        why = f"{CONDITIONAL}: " + "; ".join(result.premises or ["a premise"])
        quoted = [(f, why) for f in shown]
    # The model line comes right after the head: an older result is shortened from the end, and a follow-up
    # must still find the model it re-sends.
    return ToolOutput(
        card, "\n".join([head, model_line, *lines]), known, status, quoted_latex=quoted
    )
