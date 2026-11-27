"""The one reader of the observations a statistical request names: how many there are and what they are.

Every action that depends on observations (likelihood, score, MLE, posterior, Fisher information) reads
them here, before any computation:

* `sample_size` is a whole number (digits, at least 1) or the name of one (a letter, then letters, digits
  and underscores, not the name of the data or of a parameter, so it becomes a positive-integer symbol).
  Anything else (`-1`, `0`, `1/2`, `2n`, `n+1`) is `sample_size_invalid`: it used to become a symbol with that
  very name, and a checker that substitutes positive values for it "verified" a negative count.
* `data` determines the count. An empty list is `no_observations`; a count that disagrees with it, or a
  symbolic one next to it, is `sample_size_conflict`.
* Every concrete value must lie in the support of the family (`observation_outside_support`): integers for a
  discrete family, and inside the interval with the family's own open or closed ends. A bound that depends
  on a parameter being estimated is left to the estimator's admissibility check; a bound that is a known
  symbol and cannot be decided is recorded as an unestablished condition, never passed silently.
* Independence is a separate matter (`iid`): it is needed whenever the count is not exactly one.
"""

from __future__ import annotations

import re

import sympy as sp

from ..distributions.models import Condition
from ..latexprint import to_latex
from ..models import ALLOWED_ASSUMPTIONS, ExpressionInput, Limits, Status
from ..ops import parse_expression
from .models import StatModelInput
from .statmodel_types import Count, Reading, Refusal

_DIGITS = re.compile(r"[0-9]{1,18}")
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _decided(relation: object) -> bool | None:
    if relation == sp.true:
        return True
    if relation == sp.false:
        return False
    return None


def read_count(data: StatModelInput, reading: Reading, limits: Limits) -> Count:
    """Validate `sample_size` and `data`; raises `Refusal` before anything is computed."""
    family = reading.family
    name = data.observation or "x"
    n: sp.Expr | None = None
    symbolic = False
    if data.sample_size is not None:
        text = data.sample_size.strip()
        reserved = {name, *family.symbols, *(p.name for p in reading.params)}
        for expr in reading.exprs.values():
            reserved |= {s.name for s in expr.free_symbols}
        if _DIGITS.fullmatch(text):
            n = sp.Integer(int(text))
            if n < 1:
                raise Refusal(
                    Status.PARSE_ERROR,
                    "sample_size_invalid",
                    "the number of observations must be a whole number of at least 1",
                )
        elif _NAME.fullmatch(text) and text not in reserved:
            flags = {"integer": True, "positive": True}
            for item in data.assumptions.get(text, []):
                if item in ALLOWED_ASSUMPTIONS:
                    flags[item] = True
            n = sp.Symbol(text, **flags)
            symbolic = True
        else:
            raise Refusal(
                Status.PARSE_ERROR,
                "sample_size_invalid",
                "the number of observations is a whole number (4) or the name of one (n), not an "
                "expression; it cannot be negative, zero, fractional or the name of the data or of a "
                "parameter",
            )
    if data.data is None:
        if data.action != "fisher_information" and n is not None and not symbolic:
            raise Refusal(
                Status.PARSE_ERROR,
                "sample_size_invalid",
                "a numeric sample size needs the observations themselves for this action: give the "
                "data, or name the size with a letter such as n",
            )
        return Count(n)
    if not data.data:
        raise Refusal(
            Status.CONDITIONS_NOT_SATISFIED, "no_observations", "the list of observations is empty"
        )
    if n is not None and (symbolic or n != len(data.data)):
        raise Refusal(
            Status.CONDITIONS_NOT_SATISFIED,
            "sample_size_conflict",
            f"{len(data.data)} observations were given, which determines the count; the stated "
            "sample size disagrees (leave it out, or make it equal to the number of observations)",
        )
    values: list[sp.Rational] = []
    for text in data.data:
        value = parse_expression(ExpressionInput(input=text, format="text"), limits)
        if value.free_symbols or not value.is_real:
            raise Refusal(Status.PARSE_ERROR, "parse_error", "observations must be real numbers")
        values.append(sp.nsimplify(value))
    reading.conditions.extend(_check_support(values, data, reading))
    return Count(sp.Integer(len(values)), values)


def _check_support(
    values: list[sp.Rational], data: StatModelInput, reading: Reading
) -> list[Condition]:
    family = reading.family
    lo, hi = family.support(family.symbols)
    swap = {family.symbols[n]: e for n, e in reading.exprs.items()}
    lo, hi = lo.xreplace(swap), hi.xreplace(swap)
    estimated = {family.symbols[n] for n in data.estimate if n in family.symbols}
    lo_open, hi_open = family.support_open
    conditions: list[Condition] = []
    undecided = False
    for k, value in enumerate(values, start=1):
        where = f"observation {k} ({to_latex(value)})"
        if family.discrete and value.is_integer is not True:
            raise Refusal(
                Status.CONDITIONS_NOT_SATISFIED,
                "observation_outside_support",
                f"{where} is not an integer, but a {family.title} observation is",
            )
        if lo != -sp.oo and not (lo.free_symbols & estimated):
            ok = _decided(sp.Gt(value, lo) if lo_open else sp.Ge(value, lo))
            if ok is False:
                raise Refusal(
                    Status.CONDITIONS_NOT_SATISFIED,
                    "observation_outside_support",
                    f"{where} is outside the support of the {family.title} family "
                    f"(it must be {'above' if lo_open else 'at least'} {to_latex(lo)})",
                )
        if hi != sp.oo and not (hi.free_symbols & estimated):
            ok = _decided(sp.Lt(value, hi) if hi_open else sp.Le(value, hi))
            if ok is False:
                raise Refusal(
                    Status.CONDITIONS_NOT_SATISFIED,
                    "observation_outside_support",
                    f"{where} is outside the support of the {family.title} family "
                    f"(it must be {'below' if hi_open else 'at most'} {to_latex(hi)})",
                )
            if ok is None and not undecided:
                undecided = True
                end = "below" if hi_open else "at most"
                conditions.append(
                    Condition(
                        latex=f"\\text{{every observation is {end} }}{to_latex(hi)}",
                        established=False,
                        how="the upper end of the support is a symbol; the data cannot be "
                        "compared with it",
                        # a known parameter: the result then holds only given it (conditional)
                        parameter=next(
                            (
                                n
                                for n, s in family.symbols.items()
                                if s in family.support(family.symbols)[1].free_symbols
                            ),
                            None,
                        ),
                        role="given",
                    )
                )
    if values and hi != sp.oo and hi.free_symbols & estimated and data.action != "mle":
        # The density is written without its indicator (uniform 1/theta), so a likelihood, score or
        # posterior of these data is that formula only where the unknown bound admits every value. An
        # estimator checks this itself (admissibility); every other result holds only given it.
        top = max(values)
        relation = sp.Gt(hi, top) if hi_open else sp.Ge(hi, top)
        conditions.append(
            Condition(
                latex=to_latex(relation),
                established=False,
                how="the support ends at a parameter being estimated: the formula is the likelihood "
                "of these data only where that end admits every observation",
                parameter=next(
                    (n for n, s in family.symbols.items() if s in hi.free_symbols & estimated),
                    None,
                ),
                # a premise of the result (see `statmodel._premise_unestablished_constraints`)
                role="given",
            )
        )
    return conditions


def fits_at(
    values: list[sp.Rational], reading: Reading, point: dict[sp.Symbol, sp.Rational]
) -> bool:
    """Whether concrete data lie in the support at a sampled value of the known parameters.

    The support check of `read_count` could not decide a bound given as a symbol (Binomial `n`); a point at
    which the data do not fit has a likelihood of zero, so it is not a place to test an estimator.
    """
    family = reading.family
    _, hi = family.support(family.symbols)
    swap = {family.symbols[n]: e for n, e in reading.exprs.items()}
    bound = hi.xreplace(swap).subs(point)
    if bound == sp.oo or bound.free_symbols:
        return True
    hi_open = family.support_open[1]
    return all(
        _decided(sp.Lt(v, bound) if hi_open else sp.Le(v, bound)) is not False for v in values
    )
