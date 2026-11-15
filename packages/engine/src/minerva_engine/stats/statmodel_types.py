"""Types shared by the statistical-model derivations (kept apart to avoid import cycles)."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ..distributions.models import Condition
from ..errors import MinervaMathError
from ..models import Status
from .families import Family, Param
from .models import StatReason


class Refusal(MinervaMathError):
    """A refusal with a status and an enumerated reason."""

    def __init__(self, status: Status, reason: StatReason, message: str) -> None:
        super().__init__(message)
        self.status, self.reason = status, reason


@dataclass
class Reading:
    family: Family
    params: tuple[Param, ...]
    exprs: dict[
        str, sp.Expr
    ]  # family parameter name -> the user's expression (or the unknown itself)
    conditions: list[Condition]
