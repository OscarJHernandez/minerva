"""Point-source Coulomb kernels: bounded random sources, strengths and rescalings, as explicit components."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from hypothesis import strategies as st

source_coordinate = st.fractions(min_value=Fraction(-2), max_value=Fraction(2), max_denominator=4)
strength = st.fractions(min_value=Fraction(-4), max_value=Fraction(4), max_denominator=4).filter(
    lambda c: c != 0
)
wrong_factor = st.sampled_from(
    [Fraction(1, 1_000_000), Fraction(1, 100), Fraction(-2), Fraction(2)]
)


@dataclass(frozen=True)
class Kernel:
    strength: Fraction
    source: tuple[Fraction, Fraction, Fraction]

    def components(self) -> list[str]:
        q = " + ".join(f"(({v}) - ({a}))**2" for v, a in zip("xyz", self.source, strict=True))
        return [
            f"({self.strength})*(({v}) - ({a}))/({q})**(3/2)"
            for v, a in zip("xyz", self.source, strict=True)
        ]


@st.composite
def kernels(draw: st.DrawFn) -> Kernel:
    return Kernel(
        draw(strength), (draw(source_coordinate), draw(source_coordinate), draw(source_coordinate))
    )
