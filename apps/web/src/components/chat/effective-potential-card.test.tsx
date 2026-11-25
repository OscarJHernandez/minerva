import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Card, EffectivePotentialReport } from "@/lib/chat-types";
import fixture from "../../../../../packages/ui/fixtures/equation-rows.json";
import { CardView } from "./card-view";
import { EffectivePotentialCard } from "./effective-potential-card";

const ok = (fixture as unknown as { effective_potential: { data: EffectivePotentialReport } })
  .effective_potential.data;

describe("EffectivePotentialCard", () => {
  it("shows the radial equation, V_eff, the held momenta, the plane and how it was checked", () => {
    render(<EffectivePotentialCard report={ok} />);
    const text = document.body.textContent ?? "";
    expect(text).toContain("Radial equation");
    expect(text).toContain("Normalisation chosen by you");
    expect(text).toContain("Held fixed");
    expect(text).toContain("Derived, proved and checked");
    expect(text).toContain("only after computing");
    expect(screen.getByTestId("restrictions")).toBeInTheDocument();
  });

  it("a refusal shows its message and no formula of a result", () => {
    const refused: EffectivePotentialReport = {
      ...ok,
      status: "unsupported",
      reason: "equatorial_not_consistent",
      result_latex: null,
      veff_latex: null,
      held_fixed: [],
      restrictions_latex: [],
      message: "Fixing theta at the stated value is not consistent with the geodesic equations.",
    };
    render(<EffectivePotentialCard report={refused} />);
    const text = document.body.textContent ?? "";
    expect(text).toContain("not consistent");
    expect(text).not.toContain("Radial equation");
    expect(screen.queryByTestId("restrictions")).toBeNull();
  });

  it("the whole number of the card names the radial equation; V_eff and the momenta are rows", () => {
    const card = {
      id: "c",
      tool: "effective_potential",
      type: "effective_potential",
      title: "Effective potential",
      data: ok,
      number: 4,
      numbers: { veff: 5, "held:t": 6, "held:phi": 7 },
    } as unknown as Card;
    render(<CardView card={card} />);
    const numbers = screen.getAllByTestId("equation-number").map((e) => e.textContent);
    expect(numbers).toEqual(["(4)", "(5)", "(6)", "(7)"]);
  });
});
