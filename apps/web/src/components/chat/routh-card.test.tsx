import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Card, RouthReport } from "@/lib/chat-types";
import fixture from "../../../../../packages/ui/fixtures/equation-rows.json";
import { CardView } from "./card-view";
import { RouthCard } from "./routh-card";

const ok = (fixture as unknown as { routh: { data: RouthReport } }).routh.data;

describe("RouthCard", () => {
  it("shows R, what was held fixed, the velocity, where it holds and how it was checked", () => {
    render(<RouthCard report={ok} />);
    const text = document.body.textContent ?? "";
    expect(text).toContain("Routhian: a Lagrangian for r, theta, phi");
    expect(text).toContain("Held fixed");
    expect(text).toContain("Velocities of the removed coordinates");
    expect(text).toContain("Reduced, proved and checked");
    expect(screen.getByTestId("restrictions")).toBeInTheDocument();
    expect(text).toContain("Landau-Lifshitz §41");
  });

  it("a refusal shows its message and no formula of a result", () => {
    const refused: RouthReport = {
      ...ok,
      status: "unsupported",
      reason: "not_cyclic",
      result_latex: null,
      held_fixed: [],
      reconstruction: [],
      restrictions_latex: [],
      message: "r is not cyclic: the Lagrangian depends on it.",
    };
    render(<RouthCard report={refused} />);
    const text = document.body.textContent ?? "";
    expect(text).toContain("r is not cyclic");
    expect(text).not.toContain("Held fixed");
    expect(screen.queryByTestId("restrictions")).not.toBeInTheDocument();
  });

  it("the whole number of the card names R", () => {
    const card = {
      id: "c",
      tool: "routh_reduce",
      type: "routh",
      title: "Routh reduction",
      data: ok,
      number: 4,
      numbers: { "held:t": 5, "velocity:t": 6 },
    } as unknown as Card;
    render(<CardView card={card} />);
    const numbers = screen.getAllByTestId("equation-number").map((e) => e.textContent);
    expect(numbers).toEqual(["(4)", "(5)", "(6)"]);
  });
});
