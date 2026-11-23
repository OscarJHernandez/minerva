import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Card } from "@/lib/chat-types";
import fixture from "../../../../../packages/ui/fixtures/equation-rows.json";
import { OdeAnalysisPanel } from "../ode-analysis-panel";
import { OdeSolutionPanel } from "../ode-solution-panel";
import { CardView } from "./card-view";

/**
 * The keys the server numbers (`services/api/.../card_equations.py`) are pinned by the same fixture in
 * `test_equation_rows_contract.py`. Here every key of the fixture is given a distinct number and the panels
 * must show exactly those numbers: a row the server numbers and the panel does not mark (or the reverse)
 * fails one of the two tests.
 */
type Entry = { data: { report: unknown; methods?: unknown[] }; keys: string[] };
const entries = fixture as unknown as Record<"solve" | "ode_solution" | "ode_analysis", Entry>;

function numbersFor(keys: string[]): Record<string, number> {
  return Object.fromEntries(keys.map((k, i) => [k, 100 + i]));
}

function shown(): string[] {
  return screen.queryAllByTestId("equation-number").map((e) => e.textContent ?? "");
}

function card(kind: "solve" | "ode_solution" | "ode_analysis", numbers: Record<string, number>) {
  const entry = entries[kind];
  return {
    id: "c",
    tool: kind,
    type: kind,
    title: kind,
    data: { ...entry.data, methods: [], request: {} },
    numbers,
  } as unknown as Card;
}

describe("row numbers of the solve and differential equation cards", () => {
  it.each(["solve", "ode_solution", "ode_analysis"] as const)(
    "%s: every key the server numbers is a marked row, and nothing else is",
    (kind) => {
      const keys = entries[kind].keys;
      const numbers = numbersFor(keys);
      render(<CardView card={card(kind, numbers)} />);
      const expected = new Set(Object.values(numbers).map((n) => `(${n})`));
      expect(new Set(shown())).toEqual(expected);
      expect(shown()).toHaveLength(keys.length);
    },
  );

  it.each(["solve", "ode_solution", "ode_analysis"] as const)(
    "%s: a key the server did not number shows no number",
    (kind) => {
      const keys = entries[kind].keys;
      const numbers = numbersFor(keys);
      delete numbers[keys[0] ?? ""];
      render(<CardView card={card(kind, numbers)} />);
      expect(shown()).toHaveLength(keys.length - 1);
    },
  );

  it("shows no number without any, as in a card from before they existed", () => {
    render(<CardView card={card("solve", {})} />);
    expect(screen.queryByTestId("equation-number")).not.toBeInTheDocument();
  });

  it("the workbench panels render unchanged without a provider", () => {
    const solution = entries.ode_solution.data.report as Parameters<
      typeof OdeSolutionPanel
    >[0]["report"];
    const analysis = entries.ode_analysis.data.report as Parameters<
      typeof OdeAnalysisPanel
    >[0]["report"];
    render(
      <>
        <OdeSolutionPanel report={solution} />
        <OdeAnalysisPanel report={analysis} />
      </>,
    );
    expect(screen.queryByTestId("equation-number")).not.toBeInTheDocument();
    expect(document.body.textContent).toContain("Closed form");
  });
});
