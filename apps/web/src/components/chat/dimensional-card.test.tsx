import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { DimensionalReport, UnitsCheckReport } from "@/lib/chat-types";
import { DimensionalCard } from "./dimensional-card";
import { UnitsCard } from "./units-card";

const reading = (name: string, given: string, dim: string) => ({
  name,
  latex: name,
  meaning: "",
  given,
  dimension_text: dim,
  dimension_latex: dim,
});

const base: DimensionalReport = {
  status: "ok",
  quantities: [reading("T", "kind: period", "s"), reading("L", "kind: length", "m")],
  rank: 2,
  target: "T",
  repeating: ["L"],
  groups: [
    {
      label: "\\Pi_{1}",
      latex: "\\frac{T \\sqrt{g}}{\\sqrt{L}}",
      exponents: { T: "1" },
      contains_target: true,
      dimensionless_verified: true,
    },
  ],
  closure: "determined",
  relation_latex: "T = C\\,\\sqrt{\\frac{L}{g}}",
  cannot_enter: ["m"],
  missing: null,
  tested: [],
  notes: ["Dimensional analysis fixes the groups, not the function of them."],
  features: [],
  message: "The dimensions fix T up to a dimensionless constant.",
  duration_ms: 3,
};

describe("DimensionalCard", () => {
  it("shows how units were read, the checked groups and what cannot enter", () => {
    render(<DimensionalCard report={base} methods={[]} />);
    expect(screen.getByText("How the units were read")).toBeInTheDocument();
    expect(screen.getByText("kind: period")).toBeInTheDocument();
    expect(screen.getByText("dimensionless: checked")).toBeInTheDocument();
    expect(screen.getByText("Fixed by dimensions, up to a constant")).toBeInTheDocument();
    expect(screen.getByText(/Cannot enter by dimensions alone: m/)).toBeInTheDocument();
    expect(screen.queryByRole("group", { name: "Missing quantity" })).not.toBeInTheDocument();
  });

  it("shows the missing quantity with the constants that close it", () => {
    const report: DimensionalReport = {
      ...base,
      groups: [],
      closure: "missing",
      relation_latex: null,
      cannot_enter: [],
      missing: {
        target_dimension_text: "kg m^2 s^-2",
        target_dimension_latex: "\\mathrm{kg\\,m^{2}\\,s^{-2}}",
        unbalanced: ["m", "s"],
        missing_dimension_text: "m s^-1",
        missing_dimension_latex: "\\mathrm{m\\,s^{-1}}",
        verified: true,
        explanation: "E cannot be built from the other quantities.",
        suggestions: [
          {
            ids: ["const.c"],
            latex: ["c"],
            closure: "determined",
            relation_latex: "E = C\\,m c^{2}",
            group_latex: null,
            verified: true,
          },
        ],
      },
      tested: ["Tested each of 4 handbook constants alone: 1 close the gap."],
    };
    render(<DimensionalCard report={report} methods={[]} />);
    const box = screen.getByRole("group", { name: "Missing quantity" });
    expect(box).toHaveTextContent("A quantity is missing");
    expect(box).toHaveTextContent("const.c");
    expect(screen.getByText(/Tested each of 4 handbook constants/)).toBeInTheDocument();
  });

  it("says plainly when no constant helps and when the analysis was refused", () => {
    const none: DimensionalReport = {
      ...base,
      closure: "missing",
      groups: [],
      missing: {
        target_dimension_text: "A",
        target_dimension_latex: "A",
        unbalanced: ["A"],
        missing_dimension_text: "A",
        missing_dimension_latex: "A",
        verified: true,
        explanation: "",
        suggestions: [],
      },
    };
    const { unmount } = render(<DimensionalCard report={none} methods={[]} />);
    expect(screen.getByText(/No single handbook constant, and no pair/)).toBeInTheDocument();
    unmount();
    const refused: DimensionalReport = {
      ...base,
      status: "parse_error",
      quantities: [],
      groups: [],
      closure: null,
      relation_latex: null,
      message: "x: unknown unit 'cm'",
    };
    render(<DimensionalCard report={refused} methods={[]} />);
    expect(screen.getByText("x: unknown unit 'cm'")).toBeInTheDocument();
    expect(screen.queryByText("dimensionless: checked")).not.toBeInTheDocument();
  });
});

describe("UnitsCard", () => {
  const report: UnitsCheckReport = {
    status: "ok",
    equation_latex: "W = c m",
    consistent: false,
    sides: [
      {
        latex: "W",
        dimension_text: "kg m^2 s^-2",
        dimension_latex: "\\mathrm{kg\\,m^{2}\\,s^{-2}}",
      },
      { latex: "c m", dimension_text: "kg m s^-1", dimension_latex: "\\mathrm{kg\\,m\\,s^{-1}}" },
    ],
    missing_factor_text: "m s^-1",
    missing_factor_latex: "\\mathrm{m\\,s^{-1}}",
    problem: null,
    problem_terms: [],
    quantities: [],
    notes: [],
    message: "The sides differ.",
    duration_ms: 2,
  };

  it("flags an inconsistent equation and names the missing factor", () => {
    render(<UnitsCard report={report} methods={[]} />);
    expect(screen.getByText("Not dimensionally consistent")).toBeInTheDocument();
    expect(screen.getByText(/A factor with dimension/)).toBeInTheDocument();
  });

  it("confirms a consistent equation and falls back to the status when the check did not run", () => {
    const { unmount } = render(
      <UnitsCard
        report={{ ...report, consistent: true, missing_factor_latex: null }}
        methods={[]}
      />,
    );
    expect(screen.getByText("Dimensionally consistent")).toBeInTheDocument();
    unmount();
    render(
      <UnitsCard
        report={{
          ...report,
          status: "parse_error",
          consistent: null,
          sides: [],
          message: "no unit for y",
        }}
        methods={[]}
      />,
    );
    expect(screen.queryByText(/dimensionally consistent/i)).not.toBeInTheDocument();
    expect(screen.getByText("no unit for y")).toBeInTheDocument();
  });
});
