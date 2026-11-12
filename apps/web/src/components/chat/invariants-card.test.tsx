import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { InvariantReport } from "@/lib/chat-types";
import { InvariantsCard } from "./invariants-card";

const verification = {
  symbolic_zero: true,
  numeric_points_checked: 12,
  numeric_points_agreed: 12,
  passed: true,
};

const report: InvariantReport = {
  status: "ok",
  lagrangian_latex: "\\frac{m r^{2} \\dot{\\theta}^{2}}{2}",
  coordinates: ["r", "theta"],
  angles: ["theta"],
  degree: 2,
  power_bound: 3,
  harmonics: 1,
  unknowns: 126,
  ansatz: "Constants of the form sum f(q) times a velocity monomial. 126 unknown coefficients.",
  known: ["Momentum conjugate to theta", "Energy function"],
  found: [
    {
      name: "Constant of motion 1",
      expression_latex: "C_{1} = - k \\cos{\\theta}",
      verified: true,
      verification,
    },
  ],
  notes: ["Nothing outside this family was searched; finding none proves nothing beyond it."],
  features: [],
  message: "1 new verified constant of motion in the searched family.",
  duration_ms: 5,
};

describe("InvariantsCard", () => {
  it("shows each constant as verified, with the family that was searched", () => {
    render(<InvariantsCard report={report} />);
    expect(screen.getByText("Constant of motion 1")).toBeInTheDocument();
    expect(screen.getByText("dC/dt = 0 verified")).toBeInTheDocument();
    expect(screen.getByText(/12 of 12 sample points give 0/)).toBeInTheDocument();
    expect(screen.getByText(/126 unknown coefficients/)).toBeInTheDocument();
    expect(screen.getByText(/finding none proves nothing/)).toBeInTheDocument();
    expect(screen.getByText(/Already known and left out of the count/)).toBeInTheDocument();
  });

  it("says plainly when nothing was found", () => {
    render(<InvariantsCard report={{ ...report, found: [], message: "0 new constants." }} />);
    expect(screen.queryByText("Constant of motion 1")).not.toBeInTheDocument();
    expect(screen.getByText("0 new constants.")).toBeInTheDocument();
  });

  it("shows a refusal without a search description", () => {
    render(
      <InvariantsCard
        report={{
          ...report,
          status: "unsupported",
          found: [],
          ansatz: "",
          known: [],
          notes: [],
          message: "The Lagrangian depends explicitly on time.",
        }}
      />,
    );
    expect(screen.getByText(/depends explicitly on time/)).toBeInTheDocument();
    expect(screen.queryByText("What was searched")).not.toBeInTheDocument();
  });
});
