import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { DiscoveryReport } from "@/lib/chat-types";
import { DiscoveryCard } from "./discovery-card";

const verification = {
  symbolic_zero: true,
  numeric_points_checked: 12,
  numeric_points_agreed: 12,
  passed: true,
};

const report: DiscoveryReport = {
  status: "ok",
  system: "lagrangian",
  function_latex: "\\frac{m r^{2} \\dot{\\theta}^{2}}{2}",
  coordinates: ["r", "theta"],
  angles: ["theta"],
  budget: "standard",
  constants: [
    {
      id: "C1",
      name: "Momentum conjugate to theta",
      expression_latex: "C1 = m r^{2} \\dot{\\theta}",
      kind: "known",
      source: "cyclic",
      tier: null,
      momentum_degree: 1,
      independence: "independent",
      verified: true,
      verification,
      reason: "",
    },
    {
      id: "C3",
      name: "Constant of motion 1",
      expression_latex: "C3 = - k \\cos{\\theta}",
      kind: "new",
      source: "search",
      tier: 3,
      momentum_degree: 2,
      independence: "independent",
      verified: true,
      verification,
      reason: "",
    },
  ],
  tiers: [
    {
      index: 0,
      code: "d1p1h1t0",
      description: "polynomials of degree at most 1 in the velocities",
      unknowns: 27,
      equations: 50,
      rank: 26,
      nullity: 1,
      new_directions: 0,
      accepted: 0,
      rejected: 0,
      duration_ms: 70,
      outcome: "searched",
    },
  ],
  unexplored: ["raising the power bound to 4 needs 450 unknown coefficients (limit 350)"],
  stop_reason: "unknown_cap",
  cause: "",
  independence: {
    constants: 4,
    rank_lower: 3,
    rank_upper: 3,
    rank_proved: true,
    max_possible: 3,
    relations: [{ latex: "C_{3}^{2} + C_{4}^{2} = 1", factors: 2, exact: true }],
    notes: [],
  },
  parameter_conditions: [],
  limitations: ["A tier is complete within its family only."],
  notes: [],
  message: "1 new verified constant of motion in the searched families.",
  peak_memory_mb: 70,
  duration_ms: 900,
};

describe("DiscoveryCard", () => {
  it("shows known and new constants, independence with its proof, and what was searched", () => {
    render(<DiscoveryCard report={report} />);
    expect(screen.getByText("Newly found")).toBeInTheDocument();
    expect(screen.getByText("Already known")).toBeInTheDocument();
    expect(screen.getAllByText("dC/dt = 0 verified")).toHaveLength(2);
    expect(
      screen.getByText(/3 of the 4 are functionally independent \(proved/),
    ).toBeInTheDocument();
    expect(screen.getByText(/What was searched \(1 family\)/)).toBeInTheDocument();
    expect(screen.getByText(/needs 450 unknown coefficients/)).toBeInTheDocument();
    expect(screen.getByText(/complete within its family only/)).toBeInTheDocument();
    expect(screen.queryByText(/within the explored families/)).not.toBeInTheDocument();
  });

  it("says no additional constant was found within the explored families, not that none exist", () => {
    render(
      <DiscoveryCard
        report={{
          ...report,
          constants: report.constants.filter((c) => c.kind === "known"),
          message: "0 new verified constants of motion in the searched families.",
        }}
      />,
    );
    expect(screen.getByText(/within the explored families/)).toBeInTheDocument();
    expect(screen.getByText(/does not mean the system has none/)).toBeInTheDocument();
  });

  it("states a rank that is only a lower bound as such", () => {
    render(
      <DiscoveryCard
        report={{
          ...report,
          independence: { ...report.independence!, rank_proved: false, rank_upper: 4 },
        }}
      />,
    );
    expect(screen.getByText(/At least 3 and at most 4/)).toBeInTheDocument();
    expect(screen.getByText(/only the lower bound is proved/)).toBeInTheDocument();
  });

  it("shows a refusal without inventing constants", () => {
    render(
      <DiscoveryCard
        report={{
          ...report,
          status: "unsupported",
          constants: [],
          tiers: [],
          independence: null,
          function_latex: null,
          message: "The equations of motion are not rational.",
        }}
      />,
    );
    expect(screen.getByText("The equations of motion are not rational.")).toBeInTheDocument();
    expect(screen.queryByText("Newly found")).not.toBeInTheDocument();
    expect(screen.queryByText(/within the explored families/)).not.toBeInTheDocument();
  });
});
