import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ExpansionReport } from "@/lib/chat-types";
import { okSeries } from "@/test/fixtures";
import { ExpansionCard } from "./expansion-card";

const base: ExpansionReport = {
  status: "ok",
  lagrangian_latex: "L",
  coordinates: ["theta"],
  equilibrium: [{ name: "theta", latex: "0", name_latex: "\\theta" }],
  order: 3,
  equilibrium_verification: {
    symbolic_zero: true,
    numeric_points_checked: 8,
    numeric_points_agreed: 8,
    passed: true,
    note: "every acceleration vanishes at the point, at zero velocity",
  },
  equations: [
    {
      coordinate: "theta",
      full_latex:
        "\\ddot{\\delta \\theta} = - \\frac{\\delta \\theta g}{l} + O\\left(\\delta^{4}\\right)",
      terms: [{ degree: 1, latex: "- \\frac{\\delta \\theta g}{l}" }],
      verification: okSeries.verification,
    },
  ],
  stiffness_latex: "\\left[\\begin{matrix}- \\frac{g}{l}\\end{matrix}\\right]",
  velocity_latex: null,
  characteristic_latex: "\\frac{g + l \\lambda^{2}}{l} = 0",
  modes: [
    {
      omega_squared_latex: "\\omega^2 = \\frac{g}{l}",
      sign: "positive",
      omega_latex: "\\omega = \\frac{\\sqrt{g}}{\\sqrt{l}}",
      multiplicity: 1,
      eigenvector_latex: "\\left[\\begin{matrix}1\\end{matrix}\\right]",
      verified: true,
    },
  ],
  notes: [],
  message: "1 acceleration expanded to order 3 about the equilibrium.",
  duration_ms: 12,
};

describe("ExpansionCard", () => {
  it("shows the verified expansion, the linear part and the modes", () => {
    render(<ExpansionCard report={base} />);
    expect(screen.getByText("Expanded and verified")).toBeInTheDocument();
    expect(screen.getByText("Accelerations to degree 3")).toBeInTheDocument();
    expect(screen.getByText(/Cauchy-integral estimate/)).toBeInTheDocument();
    expect(screen.getByText("Linear part")).toBeInTheDocument();
    expect(screen.getByText("Characteristic polynomial")).toBeInTheDocument();
    expect(
      screen.getByText(/ω² > 0 \(follows from the declared assumptions\)/),
    ).toBeInTheDocument();
    expect(screen.getByText("root checked by substitution")).toBeInTheDocument();
    expect(screen.getByText(/says nothing about large motion/)).toBeInTheDocument();
  });

  it("calls a negative ω² a statement about the linearised equations only", () => {
    render(
      <ExpansionCard
        report={{
          ...base,
          modes: [{ ...base.modes[0]!, sign: "negative", omega_latex: null }],
        }}
      />,
    );
    expect(screen.getByText(/linearised equations only/)).toBeInTheDocument();
    expect(screen.queryByText(/stable/i)).not.toBeInTheDocument();
  });

  it("does not claim a sign the assumptions do not decide", () => {
    render(
      <ExpansionCard
        report={{ ...base, modes: [{ ...base.modes[0]!, sign: "unknown", omega_latex: null }] }}
      />,
    );
    expect(screen.getByText(/Sign not decided by the declared assumptions/)).toBeInTheDocument();
  });

  it("says that velocity terms rule out normal modes and still shows B", () => {
    render(
      <ExpansionCard
        report={{
          ...base,
          velocity_latex: "\\left[\\begin{matrix}0 & 2\\Omega\\end{matrix}\\right]",
          modes: [],
          notes: ["The linear part has velocity terms (damping or gyroscopic)."],
        }}
      />,
    );
    expect(screen.queryByText("Normal modes")).not.toBeInTheDocument();
    expect(screen.getByText(/velocity terms \(damping or gyroscopic\)/)).toBeInTheDocument();
  });

  it("never shows formulas for a refused or unverified expansion", () => {
    for (const status of [
      "unsupported",
      "not_verified",
      "parse_error",
      "resource_limit",
    ] as const) {
      const { unmount } = render(
        <ExpansionCard
          report={{
            ...base,
            status,
            message: "Explained here.",
            equilibrium_verification: { ...base.equilibrium_verification!, passed: false },
          }}
        />,
      );
      expect(screen.queryByText("Accelerations to degree 3")).not.toBeInTheDocument();
      expect(screen.queryByText("Linear part")).not.toBeInTheDocument();
      expect(screen.queryByText("Normal modes")).not.toBeInTheDocument();
      expect(screen.queryByText("Characteristic polynomial")).not.toBeInTheDocument();
      expect(screen.getByText("Explained here.")).toBeInTheDocument();
      expect(screen.getByText(/not an equilibrium/)).toBeInTheDocument();
      unmount();
    }
  });
});
