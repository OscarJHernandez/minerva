import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { TelescopingResult } from "@/lib/telescoping-types";
import { TelescopingCard } from "./telescoping-card";

const base: TelescopingResult = {
  status: "ok",
  outcome: "evaluated",
  reason: null,
  message: "The integral was evaluated from the derived differential equation and checked.",
  input_latex: "\\mathrm{e}^{- a x^{2}}",
  variable: "x",
  parameter: "a",
  lower_latex: "-\\infty",
  upper_latex: "\\infty",
  parameter_domain_latex: "a \\in \\left(0, \\infty\\right)",
  operator: { order: 1, coefficients_latex: ["1", "2 a"], latex: "1 + 2 a\\,\\partial_{a}" },
  certificate: { latex: "x \\mathrm{e}^{- a x^{2}}", latex_ratio: "x" },
  certificate_check: {
    symbolic_zero: true,
    numeric_points_checked: 20,
    numeric_points_agreed: 20,
    derivative_points_checked: 12,
    derivative_points_agreed: 12,
    passed: true,
    note: "L f - dG/dx simplifies to zero; both numerical legs agree",
  },
  families: [{ order: 1, degree: 1, unknowns: 4, outcome: "found" }],
  conditions: [
    {
      latex: "|f| \\text{ decays exponentially at } \\infty",
      established: true,
      how: "the real part of the exponent tends to -\\infty",
    },
  ],
  boundary: {
    lower_latex: "0",
    upper_latex: "0",
    contribution_latex: "0",
    vanishes: true,
  },
  equation: {
    latex: "I(a) + 2 a\\,I'(a) = 0",
    homogeneous: true,
    rhs_latex: "0",
    leading_nonzero: true,
  },
  solution: {
    general_latex: "I(a) = \\frac{C_{1}}{\\sqrt{a}}",
    initial_point_latex: "a = 1",
    initial_values_latex: ["I(a) = \\sqrt{\\pi}"],
    constants_latex: ["C_{1} = \\sqrt{\\pi}"],
  },
  result_latex: "\\frac{\\sqrt{\\pi}}{\\sqrt{a}}",
  result_text: "sqrt(pi)/sqrt(a)",
  value_verification: {
    ode_symbolic_zero: true,
    initial_values_hold: true,
    quadrature_points_checked: 8,
    quadrature_points_agreed: 8,
    passed: true,
    note: "satisfies the equation exactly and the initial values; confirmed by numerical quadrature",
  },
  unresolved: [],
  notes: [],
  duration_ms: 200,
};

describe("TelescopingCard", () => {
  it("shows the operator, the proved certificate, the equation and the checked value", () => {
    render(<TelescopingCard result={base} />);
    expect(screen.getByText("Evaluated and checked")).toBeInTheDocument();
    expect(screen.getByText(/Proved exactly/)).toBeInTheDocument();
    expect(screen.getByText("established")).toBeInTheDocument();
    expect(screen.getByText("Value")).toBeInTheDocument();
    expect(screen.getByText(/8\/8 parameter values agree/)).toBeInTheDocument();
  });

  it("never presents a certificate or an equation as the value of the integral", () => {
    const partial: TelescopingResult = {
      ...base,
      outcome: "equation",
      reason: "initial_value_unavailable",
      result_latex: null,
      result_text: null,
      solution: null,
      value_verification: null,
      message: "The equation was derived but no independently verified initial values exist.",
      unresolved: ["The integration constants stay undetermined."],
    };
    render(<TelescopingCard result={partial} />);
    expect(screen.getByText("Equation only: not evaluated")).toBeInTheDocument();
    expect(screen.queryByText("Value")).toBeNull();
    expect(screen.getByText(/Not established:/)).toBeInTheDocument();
  });

  it("states that a failed search proves nothing outside the searched families", () => {
    render(
      <TelescopingCard
        result={{
          ...base,
          status: "unsupported",
          outcome: "none",
          reason: "no_telescoper_found",
          operator: null,
          certificate: null,
          certificate_check: null,
          conditions: [],
          boundary: null,
          equation: null,
          solution: null,
          result_latex: null,
          value_verification: null,
          message: "No telescoping relation was found within the searched families.",
        }}
      />,
    );
    expect(screen.getByText(/proves nothing outside them/)).toBeInTheDocument();
    expect(screen.queryByText("Evaluated and checked")).toBeNull();
  });
});
