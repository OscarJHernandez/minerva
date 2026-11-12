import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { okIntegral } from "@/test/fixtures";
import { IntegralPanel } from "./integral-panel";

describe("IntegralPanel", () => {
  it("shows the verified antiderivative and how it was checked", () => {
    render(<IntegralPanel result={okIntegral} />);
    expect(screen.getByText("Integrated, numerically checked")).toBeInTheDocument();
    expect(screen.getByLabelText("Result")).toBeInTheDocument();
    expect(screen.getByText(/12 of 12 sample points agree/)).toBeInTheDocument();
    expect(screen.getByText(/real and imaginary/)).toBeInTheDocument();
    expect(screen.getByText(/simplifies to 0/)).toBeInTheDocument();
    expect(screen.getByText(/arbitrary constant/)).toBeInTheDocument();
  });

  it("never shows an antiderivative that is unsupported or unverified", () => {
    for (const status of [
      "unsupported",
      "not_verified",
      "parse_error",
      "resource_limit",
    ] as const) {
      const { unmount } = render(
        <IntegralPanel
          result={{
            ...okIntegral,
            status,
            result_latex: "SHOULD-NOT-APPEAR",
            result_text: "SHOULD-NOT-APPEAR",
            verification: null,
            message: "Explained here.",
          }}
        />,
      );
      expect(screen.queryByLabelText("Result")).not.toBeInTheDocument();
      expect(screen.queryByText(/SHOULD-NOT-APPEAR/)).not.toBeInTheDocument();
      expect(screen.getByText("Explained here.")).toBeInTheDocument();
      unmount();
    }
  });

  it("says when the symbolic check could not establish the result", () => {
    render(
      <IntegralPanel
        result={{
          ...okIntegral,
          verification: { ...okIntegral.verification!, symbolic_zero: false, branches_checked: 2 },
        }}
      />,
    );
    expect(screen.getByText(/Not established/)).toBeInTheDocument();
    expect(screen.getByText(/over 2 cases/)).toBeInTheDocument();
  });

  it("shows a definite value with its quadrature check and approximation", () => {
    render(
      <IntegralPanel
        result={{
          ...okIntegral,
          definite: true,
          lower_latex: "0",
          upper_latex: "\\pi",
          result_latex: "2",
          numeric: "2.0",
          verification: null,
          quadrature: { method: "q", points_checked: 1, points_agreed: 1, passed: true, note: "" },
        }}
      />,
    );
    expect(screen.getByText("Value")).toBeInTheDocument();
    expect(screen.getByText(/≈ 2\.0/)).toBeInTheDocument();
    expect(screen.getByText(/1 of 1 sample agree with independent numerical/)).toBeInTheDocument();
  });

  it("lists the conditions and shows no value when they are not satisfied", () => {
    render(
      <IntegralPanel
        result={{
          ...okIntegral,
          status: "conditions_not_satisfied",
          definite: true,
          result_latex: "SHOULD-NOT-APPEAR",
          numeric: "SHOULD-NOT-APPEAR",
          conditions_latex: ["a > 0"],
          verification: null,
          message: "Declare them.",
        }}
      />,
    );
    expect(screen.getByText("Conditions not satisfied")).toBeInTheDocument();
    expect(screen.getByText("Needs")).toBeInTheDocument();
    expect(screen.queryByLabelText("Result")).not.toBeInTheDocument();
    expect(screen.queryByText(/SHOULD-NOT-APPEAR/)).not.toBeInTheDocument();
  });
});
