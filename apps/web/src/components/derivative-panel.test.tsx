import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { okDerivative } from "@/test/fixtures";
import { DerivativePanel } from "./derivative-panel";

describe("DerivativePanel", () => {
  it("shows the verified result, its steps and how it was checked", () => {
    render(<DerivativePanel result={okDerivative} />);
    expect(screen.getByText("Differentiated and verified")).toBeInTheDocument();
    expect(screen.getByLabelText("Result")).toBeInTheDocument();
    expect(screen.getByText(/24 of 24 sample points agree, over 2 steps/)).toBeInTheDocument();
    expect(screen.getByText(/real and imaginary/)).toBeInTheDocument();
    expect(screen.getByText(/Intermediate derivatives \(1\)/)).toBeInTheDocument();
    expect(screen.getByText(/no independent symbolic test/)).toBeInTheDocument();
  });

  it("never shows a derivative that is unsupported or unverified", () => {
    for (const status of [
      "unsupported",
      "not_verified",
      "parse_error",
      "resource_limit",
    ] as const) {
      const { unmount } = render(
        <DerivativePanel
          result={{
            ...okDerivative,
            status,
            result_latex: "SHOULD-NOT-APPEAR",
            result_text: "SHOULD-NOT-APPEAR",
            notation_latex: "SHOULD-NOT-APPEAR",
            verification: null,
            message: "Explained here.",
          }}
        />,
      );
      expect(screen.queryByLabelText("Result")).not.toBeInTheDocument();
      expect(screen.queryByText(/SHOULD-NOT-APPEAR/)).not.toBeInTheDocument();
      expect(screen.queryByText(/Intermediate derivatives/)).not.toBeInTheDocument();
      expect(screen.getByText("Explained here.")).toBeInTheDocument();
      unmount();
    }
  });

  it("tells the reader when the derivative is exactly zero without a numeric check", () => {
    render(
      <DerivativePanel
        result={{
          ...okDerivative,
          result_latex: "0",
          steps: [{ variable: "y", order_so_far: 1, latex: "0" }],
          verification: {
            ...okDerivative.verification!,
            steps_checked: 0,
            points_checked: 0,
            points_agreed: 0,
            note: "the expression does not depend on the variable, so the derivative is exactly 0",
          },
        }}
      />,
    );
    expect(screen.getByText(/does not depend on the variable/)).toBeInTheDocument();
    expect(screen.queryByText(/sample points agree/)).not.toBeInTheDocument();
  });

  it("surfaces notes such as the Dirac-delta caveat", () => {
    render(<DerivativePanel result={{ ...okDerivative, notes: ["Contains a Dirac delta."] }} />);
    expect(screen.getByText("Contains a Dirac delta.")).toBeInTheDocument();
  });
});
