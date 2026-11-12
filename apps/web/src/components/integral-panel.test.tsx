import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { okIntegral } from "@/test/fixtures";
import { IntegralPanel } from "./integral-panel";

describe("IntegralPanel", () => {
  it("shows the verified antiderivative and how it was checked", () => {
    render(<IntegralPanel result={okIntegral} />);
    expect(screen.getByText("Integrated and verified")).toBeInTheDocument();
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
});
