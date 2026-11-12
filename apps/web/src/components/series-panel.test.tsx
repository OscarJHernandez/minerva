import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { okSeries } from "@/test/fixtures";
import { SeriesPanel } from "./series-panel";

describe("SeriesPanel", () => {
  it("shows the verified polynomial with its remainder and both checks", () => {
    render(<SeriesPanel result={okSeries} />);
    expect(screen.getByText("Expanded and verified")).toBeInTheDocument();
    expect(screen.getByLabelText("Result")).toBeInTheDocument();
    expect(screen.getByText(/in x about 0, order 3/)).toBeInTheDocument();
    expect(screen.getByText(/agree with a Cauchy-integral estimate/)).toBeInTheDocument();
    expect(screen.getByText(/shrinks at order 4, as claimed/)).toBeInTheDocument();
    expect(screen.getByText(/1 parameter set, real and imaginary directions/)).toBeInTheDocument();
  });

  it("labels the error table as a measurement, not a bound", () => {
    render(<SeriesPanel result={okSeries} />);
    expect(screen.getByText("Measured error of the truncation")).toBeInTheDocument();
    expect(screen.getByText("2.34e-3")).toBeInTheDocument();
    expect(screen.getByText(/not a bound/)).toBeInTheDocument();
  });

  it("never shows a series that is unsupported or unverified", () => {
    for (const status of [
      "unsupported",
      "not_verified",
      "parse_error",
      "resource_limit",
    ] as const) {
      const { unmount } = render(
        <SeriesPanel
          result={{
            ...okSeries,
            status,
            result_latex: "SHOULD-NOT-APPEAR",
            result_text: "SHOULD-NOT-APPEAR",
            remainder_latex: "SHOULD-NOT-APPEAR",
            verification: null,
            message: "Explained here.",
          }}
        />,
      );
      expect(screen.queryByLabelText("Result")).not.toBeInTheDocument();
      expect(screen.queryByText(/SHOULD-NOT-APPEAR/)).not.toBeInTheDocument();
      expect(screen.queryByText("Measured error of the truncation")).not.toBeInTheDocument();
      expect(screen.getByText("Explained here.")).toBeInTheDocument();
      unmount();
    }
  });

  it("states a failed leg instead of hiding it", () => {
    render(
      <SeriesPanel
        result={{
          ...okSeries,
          status: "not_verified",
          result_latex: null,
          remainder_latex: null,
          measured_errors: [],
          verification: {
            ...okSeries.verification!,
            coefficients_passed: false,
            remainder_passed: false,
            passed: false,
          },
        }}
      />,
    );
    expect(screen.getByText(/do not agree with a Cauchy-integral estimate/)).toBeInTheDocument();
    expect(screen.getByText(/was not confirmed at the claimed order/)).toBeInTheDocument();
    expect(screen.queryByLabelText("Result")).not.toBeInTheDocument();
  });
});
