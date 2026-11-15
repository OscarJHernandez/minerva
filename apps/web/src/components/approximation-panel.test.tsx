import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { comparison, notVerified, needsInterval, pade } from "@/test/approximation-fixtures";
import { ApproximationPanel } from "./approximation-panel";

describe("ApproximationPanel", () => {
  it("shows the formula with the order of what was dropped, and the evidence in words", () => {
    render(<ApproximationPanel report={pade} />);
    expect(screen.getByRole("region", { name: /^Approximation/ })).toBeInTheDocument();
    expect(screen.getByText("Series checked numerically")).toBeInTheDocument();
    const result = screen.getByTestId("approximation-result");
    expect(result.textContent).toContain("O");
  });

  it("states every warning in words, including the pole the function does not have", () => {
    render(<ApproximationPanel report={pade} />);
    const warnings = screen.getByRole("list", { name: "Warnings" });
    expect(
      within(warnings).getByText(/pole that the original function does not have/),
    ).toBeInTheDocument();
  });

  it("shows checks as passed or undecided, never silently", () => {
    render(<ApproximationPanel report={pade} />);
    const checks = screen.getByRole("list", { name: "Checks" });
    expect(within(checks).getByText("passed")).toBeInTheDocument();
    expect(within(checks).getByText("undecided")).toBeInTheDocument();
  });

  it("shows a comparison with the chosen candidate, its cost, and the tolerance as sampled", () => {
    render(<ApproximationPanel report={comparison} />);
    const table = screen.getByRole("table", { name: "Candidates" });
    const rows = within(table).getAllByRole("row");
    expect(rows).toHaveLength(4);
    expect(within(table).getByText("chosen")).toBeInTheDocument();
    expect(within(table).getByText(/Tolerance \(sampled\)/)).toBeInTheDocument();
    expect(screen.getByText(/numerical support, not a guarantee/)).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /Absolute error of the approximation/ }),
    ).toBeInTheDocument();
  });

  it("says plainly when the requested tolerance is not met", () => {
    render(<ApproximationPanel report={{ ...comparison, tolerance_met: false }} />);
    expect(screen.getByText(/tolerance is NOT met/)).toBeInTheDocument();
  });

  it("calls the error a measurement, not a bound", () => {
    render(<ApproximationPanel report={{ ...pade, candidates: [], error: comparison.error }} />);
    expect(screen.getByText(/a measurement, not a bound/)).toBeInTheDocument();
  });

  it("asks the question and shows no result when something is needed", () => {
    render(<ApproximationPanel report={needsInterval} />);
    expect(screen.getByText("Needed from you:")).toBeInTheDocument();
    expect(
      screen.getByText("On which interval should the approximation hold?"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("approximation-result")).not.toBeInTheDocument();
  });

  it("never shows a result that was not verified", () => {
    render(<ApproximationPanel report={{ ...notVerified, result_latex: "WRONG" }} />);
    expect(screen.getByText("Could not verify a result")).toBeInTheDocument();
    expect(screen.queryByTestId("approximation-result")).not.toBeInTheDocument();
    expect(screen.queryByText("WRONG")).not.toBeInTheDocument();
  });
});
