import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { okSimplify } from "@/test/fixtures";
import { ResultPanel } from "./result-panel";

describe("ResultPanel", () => {
  it("shows the verified result and how it was checked", () => {
    render(<ResultPanel result={okSimplify} />);
    expect(screen.getByText("Simplified and verified")).toBeInTheDocument();
    expect(screen.getByText(/12 of 12 sample points agree/)).toBeInTheDocument();
    expect(screen.getByText(/Difference simplifies to 0/)).toBeInTheDocument();
    expect(screen.getByLabelText("Result")).toBeInTheDocument();
  });

  it("says plainly when nothing simpler was found", () => {
    render(
      <ResultPanel
        result={{
          ...okSimplify,
          status: "unchanged",
          verification: null,
          message: "No simpler form was found. The expression is returned unchanged.",
        }}
      />,
    );
    expect(screen.getByText("No simpler form found")).toBeInTheDocument();
    expect(screen.getByText(/returned unchanged/)).toBeInTheDocument();
  });

  it("never shows a result that failed verification", () => {
    render(
      <ResultPanel
        result={{
          ...okSimplify,
          status: "not_verified",
          result_latex: "SHOULD-NOT-APPEAR",
          result_text: "SHOULD-NOT-APPEAR",
          message: "A different form was produced but it could not be verified.",
        }}
      />,
    );
    expect(screen.getByText("Could not verify a result")).toBeInTheDocument();
    expect(screen.queryByText(/SHOULD-NOT-APPEAR/)).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Result")).not.toBeInTheDocument();
  });

  it("reports parse errors and resource limits without a result", () => {
    const { rerender } = render(
      <ResultPanel
        result={{
          ...okSimplify,
          status: "parse_error",
          result_latex: null,
          input_latex: null,
          verification: null,
          message: "unsupported LaTeX command \\foo",
        }}
      />,
    );
    expect(screen.getByText("Could not read the expression")).toBeInTheDocument();
    expect(screen.getByText(/unsupported LaTeX command/)).toBeInTheDocument();
    rerender(
      <ResultPanel
        result={{
          ...okSimplify,
          status: "resource_limit",
          result_latex: null,
          verification: null,
          message: "time limit",
        }}
      />,
    );
    expect(screen.getByText("Resource limit reached")).toBeInTheDocument();
  });
});
