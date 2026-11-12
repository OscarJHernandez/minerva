import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { chainSimplify, okSimplify } from "@/test/fixtures";
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

  it("lists the steps with the identity used and how each was checked", () => {
    render(<ResultPanel result={chainSimplify} />);
    const steps = screen.getAllByRole("listitem").filter((li) => li.textContent?.match(/^\d\./));
    expect(steps).toHaveLength(2);
    expect(screen.getByText("Pythagorean identity")).toHaveAttribute(
      "href",
      "/reference/trig.pythagorean",
    );
    // A SymPy move is labelled as such, and a step without a symbolic zero says so.
    expect(screen.getByText(/SymPy: simplify/)).toBeInTheDocument();
    expect(screen.getByText(/not a handbook identity/)).toBeInTheDocument();
    expect(screen.getByText(/no symbolic proof found/)).toBeInTheDocument();
    expect(screen.getByText(/difference simplifies to 0/)).toBeInTheDocument();
    expect(screen.getByText(/Other forms/)).toBeInTheDocument();
  });

  it("shows an identity that matches but is not applied, with its conditions", () => {
    render(
      <ResultPanel
        result={{
          ...okSimplify,
          status: "conditions_not_satisfied",
          verification: null,
          result_latex: okSimplify.input_latex,
          blocked: [
            {
              rule_id: "log.product",
              direction: "reverse",
              matched_latex: "\\log x + \\log y",
              unmet_conditions: [
                { variable: "a", assumption: "positive" },
                { variable: "b", assumption: "positive" },
              ],
            },
          ],
          message: "Nothing was changed.",
        }}
      />,
    );
    expect(screen.getByText("Conditions not satisfied")).toBeInTheDocument();
    expect(screen.getByText(/not applied/)).toBeInTheDocument();
    expect(screen.getByText("log.product")).toBeInTheDocument();
    expect(screen.getAllByText(/must be positive/, { selector: "li" })).toHaveLength(2);
    expect(screen.queryByText("Steps")).not.toBeInTheDocument();
  });

  it("labels the other goals", () => {
    render(<ResultPanel result={{ ...chainSimplify, goal: "factor" }} />);
    expect(screen.getByText("Factored and verified")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Factor" })).toBeInTheDocument();
  });
});
