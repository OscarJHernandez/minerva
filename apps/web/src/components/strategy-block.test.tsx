import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { StrategyReport } from "@/lib/types";
import { StrategyBlock } from "./strategy-block";

const base: StrategyReport = {
  technique: "substitution",
  why: "composite_with_inner_derivative",
  steps: [
    { what: "substitution", latex: "u = 1 + x^{2}" },
    { what: "transformed integral", latex: "\\int \\frac{1}{2 u^{2}}\\,\\mathrm{d}u" },
    { what: "antiderivative", latex: "- \\frac{1}{2 \\left(x^{2} + 1\\right)}" },
  ],
  suggestions: [{ technique: "substitution", why: "composite_with_inner_derivative" }],
  attempts: [{ technique: "substitution", outcome: "verified", detail: "" }],
  verification: "proved",
  candidates_rejected: 0,
  conditions_latex: [],
  note: "",
};

describe("StrategyBlock", () => {
  it("names the technique, why it applies, its steps and how the result was verified", () => {
    render(<StrategyBlock strategy={base} />);
    expect(screen.getByText("Substitution")).toBeInTheDocument();
    expect(screen.getByText(/a function of an inner expression/)).toBeInTheDocument();
    expect(screen.getByText("transformed integral")).toBeInTheDocument();
    expect(screen.getByText(/Verified independently of the technique/)).toBeInTheDocument();
  });

  it("lists suggestions and a blocked technique without showing steps for it", () => {
    render(
      <StrategyBlock
        strategy={{
          ...base,
          technique: null,
          why: null,
          steps: [],
          verification: "none",
          suggestions: [{ technique: "frullani", why: "difference_at_two_scalings_over_x" }],
          attempts: [
            { technique: "frullani", outcome: "blocked", detail: "scales_not_provably_positive" },
          ],
        }}
      />,
    );
    expect(screen.getByText(/Techniques that may apply: Frullani integral/)).toBeInTheDocument();
    expect(screen.getByText(/not applied, a condition is not established/)).toBeInTheDocument();
    expect(screen.queryByText("transformed integral")).toBeNull();
  });

  it("renders nothing for a plain direct result without suggestions", () => {
    const { container } = render(
      <StrategyBlock
        strategy={{ ...base, technique: "direct", steps: [], suggestions: [], attempts: [] }}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
