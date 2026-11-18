import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { IdentityReport } from "@/lib/chat-types";
import { IdentityCard } from "./identity-card";

const base: IdentityReport = {
  status: "ok",
  verdict: null,
  input_latex: null,
  lhs_latex: "\\left(x + y\\right)^{2}",
  rhs_latex: "x^{2} + y^{2}",
  condition: null,
  counterexample: null,
  message: "message text",
};

describe("IdentityCard", () => {
  it("shows the exact point where a false identity fails", () => {
    render(
      <IdentityCard
        report={{
          ...base,
          verdict: "not_identity",
          counterexample: {
            substitution: { x: "1", y: "1" },
            lhs_value: "4.0",
            rhs_value: "2.0",
            residual: "2",
            probe: "random",
          },
        }}
      />,
    );
    expect(screen.getByText("Not an identity")).toBeInTheDocument();
    expect(screen.getByTestId("identity-witness")).toHaveTextContent("A point where");
    expect(screen.getByText("message text")).toBeInTheDocument();
  });

  it("says numerical agreement is not a proof, and a proof is a proof", () => {
    const { rerender } = render(
      <IdentityCard report={{ ...base, verdict: "numerically_equal" }} />,
    );
    expect(screen.getByText("Equal: checked numerically, not a proof")).toBeInTheDocument();
    rerender(<IdentityCard report={{ ...base, verdict: "proved_equal" }} />);
    expect(screen.getByText("Equal: proved")).toBeInTheDocument();
    expect(screen.queryByTestId("identity-witness")).not.toBeInTheDocument();
  });

  it("states the condition of a conditional result", () => {
    render(
      <IdentityCard
        report={{
          ...base,
          verdict: "conditional",
          condition: "false in general, but holds if x is positive",
        }}
      />,
    );
    expect(screen.getByTestId("identity-condition")).toHaveTextContent("holds if x is positive");
  });

  it("shows the status, not a verdict, for a refused statement", () => {
    render(
      <IdentityCard
        report={{
          ...base,
          status: "unsupported",
          lhs_latex: null,
          rhs_latex: null,
          message: "Not decided",
        }}
      />,
    );
    expect(screen.queryByText("Not an identity")).not.toBeInTheDocument();
    expect(screen.getByText("Not decided")).toBeInTheDocument();
  });
});
