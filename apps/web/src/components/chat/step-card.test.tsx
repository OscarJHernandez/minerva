import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Card } from "@/lib/chat-types";
import { isolateResult, stepResult } from "@/test/fixtures";
import { CardView } from "./card-view";

describe("CardView for controlled steps", () => {
  it("shows a substitution with its premise and the conditional label", () => {
    const card: Card = {
      id: "c1",
      tool: "substitute",
      type: "step",
      title: "Substitute",
      data: { operation: "substitute", result: stepResult() },
    };
    render(<CardView card={card} />);
    expect(screen.getByText("Holds given your relation")).toBeInTheDocument();
    expect(screen.getByTestId("premises")).toHaveTextContent("not checked by Minerva");
  });

  it("shows the solutions of an equation without a way to continue (that is the Workbench's)", () => {
    const card: Card = {
      id: "c2",
      tool: "isolate_variable",
      type: "step",
      title: "Solve for a variable",
      data: { operation: "isolate", result: isolateResult() },
    };
    render(<CardView card={card} />);
    expect(screen.getByLabelText("Solutions")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Continue with this solution" })).toBeNull();
    expect(screen.getByTestId("exceptional")).toBeInTheDocument();
  });

  it("shows a refusal as a refusal", () => {
    const card: Card = {
      id: "c3",
      tool: "isolate_variable",
      type: "step",
      title: "Solve for a variable",
      data: {
        operation: "isolate",
        result: isolateResult({
          status: "unsupported",
          solutions: [],
          reason: "form",
          message: "no",
        }),
      },
    };
    render(<CardView card={card} />);
    expect(screen.getByText("Not supported")).toBeInTheDocument();
    expect(screen.queryByLabelText("Solutions")).toBeNull();
  });
});
