import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { diagnostics, isolateResult, stepResult } from "@/test/fixtures";
import { IsolatePanel, StepResultPanel } from "./step-result-panel";

describe("StepResultPanel", () => {
  it("says that a substitution holds given the user's relation, which Minerva did not check", () => {
    render(<StepResultPanel result={stepResult()} />);
    expect(screen.getByText("Holds given your relation")).toBeInTheDocument();
    const premises = screen.getByTestId("premises");
    expect(premises).toHaveTextContent("stated by you, not checked by Minerva");
    // The substitution is shown before the arithmetic done on construction, then its value.
    expect(screen.getByRole("list", { name: "Steps" })).toHaveTextContent("1 - x");
    expect(screen.getByLabelText("Result")).toBeInTheDocument();
  });

  it("does not present a result as conditional when it holds without the relation", () => {
    render(<StepResultPanel result={stepResult({ evidence: "identity" })} />);
    expect(screen.getByText("Verified")).toBeInTheDocument();
    expect(screen.queryByTestId("premises")).not.toBeInTheDocument();
  });

  it("shows why a refused substitution was refused and no result", () => {
    render(
      <StepResultPanel
        result={stepResult({
          status: "unsupported",
          reason: "order_dependent",
          result_latex: null,
          result_text: null,
          steps: [],
          given: [],
          message: "The result depends on the order.",
        })}
      />,
    );
    expect(screen.getByText("Not supported")).toBeInTheDocument();
    expect(screen.getByText(/depends on the order of the relations/)).toBeInTheDocument();
    expect(screen.queryByLabelText("Result")).not.toBeInTheDocument();
  });

  it("reports an identity whose conditions are unproven without a result", () => {
    render(
      <StepResultPanel
        result={stepResult({
          status: "conditions_not_satisfied",
          operation: "rewrite",
          result_latex: null,
          result_text: null,
          steps: [],
          given: [],
          unmet_conditions: [{ variable: "a", assumption: "positive" }],
          reason: "conditions_unmet",
        })}
      />,
    );
    expect(screen.getByText(/a must be positive/)).toBeInTheDocument();
    expect(screen.queryByLabelText("Result")).not.toBeInTheDocument();
  });

  it("lists restrictions and conditions", () => {
    render(
      <StepResultPanel
        result={stepResult({ restrictions_latex: ["x \\neq 0"], conditions_latex: ["a \\neq 0"] })}
      />,
    );
    expect(screen.getByTestId("restrictions")).toBeInTheDocument();
    expect(screen.getByTestId("conditions")).toHaveTextContent("Valid when");
  });

  it("names rejection reasons in words and shows debug detail only when it exists", () => {
    const { rerender } = render(
      <StepResultPanel
        result={stepResult({ diagnostics: diagnostics({ rejected: { no_match: 2 } }) })}
      />,
    );
    expect(screen.getByText(/The move does not apply to this part \(2\)/)).toBeInTheDocument();
    expect(screen.queryByTestId("debug-detail")).not.toBeInTheDocument();
    rerender(
      <StepResultPanel
        result={stepResult({ diagnostics: diagnostics({ detail: ["candidate: x + 1"] }) })}
      />,
    );
    expect(screen.getByTestId("debug-detail")).toHaveTextContent("candidate: x + 1");
  });
});

describe("IsolatePanel", () => {
  it("lists every solution with the conditions and the parameter values it does not cover", () => {
    render(<IsolatePanel result={isolateResult()} />);
    expect(screen.getByText("Verified, none dropped")).toBeInTheDocument();
    const list = screen.getByRole("list", { name: "Solutions" });
    expect(within(list).getAllByRole("listitem")).toHaveLength(2);
    expect(screen.getByText(/All solutions need/)).toBeInTheDocument();
    expect(screen.getByTestId("exceptional")).toHaveTextContent("examined separately");
    expect(screen.getByText(/Nothing dropped/)).toBeInTheDocument();
  });

  it("continues with the chosen solution", () => {
    const onContinue = vi.fn();
    render(<IsolatePanel result={isolateResult()} onContinue={onContinue} />);
    const buttons = screen.getAllByRole("button", { name: "Continue with this solution" });
    expect(buttons).toHaveLength(2);
    fireEvent.click(buttons[1]!);
    expect(onContinue).toHaveBeenCalledWith(1);
  });

  it("says plainly when there is no solution", () => {
    render(
      <IsolatePanel
        result={isolateResult({
          solutions: [],
          conditions_latex: [],
          exceptional: [],
          message: "none",
        })}
      />,
    );
    expect(screen.getByTestId("no-solution")).toHaveTextContent("No solution");
  });

  it("shows an unsupported equation as such, with no solutions and no tick", () => {
    render(
      <IsolatePanel
        result={isolateResult({
          status: "unsupported",
          solutions: [],
          reason: "form",
          complete: false,
          message: "not supported",
        })}
      />,
    );
    expect(screen.getByText("Not supported")).toBeInTheDocument();
    expect(screen.queryByRole("list", { name: "Solutions" })).not.toBeInTheDocument();
    expect(screen.queryByText("Verified, none dropped")).not.toBeInTheDocument();
  });

  it("marks a family of solutions and the integer that indexes it", () => {
    render(
      <IsolatePanel
        result={isolateResult({
          solutions: [
            {
              latex: "t = k_{1}",
              text: "k_1",
              equation_text: "t = k_1",
              conditions_latex: ["r \\neq 0"],
              branch: "k_1 any integer",
              verification: null,
            },
          ],
        })}
      />,
    );
    expect(screen.getByText("k_1 any integer")).toBeInTheDocument();
  });
});
