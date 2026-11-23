import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { okDerivative } from "@/test/fixtures";
import { DerivativePanel } from "./derivative-panel";

describe("DerivativePanel", () => {
  it("shows the verified result, its steps and how it was checked", () => {
    render(<DerivativePanel result={okDerivative} />);
    expect(screen.getByText("Differentiated, numerically checked")).toBeInTheDocument();
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

  it("leads with the scoped zero and keeps the original derivative behind a disclosure", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    const raw = "m \\dot{x} \\ddot{x} + k x \\dot{x}";
    render(
      <DerivativePanel
        total
        number={7}
        result={{
          ...okDerivative,
          result_latex: raw,
          on_motion_latex: "0",
          zero_on_motion: true,
          symbolic_zero_on_motion: true,
          definitions_used: ["C_1 = x"],
          steps: [
            { variable: "t", order_so_far: 1, latex: raw },
            { variable: "t", order_so_far: 1, latex: "0" },
          ],
        }}
      />,
    );
    expect(screen.getByRole("heading", { name: "Total time derivative" })).toBeVisible();
    expect(screen.getByLabelText("Result on the equations of motion")).toBeVisible();
    expect(
      screen.getByText(/conclusion applies to motions of the supplied Lagrangian/),
    ).toBeVisible();
    expect(screen.getByLabelText("Result")).not.toBeVisible();
    expect(screen.getByTestId("equation-number")).not.toBeVisible();
    expect(screen.queryByText(/Intermediate derivatives/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("Full derivative before using the equations of motion"));
    expect(screen.getByLabelText("Result")).toBeVisible();
    expect(screen.getByTestId("equation-number")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Copy LaTeX" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(raw));
    expect(screen.getByText("Definitions substituted (1)")).toBeVisible();
  });

  it("does not turn numerical zero evidence into a symbolic proof", () => {
    render(
      <DerivativePanel
        total
        result={{
          ...okDerivative,
          zero_on_motion: true,
          on_motion_latex: "0",
          symbolic_zero_on_motion: false,
        }}
      />,
    );
    expect(screen.getByText("Numerically confirmed.")).toBeVisible();
    expect(screen.queryByText(/also a symbolic zero/)).not.toBeInTheDocument();
  });

  it("distinguishes a nonzero substitution from an independently verified formula", () => {
    render(
      <DerivativePanel
        total
        result={{ ...okDerivative, zero_on_motion: false, on_motion_latex: "x" }}
      />,
    );
    expect(screen.getByText("Not zero on the supplied equations of motion")).toBeVisible();
    expect(screen.queryByLabelText("Result on the equations of motion")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("Expression after substituting the equations of motion"));
    expect(screen.getByLabelText("Substituted expression")).toBeVisible();
    expect(screen.getByText(/this expression was not separately sampled/)).toBeVisible();
  });

  it("never promotes on-motion fields from a failed result", () => {
    render(
      <DerivativePanel
        total
        result={{
          ...okDerivative,
          status: "not_verified",
          zero_on_motion: true,
          on_motion_latex: "0",
        }}
      />,
    );
    expect(screen.queryByLabelText("Result on the equations of motion")).not.toBeInTheDocument();
    expect(screen.queryByText("Zero on the supplied equations of motion")).not.toBeInTheDocument();
  });

  it("collapses long derivatives without truncating the result or hiding a short one", () => {
    const latex = Array.from({ length: 60 }, (_, i) => `x^{${i + 1}}`).join(" + ");
    render(<DerivativePanel result={{ ...okDerivative, result_latex: latex }} />);
    expect(screen.getByText(/verified derivative is long/)).toBeVisible();
    expect(screen.getByLabelText("Result")).not.toBeVisible();
    fireEvent.click(screen.getByText("Full derivative"));
    expect(screen.getByLabelText("Result")).toBeVisible();
    expect(screen.getByLabelText("Result").querySelector("annotation")?.textContent).toBe(latex);
  });

  it("says nothing about the equations of motion when the engine did not decide", () => {
    for (const extra of [{}, { zero_on_motion: null }, { zero_on_motion: undefined }]) {
      const { unmount } = render(
        <DerivativePanel total result={{ ...okDerivative, ...extra, on_motion_latex: null }} />,
      );
      expect(screen.queryByText(/equations of motion/)).not.toBeInTheDocument();
      expect(screen.getByLabelText("Result")).toBeVisible();
      expect(screen.getByText("Intermediate derivatives (1)")).toBeInTheDocument();
      unmount();
    }
  });

  it("does not call the on-motion evaluation an intermediate derivative", () => {
    render(
      <DerivativePanel
        total
        result={{
          ...okDerivative,
          zero_on_motion: false,
          on_motion_latex: "x",
          steps: [
            { variable: "t", order_so_far: 1, latex: "a" },
            { variable: "t", order_so_far: 2, latex: "b" },
            { variable: "t", order_so_far: 1, latex: "x" },
          ],
        }}
      />,
    );
    expect(screen.getByText("Intermediate derivatives (1)")).toBeInTheDocument();
  });

  it("shows no on-motion conclusion for a refused result and still gives its message", () => {
    render(
      <DerivativePanel
        total
        result={{
          ...okDerivative,
          status: "unsupported",
          result_latex: null,
          result_text: null,
          zero_on_motion: true,
          symbolic_zero_on_motion: true,
          on_motion_latex: "0",
          message: "Not supported here.",
        }}
      />,
    );
    expect(screen.queryByText(/equations of motion/)).not.toBeInTheDocument();
    expect(screen.getByText("Not supported here.")).toBeVisible();
  });
});
