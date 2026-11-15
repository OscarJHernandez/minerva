import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  matrixFrechet,
  matrixNumericEvolution,
  matrixRepresentation,
  okMatrixIntegral,
} from "@/test/fixtures";
import { gradientRefused, gradientSymmetric } from "@/test/stat-fixtures";
import { MatrixPanel } from "./matrix-panel";

describe("MatrixPanel", () => {
  it("shows the structure, method, assumptions, steps and both checks of an evaluated result", () => {
    render(<MatrixPanel result={okMatrixIntegral} />);
    expect(
      screen.getByText("Evaluated, checked numerically and exactly", { selector: "span" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/commutation is not assumed for AB/)).toBeInTheDocument();
    expect(screen.getAllByText("sylvester_reduction", { selector: "code" }).length).toBe(2);
    expect(screen.getByLabelText("Result")).toBeInTheDocument();
    expect(screen.getByLabelText("Equation")).toBeInTheDocument();
    expect(screen.getByText(/the Sylvester operator is invertible/)).toBeInTheDocument();
    expect(screen.getByText(/4 of 4 sample points agree/)).toBeInTheDocument();
    expect(screen.getByText(/Passed: the explicit result substituted/)).toBeInTheDocument();
    expect(screen.queryByTestId("not-evaluated")).not.toBeInTheDocument();
  });

  it("names the exact leg only when it passed", () => {
    const numericOnly = {
      ...okMatrixIntegral,
      verification: { ...okMatrixIntegral.verification!, symbolic_zero: null },
    };
    render(<MatrixPanel result={numericOnly} />);
    expect(
      screen.getByText("Evaluated, checked numerically", { selector: "span" }),
    ).toBeInTheDocument();
    expect(screen.queryByText(/and exactly/)).not.toBeInTheDocument();
    expect(screen.queryByText(/verified/i, { selector: "span" })).not.toBeInTheDocument();
  });

  it("never calls a representation an evaluated integral", () => {
    render(<MatrixPanel result={matrixRepresentation} />);
    expect(screen.getByText("Represented exactly, not evaluated")).toBeInTheDocument();
    expect(screen.getByTestId("not-evaluated")).toHaveTextContent(/was not evaluated/);
    expect(screen.queryByText(/^Evaluated,/)).not.toBeInTheDocument();
  });

  it("never shows a result for a failure status", () => {
    for (const status of [
      "unsupported",
      "not_verified",
      "parse_error",
      "resource_limit",
      "conditions_not_satisfied",
    ] as const) {
      const { unmount } = render(
        <MatrixPanel
          result={{
            ...okMatrixIntegral,
            status,
            result_kind: null,
            result_latex: "SHOULD-NOT-APPEAR",
            equation_latex: "SHOULD-NOT-APPEAR",
            verification: null,
            message: "Explained here.",
          }}
        />,
      );
      expect(screen.queryByLabelText("Result")).not.toBeInTheDocument();
      expect(screen.queryByLabelText("Equation")).not.toBeInTheDocument();
      expect(screen.queryByText(/SHOULD-NOT-APPEAR/)).not.toBeInTheDocument();
      expect(screen.getByText("Explained here.")).toBeInTheDocument();
      unmount();
    }
  });

  it("lists what was not applied and why", () => {
    render(
      <MatrixPanel
        result={{
          ...okMatrixIntegral,
          status: "conditions_not_satisfied",
          result_kind: null,
          verification: null,
          blocked: [
            {
              method: "exp_merge_commuting",
              title: "Merge exponentials into one",
              needs: "AB = BA (not declared)",
              latex: "\\exp\\left(A\\right) \\exp\\left(B\\right)",
            },
          ],
        }}
      />,
    );
    expect(screen.getByText(/Not applied/)).toBeInTheDocument();
    expect(screen.getByText(/AB = BA \(not declared\)/)).toBeInTheDocument();
  });

  it("says when the exact check was not run", () => {
    render(
      <MatrixPanel
        result={{
          ...okMatrixIntegral,
          verification: { ...okMatrixIntegral.verification!, symbolic_zero: null },
        }}
      />,
    );
    expect(screen.getByText(/Not run \(the numeric check decides\)/)).toBeInTheDocument();
  });

  it("says a derivative representation was not evaluated, and what the familiar form needs", () => {
    render(<MatrixPanel result={matrixFrechet} />);
    expect(screen.getByTestId("not-evaluated")).toHaveTextContent(
      /The result was not evaluated.*exact representation/,
    );
    expect(screen.queryByText(/The integral was not evaluated/)).not.toBeInTheDocument();
    expect(screen.getByText("Exact representation")).toBeInTheDocument();
    expect(screen.getByText(/Not applied/)).toBeInTheDocument();
    expect(screen.getByText(/X X' = X' X for the exponent X/)).toBeInTheDocument();
  });

  it("labels a numerical solution an approximation, never a closed form or a proof", () => {
    render(<MatrixPanel result={matrixNumericEvolution} />);
    expect(screen.getByText("Numerical approximation, checked independently")).toBeInTheDocument();
    expect(screen.getByText("Numerical value (approximation)")).toBeInTheDocument();
    expect(screen.getByTestId("not-evaluated")).toHaveTextContent(
      /numerical approximation.*not a closed form and not a proof/,
    );
    expect(screen.queryByText(/^Evaluated,/)).not.toBeInTheDocument();
    expect(screen.getByText(/Not run \(the numeric check decides\)/)).toBeInTheDocument();
  });

  it("shows how the result was computed, without any mathematics", () => {
    render(<MatrixPanel result={okMatrixIntegral} />);
    const diagnostics = screen.getByTestId("matrix-diagnostics");
    expect(diagnostics).toHaveTextContent("integrate_polynomial: no match");
    expect(diagnostics).toHaveTextContent("sylvester_reduction: selected");
    expect(diagnostics).toHaveTextContent("2 declared (2 explicit, 0 time dependent), sizes 2×2");
    expect(diagnostics).toHaveTextContent("Invertibility");
    expect(diagnostics).toHaveTextContent("visible in the entries");
    expect(diagnostics).toHaveTextContent("16 digits, tolerance 1e-9, largest residual 2.5e-14");
    expect(diagnostics).toHaveTextContent("verify numeric 170 ms");
    expect(diagnostics).toHaveTextContent("req0123-card0123");
  });

  it("copes with a result that has no diagnostics (a saved older result)", () => {
    render(<MatrixPanel result={{ ...okMatrixIntegral, diagnostics: undefined }} />);
    expect(screen.queryByTestId("matrix-diagnostics")).not.toBeInTheDocument();
  });
});

describe("MatrixPanel: gradients", () => {
  it("says what kind of derivative a symmetric variable gets and labels the entrywise form", () => {
    render(<MatrixPanel result={gradientSymmetric} />);
    expect(
      screen.getByText("Gradient, checked against the numerical derivative"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("derivative-convention")).toHaveTextContent(
      /within the symmetric matrices/i,
    );
    expect(screen.getByTestId("derivative-convention")).toHaveTextContent(/real numbers/);
    const alt = screen.getByTestId("entrywise-alternative");
    expect(alt).toHaveTextContent(/labelled/);
    expect(alt).toHaveTextContent(/not the result above/);
    expect(alt).toHaveTextContent(/4 of 4 sample points agree/);
    expect(screen.getByText(/Not run \(the numeric check decides\)/)).toBeInTheDocument();
  });

  it("explains a refusal in plain words and shows no result", () => {
    render(<MatrixPanel result={gradientRefused} />);
    expect(screen.getByTestId("matrix-reason")).toHaveTextContent(/not declared invertible/);
    expect(screen.queryByLabelText("Result")).toBeNull();
    expect(screen.queryByTestId("derivative-convention")).toBeNull();
  });
});
