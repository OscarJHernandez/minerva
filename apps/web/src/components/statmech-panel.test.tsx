import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import {
  analysis,
  approximated,
  needsCell,
  refused,
  sweep,
  twoLevel,
} from "@/test/statmech-fixtures";
import { StatMechPanel, sweepSeries } from "./statmech-panel";

describe("StatMechPanel", () => {
  it("shows the derivation, the quantity and the independent checks, worded as evidence", () => {
    render(<StatMechPanel result={twoLevel} />);
    expect(screen.getByTestId("step-partition_function")).toBeInTheDocument();
    const u = screen.getByTestId("quantity-U");
    expect(u).toHaveTextContent("enumeration: agreed (8/8 points)");
    expect(u).toHaveTextContent("symbolic zero: undecided");
    expect(screen.getByTestId("statmech-evidence")).toHaveTextContent(
      "8/8 points agreed with an independent evaluation",
    );
    expect(screen.getByTestId("statmech-evidence")).toHaveTextContent("not a proof");
    expect(screen.getByText("Numerically checked")).toBeInTheDocument(); // never "proved"
  });

  it("shows a refusal with its reason and no formula", () => {
    render(<StatMechPanel result={refused} />);
    expect(screen.getByRole("status")).toHaveTextContent("the series diverges");
    expect(screen.getByRole("status")).toHaveTextContent("Nothing is shown as a result");
    expect(
      screen.queryByTestId("statmech-panel")?.querySelector("[data-testid^=quantity-]"),
    ).toBeNull();
  });

  it("asks for the user's convention and sends exactly that answer", async () => {
    const onAnswer = vi.fn();
    render(
      <StatMechPanel
        result={needsCell}
        needs={{ cell: true, identical: false }}
        onAnswer={onAnswer}
      />,
    );
    const button = screen.getByRole("button", { name: "Continue with this choice" });
    expect(button).toBeDisabled(); // nothing is chosen for the user
    await userEvent.type(screen.getByLabelText("Phase-space cell"), "h^3");
    await userEvent.click(button);
    expect(onAnswer).toHaveBeenCalledWith({ phase_space_cell: "h^3" });
  });

  it("needs a choice for indistinguishability before it continues", async () => {
    const onAnswer = vi.fn();
    render(
      <StatMechPanel
        result={needsCell}
        needs={{ cell: false, identical: true }}
        onAnswer={onAnswer}
      />,
    );
    const button = screen.getByRole("button", { name: "Continue with this choice" });
    expect(button).toBeDisabled();
    await userEvent.click(screen.getByLabelText(/^distinguishable$/));
    await userEvent.click(button);
    expect(onAnswer).toHaveBeenCalledWith({ identical: false });
  });

  it("separates what applies from what Minerva implements", () => {
    render(<StatMechPanel result={analysis} />);
    expect(screen.getByTestId("method-closed_form_sum")).toHaveTextContent("implemented");
    expect(screen.getByTestId("method-high_temperature_expansion")).toHaveTextContent(
      "not implemented yet",
    );
    expect(screen.getByText("Monte Carlo sampling and many-body solvers")).toBeInTheDocument();
  });

  it("tabulates and plots a sweep", () => {
    render(<StatMechPanel result={sweep} />);
    expect(within(screen.getByRole("table")).getByText("0.4")).toBeInTheDocument();
    expect(screen.getByLabelText("Sweep over beta")).toBeInTheDocument();
    expect(sweepSeries(sweep.table!)).toEqual([{ label: "U", x: [0.5, 1, 2], y: [0.4, 0.3, 0.1] }]);
  });

  it("labels an approximation as one and shows what it assumes and how it was measured", () => {
    render(<StatMechPanel result={approximated} />);
    const section = screen.getByTestId("statmech-approximation");
    expect(section).toHaveTextContent("An approximation, not an identity (high temperature)");
    expect(section).toHaveTextContent("Expansion parameter: beta");
    expect(section).toHaveTextContent("O(beta^5)");
    expect(section).toHaveTextContent("beta is small");
    expect(within(section).getByRole("table")).toHaveTextContent("-1.57699e-5");
    expect(section).toHaveTextContent("a measurement, not a bound");
    // the quantity is labelled asymptotic, never exact
    expect(screen.getByTestId("quantity-lnZ_series")).toHaveTextContent("asymptotic");
    expect(screen.getByTestId("quantity-lnZ_series")).not.toHaveTextContent(/\bexact\b/);
  });

  it("shows no approximation section for an exact derivation", () => {
    render(<StatMechPanel result={twoLevel} />);
    expect(screen.queryByTestId("statmech-approximation")).toBeNull();
  });
});
