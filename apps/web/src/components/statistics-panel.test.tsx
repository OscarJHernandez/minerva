import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import {
  expectationConditional,
  expectationFree,
  fisher,
  mleBernoulliGlobal,
  mleGaussian,
  mleMeasured,
  momentsCauchy,
  momentsNormal,
  posterior,
  sumsOk,
  sumsRefused,
} from "@/test/stat-fixtures";
import { StatisticsPanel, conditionLatex, optimumText } from "./statistics-panel";

describe("StatisticsPanel: sums", () => {
  it("words an instance check as 'checked for', never as a proof", () => {
    render(<StatisticsPanel kind="sums" result={sumsOk} />);
    expect(screen.getByText("Checked on explicit instances")).toBeInTheDocument();
    const checks = screen.getByTestId("checks");
    expect(checks).toHaveTextContent(/checked for n = 1; n = 2; n = 3; n = 5/);
    expect(checks).toHaveTextContent(/not proved for every size/);
    expect(checks).toHaveTextContent(/evidence, not a proof/);
    expect(screen.queryByText(/proved for all/i)).toBeNull();
  });

  it("shows no formula for a refusal and gives the reason in plain words", () => {
    render(<StatisticsPanel kind="sums" result={sumsRefused} />);
    const refusal = screen.getByTestId("refusal");
    expect(refusal).toHaveTextContent(/non-empty/);
    expect(refusal).toHaveTextContent(/No result is shown/);
    expect(screen.queryByLabelText("Result")).toBeNull();
    expect(screen.queryByTestId("checks")).toBeNull();
  });
});

describe("StatisticsPanel: expectation", () => {
  it("shows the premise a conditional result holds under", () => {
    render(<StatisticsPanel kind="expectation" result={expectationConditional} />);
    expect(screen.getByText("Holds given your independence")).toBeInTheDocument();
    expect(screen.getByTestId("premises")).toHaveTextContent(/X and Y are independent/);
    expect(screen.getByTestId("premises")).toHaveTextContent(/not checked by Minerva/);
  });

  it("shows no premise when none was used and keeps the covariance term", () => {
    render(<StatisticsPanel kind="expectation" result={expectationFree} />);
    expect(screen.queryByTestId("premises")).toBeNull();
    expect(screen.getByText("Checked on explicit distributions")).toBeInTheDocument();
    expect(screen.getByTestId("checks")).toHaveTextContent(
      /dependent unless independence was declared/,
    );
  });
});

describe("StatisticsPanel: statistical models", () => {
  it("separates a proved local maximum from a global one and states the admissibility condition", () => {
    render(<StatisticsPanel kind="statistics" result={mleGaussian} />);
    const estimates = screen.getByTestId("estimates");
    expect(estimates).toHaveTextContent(/Local maximum: the Hessian is negative definite/);
    expect(estimates).toHaveTextContent(/not claimed to be global/);
    expect(estimates).not.toHaveTextContent(/Global maximum/);
    expect(estimates).toHaveTextContent(/The estimate exists only if/);
    expect(estimates).toHaveTextContent(/assumed/);
    expect(screen.getByTestId("premises")).toHaveTextContent(
      /independent and identically distributed/,
    );
    expect(screen.getByText("Holds given your premises")).toBeInTheDocument();
  });

  it("says global only when concavity is proved", () => {
    const { rerender } = render(<StatisticsPanel kind="statistics" result={mleBernoulliGlobal} />);
    expect(screen.getByTestId("estimates")).toHaveTextContent(
      /Global maximum: .* concave .* \(proved\)/,
    );
    rerender(<StatisticsPanel kind="statistics" result={mleMeasured} />);
    const text = screen.getByTestId("estimates").textContent ?? "";
    expect(text).toMatch(/measured, not proved/);
    expect(text).not.toMatch(/\(proved/);
  });

  it("never uses the word proved for a measured optimum", () => {
    const e = mleMeasured.estimates[0]!;
    expect(optimumText(e)).toMatch(/measured/);
    expect(optimumText(e)).not.toMatch(/\(proved/);
    expect(optimumText({ ...e, optimum: "stationary_only", optimum_evidence: null })).toMatch(
      /not established/,
    );
  });

  it("draws the information matrix and the checks with their counts", () => {
    render(<StatisticsPanel kind="statistics" result={fisher} />);
    expect(screen.getByLabelText("Information matrix")).toBeInTheDocument();
    expect(screen.getAllByText(/4 of 4 points agree/).length).toBe(2);
  });

  it("shows the posterior family, its parameters and the dropped factors", () => {
    render(<StatisticsPanel kind="statistics" result={posterior} />);
    const box = screen.getByTestId("posterior");
    expect(box).toHaveTextContent(/Posterior: Beta/);
    // the posterior's parameter is never written as an equation with its own prior name alone
    expect(box.querySelector("annotation")?.textContent).toContain(
      "\\alpha_{\\mathrm{post}} = \\alpha + k",
    );
    expect(box).toHaveTextContent(/did not depend on the parameter|do not depend on the parameter/);
    expect(screen.getByTestId("premises")).toHaveTextContent(/prior is as stated/);
  });

  it("shows conditions as established or assumed", () => {
    render(<StatisticsPanel kind="statistics" result={momentsNormal} />);
    const list = screen.getByTestId("conditions");
    expect(within(list).getByText("established")).toBeInTheDocument();
    expect(within(list).getByText("assumed")).toBeInTheDocument();
  });

  it("keeps prose conditions out of math italics", () => {
    expect(conditionLatex("mu is a real number")).toBe("\\mu\\text{ is a real number}");
    expect(conditionLatex("v > 0")).toBe("v > 0");
  });

  it("shows a refusal with its reason and never a moment", () => {
    render(<StatisticsPanel kind="statistics" result={momentsCauchy} />);
    expect(screen.getByTestId("refusal")).toHaveTextContent(/does not converge absolutely/);
    expect(screen.getByTestId("refusal")).toHaveTextContent(/No result is shown/);
    expect(screen.queryByText("0")).toBeNull(); // the moment that must not appear
    expect(screen.queryByRole("group")).toBeNull();
  });

  it("flags a failed check", () => {
    const failed = {
      ...mleGaussian,
      checks: [
        {
          kind: "score",
          method: "the score",
          points_checked: 4,
          points_agreed: 3,
          passed: false,
          note: "",
        },
      ],
    };
    render(<StatisticsPanel kind="statistics" result={failed} />);
    expect(screen.getByTestId("checks")).toHaveTextContent("✗");
  });

  it("offers each expression of a statistical step to continue from", async () => {
    const onContinue = vi.fn();
    render(
      <StatisticsPanel
        kind="statistics"
        result={mleGaussian}
        continuations={[{ label: "estimate of mu" }, { label: "estimate of v" }]}
        onContinue={onContinue}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: "estimate of v" }));
    expect(onContinue).toHaveBeenCalledWith(1);
  });
});

describe("StatisticsPanel: no repeated text", () => {
  it("does not print the engine's check sentence a second time as a footer", () => {
    render(<StatisticsPanel kind="sums" result={sumsOk} />);
    // the sentence is part of the "How it was checked" list; the footer would repeat it verbatim
    expect(screen.getAllByText(/not proved for every size/)).toHaveLength(1);
  });
});
