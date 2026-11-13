import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { okDistributionIntegral, step } from "@/test/fixtures";
import { StepCard } from "./step-card";

const asked = step(
  "s1",
  "distribution",
  "x^2*delta(x^2 - 4)",
  {
    ...okDistributionIntegral,
    status: "conditions_not_satisfied",
    reason: "notation_unconfirmed",
    result_latex: null,
    result_text: null,
    equality: null,
    steps: [],
    verification: null,
    questions: [{ name: "delta", candidate: "dirac_delta" }],
    message: "Is δ(…) a Dirac delta or a step function?",
    request: { action: "integrate", variable: "x", lower: "-oo", upper: "oo", endpoint: null },
  },
  { status: "conditions_not_satisfied" },
);

const props = {
  number: 1,
  parentNumber: null,
  current: false,
  onContinue: vi.fn(),
  onDelete: vi.fn(),
};

describe("StepCard", () => {
  it("shows a waiting delta question open, and sends the answer with the step", async () => {
    const onAnswerNotation = vi.fn().mockResolvedValue(undefined);
    render(<StepCard item={asked} onAnswerNotation={onAnswerNotation} {...props} />);
    expect(screen.getByText("Answer to continue")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /delta\(…\) is a Dirac delta/ }));
    expect(onAnswerNotation).toHaveBeenCalledExactlyOnceWith(asked, { delta: "dirac_delta" });
  });

  it("keeps an ordinary delta step folded", () => {
    const done = step("s2", "distribution", "DiracDelta(x)", { ...okDistributionIntegral });
    render(<StepCard item={done} {...props} />);
    expect(screen.getByText("Full verification")).toBeInTheDocument();
    expect(screen.queryByText("Answer to continue")).not.toBeInTheDocument();
  });
});
