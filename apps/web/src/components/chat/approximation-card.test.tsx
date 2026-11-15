import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ApproximationToolArgs } from "@/lib/approximation-types";
import { needsInterval, pade } from "@/test/approximation-fixtures";
import { ApproximationCard } from "./approximation-card";
import { MotionRunnerProvider } from "./motion-runner";

/** What the model's tool call carries: only what the user's words contained. */
const modelRequest: ApproximationToolArgs = {
  latex: "\\exp(x)",
  method: "pade",
  variable: "x",
  point: "0",
  degree: 1,
  denominator_degree: 1,
};

function mount(
  report = pade,
  request = modelRequest,
  approximation = vi.fn<(r: ApproximationToolArgs) => Promise<void>>().mockResolvedValue(),
) {
  render(
    <MotionRunnerProvider value={{ run: vi.fn(), approximation }}>
      <ApproximationCard report={report} request={request} />
    </MotionRunnerProvider>,
  );
  return approximation;
}

describe("ApproximationCard", () => {
  it("starts the form with exactly what the user's message contained", () => {
    mount();
    const form = within(screen.getByRole("form", { name: "Run with your numbers", hidden: true }));
    expect(form.getByLabelText("Point", { selector: "input" })).toHaveValue("0");
    expect(form.getByLabelText(/^Degree/)).toHaveValue("1");
    expect(form.getByLabelText("Denominator degree")).toHaveValue("1");
    expect(form.getByLabelText("Lower end")).toHaveValue("");
    expect(form.getByLabelText("Upper end")).toHaveValue("");
  });

  it("posts the model's expression plus the numbers the user typed", async () => {
    const approximation = mount();
    const form = within(screen.getByRole("form", { name: "Run with your numbers", hidden: true }));
    await userEvent.type(form.getByLabelText("Lower end"), "-1");
    await userEvent.type(form.getByLabelText("Upper end"), "3");
    await userEvent.click(form.getByRole("button", { name: "Approximate" }));
    expect(approximation).toHaveBeenCalledOnce();
    expect(approximation.mock.calls[0]![0]).toEqual({
      ...modelRequest,
      lower: "-1",
      upper: "3",
    });
  });

  it("refuses an incomplete interval and a tolerance without a kind, with a reason", async () => {
    const approximation = mount();
    const form = within(screen.getByRole("form", { name: "Run with your numbers", hidden: true }));
    await userEvent.type(form.getByLabelText("Lower end"), "0");
    await userEvent.click(form.getByRole("button", { name: "Approximate" }));
    expect(screen.getByRole("alert", { hidden: true })).toHaveTextContent(/both ends/);
    expect(approximation).not.toHaveBeenCalled();
  });

  it("opens the form when the engine asked a question, and shows the question", () => {
    mount(needsInterval, { latex: "\\sin x", method: "chebyshev", variable: "x" });
    expect(
      screen.getByText("On which interval should the approximation hold?"),
    ).toBeInTheDocument();
    expect(screen.getByText(/Run with your numbers/).closest("details")).toHaveAttribute("open");
  });

  it("offers no form outside a conversation", () => {
    render(<ApproximationCard report={pade} request={modelRequest} />);
    expect(screen.queryByRole("form", { name: "Run with your numbers" })).not.toBeInTheDocument();
    expect(screen.getByRole("region", { name: /^Approximation/ })).toBeInTheDocument();
  });

  it("shows the server's message when running fails", async () => {
    const { ApiError } = await import("@/lib/api");
    const approximation = vi
      .fn<(r: ApproximationToolArgs) => Promise<void>>()
      .mockRejectedValue(new ApiError(429, "Too many requests."));
    mount(pade, modelRequest, approximation);
    const form = within(screen.getByRole("form", { name: "Run with your numbers", hidden: true }));
    await userEvent.click(form.getByRole("button", { name: "Approximate" }));
    expect(await screen.findByText("Too many requests.")).toBeInTheDocument();
  });
});
