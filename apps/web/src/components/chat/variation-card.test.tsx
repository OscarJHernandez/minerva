import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { FunctionalRequest, VariationRequest } from "@/lib/variation-types";
import { analyzed, analyzedRequest, notVerified } from "@/test/variation-fixtures";
import { MotionRunnerProvider } from "./motion-runner";
import { VariationCard } from "./variation-card";

/** What the model's tool call carries: the functional only, no interval, no conditions, no numbers. */
const modelRequest: FunctionalRequest = {
  integrand: analyzedRequest.integrand,
  variable: analyzedRequest.variable,
  functions: analyzedRequest.functions,
  given_functions: [],
  lower: null,
  upper: null,
  lower_term: null,
  upper_term: null,
  conditions: [],
  assumptions: {},
};

function mount(variation = vi.fn<(r: VariationRequest) => Promise<void>>().mockResolvedValue()) {
  render(
    <MotionRunnerProvider value={{ run: vi.fn(), variation }}>
      <VariationCard report={analyzed} request={modelRequest} />
    </MotionRunnerProvider>,
  );
  return variation;
}

describe("VariationCard", () => {
  it("renders the report and starts with an empty form: nothing is pre-filled for the user", () => {
    mount();
    expect(screen.getByRole("region", { name: /^Variational calculus ·/ })).toBeInTheDocument();
    const form = screen.getByRole("form", { name: "Run with your numbers" });
    expect(within(form).getByLabelText("Lower end")).toHaveValue("");
    expect(within(form).getByLabelText("Upper end")).toHaveValue("");
    expect(within(form).getByLabelText("Path for x")).toHaveValue("");
    expect(within(form).queryByRole("table")).not.toBeInTheDocument(); // no condition added
  });

  it("posts the model's functional plus exactly the numbers the user typed, with the chosen action", async () => {
    const variation = mount();
    const form = within(screen.getByRole("form", { name: "Run with your numbers" }));
    await userEvent.type(form.getByLabelText("Lower end"), "0");
    await userEvent.type(form.getByLabelText("Upper end"), "1");
    await userEvent.type(form.getByLabelText("Path for x"), "\\sin(t)/\\sin(1)");
    await userEvent.click(form.getByRole("button", { name: "Run" }));

    expect(variation).toHaveBeenCalledOnce();
    expect(variation.mock.calls[0]![0]).toEqual({
      ...modelRequest,
      format: "latex",
      action: "evaluate",
      lower: "0",
      upper: "1",
      values: {},
      definitions: {},
      paths: [{ analytic: { x: "\\sin(t)/\\sin(1)" }, sampled: {} }],
      sweep: null,
      perturbations: [],
      automatic_perturbations: true,
      with_second: false,
    });
  });

  it("sends the endpoint conditions the user added, with their values", async () => {
    const variation = mount();
    const form = within(screen.getByRole("form", { name: "Run with your numbers" }));
    await userEvent.type(form.getByLabelText("Lower end"), "0");
    await userEvent.type(form.getByLabelText("Upper end"), "1");
    await userEvent.selectOptions(form.getByLabelText("Action"), "Find a stationary path");
    await userEvent.click(form.getByRole("button", { name: "Add a condition" }));
    await userEvent.click(form.getByRole("button", { name: "Add a condition" }));
    await userEvent.selectOptions(form.getByLabelText("Condition 2 end"), "upper");
    await userEvent.type(form.getByLabelText("Condition 1 value"), "0");
    await userEvent.type(form.getByLabelText("Condition 2 value"), "1");
    await userEvent.click(form.getByLabelText(/Also classify/));
    await userEvent.click(form.getByRole("button", { name: "Run" }));
    const sent = variation.mock.calls[0]![0];
    expect(sent.action).toBe("solve");
    expect(sent.with_second).toBe(true);
    expect(sent.paths).toEqual([]); // a starting guess is optional and was not typed
    expect(sent.conditions).toEqual([
      { function: "x", order: 0, end: "lower", kind: "fixed", value: "0" },
      { function: "x", order: 0, end: "upper", kind: "fixed", value: "1" },
    ]);
  });

  it("does not post an incomplete request", async () => {
    const variation = mount();
    const form = within(screen.getByRole("form", { name: "Run with your numbers" }));
    await userEvent.click(form.getByRole("button", { name: "Run" }));
    expect(form.getByRole("alert")).toHaveTextContent(/both ends of the interval/);
    expect(variation).not.toHaveBeenCalled();
  });

  it("shows the server's error and keeps the form", async () => {
    const variation = vi
      .fn<(r: VariationRequest) => Promise<void>>()
      .mockRejectedValue(new Error("x"));
    mount(variation);
    const form = within(screen.getByRole("form", { name: "Run with your numbers" }));
    await userEvent.type(form.getByLabelText("Lower end"), "0");
    await userEvent.type(form.getByLabelText("Upper end"), "1");
    await userEvent.type(form.getByLabelText("Path for x"), "t");
    await userEvent.click(form.getByRole("button", { name: "Run" }));
    expect(await form.findByRole("alert")).toHaveTextContent(/could not be run/);
    expect(form.getByLabelText("Path for x")).toHaveValue("t");
  });

  it("offers no form outside a conversation or for a refused result", () => {
    const { unmount } = render(<VariationCard report={analyzed} request={modelRequest} />);
    expect(screen.queryByRole("form", { name: "Run with your numbers" })).not.toBeInTheDocument();
    unmount();
    render(
      <MotionRunnerProvider value={{ run: vi.fn(), variation: vi.fn() }}>
        <VariationCard report={notVerified} request={modelRequest} />
      </MotionRunnerProvider>,
    );
    expect(screen.queryByRole("form", { name: "Run with your numbers" })).not.toBeInTheDocument();
  });
});
