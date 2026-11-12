import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { LagrangianReport, MotionRequest } from "@/lib/chat-types";
import { LagrangianCard } from "./lagrangian-card";
import { MotionRunnerProvider } from "./motion-runner";

const verification = { numeric_points_checked: 6, numeric_points_agreed: 6, passed: true };

const lagrangian: LagrangianReport = {
  status: "ok",
  lagrangian_latex: "\\frac{m \\dot{x}^{2}}{2} - \\frac{k x^{2}}{2}",
  coordinates: ["x"],
  parameters: ["k", "m"],
  time_dependent: false,
  equations: [{ coordinate: "x", latex: "m \\ddot{x} + k x = 0", verification }],
  momenta: [],
  accelerations: [{ coordinate: "x", latex: "\\ddot{x} = - \\frac{k x}{m}" }],
  conserved: [],
  tested: [],
  features: [],
  notes: [],
  message: "1 equation of motion",
  duration_ms: 3,
};
const request = { lagrangian: "L", coordinates: ["x"], time: "t", assumptions: {} };

describe("LagrangianCard numerical form", () => {
  it("mentions the independent check of the equations", () => {
    render(<LagrangianCard report={lagrangian} methods={[]} request={request} />);
    expect(screen.getByText(/checked against numerical differentiation of L/)).toBeInTheDocument();
  });

  it("offers no form outside a conversation", () => {
    render(<LagrangianCard report={lagrangian} methods={[]} request={request} />);
    expect(screen.queryByRole("form", { name: "Numerical solution" })).not.toBeInTheDocument();
  });

  it("refuses to submit with a missing number, then sends the typed values", async () => {
    const run = vi.fn<(r: MotionRequest) => Promise<void>>().mockResolvedValue();
    render(
      <MotionRunnerProvider value={{ run }}>
        <LagrangianCard report={lagrangian} methods={[]} request={request} />
      </MotionRunnerProvider>,
    );
    await userEvent.click(screen.getByText(/Solve numerically/));
    const submit = screen.getByRole("button", { name: "Solve and verify" });
    await userEvent.click(submit);
    expect(screen.getByRole("alert")).toHaveTextContent("Give a value for k.");
    expect(run).not.toHaveBeenCalled();

    await userEvent.type(screen.getByLabelText("k"), "4");
    await userEvent.type(screen.getByLabelText("m"), "1");
    await userEvent.type(screen.getByLabelText("x(start)"), "1");
    await userEvent.type(screen.getByLabelText("dx/dt (start)"), "0");
    await userEvent.type(screen.getByLabelText("to"), "10");
    await userEvent.click(submit);
    expect(run).toHaveBeenCalledTimes(1);
    expect(run.mock.calls[0]?.[0]).toMatchObject({
      parameters: { k: 4, m: 1 },
      initial_positions: [1],
      initial_velocities: [0],
      t_start: 0,
      t_end: 10,
      sweep: null,
    });
  });

  it("sweeps a parameter instead of asking for its single value", async () => {
    const run = vi.fn<(r: MotionRequest) => Promise<void>>().mockResolvedValue();
    render(
      <MotionRunnerProvider value={{ run }}>
        <LagrangianCard report={lagrangian} methods={[]} request={request} />
      </MotionRunnerProvider>,
    );
    await userEvent.click(screen.getByText(/Solve numerically/));
    await userEvent.selectOptions(screen.getByLabelText(/Sweep one parameter/), "k");
    expect(screen.queryByLabelText("k")).not.toBeInTheDocument();
    await userEvent.type(screen.getByLabelText("k from"), "1");
    await userEvent.type(screen.getByLabelText("k to"), "9");
    await userEvent.type(screen.getByLabelText("m"), "1");
    await userEvent.type(screen.getByLabelText("x(start)"), "1");
    await userEvent.type(screen.getByLabelText("dx/dt (start)"), "0");
    await userEvent.type(screen.getByLabelText("to"), "10");
    await userEvent.click(screen.getByRole("button", { name: "Solve and verify" }));
    expect(run.mock.calls[0]?.[0].sweep).toEqual({ name: "k", values: [1, 5, 9] });
  });

  it("shows the server's message when the run fails", async () => {
    const run = vi.fn().mockRejectedValue(new Error("boom"));
    render(
      <MotionRunnerProvider value={{ run }}>
        <LagrangianCard report={lagrangian} methods={[]} request={request} />
      </MotionRunnerProvider>,
    );
    await userEvent.click(screen.getByText(/Solve numerically/));
    await userEvent.type(screen.getByLabelText("k"), "4");
    await userEvent.type(screen.getByLabelText("m"), "1");
    await userEvent.type(screen.getByLabelText("x(start)"), "1");
    await userEvent.type(screen.getByLabelText("dx/dt (start)"), "0");
    await userEvent.type(screen.getByLabelText("to"), "10");
    await userEvent.click(screen.getByRole("button", { name: "Solve and verify" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("could not be run");
  });
});
