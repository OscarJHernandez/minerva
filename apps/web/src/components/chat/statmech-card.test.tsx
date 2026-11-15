import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { StatAnswers, StatToolArgs } from "@/lib/statmech-types";
import { classicalArgs, needsCell } from "@/test/statmech-fixtures";
import { MotionRunnerProvider } from "./motion-runner";
import { StatMechCard } from "./statmech-card";

describe("StatMechCard", () => {
  it("posts the model's call together with the user's answer, nothing else added", async () => {
    const statmech = vi.fn<(r: StatToolArgs & StatAnswers) => Promise<void>>().mockResolvedValue();
    render(
      <MotionRunnerProvider value={{ run: vi.fn(), statmech }}>
        <StatMechCard result={needsCell} args={classicalArgs} answers={{}} />
      </MotionRunnerProvider>,
    );
    await userEvent.type(screen.getByLabelText("Phase-space cell"), "h");
    await userEvent.click(screen.getByRole("button", { name: "Continue with this choice" }));
    expect(statmech).toHaveBeenCalledOnce();
    expect(statmech.mock.calls[0]![0]).toEqual({ ...classicalArgs, phase_space_cell: "h" });
  });

  it("keeps earlier answers when it asks again", async () => {
    const statmech = vi.fn<(r: StatToolArgs & StatAnswers) => Promise<void>>().mockResolvedValue();
    const args = { ...classicalArgs, copies_count: "N" };
    render(
      <MotionRunnerProvider value={{ run: vi.fn(), statmech }}>
        <StatMechCard result={needsCell} args={args} answers={{ phase_space_cell: "h" }} />
      </MotionRunnerProvider>,
    );
    await userEvent.click(screen.getByLabelText(/^indistinguishable/));
    await userEvent.click(screen.getByRole("button", { name: "Continue with this choice" }));
    expect(statmech.mock.calls[0]![0]).toMatchObject({ phase_space_cell: "h", identical: true });
  });

  it("shows an error and keeps the form when the request fails", async () => {
    const statmech = vi.fn().mockRejectedValue(new Error("x"));
    render(
      <MotionRunnerProvider value={{ run: vi.fn(), statmech }}>
        <StatMechCard result={needsCell} args={classicalArgs} answers={{}} />
      </MotionRunnerProvider>,
    );
    await userEvent.type(screen.getByLabelText("Phase-space cell"), "h");
    await userEvent.click(screen.getByRole("button", { name: "Continue with this choice" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("x");
  });

  it("offers no way to answer outside a conversation", () => {
    render(<StatMechCard result={needsCell} args={classicalArgs} answers={{}} />);
    expect(screen.queryByRole("button", { name: "Continue with this choice" })).toBeNull();
    expect(screen.getByText(/Open a conversation or the workbench/)).toBeInTheDocument();
  });
});
