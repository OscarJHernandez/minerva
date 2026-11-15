import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { StatMechRequest } from "@/lib/statmech-types";
import { needsCell, twoLevel, twoLevelRequest } from "@/test/statmech-fixtures";
import { StatMechWorkbench } from "./statmech-workbench";

const statmech = vi.fn<(r: StatMechRequest) => Promise<typeof twoLevel>>();
vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return { ...actual, api: { statmech: (r: StatMechRequest) => statmech(r) } };
});

const noop = () => undefined;

function mount(restore: { request: StatMechRequest; nonce: number } | null = null) {
  return render(<StatMechWorkbench restore={restore} onUnauthorized={noop} onSaved={noop} />);
}

beforeEach(() => {
  statmech.mockReset();
  statmech.mockResolvedValue(twoLevel);
});

describe("StatMechWorkbench", () => {
  it("derives the chosen quantities for the typed spectrum and shows the checked result", async () => {
    mount();
    await userEvent.click(screen.getByRole("button", { name: "Derive quantities" }));
    await waitFor(() => expect(statmech).toHaveBeenCalledOnce());
    const sent = statmech.mock.calls[0]![0];
    expect(sent.action).toBe("derive");
    expect(sent.system).toMatchObject({ kind: "levels" });
    expect(sent.quantities).toEqual(["Z", "F", "U", "S"]);
    expect(await screen.findByTestId("quantity-U")).toBeInTheDocument();
  });

  it("starts from an example and shows what it contains", async () => {
    mount();
    await userEvent.selectOptions(screen.getByLabelText("Start from an example"), "qho");
    expect(screen.getByDisplayValue("hbar*omega*(n + 1/2)")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Partition function" }));
    await waitFor(() => expect(statmech).toHaveBeenCalledOnce());
    expect(statmech.mock.calls[0]![0].system).toMatchObject({ kind: "level_family" });
  });

  it("does not send an incomplete problem", async () => {
    mount();
    await userEvent.click(screen.getByRole("button", { name: "Remove level 1" }));
    await userEvent.clear(screen.getByLabelText("Energy of level 1"));
    await userEvent.click(screen.getByRole("button", { name: "Partition function" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Give at least one energy level");
    expect(statmech).not.toHaveBeenCalled();
  });

  it("asks the user for a missing convention, keeps the answer, and runs again", async () => {
    statmech.mockResolvedValueOnce(needsCell).mockResolvedValueOnce(twoLevel);
    mount();
    await userEvent.selectOptions(
      screen.getByLabelText("Start from an example"),
      "classical-oscillator",
    );
    await userEvent.clear(screen.getByLabelText(/Phase-space cell/));
    await userEvent.click(screen.getByRole("button", { name: "Partition function" }));
    expect(await screen.findByText("This is your choice")).toBeInTheDocument();
    const cells = screen.getAllByLabelText(/Phase-space cell/);
    await userEvent.type(cells[cells.length - 1]!, "h");
    await userEvent.click(screen.getByRole("button", { name: "Continue with this choice" }));
    await waitFor(() => expect(statmech).toHaveBeenCalledTimes(2));
    const second = statmech.mock.calls[1]![0];
    expect(second.system).toMatchObject({ kind: "classical", phase_space_cell: "h" });
  });

  it("loads a saved problem back into the form", async () => {
    mount({ request: twoLevelRequest, nonce: 1 });
    expect(await screen.findByDisplayValue("epsilon")).toBeInTheDocument();
  });

  it("reports a failed request and keeps the form", async () => {
    statmech.mockRejectedValue(new Error("down"));
    mount();
    await userEvent.click(screen.getByRole("button", { name: "Partition function" }));
    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(screen.getByDisplayValue("epsilon")).toBeInTheDocument();
  });
});
