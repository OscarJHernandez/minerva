import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { GeometryRequest } from "@/lib/types";
import { sphere, sphereRequest } from "@/test/geometry-fixtures";
import { GeometryCard } from "./geometry-card";
import { MotionRunnerProvider } from "./motion-runner";

describe("GeometryCard", () => {
  it("posts the card's own request back with the chosen quantity, then hides that button", async () => {
    const geometry = vi.fn<(r: GeometryRequest) => Promise<void>>().mockResolvedValue();
    render(
      <MotionRunnerProvider value={{ run: vi.fn(), geometry }}>
        <GeometryCard result={sphere} request={sphereRequest} />
      </MotionRunnerProvider>,
    );
    await userEvent.click(screen.getByRole("button", { name: "Einstein tensor" }));
    expect(geometry).toHaveBeenCalledOnce();
    const sent = geometry.mock.calls[0]![0];
    expect(sent.quantities).toEqual(["einstein"]);
    expect(sent.metric).toEqual(sphereRequest.metric); // the same metric, not retyped
    expect(sent.assumptions).toEqual({ a: ["positive"] });
    expect(screen.queryByRole("button", { name: "Einstein tensor" })).not.toBeInTheDocument();
  });

  it("shows an error and keeps the button when the request fails", async () => {
    const geometry = vi
      .fn<(r: GeometryRequest) => Promise<void>>()
      .mockRejectedValue(new Error("x"));
    render(
      <MotionRunnerProvider value={{ run: vi.fn(), geometry }}>
        <GeometryCard result={sphere} request={sphereRequest} />
      </MotionRunnerProvider>,
    );
    await userEvent.click(screen.getByRole("button", { name: "Einstein tensor" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/could not be computed/);
    expect(screen.getByRole("button", { name: "Einstein tensor" })).toBeEnabled();
  });

  it("offers no buttons without a conversation", () => {
    render(<GeometryCard result={sphere} request={sphereRequest} />);
    expect(screen.queryByText(/Also compute/)).not.toBeInTheDocument();
  });
});
