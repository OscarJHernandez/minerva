import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { GeometryRequest } from "@/lib/types";
import { sphere, sphereRequest } from "@/test/geometry-fixtures";
import { GeometryWorkbench } from "./geometry-workbench";

const geometry = vi.fn<(r: GeometryRequest) => Promise<typeof sphere>>();
vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return { ...actual, api: { geometry: (r: GeometryRequest) => geometry(r) } };
});

const noop = () => undefined;

function mount(restore: { request: GeometryRequest; nonce: number } | null = null) {
  return render(<GeometryWorkbench restore={restore} onUnauthorized={noop} onSaved={noop} />);
}

async function chooseExample(label: string) {
  await userEvent.selectOptions(screen.getByLabelText("Start from an example"), label);
}

beforeEach(() => {
  geometry.mockReset();
  geometry.mockResolvedValue(sphere);
});

describe("GeometryWorkbench", () => {
  it("computes the chosen quantities for the typed metric and shows the verified result", async () => {
    mount();
    await chooseExample("Sphere of radius a");
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    await waitFor(() => expect(geometry).toHaveBeenCalledOnce());
    const sent = geometry.mock.calls[0]![0];
    expect(sent.coordinates).toEqual(["\\theta", "\\phi"]);
    expect(sent.metric[1]![1]).toBe("a^2 \\sin^2\\theta");
    expect(sent.metric[0]![1]).toBe("0"); // an empty off-diagonal entry means 0
    expect(sent.assumptions).toEqual({ a: ["positive"] });
    expect(sent.quantities).toEqual(["christoffel"]);
    expect(await screen.findByText("Ricci scalar R")).toBeInTheDocument();
  });

  it("keeps the metric for a follow-up quantity instead of asking for it again", async () => {
    mount();
    await chooseExample("Sphere of radius a");
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    await screen.findByText("Ricci scalar R");
    await userEvent.click(screen.getByRole("button", { name: "Einstein tensor" }));
    await waitFor(() => expect(geometry).toHaveBeenCalledTimes(2));
    const second = geometry.mock.calls[1]![0];
    expect(second.quantities).toEqual(["einstein"]);
    expect(second.metric).toEqual(geometry.mock.calls[0]![0].metric);
    expect(screen.getAllByRole("region", { name: "Differential geometry" })).toHaveLength(2);
  });

  it("explains what is missing instead of sending an incomplete metric", async () => {
    mount();
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/diagonal entry is empty/);
    expect(geometry).not.toHaveBeenCalled();
  });

  it("asks for the tensor field only when a tensor quantity is chosen", async () => {
    mount();
    await chooseExample("Plane, polar coordinates");
    expect(screen.queryByText("Tensor field")).not.toBeInTheDocument();
    await userEvent.click(screen.getByLabelText("Covariant derivative"));
    expect(screen.getByText("Tensor field")).toBeInTheDocument();
    expect(screen.getByLabelText("Tensor component 1")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/every component of the tensor field/);
    expect(geometry).not.toHaveBeenCalled();
    await userEvent.type(screen.getByLabelText("Tensor component 1"), "1");
    await userEvent.type(screen.getByLabelText("Tensor component 2"), "0");
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    await waitFor(() => expect(geometry).toHaveBeenCalledOnce());
    expect(geometry.mock.calls[0]![0].tensor).toEqual({ variance: ["up"], components: ["1", "0"] });
  });

  it("loads a saved problem into the form", async () => {
    mount({ request: sphereRequest, nonce: 1 });
    await waitFor(() =>
      expect(screen.getByLabelText("Metric component 2,2")).toHaveValue("a^2 \\sin^2\\theta"),
    );
    expect(screen.getByLabelText("Coordinate 1")).toHaveValue("theta");
  });

  it("shows a server error without a result", async () => {
    const { ApiError } = await import("@/lib/api");
    geometry.mockRejectedValue(new ApiError(429, "Too many requests; slow down."));
    mount();
    await chooseExample("Plane, polar coordinates");
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Too many requests");
    expect(screen.queryByRole("region", { name: "Differential geometry" })).not.toBeInTheDocument();
  });
});
