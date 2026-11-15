import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { curl } from "@/test/vector-fixtures";
import type { VectorRequest, VectorResult } from "@/lib/types";
import { VectorWorkbench } from "./vector-workbench";

const vector = vi.fn<(r: VectorRequest, link?: unknown) => Promise<VectorResult>>();
vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return { ...actual, api: { vector: (r: VectorRequest, l?: unknown) => vector(r, l) } };
});

const noop = () => undefined;
const mount = () => render(<VectorWorkbench restore={null} onUnauthorized={noop} onSaved={noop} />);

beforeEach(() => {
  vector.mockReset();
  vector.mockResolvedValue(curl);
});

describe("VectorWorkbench", () => {
  it("sends the declarations as typed, saves the run and shows the verified result", async () => {
    mount();
    await userEvent.type(screen.getByLabelText("Expression"), "curl(E)");
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    await waitFor(() => expect(vector).toHaveBeenCalledOnce());
    const [request, link] = vector.mock.calls[0]!;
    expect(request.expression).toBe("curl(E)");
    expect(request.fields.map((f) => f.name)).toEqual(["E", "B"]);
    expect(request.fields.every((f) => f.smoothness === null)).toBe(true); // never declared for the user
    expect(link).toEqual({ save: true });
    expect(await screen.findByTestId("vector-panel")).toBeInTheDocument();
  });

  it("states what is missing instead of sending an incomplete problem", async () => {
    mount();
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/Enter the expression/);
    expect(vector).not.toHaveBeenCalled();
  });

  it("adopts premises only when the user ticks them and declares smoothness only when chosen", async () => {
    mount();
    await userEvent.click(screen.getByRole("radio", { name: "Electromagnetic derivation" }));
    expect(screen.getByTestId("needs")).toHaveTextContent(/curl\(E\) = -dt\(B\)/);
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    await waitFor(() => expect(vector).toHaveBeenCalledOnce());
    let request = vector.mock.calls[0]![0];
    expect(request.action).toBe("em_derive");
    expect(request.premise_ids).toEqual([]);
    expect(request.fields.every((f) => f.smoothness === null)).toBe(true);

    await userEvent.click(screen.getByRole("checkbox", { name: "Faraday's law" }));
    await userEvent.selectOptions(screen.getByLabelText("Smoothness"), "c2");
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    await waitFor(() => expect(vector).toHaveBeenCalledTimes(2));
    request = vector.mock.calls[1]![0];
    expect(request.premise_ids).toEqual(["em.maxwell.faraday"]);
    expect(request.fields.find((f) => f.name === "E")!.smoothness).toBe("c2");
  });

  it("offers a result as the next expression, but not one that holds only given premises", async () => {
    mount();
    await userEvent.type(screen.getByLabelText("Expression"), "curl(E)");
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    await screen.findByTestId("vector-panel");
    await userEvent.click(
      screen.getByRole("button", { name: /Use this result as the expression/ }),
    );
    expect(screen.getByLabelText("Expression")).toHaveValue(curl.result_latex);
  });

  it("hides the continue button for a conditional result", async () => {
    vector.mockResolvedValue({ ...curl, evidence: "conditional", given: ["curl(E) = -dt(B)"] });
    mount();
    await userEvent.type(screen.getByLabelText("Expression"), "curl(E)");
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    await screen.findByTestId("vector-panel");
    expect(screen.queryByRole("button", { name: /Use this result/ })).not.toBeInTheDocument();
  });
});
