import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { okMatrixIntegral } from "@/test/fixtures";
import type { MatrixRequest } from "@/lib/types";
import { MatrixWorkbench } from "./matrix-workbench";

const matrix = vi.fn<(c: unknown, r: MatrixRequest) => Promise<typeof okMatrixIntegral>>();
vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return { ...actual, api: { matrix: (c: unknown, r: MatrixRequest) => matrix(c, r) } };
});

const noop = () => undefined;
const mount = () => render(<MatrixWorkbench restore={null} onUnauthorized={noop} onSaved={noop} />);

beforeEach(() => {
  matrix.mockReset();
  matrix.mockResolvedValue(okMatrixIntegral);
});

describe("MatrixWorkbench", () => {
  it("sends the declarations exactly as typed and shows the verified result", async () => {
    mount();
    await userEvent.type(screen.getByLabelText("Expression"), "\\exp(A)\\exp(B)");
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    await waitFor(() => expect(matrix).toHaveBeenCalledOnce());
    const [common, request] = matrix.mock.calls[0]!;
    expect(common).toMatchObject({ input: "\\exp(A)\\exp(B)", format: "latex" });
    expect(request.matrices.map((m) => m.name)).toEqual(["A", "B"]);
    expect(request.commuting).toEqual([]); // nothing declared that the user did not state
    expect(await screen.findByRole("region", { name: "Matrix calculus" })).toBeInTheDocument();
  });

  it("sends plain text when the user chooses it", async () => {
    mount();
    await userEvent.selectOptions(screen.getByLabelText("Input syntax"), "Plain text");
    await userEvent.type(screen.getByLabelText("Expression"), "exp(A)*exp(B)");
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    await waitFor(() => expect(matrix).toHaveBeenCalledOnce());
    expect(matrix.mock.calls[0]![0]).toMatchObject({ format: "text" });
  });

  it("states what is missing instead of sending an incomplete problem", async () => {
    mount();
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/Enter the expression/);
    expect(matrix).not.toHaveBeenCalled();
  });

  it("shows every run with its own heading and keeps the declarations for the next one", async () => {
    mount();
    await userEvent.type(screen.getByLabelText("Expression"), "\\exp(A)");
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    await screen.findByRole("region", { name: "Matrix calculus" });
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    await waitFor(() => expect(matrix).toHaveBeenCalledTimes(2));
    const regions = await screen.findAllByRole("region", { name: "Matrix calculus" });
    expect(regions).toHaveLength(2);
    expect(new Set(regions.map((r) => r.getAttribute("aria-labelledby"))).size).toBe(2);
    expect(matrix.mock.calls[1]![1].matrices).toEqual(matrix.mock.calls[0]![1].matrices);
  });

  it("loads an example, including explicit entries", async () => {
    mount();
    await userEvent.selectOptions(
      screen.getByLabelText("Start from an example"),
      "exp(A) exp(B) for noncommuting A, B",
    );
    await userEvent.click(screen.getByRole("button", { name: "Compute" }));
    await waitFor(() => expect(matrix).toHaveBeenCalledOnce());
    const [, request] = matrix.mock.calls[0]!;
    expect(request.action).toBe("evaluate");
    expect(request.matrices[0]!.entries).toEqual([
      ["0", "1"],
      ["0", "0"],
    ]);
  });
});
