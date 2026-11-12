import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api";
import type { ExtractedFormula } from "@/lib/chat-types";
import { ImageFormulasPanel } from "./image-formulas-panel";

const readFormulaImage = vi.fn();
const prepareImage = vi.fn();

vi.mock("@/lib/chat-api", () => ({
  readFormulaImage: (...args: unknown[]) => readFormulaImage(...args),
}));
vi.mock("@/lib/image-prep", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/image-prep")>()),
  prepareImage: (...args: unknown[]) => prepareImage(...args),
}));
// MathLive does not run in jsdom: a plain input stands in for the editor.
vi.mock("./math-field", () => ({
  MathField: ({
    initial,
    label,
    onChange,
  }: {
    initial?: string;
    label: string;
    onChange: (v: string) => void;
  }) => (
    <input aria-label={label} defaultValue={initial} onChange={(e) => onChange(e.target.value)} />
  ),
}));

const GOOD: ExtractedFormula = {
  latex: "x^2-1",
  status: "ok",
  read_as: "x^{2} - 1",
  message: "",
};
const BAD: ExtractedFormula = {
  latex: "\\sin 2x",
  status: "parse_error",
  read_as: null,
  message: "ambiguous: write \\sin(2x)",
};

function setup(formulas: ExtractedFormula[] | Error) {
  prepareImage.mockResolvedValue({ base64: "AAAA", previewUrl: "blob:preview" });
  if (formulas instanceof Error) readFormulaImage.mockRejectedValue(formulas);
  else readFormulaImage.mockResolvedValue(formulas);
  const onInsert = vi.fn();
  const onClose = vi.fn();
  const file = new File(["x"], "f.png", { type: "image/png" });
  render(<ImageFormulasPanel file={file} onInsert={onInsert} onClose={onClose} />);
  return { onInsert, onClose };
}

beforeEach(() => {
  vi.clearAllMocks();
  URL.revokeObjectURL = vi.fn();
});

describe("ImageFormulasPanel", () => {
  it("shows the formulas, how Minerva read them, and warns about the unreadable one", async () => {
    setup([GOOD, BAD]);
    expect(screen.getByText("Reading the image…")).toBeInTheDocument();
    expect(await screen.findByText("Is this what the image says?")).toBeInTheDocument();
    expect(screen.getByText(/Minerva reads this as/)).toBeInTheDocument();
    expect(screen.getByText(/Minerva cannot read this formula: ambiguous/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "The image you added" })).toHaveAttribute(
      "src",
      "blob:preview",
    );
  });

  it("inserts only the selected formulas; unreadable ones start unselected", async () => {
    const { onInsert } = setup([GOOD, BAD]);
    await screen.findByText("Is this what the image says?");
    expect(screen.getByLabelText("Use formula 1")).toBeChecked();
    expect(screen.getByLabelText("Use formula 2")).not.toBeChecked();
    await userEvent.click(screen.getByRole("button", { name: "Use selected (1)" }));
    expect(onInsert).toHaveBeenCalledWith(["x^2-1"]);
  });

  it("lets the user choose the unreadable formula anyway", async () => {
    const { onInsert } = setup([GOOD, BAD]);
    await screen.findByText("Is this what the image says?");
    await userEvent.click(screen.getByLabelText("Use formula 2"));
    await userEvent.click(screen.getByRole("button", { name: "Use selected (2)" }));
    expect(onInsert).toHaveBeenCalledWith(["x^2-1", "\\sin 2x"]);
  });

  it("inserts nothing on Discard", async () => {
    const { onInsert, onClose } = setup([GOOD]);
    await screen.findByText("Is this what the image says?");
    await userEvent.click(screen.getByRole("button", { name: "Discard" }));
    expect(onClose).toHaveBeenCalled();
    expect(onInsert).not.toHaveBeenCalled();
  });

  it("an edited formula is inserted as edited and no longer claims a parser reading", async () => {
    const { onInsert } = setup([GOOD]);
    await screen.findByText("Is this what the image says?");
    await userEvent.click(screen.getByRole("button", { name: "Edit" }));
    const field = screen.getByLabelText("Edit formula 1");
    await userEvent.clear(field);
    await userEvent.type(field, "x^3");
    expect(screen.getByText(/Edited by you/)).toBeInTheDocument();
    expect(screen.queryByText(/Minerva reads this as/)).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Use selected (1)" }));
    expect(onInsert).toHaveBeenCalledWith(["x^3"]);
  });

  it("says so when no formula is found", async () => {
    setup([]);
    expect(await screen.findByText(/No formula was found/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Use selected/ })).not.toBeInTheDocument();
  });

  it("shows the server's message when reading fails", async () => {
    setup(new ApiError(502, "The image model host is not reachable."));
    expect(await screen.findByRole("alert")).toHaveTextContent("not reachable");
  });

  it("asks to sign in again on 401", async () => {
    setup(new ApiError(401, "Not authenticated"));
    expect(await screen.findByRole("alert")).toHaveTextContent("Sign in again");
  });

  it("frees the preview and aborts the request when it closes", async () => {
    prepareImage.mockResolvedValue({ base64: "AAAA", previewUrl: "blob:preview" });
    let signal: AbortSignal | undefined;
    readFormulaImage.mockImplementation((_b: string, s: AbortSignal) => {
      signal = s;
      return new Promise(() => {});
    });
    const file = new File(["x"], "f.png", { type: "image/png" });
    const { unmount } = render(
      <ImageFormulasPanel file={file} onInsert={vi.fn()} onClose={vi.fn()} />,
    );
    await waitFor(() => expect(signal).toBeDefined());
    unmount();
    expect(signal?.aborted).toBe(true);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:preview");
  });
});
