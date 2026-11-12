import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Composer } from "./composer";

vi.mock("./image-formulas-panel", () => ({
  ImageFormulasPanel: ({ file }: { file: File }) => <div role="dialog">reading {file.name}</div>,
}));
vi.mock("./formula-popover", () => ({ FormulaPopover: () => null }));

const props = { disabled: false, busy: false, onSend: vi.fn(), onStop: vi.fn() };

describe("Composer: add image", () => {
  it("has no image button when the server cannot read images", () => {
    render(<Composer {...props} />);
    expect(screen.queryByRole("button", { name: "Add image" })).not.toBeInTheDocument();
  });

  it("shows the button when images are enabled", () => {
    render(<Composer {...props} imagesEnabled />);
    expect(screen.getByRole("button", { name: "Add image" })).toBeInTheDocument();
  });

  it("opens the confirmation panel for a chosen picture", () => {
    render(<Composer {...props} imagesEnabled />);
    const input = screen.getByLabelText("Choose an image of a formula");
    fireEvent.change(input, {
      target: { files: [new File(["x"], "eq.png", { type: "image/png" })] },
    });
    expect(screen.getByRole("dialog")).toHaveTextContent("reading eq.png");
  });

  it("refuses a file that is not a png, jpeg or webp", () => {
    render(<Composer {...props} imagesEnabled />);
    fireEvent.change(screen.getByLabelText("Choose an image of a formula"), {
      target: { files: [new File(["<svg/>"], "eq.svg", { type: "image/svg+xml" })] },
    });
    expect(screen.getByRole("alert")).toHaveTextContent("PNG, JPEG or WebP");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("opens the panel for a pasted picture, but lets plain text paste", () => {
    render(<Composer {...props} imagesEnabled />);
    const area = screen.getByLabelText("Message");
    const image = new File(["x"], "shot.png", { type: "image/png" });
    fireEvent.paste(area, { clipboardData: { files: [image], getData: () => "" } });
    expect(screen.getByRole("dialog")).toHaveTextContent("reading shot.png");
  });

  it("ignores pasted pictures when images are off", () => {
    render(<Composer {...props} />);
    const image = new File(["x"], "shot.png", { type: "image/png" });
    fireEvent.paste(screen.getByLabelText("Message"), {
      clipboardData: { files: [image], getData: () => "" },
    });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
