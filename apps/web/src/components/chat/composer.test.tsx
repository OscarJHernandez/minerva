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

describe("Composer: multiline editing", () => {
  function setup() {
    const onSend = vi.fn();
    render(<Composer {...props} onSend={onSend} />);
    return { onSend, area: screen.getByLabelText("Message") as HTMLTextAreaElement };
  }

  it("starts with room for three lines", () => {
    const { area } = setup();
    expect(area.rows).toBe(3);
    expect(area).toHaveClass("resize-y");
  });

  it("does not send on Enter, so a new line can be written", () => {
    const { onSend, area } = setup();
    fireEvent.change(area, { target: { value: "line one" } });
    const notPrevented = fireEvent.keyDown(area, { key: "Enter" });
    expect(notPrevented).toBe(true); // the textarea keeps its default: a new line
    fireEvent.keyDown(area, { key: "Enter", shiftKey: true });
    expect(onSend).not.toHaveBeenCalled();
  });

  it.each([
    ["Ctrl", { ctrlKey: true }],
    ["Cmd", { metaKey: true }],
  ])("sends on %s+Enter, keeping the line breaks, and clears", (_name, mods) => {
    const { onSend, area } = setup();
    fireEvent.change(area, { target: { value: "first\nsecond $x^2$" } });
    fireEvent.keyDown(area, { key: "Enter", ...mods });
    expect(onSend).toHaveBeenCalledWith("first\nsecond $x^2$");
    expect(area.value).toBe("");
  });

  it("does not send an empty message or while busy", () => {
    const onSend = vi.fn();
    const { rerender } = render(<Composer {...props} onSend={onSend} />);
    const area = screen.getByLabelText("Message");
    fireEvent.keyDown(area, { key: "Enter", ctrlKey: true });
    fireEvent.change(area, { target: { value: "hi" } });
    rerender(<Composer {...props} onSend={onSend} busy />);
    fireEvent.keyDown(area, { key: "Enter", ctrlKey: true });
    expect(onSend).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
  });

  it("keeps Send visible and enabled only with text", () => {
    const { area } = setup();
    const send = screen.getByRole("button", { name: "Send" });
    expect(send).toBeDisabled();
    fireEvent.change(area, { target: { value: "hello" } });
    expect(send).toBeEnabled();
  });

  it("grows with the content but never past 40% of the window", () => {
    const { area } = setup();
    Object.defineProperty(window, "innerHeight", { value: 800, configurable: true });
    Object.defineProperty(area, "scrollHeight", { value: 900, configurable: true });
    fireEvent.change(area, { target: { value: "a\nb\nc\nd\ne\nf" } });
    expect(area.style.height).toBe("320px");
    Object.defineProperty(area, "scrollHeight", { value: 150, configurable: true });
    fireEvent.change(area, { target: { value: "a\nb\nc\nd" } });
    expect(area.style.height).toBe("150px");
  });
});
