import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ConversationSummary } from "@/lib/chat-types";
import { ConversationList } from "./conversation-list";

const items: ConversationSummary[] = [
  { id: "a", title: "Equations of motion", updated_at: "2026-11-18T10:00:00Z" },
  { id: "b", title: "Integral of $x^2$", updated_at: "2026-11-18T09:00:00Z" },
];

// jsdom cannot compute the accessible name of the MathML KaTeX renders, so these tests find the buttons by
// label or text rather than by role and name.
function mount(props: Partial<React.ComponentProps<typeof ConversationList>> = {}) {
  const handlers = { onOpen: vi.fn(), onNew: vi.fn(), onDelete: vi.fn(), onRename: vi.fn() };
  render(<ConversationList items={items} activeId="a" {...handlers} {...props} />);
  return handlers;
}

function titleButton(index: number): HTMLElement {
  const button = document.querySelectorAll("li")[index]?.querySelector("button");
  if (!button) throw new Error(`no row ${index}`);
  return button;
}

const rename = (title: string) => screen.getByLabelText(`Rename conversation ${title}`);
const editor = () => screen.getByLabelText(/Title of conversation/) as HTMLInputElement;

describe("ConversationList rename", () => {
  it("shows no text 'Rename' button: the rename action is an icon with a label", () => {
    mount();
    expect(screen.queryByText("Rename")).toBeNull();
    expect(rename("Equations of motion")).toBeTruthy();
  });

  it("keeps the actions of the open conversation visible and hides the others until wanted", () => {
    mount();
    const open = screen.getByLabelText("Delete conversation Equations of motion");
    const other = screen.getByLabelText("Delete conversation Integral of $x^2$");
    expect(open.className).toContain("opacity-100");
    expect(other.className).toContain("opacity-0");
    expect(other.className).toContain("group-hover:opacity-100");
    expect(other.className).toContain("group-focus-within:opacity-100");
    expect(rename("Integral of $x^2$").className).toContain("group-focus-within:opacity-100");
  });

  it("renames on double-click: Enter saves the edited title", async () => {
    const user = userEvent.setup();
    const { onRename } = mount();
    await user.dblClick(titleButton(0));
    expect(editor().value).toBe("Equations of motion");
    await user.clear(editor());
    await user.type(editor(), "Pendulum{Enter}");
    expect(onRename).toHaveBeenCalledTimes(1);
    expect(onRename).toHaveBeenCalledWith("a", "Pendulum");
    expect(screen.queryByLabelText(/Title of conversation/)).toBeNull();
  });

  it("renames with F2 on the focused row", async () => {
    const user = userEvent.setup();
    const { onRename } = mount();
    titleButton(0).focus();
    await user.keyboard("{F2}");
    await user.keyboard("{Control>}a{/Control}Kepler{Enter}");
    expect(onRename).toHaveBeenCalledWith("a", "Kepler");
  });

  it("renames from the pencil icon, editing the raw source of a title with math", async () => {
    const user = userEvent.setup();
    const { onRename } = mount();
    await user.click(rename("Integral of $x^2$"));
    expect(editor().value).toBe("Integral of $x^2$");
    await user.type(editor(), "{Backspace}{Backspace}3$");
    await user.keyboard("{Enter}");
    expect(onRename).toHaveBeenCalledWith("b", "Integral of $x^3$");
  });

  it("Escape cancels without saving, and the blur that follows does not save either", async () => {
    const user = userEvent.setup();
    const { onRename } = mount();
    await user.dblClick(titleButton(0));
    await user.type(editor(), " changed{Escape}");
    expect(screen.queryByLabelText(/Title of conversation/)).toBeNull();
    await user.click(document.body);
    expect(onRename).not.toHaveBeenCalled();
  });

  it("saves on blur, once", async () => {
    const user = userEvent.setup();
    const { onRename } = mount();
    await user.dblClick(titleButton(0));
    await user.type(editor(), " II");
    await user.click(document.body);
    expect(onRename).toHaveBeenCalledTimes(1);
    expect(onRename).toHaveBeenCalledWith("a", "Equations of motion II");
  });

  it("does not call rename for an unchanged or an empty title", async () => {
    const user = userEvent.setup();
    const { onRename } = mount();
    await user.dblClick(titleButton(0));
    await user.keyboard("{Enter}");
    await user.dblClick(titleButton(0));
    await user.clear(editor());
    await user.keyboard("   {Enter}");
    expect(onRename).not.toHaveBeenCalled();
  });

  it("offers no rename at all when the caller gave no handler", async () => {
    const user = userEvent.setup();
    mount({ onRename: undefined });
    expect(screen.queryByLabelText(/Rename conversation/)).toBeNull();
    await user.dblClick(titleButton(0));
    expect(screen.queryByLabelText(/Title of conversation/)).toBeNull();
  });

  it("a single click still opens the conversation, and delete still deletes", async () => {
    const user = userEvent.setup();
    const { onOpen, onDelete } = mount();
    await user.click(titleButton(1));
    expect(onOpen).toHaveBeenCalledWith("b");
    await user.click(screen.getByLabelText("Delete conversation Integral of $x^2$"));
    expect(onDelete).toHaveBeenCalledWith("b");
  });
});
