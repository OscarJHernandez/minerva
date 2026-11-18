import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ALL_EXAMPLES, type HomepageExample } from "@/lib/homepage-examples";
import { ExampleCards } from "./example-cards";

/** Every card, as if approved: the component's behaviour does not depend on the release state. */
const ALL: HomepageExample[] = ALL_EXAMPLES.map((e) => ({ ...e, approved: true }));

describe("ExampleCards", () => {
  it("draws one card per example in a responsive two-column grid", () => {
    render(<ExampleCards onPick={() => {}} disabled={false} examples={ALL} />);
    const grid = screen.getByTestId("example-cards");
    expect(grid.children).toHaveLength(6);
    expect(grid.className).toMatch(/sm:grid-cols-2/);
    expect(grid.className).not.toMatch(/grid-cols-3/);
    for (const ex of ALL) {
      expect(screen.getByRole("button", { name: ex.title })).toHaveAttribute(
        "data-example-id",
        ex.id,
      );
    }
  });

  it("renders every displayed expression with KaTeX and never as marked source", () => {
    const { container } = render(
      <ExampleCards onPick={() => {}} disabled={false} examples={ALL} />,
    );
    expect(container.querySelector("[data-math-error]")).toBeNull();
    expect(container.querySelectorAll(".katex")).toHaveLength(6);
  });

  it("sends the exact prompt of the card on a click", async () => {
    const onPick = vi.fn();
    render(<ExampleCards onPick={onPick} disabled={false} examples={ALL} />);
    for (const ex of ALL) {
      await userEvent.click(screen.getByRole("button", { name: ex.title }));
      expect(onPick).toHaveBeenLastCalledWith(ex.prompt);
    }
    expect(onPick).toHaveBeenCalledTimes(6);
  });

  it("works from the keyboard: Tab reaches the first card, Enter and Space send its prompt", async () => {
    const onPick = vi.fn();
    render(<ExampleCards onPick={onPick} disabled={false} examples={ALL} />);
    await userEvent.tab();
    expect(screen.getByRole("button", { name: ALL[0]!.title })).toHaveFocus();
    await userEvent.keyboard("{Enter}");
    expect(onPick).toHaveBeenLastCalledWith(ALL[0]!.prompt);
    await userEvent.keyboard(" ");
    expect(onPick).toHaveBeenCalledTimes(2);
    await userEvent.tab();
    expect(screen.getByRole("button", { name: ALL[1]!.title })).toHaveFocus();
  });

  it("sends nothing while the chat is busy", async () => {
    const onPick = vi.fn();
    render(<ExampleCards onPick={onPick} disabled examples={ALL} />);
    const first = screen.getByRole("button", { name: ALL[0]!.title });
    expect(first).toBeDisabled();
    await userEvent.click(first);
    expect(onPick).not.toHaveBeenCalled();
  });

  it("does not show an expected answer on any card", () => {
    const { container } = render(
      <ExampleCards onPick={() => {}} disabled={false} examples={ALL} />,
    );
    const text = container.textContent ?? "";
    // The problems are stated; their results are what Minerva is asked to find.
    expect(text).not.toMatch(/4π\s*δ|4\\pi|π\/4|\\frac\{\\pi\}\{4\}/);
    expect(text).not.toMatch(/δ\(t\)\s*[−-]\s*γ|energy, angular momentum, and/i);
    for (const ex of ALL) {
      expect(ex.description).not.toMatch(/=|equals|is 4π|is π/);
    }
  });

  it("draws nothing for an empty list, so an unapproved card is never shown", () => {
    const { container } = render(<ExampleCards onPick={() => {}} disabled={false} examples={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
});
