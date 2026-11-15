import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Segment } from "@/lib/chat-types";
import { FormulaActionsProvider } from "./formula-actions";
import { RichText } from "./rich-text";

const seg = (type: Segment["type"], value: string, status: Segment["status"] = null): Segment => ({
  type,
  value,
  status,
});

describe("RichText", () => {
  it("shows what Minerva could and could not vouch for in the model's formulas", () => {
    const { container } = render(
      <RichText
        segments={[
          seg("text", "See "),
          seg("math", "x+1", "from_tool"),
          seg("math", "y^2 = 4", "unchecked"),
          seg("math", "a=b", "checked"),
          seg("math", "1=2", "contradicted"),
          {
            ...seg("math", "q", "conditional"),
            note: "false in general, but holds if x is positive",
          },
          seg("display_math", "z", "unchecked"),
        ]}
      />,
    );
    expect(screen.getByText("?")).toBeInTheDocument(); // unchecked equation, inline
    expect(screen.getByText("checked")).toBeInTheDocument();
    expect(screen.getByText("contradicted")).toBeInTheDocument();
    expect(screen.getByText("conditional")).toBeInTheDocument();
    expect(screen.getByText(/Not verified by Minerva/)).toBeInTheDocument();
    expect(container.querySelectorAll('[data-status="from_tool"]')).toHaveLength(1);
    expect(container.querySelectorAll('[data-status="contradicted"]')).toHaveLength(1);
  });

  it("says why a relation was not checked and what a contradiction disagrees with", () => {
    const { container } = render(
      <RichText
        segments={[
          {
            ...seg("display_math", "\\theta = \\phi + \\omega t", "unchecked"),
            note: "a definition or relation between quantities, not an identity; not checked",
          },
          {
            ...seg("math", "a = b", "contradicted"),
            note: "does not agree with c = d (3 of 12 sample points disagree)",
          },
        ]}
      />,
    );
    expect(screen.getByText(/Not verified by Minerva.*not an identity/)).toBeInTheDocument();
    expect(screen.queryByText(/says this is false/)).toBeNull();
    const titles = [...container.querySelectorAll("[title]")].map((e) => e.getAttribute("title"));
    expect(titles.some((t) => t?.includes("does not agree with c = d"))).toBe(true);
    expect(screen.getByText("contradicted")).toBeInTheDocument();
  });

  it("does not flag a lone symbol, but still flags an unchecked equation", () => {
    const { container } = render(
      <RichText
        segments={[
          seg("math", "\\theta", "unchecked"),
          seg("math", "k > 0", "unchecked"),
          seg("math", "r", "unchecked"),
        ]}
      />,
    );
    expect(container.querySelectorAll('[data-status="symbol"]')).toHaveLength(2);
    expect(container.querySelectorAll('[data-status="unchecked"]')).toHaveLength(1);
    expect(container.querySelectorAll("sup")).toHaveLength(1);
  });

  it("shows formulas nobody has checked yet (user text, streaming) with no verification mark", () => {
    const { container } = render(
      <RichText segments={[seg("math", "y^2 = 4"), seg("display_math", "a = b \\\\ c = d")]} />,
    );
    expect(container.querySelectorAll("[data-status]")).toHaveLength(0);
    expect(container.querySelector('[data-status="unchecked"]')).toBeNull();
    expect(container.querySelectorAll("sup")).toHaveLength(0);
    expect(screen.queryByText(/Not verified by Minerva/)).toBeNull();
  });

  it("renders bullet lists, and bold that encloses formulas", () => {
    const { container } = render(
      <RichText
        segments={[
          seg("text", "Equations:\n* **Radial ("),
          seg("math", "r", "unchecked"),
          seg("text", "):** done\n- plain item\n\nAfter."),
        ]}
      />,
    );
    const items = container.querySelectorAll("li");
    expect(items).toHaveLength(2);
    expect(items[0]?.querySelector("strong")?.textContent).toContain("Radial (");
    expect(container.textContent).not.toContain("**");
    expect(container.querySelectorAll("p")).toHaveLength(2);
  });

  it("renders bold and code but no other markup, and never raw HTML", () => {
    const { container } = render(
      <RichText segments={[seg("text", "**Bold** and `code` <img src=x onerror=alert(1)>")]} />,
    );
    expect(container.querySelector("strong")?.textContent).toBe("Bold");
    expect(container.querySelector("code")?.textContent).toBe("code");
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("<img src=x onerror=alert(1)>");
  });

  it("shows a quoted formula as quoted, not as verified, and says why", () => {
    const { container } = render(
      <RichText
        segments={[
          seg("text", "You gave "),
          { ...seg("math", "x^2", "quoted"), note: "your input, as read by Minerva" },
          seg("text", " and the result is "),
          seg("math", "x+1", "from_tool"),
          { ...seg("display_math", "a=b", "quoted"), note: "a handbook entry that is only stated" },
        ]}
      />,
    );
    expect(container.querySelectorAll('[data-status="quoted"]')).toHaveLength(2);
    expect(screen.getByText("quoted")).toBeInTheDocument();
    expect(
      screen.getByText(/Quoted from a tool result, not a verified result: a handbook entry/),
    ).toBeInTheDocument();
    const inline = container.querySelector('span[data-status="quoted"]');
    expect(inline?.getAttribute("title")).toContain("your input as read by Minerva");
    expect(inline?.getAttribute("title")).toContain("It is not marked as checked");
    expect(container.querySelectorAll('[data-status="from_tool"]')).toHaveLength(1);
  });
});

describe("copying a formula from a reply", () => {
  it("copies the exact LaTeX of a display formula and can send it to the message box", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    const editInChat = vi.fn();
    const latex = "L = \\frac{1}{2} m \\dot{r}^2";
    render(
      <FormulaActionsProvider value={{ editInChat }}>
        <RichText segments={[seg("display_math", latex, "unchecked")]} />
      </FormulaActionsProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Copy LaTeX" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(latex));
    fireEvent.click(screen.getByRole("button", { name: "Edit in chat" }));
    expect(editInChat).toHaveBeenCalledWith(latex);
  });

  it("offers copying on inline equations but not on a lone symbol, and no edit outside a chat", () => {
    render(
      <RichText segments={[seg("math", "a = b", "unchecked"), seg("math", "r", "unchecked")]} />,
    );
    expect(screen.getAllByRole("button", { name: "Copy LaTeX" })).toHaveLength(1);
    expect(screen.queryByRole("button", { name: "Edit in chat" })).toBeNull();
  });
});
