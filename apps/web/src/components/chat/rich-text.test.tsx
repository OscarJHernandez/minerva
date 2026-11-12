import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Segment } from "@/lib/chat-types";
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
});
