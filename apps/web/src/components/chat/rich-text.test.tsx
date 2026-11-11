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
          seg("math", "y^2", "unchecked"),
          seg("math", "a=b", "checked"),
          seg("math", "1=2", "contradicted"),
          seg("display_math", "z", "unchecked"),
        ]}
      />,
    );
    expect(screen.getAllByText("unchecked")).toHaveLength(1);
    expect(screen.getByText("checked")).toBeInTheDocument();
    expect(screen.getByText("contradicted")).toBeInTheDocument();
    expect(screen.getByText(/Not verified by Minerva/)).toBeInTheDocument();
    expect(container.querySelectorAll('[data-status="from_tool"]')).toHaveLength(1);
    expect(container.querySelectorAll('[data-status="contradicted"]')).toHaveLength(1);
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
