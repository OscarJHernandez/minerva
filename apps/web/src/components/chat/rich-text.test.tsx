import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Segment } from "@/lib/chat-types";
import { RichText } from "./rich-text";

const seg = (type: Segment["type"], value: string, verified: boolean | null = null): Segment => ({
  type,
  value,
  verified,
});

describe("RichText", () => {
  it("marks formulas the model wrote on its own, and only those", () => {
    const { container } = render(
      <RichText
        segments={[
          seg("text", "See "),
          seg("math", "x+1", true),
          seg("text", " and "),
          seg("math", "y^2", false),
          seg("display_math", "z", false),
        ]}
      />,
    );
    expect(screen.getAllByText("unchecked")).toHaveLength(1);
    expect(screen.getByText(/Not verified by Minerva/)).toBeInTheDocument();
    const verified = container.querySelectorAll('[data-verified="true"]');
    const unchecked = container.querySelectorAll('[data-verified="false"]');
    expect(verified).toHaveLength(1);
    expect(unchecked).toHaveLength(2);
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
