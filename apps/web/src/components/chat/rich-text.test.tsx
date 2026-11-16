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

  it("renders bold and code inline, and never raw HTML", () => {
    const { container } = render(
      <RichText segments={[seg("text", "**Bold** and `code` <img src=x onerror=alert(1)>")]} />,
    );
    expect(container.querySelector("strong")?.textContent).toBe("Bold");
    expect(container.querySelector("code")?.textContent).toBe("code");
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("<img src=x onerror=alert(1)>");
  });

  it("renders *italic* and _italic_, and leaves products, snake_case and stray stars alone", () => {
    const { container } = render(
      <RichText
        segments={[
          seg(
            "text",
            "from Greek *brachistos* and _chronos_; then 2 * 3 * 4, snake_case_name, a lone * star and **bold *both***",
          ),
        ]}
      />,
    );
    const italics = [...container.querySelectorAll("em")].map((e) => e.textContent);
    expect(italics).toContain("brachistos");
    expect(italics).toContain("chronos");
    expect(container.textContent).toContain("2 * 3 * 4");
    expect(container.textContent).toContain("snake_case_name");
    expect(container.textContent).toContain("a lone * star");
    expect(container.textContent).not.toContain("*brachistos*");
  });

  it("does not read a star between a formula and text as emphasis", () => {
    const { container } = render(
      <RichText
        segments={[seg("text", "a "), seg("math", "x*y", "unchecked"), seg("text", " * b")]}
      />,
    );
    expect(container.querySelector("em")).toBeNull();
    expect(container.textContent).toContain("* b");
  });

  it("renders a ### heading, not the hashes", () => {
    const { container } = render(
      <RichText segments={[seg("text", "Intro\n### The Problem Statement\nBody")]} />,
    );
    expect(container.textContent).not.toContain("###");
    expect(container.querySelector("h1,h2,h3,h4,h5,h6")?.textContent).toBe("The Problem Statement");
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

describe("RichText, block markup", () => {
  it("renders headings of any level, also straight after a list or a formula", () => {
    const { container } = render(
      <RichText
        segments={[
          seg(
            "text",
            "Intro\n### 1. The Geometric Connection\n- item\n## **Bold** title\n#### Small\n",
          ),
          seg("display_math", "a = b", "unchecked"),
          seg("text", "\n### After the formula\n#hashtag and ####### seven"),
        ]}
      />,
    );
    const headings = [...container.querySelectorAll("h3, h4")];
    expect(headings.map((h) => h.textContent)).toEqual([
      "1. The Geometric Connection",
      "Bold title",
      "Small",
      "After the formula",
    ]);
    expect(headings.map((h) => h.tagName)).toEqual(["H3", "H3", "H4", "H3"]);
    expect(headings[1]?.querySelector("strong")?.textContent).toBe("Bold");
    expect(container.querySelectorAll("li")).toHaveLength(1); // the heading closed the list
    expect(container.textContent).not.toMatch(/#{2,6} (1\.|After|\*\*|Bold|Small)/);
    expect(container.textContent).toContain("#hashtag and ####### seven"); // not a heading
  });

  it("renders numbered lists, starting where the model started and apart from bullets", () => {
    const { container } = render(
      <RichText segments={[seg("text", "3. third\n4. fourth\n- bullet\n1) one\n")]} />,
    );
    const lists = [...container.querySelectorAll("ol, ul")];
    expect(lists.map((l) => l.tagName)).toEqual(["OL", "UL", "OL"]);
    expect(lists[0]?.getAttribute("start")).toBe("3");
    expect(lists[0]?.querySelectorAll("li")).toHaveLength(2);
    expect(container.textContent).not.toContain("3.");
  });

  it("renders a table with bold and formulas in its cells, and a bar inside a formula is not a cell", () => {
    const { container } = render(
      <RichText
        segments={[
          seg("text", "Before\n| Feature | Time |\n| :--- | ---: |\n| **Goal** | Minimize "),
          seg("math", "T", "unchecked"),
          seg("text", " |\n| Size | "),
          seg("math", "|x|", "unchecked"),
          seg("text", " |\nAfter"),
        ]}
      />,
    );
    const table = container.querySelector("table");
    expect(table).not.toBeNull();
    expect([...table!.querySelectorAll("th")].map((c) => c.textContent)).toEqual([
      "Feature",
      "Time",
    ]);
    const rows = [...table!.querySelectorAll("tbody tr")];
    expect(rows).toHaveLength(2);
    expect(rows[0]?.querySelectorAll("td")).toHaveLength(2);
    expect(rows[0]?.querySelector("strong")?.textContent).toBe("Goal");
    expect(rows[1]?.querySelectorAll("td")).toHaveLength(2); // `|x|` stayed inside its cell
    expect(table!.querySelectorAll("th")[1]?.className).toContain("text-right");
    expect(table!.querySelectorAll("th")[0]?.className).toContain("text-left");
    expect(container.textContent).not.toContain("| Feature");
    expect(container.textContent).toContain("Before");
    expect(container.textContent).toContain("After");
    expect(container.parentElement?.querySelector(".overflow-x-auto")).not.toBeNull();
  });

  it("leaves lines that start with a bar but have no separator row as plain text", () => {
    const { container } = render(
      <RichText segments={[seg("text", "| not | a table |\n| really | no |\n")]} />,
    );
    expect(container.querySelector("table")).toBeNull();
    expect(container.querySelectorAll("p")).toHaveLength(2);
  });
});

describe("RichText, formula buttons", () => {
  it("offers copy and edit for any inline formula but not for a lone symbol, out of the line", () => {
    const { container } = render(
      <RichText
        segments={[
          seg("text", "We have "),
          seg("math", "f'(r_0)", "unchecked"),
          seg("text", " and "),
          seg("math", "r_0", "unchecked"),
          seg("text", " and "),
          seg("math", "\\theta", "unchecked"),
        ]}
      />,
    );
    const bars = container.querySelectorAll('[data-testid="formula-actions"]');
    expect(bars).toHaveLength(1);
    // positioned out of the flow, so it takes no room beside the formula
    expect(bars[0]?.className).toContain("absolute");
    expect(screen.getAllByRole("button", { name: "Copy LaTeX" })).toHaveLength(1);
  });

  it("opens the bar on a tap on the formula and closes it on a tap elsewhere", () => {
    const { container } = render(
      <div>
        <RichText segments={[seg("math", "y(x)", "unchecked")]} />
        <p data-testid="elsewhere">elsewhere</p>
      </div>,
    );
    const bar = container.querySelector('[data-testid="formula-actions"]')!;
    expect(bar.className).toContain("opacity-0");
    fireEvent.click(container.querySelector("[data-status]")!);
    expect(bar.className).toContain("opacity-100");
    expect(bar.className).not.toContain("opacity-0");
    fireEvent.pointerDown(screen.getByTestId("elsewhere"));
    expect(bar.className).toContain("opacity-0");
  });

  it("does not open the bar when the mark's explanation is opened", () => {
    const { container } = render(<RichText segments={[seg("math", "a = b", "unchecked")]} />);
    fireEvent.click(screen.getByRole("button", { name: /what this mark means/ }));
    expect(screen.getByTestId("mark-explanation")).toBeInTheDocument();
    expect(container.querySelector('[data-testid="formula-actions"]')?.className).toContain(
      "opacity-0",
    );
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
