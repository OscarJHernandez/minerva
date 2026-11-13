import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryItem } from "@/lib/types";
import { identities, isolateResult, match, step, stepResult, subterm } from "@/test/fixtures";
import { Workbench } from "./workbench";

const router = { push: vi.fn() };
vi.mock("next/navigation", () => ({ useRouter: () => router }));
vi.mock("../math-input", () => ({
  MathInput: ({ value, onChange }: { value: string; onChange: (v: string) => void }) => (
    <textarea
      aria-label="Mathematical expression"
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  ),
}));

let stored: HistoryItem[] = [];
let nextId = 1;
let lookupResult = identities([]);

type Body = { input: string; format?: "latex" | "text"; parent_id?: string; assumptions: never };

/** A fake endpoint: stores the step it would have saved (with the request, as the API does). */
function saving(
  operation: HistoryItem["operation"],
  result: (c: Body, ...rest: unknown[]) => Record<string, unknown>,
) {
  return async (c: Body, ...rest: unknown[]) => {
    const id = `s${nextId++}`;
    const made = result(c, ...rest);
    stored.push(
      step(id, operation, c.input, made, {
        format: c.format ?? "latex",
        parent_id: c.parent_id ?? null,
        assumptions: c.assumptions,
        status: made.status as HistoryItem["status"],
      }),
    );
    return { calculation_id: id };
  };
}

const defaultSubstitute = (_c: Body, relations: unknown) =>
  ({ ...stepResult(), request: { relations } }) as unknown as Record<string, unknown>;

const fakes = {
  substitute: saving("substitute", defaultSubstitute),
  rewrite: saving(
    "rewrite",
    (_c, body) =>
      ({
        ...stepResult({
          operation: "rewrite",
          result_text: "x**2 + 2*x + 1",
          result_latex: "x^{2} + 2 x + 1",
        }),
        request: body,
      }) as unknown as Record<string, unknown>,
  ),
  isolate: saving(
    "isolate",
    (_c, variable) =>
      ({ ...isolateResult(), request: { variable } }) as unknown as Record<string, unknown>,
  ),
};
let parts = [subterm(1, "\\left(x + 1\\right)^{2}"), subterm(2, "x + 1")];

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      createDerivation: vi.fn(async () => ({ id: "d1", updated_at: "", steps: [] })),
      derivation: vi.fn(async () => ({ id: "d1", updated_at: "", steps: [...stored] })),
      derivations: vi.fn(async () => []),
      history: vi.fn(async () => []),
      deleteHistoryItem: vi.fn(),
      deleteDerivation: vi.fn(),
      clearHistory: vi.fn(),
      identities: vi.fn(async () => lookupResult),
      simplify: vi.fn(),
      differentiate: vi.fn(),
      integrate: vi.fn(),
      series: vi.fn(),
      applyIdentity: vi.fn(),
      substitute: vi.fn((...a: unknown[]) => fakes.substitute(...(a as [Body]))),
      rewrite: vi.fn((...a: unknown[]) => fakes.rewrite(...(a as [Body]))),
      isolate: vi.fn((...a: unknown[]) => fakes.isolate(...(a as [Body]))),
      subterms: vi.fn(async () =>
        stepResult({ operation: "subterms", subterms: parts, steps: [], given: [] }),
      ),
    },
  };
});

import { api } from "@/lib/api";

function type(text: string) {
  fireEvent.change(screen.getByLabelText("Mathematical expression"), { target: { value: text } });
}

beforeEach(() => {
  stored = [];
  nextId = 1;
  lookupResult = identities([]);
  parts = [subterm(1, "\\left(x + 1\\right)^{2}"), subterm(2, "x + 1")];
  vi.clearAllMocks();
});

describe("Substitute", () => {
  it("sends the relations, shows the premise as the user's own, and continues from the result", async () => {
    render(<Workbench />);
    type("x^2 + y^2");
    fireEvent.click(screen.getByRole("button", { name: "Substitute" }));
    fireEvent.change(screen.getByLabelText("Relations"), { target: { value: "y^2 = 1 - x^2" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    const card = await screen.findByRole("listitem", { name: /Step 1: Substitute a relation/ });
    expect(vi.mocked(api.substitute).mock.calls[0]?.[1]).toEqual(["y^2 = 1 - x^2"]);
    expect(vi.mocked(api.substitute).mock.calls[0]?.[2]).toBeUndefined();
    expect(within(card).getAllByText("Holds given your relation").length).toBeGreaterThan(0);
    expect(within(card).getAllByTestId("premises")[0]).toHaveTextContent("not checked by Minerva");
    const editor = (await screen.findByLabelText(
      "Expression (text syntax)",
    )) as HTMLTextAreaElement;
    expect(editor.value).toBe("1");
  });

  it("sends the chosen order when there are several relations", async () => {
    render(<Workbench />);
    type("x + y");
    fireEvent.click(screen.getByRole("button", { name: "Substitute" }));
    fireEvent.change(screen.getByLabelText("Relations"), { target: { value: "x = y\ny = 1" } });
    fireEvent.change(screen.getByLabelText("Order of the relations"), {
      target: { value: "given" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByRole("listitem", { name: /Step 1/ });
    expect(vi.mocked(api.substitute).mock.calls[0]?.[1]).toEqual(["x = y", "y = 1"]);
    expect(vi.mocked(api.substitute).mock.calls[0]?.[2]).toBe("given");
  });

  it.each([
    ["", /at least one relation/],
    ["x", /one equals sign/],
    ["a = 1\nb = 2\nc = 3\nd = 4\ne = 5", /at most 4/],
  ])("does not send relations %j", (relations, message) => {
    render(<Workbench />);
    type("x");
    fireEvent.click(screen.getByRole("button", { name: "Substitute" }));
    fireEvent.change(screen.getByLabelText("Relations"), { target: { value: relations } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(screen.getByRole("alert")).toHaveTextContent(message);
    expect(api.substitute).not.toHaveBeenCalled();
  });

  it("shows a refused substitution with its reason and nothing to continue from", async () => {
    fakes.substitute = saving(
      "substitute",
      () =>
        stepResult({
          status: "unsupported",
          reason: "order_dependent",
          result_latex: null,
          result_text: null,
          steps: [],
          given: [],
          message: "The result depends on the order.",
        }) as unknown as Record<string, unknown>,
    );
    render(<Workbench />);
    type("x + y");
    fireEvent.click(screen.getByRole("button", { name: "Substitute" }));
    fireEvent.change(screen.getByLabelText("Relations"), { target: { value: "x = y\ny = 1" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    const card = await screen.findByRole("listitem", { name: /Step 1/ });
    expect(within(card).getAllByText(/depends on the order/).length).toBeGreaterThan(0);
    expect(
      within(card).queryByRole("button", { name: "Continue from here" }),
    ).not.toBeInTheDocument();
    fakes.substitute = saving(
      "substitute",
      () => stepResult() as unknown as Record<string, unknown>,
    );
  });
});

describe("Collect and Solve for", () => {
  it("collects by the listed things", async () => {
    render(<Workbench />);
    type("a x + b x");
    fireEvent.click(screen.getByRole("button", { name: "Collect" }));
    fireEvent.change(screen.getByLabelText("Collect by"), { target: { value: "x, y" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByRole("listitem", { name: /Step 1: Collect terms/ });
    expect(vi.mocked(api.rewrite).mock.calls[0]?.[1]).toMatchObject({
      action: "collect",
      by: ["x", "y"],
    });
    expect(vi.mocked(api.rewrite).mock.calls[0]?.[1]).not.toHaveProperty(
      "selection",
      expect.anything(),
    );
  });

  it("lists the solutions of an equation and continues only with the one the user picks", async () => {
    render(<Workbench />);
    type("a x^2 = b");
    fireEvent.click(screen.getByRole("button", { name: "Solve for" }));
    fireEvent.change(screen.getByLabelText("Unknown to solve for"), { target: { value: "x" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    const card = await screen.findByRole("listitem", { name: /Step 1: Solve for x/ });
    expect(vi.mocked(api.isolate).mock.calls[0]?.[1]).toBe("x");
    expect(within(card).getByRole("list", { name: "Solutions" })).toBeInTheDocument();
    // Two solutions: nothing is continued on the user's behalf.
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    fireEvent.click(
      within(card).getAllByRole("button", { name: "Continue with this solution" })[1]!,
    );
    const editor = (await screen.findByLabelText(
      "Expression (text syntax)",
    )) as HTMLTextAreaElement;
    expect(editor.value).toBe("-sqrt(b/a)");
    expect(screen.getByRole("status")).toHaveTextContent("Continuing from the result of step 1");
  });

  it("does not run solve with a bad unknown", () => {
    render(<Workbench />);
    type("x = 1");
    fireEvent.click(screen.getByRole("button", { name: "Solve for" }));
    fireEvent.change(screen.getByLabelText("Unknown to solve for"), { target: { value: "1x" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/unknown/);
    expect(api.isolate).not.toHaveBeenCalled();
  });
});

describe("Part", () => {
  async function chooseSecond() {
    fireEvent.click(screen.getByRole("button", { name: "Part…" }));
    fireEvent.click(screen.getByRole("button", { name: "List parts" }));
    await screen.findByLabelText("Parts");
    fireEvent.click(screen.getByTestId("part-2"));
  }

  it("lists the parts from the server, and a chosen part is what Expand acts on", async () => {
    render(<Workbench />);
    type("(x+1)^2 (y+1)^2");
    await chooseSecond();
    expect(screen.getByTestId("chosen-part")).toBeInTheDocument();
    expect(vi.mocked(api.subterms).mock.calls[0]?.[1]).toEqual({});
    fireEvent.click(screen.getByRole("button", { name: "Expand" }));
    await screen.findByRole("listitem", { name: /Step 1: Expand \(one part\)/ });
    expect(vi.mocked(api.rewrite).mock.calls[0]?.[1]).toEqual({
      action: "expand",
      selection: { path: [1], args: null, expected_latex: "x + 1" },
    });
  });

  it("asks for the typed part when narrowing the list", async () => {
    render(<Workbench />);
    type("x^2 - 1 + y^2");
    fireEvent.click(screen.getByRole("button", { name: "Part…" }));
    fireEvent.change(screen.getByLabelText("Part to look for (optional)"), {
      target: { value: "x^2 - 1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "List parts" }));
    await screen.findByLabelText("Parts");
    expect(vi.mocked(api.subterms).mock.calls[0]?.[1]).toEqual({ find: "x^2 - 1" });
  });

  it("drops the choice when the expression changes, because the path no longer means the same", async () => {
    render(<Workbench />);
    type("(x+1)^2 (y+1)^2");
    await chooseSecond();
    expect(screen.getByTestId("chosen-part")).toBeInTheDocument();
    type("(x+1)^2 (y+2)^2");
    expect(screen.queryByTestId("chosen-part")).not.toBeInTheDocument();
    expect(screen.getByText(/list the parts again/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Expand" })).toBeDisabled();
  });

  it("applies an identity to the chosen part only", async () => {
    lookupResult = identities([match("trig.double_sin", "applicable")]);
    render(<Workbench />);
    type("sin(2x) + sin(2y)");
    fireEvent.click(
      within(screen.getByRole("toolbar")).getByRole("button", { name: "Identities" }),
    );
    await screen.findByRole("button", { name: /Apply/ });
    await chooseSecond();
    fireEvent.click(screen.getByRole("button", { name: /Apply/ }));
    await waitFor(() => expect(api.rewrite).toHaveBeenCalled());
    expect(api.applyIdentity).not.toHaveBeenCalled();
    expect(vi.mocked(api.rewrite).mock.calls[0]?.[1]).toMatchObject({
      action: "rule",
      rule_id: "trig.double_sin",
      selection: { path: [1], expected_latex: "x + 1" },
    });
  });

  it("collects inside the chosen part", async () => {
    render(<Workbench />);
    type("a x + sin(c x + d x)");
    await chooseSecond();
    fireEvent.click(screen.getByRole("button", { name: "Collect" }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByRole("listitem", { name: /Step 1/ });
    expect(vi.mocked(api.rewrite).mock.calls[0]?.[1]).toMatchObject({
      action: "collect",
      selection: { path: [1], expected_latex: "x + 1" },
    });
  });
});
