import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryItem } from "@/lib/types";
import { identities, match, step } from "@/test/fixtures";
import { Workbench } from "./workbench";

// A stable router, as in Next: a new object per render would re-run the initial load forever.
const router = { push: vi.fn() };
vi.mock("next/navigation", () => ({ useRouter: () => router }));
// MathLive needs a real browser; the Workbench only needs something that reports LaTeX.
vi.mock("../math-input", () => ({
  MathInput: ({ value, onChange }: { value: string; onChange: (v: string) => void }) => (
    <textarea
      aria-label="Mathematical expression"
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  ),
}));

const calls: { name: string; args: unknown[] }[] = [];
let stored: HistoryItem[] = [];
let nextId = 1;
let lookupResult = identities([]);

function record(
  name: string,
  operation: HistoryItem["operation"],
  result: Record<string, unknown>,
  status: HistoryItem["status"] = "ok",
) {
  return async (
    c: {
      input: string;
      format?: "latex" | "text";
      assumptions: never;
      parent_id?: string;
      derivation_id?: string;
    },
    ...rest: unknown[]
  ) => {
    calls.push({ name, args: [c, ...rest] });
    const id = `s${nextId++}`;
    stored.push(
      step(
        id,
        operation,
        c.input,
        { ...result, status },
        {
          format: c.format ?? "latex",
          parent_id: c.parent_id ?? null,
          assumptions: c.assumptions,
          status,
        },
      ),
    );
    return { calculation_id: id };
  };
}

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      createDerivation: vi.fn(async () => ({ id: "d1", updated_at: "", steps: [] })),
      derivation: vi.fn(async () => ({ id: "d1", updated_at: "", steps: [...stored] })),
      derivations: vi.fn(async () => []),
      history: vi.fn(async () => []),
      deleteHistoryItem: vi.fn(async () => undefined),
      deleteDerivation: vi.fn(),
      clearHistory: vi.fn(),
      identities: vi.fn(async () => lookupResult),
      simplify: vi.fn((...a: unknown[]) => fakes.simplify(...(a as [never]))),
      differentiate: vi.fn((...a: unknown[]) => fakes.differentiate(...(a as [never]))),
      integrate: vi.fn(),
      series: vi.fn(),
      distribution: vi.fn((...a: unknown[]) => fakes.distribution(...(a as [never]))),
      applyIdentity: vi.fn((...a: unknown[]) => fakes.applyIdentity(...(a as [never]))),
    },
  };
});

import { api } from "@/lib/api";

const fakes = {
  simplify: record("simplify", "simplify", { result_text: "x + 1", result_latex: "x + 1" }),
  differentiate: record("differentiate", "differentiate", {
    result_text: "1",
    result_latex: "1",
    variables: [{ name: "x", order: 1 }],
    steps: [],
    notes: [],
    verification: null,
  }),
  distribution: record("distribution", "distribution", {
    result_text: "2",
    result_latex: "2",
    action: "integrate",
    steps: [],
    blocked: [],
    notes: [],
    equality: "value",
    reason: null,
    diagnostics: { reason: null, events: [] },
    verification: {
      pairings_checked: 1,
      pairings_agreed: 1,
      conditions_established: true,
      passed: true,
      test_functions: [],
      widths: [],
      parameter_samples: 1,
      convergence_observed: true,
      method: "m",
      note: "n",
    },
  }),
  applyIdentity: record("apply_identity", "apply_identity", {
    result_text: "log(x) + log(y)",
    result_latex: "\\log x + \\log y",
    entry: { title: "Log of a product" },
  }),
};

function type(text: string) {
  fireEvent.change(screen.getByLabelText("Mathematical expression"), { target: { value: text } });
}

beforeEach(() => {
  calls.length = 0;
  stored = [];
  nextId = 1;
  lookupResult = identities([]);
  vi.clearAllMocks();
});

describe("Workbench derivation", () => {
  it("adds a step, then continues from its text form with the step as parent", async () => {
    render(<Workbench />);
    type("\\sin^2 x + \\cos^2 x + x");
    fireEvent.click(screen.getByRole("button", { name: "Simplify" }));
    const first = await screen.findByRole("listitem", { name: /Step 1: Simplify/ });
    expect(within(first).getByText("new input")).toBeInTheDocument();
    // The editor now holds the result as text, ready for the next operation.
    const editor = (await screen.findByLabelText(
      "Expression (text syntax)",
    )) as HTMLTextAreaElement;
    expect(editor.value).toBe("x + 1");
    expect(screen.getByRole("status")).toHaveTextContent("Continuing from the result of step 1");

    fireEvent.click(screen.getByRole("button", { name: "Differentiate" }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByRole("listitem", { name: /Step 2: Differentiate by x/ });

    const [input, variables] = vi.mocked(api.differentiate).mock.calls[0] as unknown as [
      Record<string, unknown>,
      unknown,
    ];
    expect(input).toMatchObject({
      input: "x + 1",
      format: "text",
      derivation_id: "d1",
      parent_id: "s1",
    });
    expect(variables).toEqual([{ name: "x", order: 1 }]);
    expect(screen.getByText("from step 1")).toBeInTheDocument();
  });

  it("makes an edited result a new input, not a continuation", async () => {
    render(<Workbench />);
    type("x + x");
    fireEvent.click(screen.getByRole("button", { name: "Simplify" }));
    const editor = await screen.findByLabelText("Expression (text syntax)");
    fireEvent.change(editor, { target: { value: "x + 2" } });
    expect(screen.getByRole("status")).toHaveTextContent(/new input, not a continuation/);
    fireEvent.click(screen.getByRole("button", { name: "Simplify" }));
    await waitFor(() => expect(api.simplify).toHaveBeenCalledTimes(2));
    expect(vi.mocked(api.simplify).mock.calls[1]?.[0]).toMatchObject({ input: "x + 2" });
    expect(vi.mocked(api.simplify).mock.calls[1]?.[0]).not.toHaveProperty("parent_id", "s1");
  });

  it("shows only the settings of the chosen operation", () => {
    render(<Workbench />);
    type("x");
    fireEvent.click(screen.getByRole("button", { name: "Integrate" }));
    expect(screen.getByLabelText("Lower bound (optional)")).toBeInTheDocument();
    expect(screen.queryByLabelText("Order of the derivative")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Series" }));
    expect(screen.getByLabelText("Point to expand about")).toBeInTheDocument();
    expect(screen.queryByLabelText("Lower bound (optional)")).not.toBeInTheDocument();
  });

  it("does not run an operation with bad settings", () => {
    render(<Workbench />);
    type("x");
    fireEvent.click(screen.getByRole("button", { name: "Integrate" }));
    fireEvent.change(screen.getByLabelText("Lower bound (optional)"), { target: { value: "0" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/both bounds/);
    expect(api.integrate).not.toHaveBeenCalled();
  });

  it("runs a distribution step with the variable declared real and the chosen limits", async () => {
    render(<Workbench />);
    type("x^2 \\operatorname{DiracDelta}(x^2-4)");
    fireEvent.click(screen.getByRole("button", { name: "Deltas" }));
    fireEvent.change(screen.getByLabelText("Distribution action"), {
      target: { value: "integrate" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/both limits/);
    expect(api.distribution).not.toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText("Lower limit"), { target: { value: "-oo" } });
    fireEvent.change(screen.getByLabelText("Upper limit"), { target: { value: "oo" } });
    fireEvent.change(screen.getByLabelText("Convention for a delta on a limit"), {
      target: { value: "half" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByRole("listitem", { name: /Step 1: Integrate against deltas/ });
    const [input, request] = vi.mocked(api.distribution).mock.calls[0] as unknown as [
      Record<string, unknown>,
      Record<string, unknown>,
    ];
    expect(input.assumptions).toEqual({ x: ["real"] });
    expect(request).toEqual({
      action: "integrate",
      variable: "x",
      lower: "-oo",
      upper: "oo",
      endpoint: "half",
    });
  });

  it("keeps a failed step visible but offers no way to continue from it", async () => {
    fakes.simplify = record(
      "simplify",
      "simplify",
      { result_text: null, result_latex: null, message: "Could not verify." },
      "not_verified",
    );
    render(<Workbench />);
    type("x");
    fireEvent.click(screen.getByRole("button", { name: "Simplify" }));
    const card = await screen.findByRole("listitem", { name: /Step 1/ });
    expect(within(card).getAllByText("Could not verify.").length).toBeGreaterThan(0);
    expect(
      within(card).queryByRole("button", { name: "Continue from here" }),
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    fakes.simplify = record("simplify", "simplify", {
      result_text: "x + 1",
      result_latex: "x + 1",
    });
  });

  it("offers Apply only for applicable identities and sends the rule id", async () => {
    lookupResult = identities([
      match("log.product", "applicable", { rewritten_latex: "\\log x + \\log y" }),
      match("log.blocked", "conditions_not_satisfied", {
        unmet_conditions: [{ variable: "a", assumption: "positive" }],
      }),
    ]);
    render(<Workbench />);
    type("\\log(xy)");
    fireEvent.click(screen.getByRole("button", { name: "Find identities for the expression" }));
    const apply = await screen.findAllByRole("button", { name: "Apply as next step" });
    expect(apply).toHaveLength(1);
    expect(vi.mocked(api.identities).mock.calls[0]?.[0]).toMatchObject({ save: false });
    fireEvent.click(apply[0]!);
    await screen.findByRole("listitem", { name: /Step 1: Log of a product/ });
    expect(vi.mocked(api.applyIdentity).mock.calls[0]?.[1]).toBe("log.product");
    expect(vi.mocked(api.applyIdentity).mock.calls[0]?.[0]).toMatchObject({ derivation_id: "d1" });
  });

  it("drops an identity lookup once the expression changes", async () => {
    lookupResult = identities([match("log.product", "applicable", { rewritten_latex: "x" })]);
    render(<Workbench />);
    type("\\log(xy)");
    fireEvent.click(screen.getByRole("button", { name: "Find identities for the expression" }));
    await screen.findByRole("button", { name: "Apply as next step" });
    type("\\log(xyz)");
    expect(screen.queryByRole("button", { name: "Apply as next step" })).not.toBeInTheDocument();
    expect(screen.getByText(/Search again/)).toBeInTheDocument();
  });
});
