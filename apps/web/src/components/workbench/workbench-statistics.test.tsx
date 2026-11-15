import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryItem } from "@/lib/types";
import { step } from "@/test/fixtures";
import { expectationConditional, expectationFree, mleGaussian, sumsOk } from "@/test/stat-fixtures";
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
let expectationResult = expectationFree;

type Body = { input: string; format?: "latex" | "text"; parent_id?: string; assumptions: never };

function save(operation: HistoryItem["operation"], c: Body, result: object, request: object) {
  const id = `s${nextId++}`;
  stored.push(
    step(
      id,
      operation,
      c.input,
      { ...result, request },
      {
        format: c.format ?? "latex",
        parent_id: c.parent_id ?? null,
        assumptions: c.assumptions,
      },
    ),
  );
  return { calculation_id: id };
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
      deleteHistoryItem: vi.fn(),
      deleteDerivation: vi.fn(),
      clearHistory: vi.fn(),
      sums: vi.fn(async (c: Body, req: object) => save("sums", c, sumsOk, req)),
      expectation: vi.fn(async (c: Body, req: object) =>
        save("expectation", c, expectationResult, req),
      ),
      statistics: vi.fn(async () => mleGaussian),
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
  expectationResult = expectationFree;
  vi.clearAllMocks();
});

describe("Sums", () => {
  it("runs the chosen action, shows how it was checked and continues from the result", async () => {
    render(<Workbench />);
    type("Sum((x[i] - mu)**2, (i, 1, n))");
    fireEvent.click(screen.getByRole("button", { name: "Sums" }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    const card = await screen.findByRole("listitem", { name: /Step 1: Linearity of finite sums/ });
    expect(vi.mocked(api.sums).mock.calls[0]?.[1]).toEqual({ action: "expand", variable: null });
    expect(within(card).getAllByText("Checked on explicit instances").length).toBeGreaterThan(0);
    expect(within(card).getByTestId("checks")).toHaveTextContent(/not proved for every size/);
    const editor = (await screen.findByLabelText(
      "Expression (text syntax)",
    )) as HTMLTextAreaElement;
    expect(editor.value).toBe(sumsOk.result_text);
  });

  it("asks for the variable only when differentiating", () => {
    render(<Workbench />);
    type("Sum(x[i], (i, 1, n))");
    fireEvent.click(screen.getByRole("button", { name: "Sums" }));
    expect(screen.queryByLabelText("Variable")).toBeNull();
    fireEvent.change(screen.getByLabelText("Sums action"), { target: { value: "differentiate" } });
    expect(screen.getByLabelText("Variable")).toBeInTheDocument();
  });
});

describe("Expectation", () => {
  it("assumes no independence unless the user states a group", async () => {
    render(<Workbench />);
    type("Var(X + Y)");
    fireEvent.click(screen.getByRole("button", { name: "Expectation" }));
    expect(screen.getByLabelText("Independent groups")).toHaveValue("");
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByRole("listitem", { name: /Step 1/ });
    expect(vi.mocked(api.expectation).mock.calls[0]?.[1]).toEqual({
      action: "expand",
      random: ["X", "Y"],
      independent: [],
    });
    expect(screen.queryByTestId("premises")).toBeNull();
  });

  it("sends the stated groups and shows the result as conditional on them", async () => {
    expectationResult = expectationConditional;
    render(<Workbench />);
    type("Var(X + Y)");
    fireEvent.click(screen.getByRole("button", { name: "Expectation" }));
    fireEvent.change(screen.getByLabelText("Independent groups"), { target: { value: "X, Y" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    const card = await screen.findByRole("listitem", { name: /Step 1/ });
    expect(vi.mocked(api.expectation).mock.calls[0]?.[1]).toMatchObject({
      independent: [["X", "Y"]],
    });
    expect(within(card).getByTestId("premises")).toHaveTextContent(/X and Y are independent/);
  });

  it("refuses a group naming a variable that is not random, before sending", () => {
    render(<Workbench />);
    type("Var(X)");
    fireEvent.click(screen.getByRole("button", { name: "Expectation" }));
    fireEvent.change(screen.getByLabelText("Random variables"), { target: { value: "X" } });
    fireEvent.change(screen.getByLabelText("Independent groups"), { target: { value: "X, Q" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/Q is in an independence group/);
    expect(api.expectation).not.toHaveBeenCalled();
  });
});

describe("Statistics mode", () => {
  it("sends independence only when ticked and shows the result with its premises", async () => {
    render(<Workbench />);
    fireEvent.click(screen.getByRole("button", { name: "Statistics" }));
    await userEvent.selectOptions(
      screen.getByLabelText("Start from an example"),
      "Maximum likelihood: Gaussian mean and variance",
    );
    const box = screen.getByRole("checkbox", { name: /independent and identically distributed/ });
    expect(box).toBeChecked(); // the example states it explicitly
    await userEvent.click(box); // the user withdraws it
    await userEvent.click(screen.getByRole("button", { name: "Derive" }));
    await waitFor(() => expect(api.statistics).toHaveBeenCalledOnce());
    expect(vi.mocked(api.statistics).mock.calls[0]?.[0]).toMatchObject({
      family: "normal",
      action: "mle",
      estimate: ["mu", "v"],
      iid: null,
    });
    const panel = await screen.findByRole("region", { name: "Maximum likelihood" });
    expect(within(panel).getByTestId("premises")).toHaveTextContent(/independent and identically/);
    expect(within(panel).getByTestId("estimates")).toHaveTextContent(/Local maximum/);
  });

  it("starts with independence off", () => {
    render(<Workbench />);
    fireEvent.click(screen.getByRole("button", { name: "Statistics" }));
    expect(
      screen.getByRole("checkbox", { name: /independent and identically distributed/ }),
    ).not.toBeChecked();
  });

  it("says what is missing instead of sending an incomplete model", async () => {
    render(<Workbench />);
    fireEvent.click(screen.getByRole("button", { name: "Statistics" }));
    await userEvent.click(screen.getByRole("radio", { name: "Moments" }));
    await userEvent.click(screen.getByRole("button", { name: "Derive" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/Give a value for mu/);
    expect(api.statistics).not.toHaveBeenCalled();
  });
});
