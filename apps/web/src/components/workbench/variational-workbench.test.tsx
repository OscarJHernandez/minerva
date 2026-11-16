import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryItem } from "@/lib/types";
import type { VariationReport, VariationRequest } from "@/lib/variation-types";
import { step } from "@/test/fixtures";
import {
  evaluated,
  evaluatedRequest,
  firstHarmonic,
  firstHarmonicRequest,
} from "@/test/variation-fixtures";
import { VariationalWorkbench, type VariationRestore } from "./variational-workbench";

let stored: HistoryItem[] = [];
let nextId = 1;
const reports: Record<string, VariationReport> = { first: firstHarmonic, evaluate: evaluated };

const variation =
  vi.fn<
    (r: VariationRequest, link?: { derivation_id?: string; parent_id?: string }) => Promise<unknown>
  >();
const createDerivation = vi.fn();
vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      createDerivation: () => createDerivation(),
      variation: (r: VariationRequest, link?: { derivation_id?: string; parent_id?: string }) =>
        variation(r, link),
      derivation: vi.fn(async () => ({ id: "d1", updated_at: "", steps: [...stored] })),
      deleteHistoryItem: vi.fn(async () => undefined),
    },
  };
});

const noop = () => undefined;
const mount = (restore: VariationRestore | null = null, onSaved = noop) =>
  render(<VariationalWorkbench restore={restore} onUnauthorized={noop} onSaved={onSaved} />);

const harmonic = (action: string, request: VariationRequest) =>
  step(
    `s${nextId++}`,
    "variation",
    request.integrand,
    { ...(reports[action] ?? firstHarmonic), request },
    { format: "latex", derivation_id: "d1" },
  );

beforeEach(() => {
  stored = [];
  nextId = 1;
  variation.mockReset();
  createDerivation.mockReset();
  createDerivation.mockResolvedValue({ id: "d1", updated_at: "", steps: [] });
  variation.mockImplementation(async (r, link) => {
    const item = harmonic(r.action, r);
    item.parent_id = link?.parent_id ?? null;
    stored.push(item);
    return { calculation_id: item.id };
  });
});

async function chooseExample(label: string) {
  await userEvent.selectOptions(screen.getByLabelText("Start from an example"), label);
}

describe("VariationalWorkbench", () => {
  it("runs the first variation as the first step, then a numerical step as the next one", async () => {
    const onSaved = vi.fn();
    mount(null, onSaved);
    await chooseExample("Harmonic oscillator, fixed ends");
    await userEvent.click(screen.getByRole("button", { name: "First variation" }));

    await waitFor(() => expect(variation).toHaveBeenCalledOnce());
    expect(createDerivation).toHaveBeenCalledOnce();
    const [first, link1] = variation.mock.calls[0]!;
    expect(first).toMatchObject({
      action: "first",
      integrand: "\\frac12\\dot{x}^2-\\frac12 x^2",
      functions: ["x"],
      lower: "0",
      upper: "1",
    });
    expect(first.conditions).toHaveLength(2);
    expect(link1).toEqual({ derivation_id: "d1", parent_id: undefined });
    expect(await screen.findAllByRole("region", { name: /^Variational calculus ·/ })).toHaveLength(
      1,
    );
    expect(onSaved).toHaveBeenCalled();

    // Pasted, not typed: sixteen keystrokes re-render the whole workbench sixteen times, which
    // pushed this test past the default 5 s on a loaded CI runner.
    await userEvent.click(screen.getByLabelText("Path for x"));
    await userEvent.paste("\\sin(t)/\\sin(1)");
    await userEvent.click(screen.getByRole("button", { name: "Evaluate the action" }));
    await waitFor(() => expect(variation).toHaveBeenCalledTimes(2));
    expect(createDerivation).toHaveBeenCalledOnce(); // the same derivation
    const [second, link2] = variation.mock.calls[1]!;
    expect(second.action).toBe("evaluate");
    expect(second.paths).toEqual([{ analytic: { x: "\\sin(t)/\\sin(1)" }, sampled: {} }]);
    expect(second.values).toEqual({}); // nothing the user did not type
    expect(link2).toEqual({ derivation_id: "d1", parent_id: stored[0]!.id });
    await waitFor(() =>
      expect(screen.getAllByRole("region", { name: /^Variational calculus ·/ })).toHaveLength(2),
    );
    const steps = screen.getAllByRole("listitem", { name: /^Step/ });
    expect(steps.map((s) => s.getAttribute("aria-label"))).toEqual(["Step 1", "Step 2"]);
  }, 20_000);

  it("states what is missing instead of sending an incomplete numerical step", async () => {
    mount();
    await chooseExample("Harmonic oscillator, fixed ends");
    await userEvent.click(screen.getByRole("button", { name: "Evaluate the action" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/trial path/);
    await userEvent.click(screen.getByRole("button", { name: "Verify by finite differences" }));
    expect(variation).not.toHaveBeenCalled();
    expect(createDerivation).not.toHaveBeenCalled();
  });

  it("refuses an empty integrand and a fixed condition without a value for a solve", async () => {
    mount();
    await userEvent.click(screen.getByRole("button", { name: "Analyze" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/Enter the integrand/);
    await chooseExample("Cantilever (clamped at 0, free at 1)");
    await userEvent.clear(screen.getByLabelText("Condition 1 value"));
    await userEvent.click(screen.getByRole("button", { name: "Find stationary path" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/Give the value/);
    expect(variation).not.toHaveBeenCalled();
  });

  it("sends the solve request with the second-variation option only when ticked", async () => {
    mount();
    await chooseExample("Harmonic oscillator, fixed ends");
    await userEvent.click(screen.getByLabelText(/Also classify/));
    await userEvent.click(screen.getByRole("button", { name: "Find stationary path" }));
    await waitFor(() => expect(variation).toHaveBeenCalledOnce());
    expect(variation.mock.calls[0]![0]).toMatchObject({
      action: "solve",
      with_second: true,
      paths: [],
    });
  });

  it("shows a server refusal as the message only", async () => {
    variation.mockResolvedValueOnce({ calculation_id: "x" });
    stored.push(
      step("s9", "variation", "L", {
        status: "not_verified",
        message: "No stationary path was confirmed from this starting guess.",
        action: "solve",
        analysis: null,
        first: null,
        request: firstHarmonicRequest,
      }),
    );
    mount();
    await chooseExample("Harmonic oscillator, fixed ends");
    await userEvent.click(screen.getByRole("button", { name: "First variation" }));
    expect(await screen.findByTestId("variation-message")).toHaveTextContent(
      "No stationary path was confirmed",
    );
  });

  it("restores the form from the first step and the numbers from the latest, and shows every step", async () => {
    const first = harmonic("first", firstHarmonicRequest);
    const second = harmonic("evaluate", {
      ...evaluatedRequest,
      values: { m: "2" },
    } as VariationRequest);
    const restore: VariationRestore = { nonce: 1, derivationId: "d1", steps: [first, second] };
    mount(restore);
    await waitFor(() =>
      expect(screen.getByLabelText(/Integrand L/)).toHaveValue("\\frac12\\dot{x}^2-\\frac12 x^2"),
    );
    expect(screen.getByLabelText("Unknown functions")).toHaveValue("x");
    expect(screen.getByLabelText("Path for x")).toHaveValue("\\sin(t)/\\sin(1)");
    expect(screen.getAllByRole("region", { name: /^Variational calculus ·/ })).toHaveLength(2);
    // continuing appends a step to the same derivation, after the last one
    await userEvent.click(screen.getByRole("button", { name: "Verify by finite differences" }));
    await waitFor(() => expect(variation).toHaveBeenCalledOnce());
    expect(variation.mock.calls[0]![1]).toEqual({ derivation_id: "d1", parent_id: second.id });
    expect(createDerivation).not.toHaveBeenCalled();
  });

  it("restores only the functional of a single saved calculation, with no steps", async () => {
    mount({ nonce: 2, derivationId: null, steps: [], request: firstHarmonicRequest });
    await waitFor(() =>
      expect(screen.getByLabelText(/Integrand L/)).toHaveValue("\\frac12\\dot{x}^2-\\frac12 x^2"),
    );
    expect(
      screen.queryByRole("region", { name: /^Variational calculus ·/ }),
    ).not.toBeInTheDocument();
    expect(
      within(screen.getByRole("region", { name: "Derivation" })).getByText(/Each action/),
    ).toBeInTheDocument();
  });
});
