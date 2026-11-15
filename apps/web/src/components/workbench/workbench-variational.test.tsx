import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { step } from "@/test/fixtures";
import {
  evaluated,
  evaluatedRequest,
  firstHarmonic,
  firstHarmonicRequest,
} from "@/test/variation-fixtures";
import { Workbench } from "./workbench";

const router = { push: vi.fn() };
vi.mock("next/navigation", () => ({ useRouter: () => router }));
vi.mock("../math-input", () => ({
  MathInput: () => <textarea aria-label="Mathematical expression" />,
}));

const steps = [
  step(
    "v1",
    "variation",
    firstHarmonicRequest.integrand,
    { ...firstHarmonic, request: firstHarmonicRequest },
    { format: "latex", derivation_id: "dv" },
  ),
  step(
    "v2",
    "variation",
    firstHarmonicRequest.integrand,
    { ...evaluated, request: evaluatedRequest },
    { format: "latex", derivation_id: "dv", parent_id: "v1" },
  ),
];
const single = step(
  "v3",
  "variation",
  firstHarmonicRequest.integrand,
  { ...firstHarmonic, request: firstHarmonicRequest },
  { format: "latex", derivation_id: null },
);

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      createDerivation: vi.fn(),
      derivation: vi.fn(async () => ({ id: "dv", updated_at: "", steps })),
      derivations: vi.fn(async () => [
        {
          id: "dv",
          updated_at: "2026-11-12T10:00:00Z",
          steps: 2,
          first_input: "\\frac12\\dot{x}^2-\\frac12 x^2",
          first_format: "latex",
        },
      ]),
      history: vi.fn(async () => [single]),
      deleteHistoryItem: vi.fn(),
      deleteDerivation: vi.fn(),
      clearHistory: vi.fn(),
    },
  };
});

beforeEach(() => router.push.mockReset());

describe("Workbench: variational mode", () => {
  it("offers the mode", async () => {
    render(<Workbench />);
    const group = screen.getByRole("group", { name: "Workbench mode" });
    await userEvent.click(within(group).getByRole("button", { name: "Variational calculus" }));
    expect(screen.getByTestId("variational-workbench")).toBeInTheDocument();
  });

  it("reopens a saved variational derivation in its own mode with the form and every step", async () => {
    render(<Workbench />);
    await userEvent.click(screen.getByRole("button", { name: /^Saved/ }));
    await userEvent.click(
      await screen.findByRole("button", { name: /Open derivation of 2 steps/ }),
    );
    await waitFor(() => expect(screen.getByTestId("variational-workbench")).toBeInTheDocument());
    await waitFor(() =>
      expect(screen.getByLabelText(/Integrand L/)).toHaveValue("\\frac12\\dot{x}^2-\\frac12 x^2"),
    );
    expect(screen.getByLabelText("Path for x")).toHaveValue("\\sin(t)/\\sin(1)");
    expect(screen.getAllByRole("region", { name: /^Variational calculus ·/ })).toHaveLength(2);
  });

  it("reopens a single saved calculation by restoring its form", async () => {
    render(<Workbench />);
    await userEvent.click(screen.getByRole("button", { name: /^Saved/ }));
    await screen.findByRole("heading", { name: "History" });
    await userEvent.click(screen.getByRole("button", { name: /^Open variation:/ }));
    await waitFor(() =>
      expect(screen.getByLabelText(/Integrand L/)).toHaveValue("\\frac12\\dot{x}^2-\\frac12 x^2"),
    );
    expect(
      screen.queryByRole("region", { name: /^Variational calculus ·/ }),
    ).not.toBeInTheDocument();
  });
});
