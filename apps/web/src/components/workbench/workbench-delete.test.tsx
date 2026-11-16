import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryItem } from "@/lib/types";
import { step } from "@/test/fixtures";
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

// A small stand-in for the server: what a delete hides and a restore brings back.
let live: HistoryItem[] = [];
let hidden: HistoryItem[] = [];
let release: (() => void) | null = null; // when set, a delete waits for it (a slow request)
let failNext = false;

const simplified = (id: string, input: string, out: string, parent: string | null) =>
  step(
    id,
    "simplify",
    input,
    { result_text: out, result_latex: out },
    { parent_id: parent, created_at: `2026-11-12T10:00:0${id.slice(1)}Z` },
  );

function below(id: string): HistoryItem[] {
  const ids = new Set([id]);
  for (let grew = true; grew;) {
    grew = false;
    for (const s of live) {
      if (s.parent_id && ids.has(s.parent_id) && !ids.has(s.id)) {
        ids.add(s.id);
        grew = true;
      }
    }
  }
  return live.filter((s) => ids.has(s.id));
}

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      inspect: vi.fn(async () => ({ status: "ok", message: "", read_as: null, converted: null })),
      createDerivation: vi.fn(async () => ({ id: "d1", updated_at: "", steps: [] })),
      derivation: vi.fn(async () => ({ id: "d1", updated_at: "", steps: [...live] })),
      derivations: vi.fn(async () => [
        {
          id: "d1",
          updated_at: "2026-11-12T10:00:00Z",
          steps: live.length,
          first_input: "x+x",
          first_format: "text",
        },
      ]),
      history: vi.fn(async () => []),
      deleteHistoryItem: vi.fn(async (id: string) => {
        if (release) await new Promise<void>((resolve) => (release = resolve));
        if (failNext) {
          failNext = false;
          throw new actual.ApiError(500, "The server could not delete the step.");
        }
        const gone = below(id);
        hidden = gone;
        live = live.filter((s) => !gone.includes(s));
        return { ids: gone.map((s) => s.id) };
      }),
      restoreHistoryItem: vi.fn(async () => {
        live = [...live, ...hidden].sort((a, b) => a.created_at.localeCompare(b.created_at));
        const ids = hidden.map((s) => s.id);
        hidden = [];
        return { ids };
      }),
      deleteDerivation: vi.fn(async () => undefined),
      restoreDerivation: vi.fn(async () => ({ ids: [] })),
      clearHistory: vi.fn(),
      identities: vi.fn(),
    },
  };
});

import { api } from "@/lib/api";

const KEY_PREFIX = "minerva:draft:";

/** Open the saved derivation so the three steps are on screen. */
async function open() {
  render(<Workbench />);
  await userEvent.click(screen.getByRole("button", { name: /^Saved/ }));
  await userEvent.click(await screen.findByRole("button", { name: /Open derivation of/ }));
  await screen.findByRole("button", { name: "Delete step 3" });
}

beforeEach(() => {
  live = [
    simplified("s1", "x+x", "2*x", null),
    simplified("s2", "2*x", "2*x", "s1"),
    simplified("s3", "2*x", "2*x", "s2"),
  ];
  hidden = [];
  release = null;
  failNext = false;
  vi.clearAllMocks();
  sessionStorage.clear();
  void KEY_PREFIX;
});

describe("Workbench: deleting steps", () => {
  it("removes the step and the steps that continued it, then offers an undo that brings them back", async () => {
    await open();
    await userEvent.click(screen.getByRole("button", { name: "Delete step 2" }));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Delete step 3" })).toBeNull());
    expect(screen.queryByRole("button", { name: "Delete step 2" })).toBeNull();
    expect(screen.getByRole("button", { name: "Delete step 1" })).toBeInTheDocument();
    expect(await screen.findByRole("status")).toHaveTextContent("Step and 1 after it deleted.");

    await userEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(api.restoreHistoryItem).toHaveBeenCalledWith("s2");
    await screen.findByRole("button", { name: "Delete step 3" });
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("changes the screen at once, before the server has answered", async () => {
    await open();
    release = () => undefined; // the next delete is slow
    await userEvent.click(screen.getByRole("button", { name: "Delete step 3" }));
    expect(screen.queryByRole("button", { name: "Delete step 3" })).toBeNull();
    await act(async () => release?.());
    await screen.findByRole("status");
  });

  it("is not undone by typing while the request is running", async () => {
    await open();
    release = () => undefined;
    await userEvent.click(screen.getByRole("button", { name: "Delete step 3" }));
    // an edit (which invalidates in-flight work) while the delete is still on its way
    fireEvent.change(screen.getByLabelText("Expression (text syntax)"), {
      target: { value: "y+y" },
    });
    await act(async () => release?.());
    await screen.findByRole("status");
    expect(screen.queryByRole("button", { name: "Delete step 3" })).toBeNull();
    expect(screen.getByRole("button", { name: "Delete step 2" })).toBeInTheDocument();
  });

  it("sends one request however often the button is pressed", async () => {
    await open();
    release = () => undefined;
    const button = screen.getByRole("button", { name: "Delete step 3" });
    fireEvent.click(button);
    fireEvent.click(button);
    fireEvent.click(button);
    await act(async () => release?.());
    expect(api.deleteHistoryItem).toHaveBeenCalledTimes(1);
  });

  it("puts the steps back and says why when the server refuses", async () => {
    await open();
    failNext = true;
    await userEvent.click(screen.getByRole("button", { name: "Delete step 2" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("could not delete the step");
    // both steps are back, in their order
    expect(screen.getByRole("button", { name: "Delete step 2" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Delete step 3" })).toBeInTheDocument();
  });

  it("stops continuing from a step that was deleted", async () => {
    await open(); // opening continues from the last step (s3)
    await userEvent.click(screen.getByRole("button", { name: "Delete step 3" }));
    await screen.findByRole("status");
    expect(screen.queryByText(/Continuing from step 3/i)).toBeNull();
  });
});

describe("Workbench: deleting a derivation", () => {
  it("deletes without a confirm dialog, resets the open derivation and offers an undo", async () => {
    const confirm = vi.spyOn(window, "confirm");
    await open();
    await userEvent.click(screen.getByRole("button", { name: "Delete this derivation" }));
    expect(api.deleteDerivation).toHaveBeenCalledWith("d1");
    expect(confirm).not.toHaveBeenCalled();
    await waitFor(() => expect(screen.queryByRole("button", { name: "Delete step 1" })).toBeNull());
    await userEvent.click(await screen.findByRole("button", { name: "Undo" }));
    expect(api.restoreDerivation).toHaveBeenCalledWith("d1");
  });

  it("shows a failed delete in every mode, not only in the expression one", async () => {
    vi.mocked(api.deleteDerivation).mockRejectedValueOnce(new Error("boom"));
    render(<Workbench />);
    await userEvent.click(screen.getByRole("button", { name: /^Saved/ }));
    const group = screen.getByRole("group", { name: "Workbench mode" });
    await userEvent.click(within(group).getByRole("button", { name: "Matrices" }));
    await userEvent.click(await screen.findByRole("button", { name: "Delete this derivation" }));
    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });
});
