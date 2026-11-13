import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatStatus } from "@/lib/chat-types";
import { modelSettings, providerChoice } from "@/lib/model-settings";
import { ModelPicker } from "./model-picker";
import { ModelSettingsPanel } from "./model-settings-panel";

const status = (over: Partial<ChatStatus["providers"] & object> = {}): ChatStatus => ({
  enabled: true,
  mode: "ollama",
  model: "minerva",
  providers: {
    default: "ollama",
    ollama: { available: true, model: "minerva", reachable: true },
    openrouter: {
      available: true,
      byok: true,
      server_key_for_you: false,
      default_model: "openai/gpt-4.1-mini",
      models: ["openai/gpt-4.1-mini", "google/gemini-2.5-flash"],
      zdr: false,
      data_collection: "deny",
    },
    ...over,
  },
});

beforeEach(() => modelSettings.clear());
afterEach(() => vi.restoreAllMocks());

describe("model settings", () => {
  it("uses the server default until the user picks something", () => {
    expect(providerChoice(modelSettings.get())).toBeUndefined();
    modelSettings.useLocal();
    expect(providerChoice(modelSettings.get())).toEqual({ kind: "ollama" });
  });

  it("will not use OpenRouter without a key, a model and consent", () => {
    render(<ModelSettingsPanel status={status()} onClose={() => undefined} />);
    const use = screen.getByRole("button", { name: "Use OpenRouter" });
    expect(use).toBeDisabled();
    fireEvent.change(screen.getByLabelText(/API key/), { target: { value: "sk-or-abc" } });
    expect(use).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox"));
    expect(use).toBeEnabled();
    fireEvent.click(use);
    expect(providerChoice(modelSettings.get())).toEqual({
      kind: "openrouter",
      model: "openai/gpt-4.1-mini",
      api_key: "sk-or-abc",
      cloud_consent: true,
    });
  });

  it("switches back to the local model in one click", () => {
    modelSettings.useCloud("openai/gpt-4.1-mini");
    render(<ModelSettingsPanel status={status()} onClose={() => undefined} />);
    fireEvent.click(screen.getByRole("button", { name: "Use local model" }));
    expect(modelSettings.get().provider).toBe("ollama");
  });

  it("rejects a malformed custom model id", () => {
    render(<ModelSettingsPanel status={status()} onClose={() => undefined} />);
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "__custom__" } });
    fireEvent.change(screen.getByLabelText(/Custom model id/), { target: { value: "bad id;" } });
    fireEvent.change(screen.getByLabelText(/API key/), { target: { value: "k" } });
    fireEvent.click(screen.getByRole("checkbox"));
    expect(screen.getByRole("button", { name: "Use OpenRouter" })).toBeDisabled();
  });

  it("keeps the key and consent out of every browser storage", () => {
    const local = vi.spyOn(Storage.prototype, "setItem");
    render(<ModelSettingsPanel status={status()} onClose={() => undefined} />);
    fireEvent.change(screen.getByLabelText(/API key/), { target: { value: "sk-or-secret" } });
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(screen.getByRole("button", { name: "Use OpenRouter" }));
    const written = local.mock.calls.map((c) => c.join("="));
    expect(written.join()).not.toContain("sk-or-secret");
    expect(document.cookie).not.toContain("sk-or-secret");
    modelSettings.clear();
    expect(modelSettings.get().apiKey).toBe("");
  });

  it("tests the connection and shows the result", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ ok: true, message: "Connected.", credit_remaining: 3 }), {
        status: 200,
      }),
    );
    render(<ModelSettingsPanel status={status()} onClose={() => undefined} />);
    fireEvent.change(screen.getByLabelText(/API key/), { target: { value: "sk-or-abc" } });
    fireEvent.click(screen.getByRole("button", { name: "Test connection" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Connected."));
  });

  it("offers no OpenRouter when the server has not enabled it", () => {
    const s = status();
    s.providers!.openrouter.available = false;
    render(<ModelSettingsPanel status={s} onClose={() => undefined} />);
    expect(screen.queryByText("OpenRouter")).not.toBeInTheDocument();
    expect(screen.getByText(/not enabled on this server/)).toBeInTheDocument();
  });

  it("the picker chip names the provider and model", () => {
    render(<ModelPicker status={status()} open={false} onToggle={() => undefined} />);
    expect(screen.getByRole("button")).toHaveTextContent("Local · minerva");
  });
});
