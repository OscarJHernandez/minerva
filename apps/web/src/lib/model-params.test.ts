import { describe, expect, it } from "vitest";
import type { ModelParams, TurnDebug } from "./chat-types";
import { cleanOverrides, reachedWindow, settingsSummary } from "./model-params";

const defaults: ModelParams = {
  num_ctx: 32768,
  num_predict: null,
  temperature: 0.2,
  top_p: null,
  top_k: null,
  think: false,
  history_messages: 20,
  history_tokens: null,
  toolset: "auto",
};

describe("cleanOverrides", () => {
  it("keeps only what differs from the server default", () => {
    expect(cleanOverrides({ temperature: 0.7, num_ctx: 32768, toolset: "auto" }, defaults)).toEqual(
      { temperature: 0.7 },
    );
  });

  it("drops cleared and invalid fields", () => {
    expect(
      cleanOverrides({ temperature: undefined, top_k: Number.NaN, think: true }, defaults),
    ).toEqual({ think: true });
  });
});

function debug(over: Partial<TurnDebug>): TurnDebug {
  return {
    request_id: "r",
    api_version: "1",
    mode: "ollama",
    model: "m",
    prompt_sha: "abc",
    max_steps: 6,
    max_seconds: 180,
    duration_ms: 10,
    llm_calls: [],
    claims: [],
    error: null,
    ...over,
  };
}

describe("settingsSummary", () => {
  it("says nothing for a turn stored before settings were recorded", () => {
    expect(settingsSummary(debug({}))).toEqual([]);
    expect(settingsSummary(null)).toEqual([]);
  });

  it("lists the parameters, the groups and the request against the window", () => {
    const lines = settingsSummary(
      debug({
        params: { num_ctx: 32768, num_predict: null, temperature: 0.2 },
        groups: ["matrix"],
        context_window: 32768,
        prompt_estimate_tokens: 6000,
        llm_calls: [{ ms: 1, text_chars: 0, tool_calls: 0, prompt_tokens: 6175 }],
        history_dropped: 2,
      }),
    );
    expect(lines[0]).toContain("num_predict model default");
    expect(lines).toContain("tool groups loaded: matrix");
    expect(lines.join("\n")).toContain("host counted 6175");
    expect(lines.join("\n")).toContain("2 earlier message(s) left out");
    expect(lines.join("\n")).not.toContain("AT THE WINDOW");
  });

  it("flags a prompt that reached the window", () => {
    const lines = settingsSummary(
      debug({
        context_window: 8192,
        llm_calls: [{ ms: 1, text_chars: 0, tool_calls: 0, prompt_tokens: 8192 }],
      }),
    );
    expect(lines.join("\n")).toContain("AT THE WINDOW");
  });
});

describe("reachedWindow", () => {
  it("recognises a prompt the host cut to about half the window", () => {
    expect(reachedWindow(8195, 16384)).toBe(true);
    expect(reachedWindow(16387, 32768)).toBe(true);
    expect(reachedWindow(16384, 16384)).toBe(true);
    expect(reachedWindow(6175, 32768)).toBe(false);
  });
});
