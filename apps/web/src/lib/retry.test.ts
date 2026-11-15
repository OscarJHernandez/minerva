import { describe, expect, it } from "vitest";
import type { StoredMessage } from "./chat-types";
import { retryLabel, untilLastUser } from "./retry";

const m = (role: "user" | "assistant", text: string): StoredMessage => ({
  id: text,
  role,
  created_at: "2026-11-15T00:00:00Z",
  content: { text },
});

describe("untilLastUser", () => {
  it("drops the reply after the last user message", () => {
    const all = [m("user", "a"), m("assistant", "b"), m("user", "c"), m("assistant", "d")];
    expect(untilLastUser(all).map((x) => x.content.text)).toEqual(["a", "b", "c"]);
  });

  it("replaces the text of that message when it is being edited", () => {
    const all = [m("user", "a"), m("assistant", "b"), m("user", "c"), m("assistant", "d")];
    const edited = untilLastUser(all, "changed");
    expect(edited.map((x) => x.content.text)).toEqual(["a", "b", "changed"]);
    expect(edited[2]?.role).toBe("user");
  });

  it("leaves a conversation with no user message alone", () => {
    const only = [m("assistant", "b")];
    expect(untilLastUser(only, "x")).toBe(only);
  });
});

describe("retryLabel", () => {
  it("says Retry for a reply that ended early and Regenerate for a finished one", () => {
    for (const reason of ["model_error", "time_limit", "stopped", "malformed_output"]) {
      expect(retryLabel(reason)).toBe("Retry");
    }
    expect(retryLabel("completed")).toBe("Regenerate");
    expect(retryLabel(undefined)).toBe("Regenerate");
  });
});
