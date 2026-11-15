import { describe, expect, it } from "vitest";
import type { StoredMessage } from "./chat-types";
import { retryLabel, untilLastUser, withUserDetails } from "./retry";

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

describe("withUserDetails", () => {
  it("adds the equations and references the server found to the last user message", () => {
    const all = [m("user", "a"), m("assistant", "b"), m("user", "use (1) and $$x$$")];
    const out = withUserDetails(all, {
      equations: [{ number: 2, latex: "x" }],
      references: [{ number: 1, latex: "y", source: "written_by_model" }],
    });
    expect(out[2]?.content.equations).toEqual([{ number: 2, latex: "x" }]);
    expect(out[2]?.content.references?.[0]?.number).toBe(1);
    expect(out[0]).toBe(all[0]); // nothing else is touched
    expect(out[1]).toBe(all[1]);
  });

  it("changes nothing when there is nothing to add or no user message", () => {
    const all = [m("user", "a")];
    expect(withUserDetails(all, {})[0]?.content).toEqual({ text: "a" });
    const none = [m("assistant", "b")];
    expect(withUserDetails(none, { equations: [{ number: 1, latex: "x" }] })).toBe(none);
  });
});
