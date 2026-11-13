import { describe, expect, it } from "vitest";
import type { Card, StoredMessage } from "./chat-types";
import { buildDebugReport } from "./debug-report";
import { diagnostics, isolateResult, stepResult } from "@/test/fixtures";

function turn(card: Card) {
  const user: StoredMessage = {
    id: "u1",
    role: "user",
    content: { text: "use my relation" },
    created_at: "2026-01-02T03:00:00Z",
  };
  const assistant: StoredMessage = {
    id: "a1",
    role: "assistant",
    created_at: "2026-01-02T03:00:05Z",
    content: {
      text: "Given your relation: $1$",
      truncated: false,
      trace: [
        {
          id: "t1",
          name: "substitute",
          arguments: { latex: "x^2+y^2", relations: ["y^2 = 1 - x^2"] },
          for_model: "status=ok; result=$1$",
          duration_ms: 200,
          status: "ok",
        },
      ],
      blocks: [{ kind: "card", segments: [], card }],
    },
  };
  return buildDebugReport({
    title: "t",
    messages: [user, assistant],
    status: { enabled: true, mode: "scripted", model: "scripted" },
    now: new Date("2026-01-02T03:04:05Z"),
  });
}

describe("debug report for controlled steps", () => {
  it("summarises a substitution with its evidence class and the diagnostics counters", () => {
    const report = turn({
      id: "t1",
      tool: "substitute",
      type: "step",
      title: "Substitute",
      data: {
        operation: "substitute",
        result: stepResult({ diagnostics: diagnostics({ rejected: { pattern_not_found: 1 } }) }),
      },
    });
    expect(report).toContain("- card: step");
    expect(report).toContain("- operation: substitute");
    expect(report).toContain("- evidence: conditional");
    expect(report).toContain("verify 2 ms");
    expect(report).toContain('rejected {"pattern_not_found":1}');
    expect(report).toContain("check congruence: passed");
  });

  it("summarises an equation by its solutions and conditions, not by a single result", () => {
    const report = turn({
      id: "t1",
      tool: "isolate_variable",
      type: "step",
      title: "Solve",
      data: { operation: "isolate", result: isolateResult() },
    });
    expect(report).toContain("- solutions: 2");
    expect(report).toContain("- conditions on all: a \\neq 0");
    expect(report).not.toContain("- evidence:");
  });
});
