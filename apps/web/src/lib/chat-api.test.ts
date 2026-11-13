import { afterEach, describe, expect, it, vi } from "vitest";
import { sendMessage } from "./chat-api";

afterEach(() => vi.restoreAllMocks());

function stream() {
  return new Response(new ReadableStream({ start: (c) => c.close() }), { status: 200 });
}

describe("sendMessage", () => {
  it("sends only the content when no provider was chosen (server default)", async () => {
    const f = vi.spyOn(globalThis, "fetch").mockResolvedValue(stream());
    await sendMessage("c1", "hi", () => undefined);
    expect(JSON.parse(String(f.mock.calls[0]![1]!.body))).toEqual({ content: "hi" });
  });

  it("sends the provider choice in the body, never in the URL", async () => {
    const f = vi.spyOn(globalThis, "fetch").mockResolvedValue(stream());
    await sendMessage("c1", "hi", () => undefined, undefined, {
      kind: "openrouter",
      model: "a/b",
      api_key: "sk-or-secret",
      cloud_consent: true,
    });
    expect(String(f.mock.calls[0]![0])).not.toContain("sk-or-secret");
    expect(JSON.parse(String(f.mock.calls[0]![1]!.body)).provider.api_key).toBe("sk-or-secret");
  });
});
