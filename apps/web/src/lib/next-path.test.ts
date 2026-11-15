import { describe, expect, it } from "vitest";
import { conversationPath, isConversationId, safeNext } from "./next-path";

describe("safeNext", () => {
  it("accepts a path on this site, with its query", () => {
    expect(safeNext("/c/0d5c3a3e-6f50-4f43-8c1f-2b0c3d0e1a2b")).toBe(
      "/c/0d5c3a3e-6f50-4f43-8c1f-2b0c3d0e1a2b",
    );
    expect(safeNext("/workbench?mode=matrix")).toBe("/workbench?mode=matrix");
  });

  it("refuses anything that could leave the site", () => {
    for (const bad of [
      "https://evil.test/",
      "//evil.test/x",
      "/\\evil.test",
      "javascript:alert(1)",
      "evil",
      "",
      "/a\nb",
    ]) {
      expect(safeNext(bad)).toBe("/");
    }
    expect(safeNext(null)).toBe("/");
    expect(safeNext(undefined)).toBe("/");
  });

  it("never sends the user back to the login page", () => {
    expect(safeNext("/login")).toBe("/");
    expect(safeNext("/register?x=1")).toBe("/");
  });
});

describe("conversation addresses", () => {
  it("builds and recognises them", () => {
    const id = "0d5c3a3e-6f50-4f43-8c1f-2b0c3d0e1a2b";
    expect(conversationPath(id)).toBe(`/c/${id}`);
    expect(isConversationId(id)).toBe(true);
    expect(isConversationId("not-an-id")).toBe(false);
    expect(isConversationId("../../etc/passwd")).toBe(false);
  });
});
