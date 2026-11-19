import { NextRequest } from "next/server";
import { describe, expect, it } from "vitest";
import { proxy } from "./proxy";

const asked = (path: string, cookie = false) =>
  proxy(
    new NextRequest(`http://localhost:3000${path}`, {
      headers: cookie ? { cookie: "minerva_session=abc" } : {},
    }),
  );

describe("proxy: where a logged-out visitor is sent", () => {
  it("keeps an exploration through login", () => {
    const response = asked("/explore/schwarzschild-curvature");
    expect(response.status).toBe(307);
    const target = new URL(response.headers.get("location")!);
    expect(target.pathname).toBe("/login");
    expect(target.searchParams.get("next")).toBe("/explore/schwarzschild-curvature");
  });

  it("keeps the query of the home page, and adds none to a bare visit", () => {
    const withQuery = new URL(asked("/?debug=1").headers.get("location")!);
    expect(withQuery.searchParams.get("next")).toBe("/?debug=1");
    const bare = new URL(asked("/").headers.get("location")!);
    expect(bare.pathname).toBe("/login");
    expect(bare.searchParams.has("next")).toBe(false);
  });

  it("lets a logged-in visitor reach the exploration", () => {
    const response = asked("/explore/schwarzschild-curvature", true);
    expect(response.headers.get("location")).toBeNull();
    expect(response.headers.get("Content-Security-Policy")).toContain("default-src 'self'");
  });
});
