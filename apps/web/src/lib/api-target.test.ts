import { describe, expect, it } from "vitest";
import { apiTarget } from "./api-target";

const API = "http://api:8100";

describe("apiTarget", () => {
  it("forwards /api/v1 paths with their query string", () => {
    expect(apiTarget("/api/v1/auth/me", "", API)?.href).toBe("http://api:8100/v1/auth/me");
    expect(apiTarget("/api/v1/history", "?limit=5", API)?.href).toBe(
      "http://api:8100/v1/history?limit=5",
    );
  });

  it("refuses everything outside /api/v1", () => {
    for (const p of ["/api/healthz", "/api/", "/api", "/other", "/api/v2/x"]) {
      expect(apiTarget(p, "", API)).toBeNull();
    }
  });

  it("refuses dot-segment traversal out of /v1", () => {
    for (const p of [
      "/api/v1/../docs",
      "/api/v1/../../readyz",
      "/api/v1/%2e%2e/openapi.json",
      "/api/v1/..%2fhealthz",
      "/api/v1/./../docs",
    ]) {
      const t = apiTarget(p, "", API);
      expect(t === null || t.pathname.startsWith("/v1/")).toBe(true);
    }
    expect(apiTarget("/api/v1/../docs", "", API)).toBeNull();
    expect(apiTarget("/api/v1/%2e%2e/openapi.json", "", API)).toBeNull();
  });
});
