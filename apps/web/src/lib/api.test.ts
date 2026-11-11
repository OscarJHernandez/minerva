import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, extractDetail } from "./api";

afterEach(() => vi.unstubAllGlobals());

function stubFetch(status: number, body: unknown) {
  const fn = vi
    .fn()
    .mockResolvedValue(new Response(body === undefined ? null : JSON.stringify(body), { status }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

describe("extractDetail", () => {
  it("reads string and validation-style details", () => {
    expect(extractDetail({ detail: "Not authenticated" }, 401)).toBe("Not authenticated");
    expect(
      extractDetail({ detail: [{ msg: "String should have at least 10 characters" }] }, 422),
    ).toBe("String should have at least 10 characters");
    expect(extractDetail(null, 500)).toBe("Request failed (500)");
  });
});

describe("api client", () => {
  it("sends same-origin requests to /api with JSON bodies", async () => {
    const fetchFn = stubFetch(200, { id: "1", email: "a@b.c" });
    await api.login("a@b.c", "pw");
    const [url, init] = fetchFn.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/v1/auth/login");
    expect(init.credentials).toBe("same-origin");
    expect(JSON.parse(init.body as string)).toEqual({ email: "a@b.c", password: "pw" });
  });

  it("always asks for LaTeX input on compute calls", async () => {
    const fetchFn = stubFetch(200, {});
    await api.simplify({ input: "x", assumptions: { x: ["positive"] } });
    const [, init] = fetchFn.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(init.body as string)).toMatchObject({ format: "latex", input: "x" });
  });

  it("throws ApiError with the server's message", async () => {
    stubFetch(401, { detail: "Invalid email or password." });
    await expect(api.login("a@b.c", "x")).rejects.toMatchObject({
      name: "ApiError",
      status: 401,
      message: "Invalid email or password.",
    });
    expect(new ApiError(500, "x")).toBeInstanceOf(Error);
  });

  it("treats 204 as success without a body", async () => {
    stubFetch(204, undefined);
    await expect(api.logout()).resolves.toBeUndefined();
  });
});
