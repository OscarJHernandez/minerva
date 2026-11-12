import { describe, expect, it } from "vitest";
import { checkImageFile, fitWithin, MAX_SIDE, MAX_SOURCE_BYTES } from "./image-prep";

describe("fitWithin", () => {
  it("keeps small images as they are", () => {
    expect(fitWithin(800, 600)).toEqual({ width: 800, height: 600 });
  });

  it("scales the longer side down to the limit, keeping the ratio", () => {
    expect(fitWithin(3200, 1600)).toEqual({ width: MAX_SIDE, height: 800 });
    expect(fitWithin(1000, 4000, 2000)).toEqual({ width: 500, height: 2000 });
  });

  it("never produces an empty side", () => {
    expect(fitWithin(100000, 1, 1600).height).toBe(1);
  });
});

describe("checkImageFile", () => {
  it("accepts png, jpeg and webp", () => {
    for (const type of ["image/png", "image/jpeg", "image/webp"]) {
      expect(checkImageFile({ type, size: 1000 })).toBeNull();
    }
  });

  it("rejects other types, including svg and gif", () => {
    for (const type of ["image/svg+xml", "image/gif", "application/pdf", ""]) {
      expect(checkImageFile({ type, size: 1000 })).toMatch(/PNG, JPEG or WebP/);
    }
  });

  it("rejects very large files", () => {
    expect(checkImageFile({ type: "image/png", size: MAX_SOURCE_BYTES + 1 })).toMatch(/too large/);
  });
});
