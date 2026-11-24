import { describe, expect, it } from "vitest";
import fixture from "../../../../packages/ui/fixtures/statmech-occupations-split.json";
import { occupationKey, splitOccupations } from "./statmech-occupations";

describe("splitOccupations", () => {
  it.each(fixture.cases)("$name", ({ latex, rows }) => {
    expect(splitOccupations(latex)).toEqual(rows);
  });
});

describe("occupationKey", () => {
  it("counts the repeats of the quantity, as on the server", () => {
    const qs = [{ name: "occupations" }, { name: "Z" }, { name: "occupations" }];
    expect(occupationKey(qs, 0, 2)).toBe("occupation:2");
    expect(occupationKey(qs, 2, 1)).toBe("occupation:1:2");
  });
});
