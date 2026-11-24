import { describe, expect, it } from "vitest";
import fixture from "../../../../packages/ui/fixtures/perturbation-solution-split.json";
import { splitSolution } from "./perturbation-solution";

describe("splitSolution", () => {
  it.each(fixture.cases)("$name", ({ latex, relation, definition }) => {
    const got = splitSolution(latex);
    expect(got === null ? null : got.relation).toEqual(relation);
    expect(got === null ? null : got.definition).toEqual(definition);
  });
});
