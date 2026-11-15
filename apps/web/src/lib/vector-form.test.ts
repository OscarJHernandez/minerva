import { describe, expect, it } from "vitest";
import { blankForm, buildVectorRequest, restoreVectorForm } from "./vector-form";

describe("buildVectorRequest", () => {
  it("rejects wrong component counts, repeated names and coordinate names", () => {
    const form = blankForm();
    form.expression = "curl(E)";
    form.fields[0]!.components = "1, 2";
    expect(buildVectorRequest(form)).toMatchObject({ ok: false });
    form.fields[0]!.components = "";
    form.fields[1]!.name = "E";
    expect(buildVectorRequest(form)).toMatchObject({ ok: false, error: /twice/ });
    form.fields[1]!.name = "x";
    expect(buildVectorRequest(form)).toMatchObject({ ok: false, error: /coordinate/ });
  });

  it("requires equations to have two sides and an identity name", () => {
    const form = blankForm();
    form.expression = "curl(E)";
    form.premises = "curl(E)";
    expect(buildVectorRequest(form)).toMatchObject({ ok: false });
    form.premises = "";
    form.action = "apply_identity";
    expect(buildVectorRequest(form)).toMatchObject({ ok: false, error: /identity/ });
  });

  it("builds an electromagnetic derivation from the template and never declares smoothness itself", () => {
    const form = blankForm();
    form.action = "em_derive";
    const built = buildVectorRequest(form);
    expect(built.ok && built.request.fields.map((f) => f.smoothness)).toEqual([null, null]);
    form.emSmoothness = "c2";
    const withSmooth = buildVectorRequest(form);
    expect(withSmooth.ok && withSmooth.request.fields.map((f) => f.smoothness)).toEqual([
      "c2",
      "c2",
    ]);
  });

  it("restores a saved request into the form", () => {
    const form = blankForm();
    form.expression = "curl(E)";
    form.fields[0]!.smoothness = "c2";
    form.premises = "div(E) = 0\nB = curl(A)";
    form.action = "substitute";
    const built = buildVectorRequest(form);
    if (!built.ok) throw new Error(built.error);
    const back = restoreVectorForm(built.request, "curl(E)", "text", {});
    expect(back.fields[0]!.smoothness).toBe("c2");
    expect(back.premiseIds).toEqual(["em.maxwell.gauss_vacuum"]); // a handbook equation: a ticked box
    expect(back.premises).toBe("B = curl(A)");
    expect(back.action).toBe("substitute");
  });
});
