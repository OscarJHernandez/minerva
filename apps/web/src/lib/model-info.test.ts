import { describe, expect, it } from "vitest";
import type { ModelInfo } from "./chat-types";
import { modelFacts, modelLabel, modelNotes } from "./model-info";

const gemma: ModelInfo = {
  provider: "ollama",
  alias: "minerva",
  hosted: false,
  known: true,
  base_model: "gemma4:26b",
  family: "gemma4",
  parameter_size: "25.2B",
  quantization: "Q4_K_M",
  digest: "149a0ff8a1ca",
  can_think: true,
  loaded_context: 32768,
};

describe("the model label", () => {
  it("names the weights, never the alias alone", () => {
    expect(modelLabel(gemma, "minerva")).toBe("gemma4:26b · local");
    expect(modelLabel(gemma, "minerva")).not.toBe("minerva");
  });
  it("says so when the host did not report what the model is", () => {
    const unknown: ModelInfo = {
      provider: "ollama",
      alias: "minerva",
      hosted: false,
      known: false,
    };
    expect(modelLabel(unknown)).toBe("minerva · identity not reported");
    expect(modelLabel(null, "minerva")).toBe("minerva · identity not reported");
  });
  it("marks a provider's model as hosted", () => {
    const hosted: ModelInfo = {
      provider: "openrouter",
      alias: "openai/gpt-4.1-mini",
      hosted: true,
      known: false,
    };
    expect(modelLabel(hosted)).toBe("OpenRouter · openai/gpt-4.1-mini");
  });
  it("labels the development stand-in as one", () => {
    expect(
      modelLabel({ provider: "scripted", alias: "scripted", hosted: false, known: false }),
    ).toMatch(/scripted/);
  });
});

describe("the model facts", () => {
  it("lists what the host reported and leaves out the rest", () => {
    const facts = Object.fromEntries(modelFacts(gemma));
    expect(facts["Base model"]).toBe("gemma4:26b");
    expect(facts["Digest"]).toBe("149a0ff8a1ca");
    expect(facts["Quantisation"]).toBe("Q4_K_M");
    expect(facts["Thinking"]).toBe("supported");
    expect(
      modelFacts({ provider: "ollama", alias: "m", hosted: false, known: false }).map(([k]) => k),
    ).toEqual(["Provider", "Configured name"]);
  });
  it("states how a reply was asked for", () => {
    const facts = Object.fromEntries(
      modelFacts({ ...gemma, thinking_requested: true, reply_budget: 4096 }),
    );
    expect(facts["Thinking"]).toBe("on");
    expect(facts["Reply limit"]).toBe("4096 tokens");
  });
  it("says when the model changed while the reply was written", () => {
    const notes = modelNotes({
      ...gemma,
      changed_during_reply: true,
      digest_after: "bbbbbbbbbbbb",
    });
    expect(notes.join(" ")).toMatch(/changed the model while this reply was written/);
  });
});
