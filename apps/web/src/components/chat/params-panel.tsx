"use client";

import type { ModelParams, ParamsInfo, Toolset } from "@/lib/chat-types";
import { cleanOverrides, PARAM_LIMITS, type Overrides } from "@/lib/model-params";

const LABELS: Record<keyof ModelParams, string> = {
  num_ctx: "Context window (tokens)",
  num_predict: "Longest reply (0: model's own)",
  temperature: "Temperature",
  top_p: "Top p",
  top_k: "Top k",
  think: "Let the model think first",
  history_messages: "Messages replayed",
  history_tokens: "Replayed history (tokens)",
  toolset: "Tools sent",
};

const NUMERIC = [
  "num_ctx",
  "num_predict",
  "temperature",
  "top_p",
  "top_k",
  "history_messages",
  "history_tokens",
] as const;

/**
 * Model parameters for the next turn (debug mode only). Empty means the server's default, which is
 * shown as a hint. The server enforces the ranges and who may change what; a field this account may
 * not change is shown read-only. Nothing mathematical is here: these change how the model writes and
 * what it is shown, never what Minerva accepts as a result.
 */
export function ParamsPanel({
  info,
  value,
  onChange,
}: {
  info: ParamsInfo;
  value: Overrides;
  onChange: (next: Overrides) => void;
}) {
  const editable = new Set<string>(info.editable);
  const set = (key: keyof ModelParams, v: unknown) =>
    onChange(cleanOverrides({ ...value, [key]: v }, info.defaults));
  const used = { ...info.defaults, ...value };
  return (
    <fieldset
      className="grid gap-x-4 gap-y-2 rounded border border-rule bg-sunk p-3 text-xs sm:grid-cols-2"
      aria-label="Model parameters for the next turn"
    >
      <legend className="px-1 text-muted">
        Model parameters for the next turn · empty: server default
      </legend>
      {NUMERIC.map((key) => {
        const limit = PARAM_LIMITS[key];
        const fallback = info.defaults[key];
        const max = key === "num_ctx" ? info.num_ctx_max : limit.max;
        return (
          <label key={key} className="flex items-center justify-between gap-2">
            <span>{LABELS[key]}</span>
            <input
              type="number"
              inputMode="decimal"
              className="w-28 rounded border border-rule bg-paper px-2 py-1 text-ink disabled:opacity-60"
              min={limit.min}
              max={max}
              step={limit.step}
              disabled={!editable.has(key)}
              placeholder={fallback === null ? "model default" : String(fallback)}
              value={value[key] ?? ""}
              onChange={(e) => set(key, e.target.value === "" ? undefined : Number(e.target.value))}
            />
          </label>
        );
      })}
      <label className="flex items-center justify-between gap-2">
        <span>{LABELS.think}</span>
        <input
          type="checkbox"
          disabled={!editable.has("think")}
          checked={used.think}
          onChange={(e) => set("think", e.target.checked)}
        />
      </label>
      <label className="flex items-center justify-between gap-2">
        <span>{LABELS.toolset}</span>
        <select
          className="rounded border border-rule bg-paper px-2 py-1 text-ink disabled:opacity-60"
          disabled={!editable.has("toolset")}
          value={used.toolset}
          onChange={(e) => set("toolset", e.target.value as Toolset)}
        >
          <option value="auto">auto (core + what is needed)</option>
          <option value="core">core only</option>
          <option value="all">all tools</option>
        </select>
      </label>
      <div className="sm:col-span-2">
        <button
          type="button"
          className="text-accent hover:underline disabled:opacity-50"
          disabled={Object.keys(value).length === 0}
          onClick={() => onChange({})}
        >
          Reset to server defaults
        </button>
      </div>
    </fieldset>
  );
}
