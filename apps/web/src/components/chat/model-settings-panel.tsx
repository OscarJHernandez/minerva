"use client";

import { useState } from "react";
import { ApiError } from "@/lib/api";
import { chatApi } from "@/lib/chat-api";
import type { ChatStatus, ConnectionResult } from "@/lib/chat-types";
import { isValidModelId, modelSettings, useModelSettings } from "@/lib/model-settings";

const CUSTOM = "__custom__";

/** Provider and model for this tab: the server's local model, or OpenRouter with the user's own key. */
export function ModelSettingsPanel({
  status,
  onClose,
}: {
  status: ChatStatus;
  onClose: () => void;
}) {
  const settings = useModelSettings();
  const providers = status.providers;
  const or = providers?.openrouter;
  const local = providers?.ollama;
  const models = or?.models ?? [];
  const initialModel = settings.model || modelSettings.savedModel() || or?.default_model || "";
  const [choice, setChoice] = useState(
    models.includes(initialModel) || !initialModel ? initialModel || (models[0] ?? "") : CUSTOM,
  );
  const [custom, setCustom] = useState(models.includes(initialModel) ? "" : initialModel);
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<ConnectionResult | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  const model = choice === CUSTOM ? custom.trim() : choice;
  const modelOk = isValidModelId(model);
  const hasKey = settings.apiKey.trim().length > 0 || (or?.server_key_for_you ?? false);
  const cloudActive = settings.touched
    ? settings.provider === "openrouter"
    : providers?.default === "openrouter";

  async function test() {
    setTesting(true);
    setResult(null);
    setProblem(null);
    try {
      setResult(await chatApi.testOpenRouter(settings.apiKey.trim() || undefined, model));
    } catch (e) {
      setProblem(e instanceof ApiError ? e.message : "The test could not run. Try again.");
    } finally {
      setTesting(false);
    }
  }

  return (
    <div
      className="mb-2 max-h-[60vh] overflow-y-auto rounded-lg border border-rule bg-raised p-3 text-sm shadow-sm"
      role="dialog"
      aria-label="Model settings"
    >
      <div className="mb-2 flex items-center justify-between">
        <h2 className="font-medium">Model settings</h2>
        <button type="button" onClick={onClose} className="text-xs text-muted hover:text-ink">
          Close
        </button>
      </div>

      <section aria-label="Local model" className="rounded border border-rule p-2.5">
        <div className="flex items-center justify-between gap-2">
          <div>
            <div className="font-medium">Local (Ollama)</div>
            <div className="text-xs text-muted">
              {local?.model || "Server model"} ·{" "}
              {local?.reachable === false ? (
                <span className="text-bad">not reachable</span>
              ) : (
                <span className="text-ok">connected</span>
              )}
            </div>
          </div>
          {cloudActive ? (
            <button
              type="button"
              onClick={() => modelSettings.useLocal()}
              disabled={!local?.available}
              className="shrink-0 rounded border border-rule px-3 py-1.5 hover:bg-sunk disabled:opacity-40"
            >
              Use local model
            </button>
          ) : (
            <span className="text-xs text-ok">In use</span>
          )}
        </div>
        <p className="mt-1 text-xs text-muted">
          Runs on the server owner&apos;s own machine. Nothing leaves it.
        </p>
      </section>

      {or?.available ? (
        <section aria-label="OpenRouter" className="mt-2 rounded border border-rule p-2.5">
          <div className="flex items-center justify-between">
            <div className="font-medium">OpenRouter</div>
            {cloudActive && <span className="text-xs text-ok">In use</span>}
          </div>
          {or.byok && (
            <label className="mt-2 block">
              <span className="text-xs text-muted">API key (kept in this tab only)</span>
              <input
                type="password"
                autoComplete="off"
                spellCheck={false}
                value={settings.apiKey}
                onChange={(e) => modelSettings.setApiKey(e.target.value)}
                placeholder={
                  or.server_key_for_you ? "Optional: you may use the server's key" : "sk-or-…"
                }
                className="mt-0.5 block w-full rounded border border-rule bg-paper px-2 py-1.5"
              />
            </label>
          )}
          <label className="mt-2 block">
            <span className="text-xs text-muted">Model</span>
            <select
              value={choice}
              onChange={(e) => setChoice(e.target.value)}
              className="mt-0.5 block w-full rounded border border-rule bg-paper px-2 py-1.5"
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
              <option value={CUSTOM}>Custom model id…</option>
            </select>
          </label>
          {choice === CUSTOM && (
            <label className="mt-2 block">
              <span className="text-xs text-muted">
                Custom model id (must support tool calling)
              </span>
              <input
                type="text"
                value={custom}
                onChange={(e) => setCustom(e.target.value)}
                placeholder="vendor/model-name"
                aria-invalid={custom !== "" && !modelOk}
                className="mt-0.5 block w-full rounded border border-rule bg-paper px-2 py-1.5"
              />
            </label>
          )}
          <p className="mt-2 rounded bg-warn-soft px-2 py-1.5 text-xs text-warn">
            Using OpenRouter sends this conversation&apos;s relevant history and Minerva&apos;s tool
            results to OpenRouter and the company serving the model. It is billed to the key owner.
            {or.zdr
              ? " This server asks for zero-data-retention routing, which reduces retention but is not the same as local processing."
              : " Data-collection restrictions reduce retention but do not make this equivalent to local processing."}
          </p>
          <label className="mt-2 flex items-start gap-2 text-xs">
            <input
              type="checkbox"
              checked={settings.consent}
              onChange={(e) => modelSettings.setConsent(e.target.checked)}
              className="mt-0.5"
            />
            I understand and want to use OpenRouter in this tab.
          </label>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={test}
              disabled={testing || !modelOk || !hasKey}
              className="rounded border border-rule px-3 py-1.5 hover:bg-sunk disabled:opacity-40"
            >
              {testing ? "Testing…" : "Test connection"}
            </button>
            <button
              type="button"
              onClick={() => {
                modelSettings.useCloud(model);
                onClose();
              }}
              disabled={!modelOk || !hasKey || !settings.consent}
              className="rounded bg-accent px-3 py-1.5 font-medium text-accent-ink disabled:opacity-40"
            >
              Use OpenRouter
            </button>
          </div>
          {result && (
            <p role="status" className={`mt-2 text-xs ${result.ok ? "text-ok" : "text-bad"}`}>
              {result.message}
              {result.ok && typeof result.credit_remaining === "number"
                ? ` Credit left: $${result.credit_remaining.toFixed(2)}.`
                : ""}
            </p>
          )}
          {problem && (
            <p role="alert" className="mt-2 text-xs text-bad">
              {problem}
            </p>
          )}
        </section>
      ) : (
        <p className="mt-2 text-xs text-muted">
          OpenRouter is not enabled on this server. The local model answers.
        </p>
      )}
    </div>
  );
}
