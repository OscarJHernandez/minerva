import "server-only";
import type { EntryOut, TocChapter } from "./types";

const API_URL = process.env.MINERVA_API_URL ?? "http://localhost:8100";

async function get<T>(path: string): Promise<T | null> {
  const response = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`API request failed (${response.status})`);
  return (await response.json()) as T;
}

export const reference = {
  toc: async () => (await get<TocChapter[]>("/v1/reference/toc")) ?? [],
  entries: async (params: { q?: string; chapter?: string }) => {
    const qs = new URLSearchParams();
    if (params.q) qs.set("q", params.q);
    if (params.chapter) qs.set("chapter", params.chapter);
    qs.set("limit", "100");
    return (await get<EntryOut[]>(`/v1/reference/entries?${qs}`)) ?? [];
  },
  entry: (id: string) => get<EntryOut>(`/v1/reference/entries/${encodeURIComponent(id)}`),
};
