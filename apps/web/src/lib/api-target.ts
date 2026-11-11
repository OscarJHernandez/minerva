/**
 * Maps a browser path `/api/v1/...` to the API URL, or returns null if it is not allowed.
 *
 * The check runs on the *normalised* target, because `new URL` resolves `..` and `%2e%2e`
 * segments: `/api/v1/../docs` would otherwise pass a prefix test on the raw path and reach
 * API endpoints outside `/v1/`.
 */
export function apiTarget(pathname: string, search: string, apiUrl: string): URL | null {
  if (!pathname.startsWith("/api/v1/")) return null;
  let target: URL;
  try {
    target = new URL(`${pathname.slice(4)}${search}`, apiUrl);
  } catch {
    return null;
  }
  if (!target.pathname.startsWith("/v1/")) return null;
  if (target.origin !== new URL(apiUrl).origin) return null;
  return target;
}
