/**
 * Where to go after logging in: the page the user was on, never anywhere else. Only a path on this site
 * is accepted (a leading "/" and no second "/" or backslash, which a browser would read as another host),
 * and the login and register pages themselves are not a destination.
 */
export function safeNext(value: string | null | undefined): string {
  if (!value || !value.startsWith("/") || value.startsWith("//") || value.includes("\\"))
    return "/";
  if (/[\u0000-\u001f]/.test(value)) return "/";
  const path = value.split(/[?#]/)[0] ?? "/";
  if (path === "/login" || path === "/register") return "/";
  return value;
}

/** The address of a conversation. */
export const conversationPath = (id: string) => `/c/${id}`;

/** A conversation id as the API makes them (a UUID); anything else in the address is ignored. */
export const isConversationId = (value: string) =>
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value);
