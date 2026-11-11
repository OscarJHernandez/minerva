import { NextResponse, type NextRequest } from "next/server";

const SESSION_COOKIE = "minerva_session";
const API_URL = () => process.env.MINERVA_API_URL ?? "http://localhost:8100";

/**
 * Next.js 16 proxy (formerly middleware):
 *  - `/api/v1/*` is forwarded to the FastAPI service, so the browser only ever talks to one
 *    origin and the httpOnly session cookie stays same-origin. Nothing else is forwarded.
 *  - The workspace (`/`) redirects to `/login` when no session cookie is present. This is a
 *    convenience only; the API checks the session on every call and is the real authority.
 *  - Every HTML response gets a nonce-based Content-Security-Policy.
 */
export function proxy(request: NextRequest) {
  const { pathname, search } = request.nextUrl;

  if (pathname.startsWith("/api/")) {
    if (!pathname.startsWith("/api/v1/")) return new NextResponse(null, { status: 404 });
    return NextResponse.rewrite(new URL(`${pathname.slice(4)}${search}`, API_URL()));
  }

  if (pathname === "/" && !request.cookies.has(SESSION_COOKIE)) {
    return NextResponse.redirect(new URL("/login", request.url));
  }

  const nonce = btoa(crypto.randomUUID());
  const dev = process.env.NODE_ENV !== "production";
  const csp = [
    "default-src 'self'",
    `script-src 'self' 'nonce-${nonce}' 'strict-dynamic'${dev ? " 'unsafe-eval'" : ""}`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    "connect-src 'self'",
    "worker-src 'self' blob:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ].join("; ");

  const headers = new Headers(request.headers);
  headers.set("x-nonce", nonce);
  headers.set("Content-Security-Policy", csp);
  const response = NextResponse.next({ request: { headers } });
  response.headers.set("Content-Security-Policy", csp);
  return response;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|icon.svg|robots.txt).*)"],
};
