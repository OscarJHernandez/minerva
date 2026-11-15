import type { NextConfig } from "next";
import path from "node:path";

const config: NextConfig = {
  output: "standalone",
  // Monorepo: trace dependencies from the repository root so the standalone build is complete.
  outputFileTracingRoot: path.join(import.meta.dirname, "../.."),
  poweredByHeader: false,
  // The chat reply is a server-sent stream proxied from the API. Next compresses proxied responses, which
  // holds the stream back and delivers the whole reply in one piece at the end (no live text, and Stop
  // keeps nothing). Compression is left to what sits in front (Cloudflare, nginx), which passes event
  // streams through as they come.
  compress: false,
  reactStrictMode: true,
  transpilePackages: ["@minerva/ui"],
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        ],
      },
    ];
  },
};

export default config;
