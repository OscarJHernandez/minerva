import type { NextConfig } from "next";

// Static export: the landing page is plain files served by nginx, no server, no user data.
const config: NextConfig = {
  output: "export",
  poweredByHeader: false,
  reactStrictMode: true,
  transpilePackages: ["@minerva/ui"],
  images: { unoptimized: true },
  trailingSlash: false,
};

export default config;
