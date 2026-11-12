import type { Metadata } from "next";
import "./globals.css";
import { Header } from "@/components/header";

// Pages are rendered per request so the proxy's CSP nonce reaches every script tag.
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: { default: "Minerva", template: "%s · Minerva" },
  description: "Mathematical computing workspace: checked simplification and a citable handbook.",
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-paper text-ink">
        <Header />
        <main className="mx-auto w-full max-w-6xl px-4 sm:px-6">{children}</main>
      </body>
    </html>
  );
}
