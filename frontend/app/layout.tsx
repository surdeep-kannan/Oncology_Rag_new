import type { Metadata } from "next";
import "./globals.css";
import { AppProviders } from "@/components/providers/AppProviders";
import { ThemeSync } from "@/components/layout/ThemeSync";

export const metadata: Metadata = {
  title: "MedAxis AI",
  description: "Premium medical LLM interface with prompt optimization and streaming chat."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <AppProviders>
          <ThemeSync />
          {children}
        </AppProviders>
      </body>
    </html>
  );
}
