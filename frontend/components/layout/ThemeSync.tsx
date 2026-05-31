"use client";

import { useEffect } from "react";
import { useChatStore } from "@/store/chatStore";

export function ThemeSync() {
  const theme = useChatStore((state) => state.settings.theme);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  return null;
}
