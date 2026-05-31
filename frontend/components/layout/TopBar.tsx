"use client";

import { MoonStar, Settings2, SunMedium, UserRound } from "lucide-react";
import { APP_NAME } from "@/lib/constants";
import { useChatStore } from "@/store/chatStore";

export function TopBar({ onOpenSettings }: { onOpenSettings: () => void }) {
  const { settings, updateSettings } = useChatStore();

  return (
    <header className="flex items-center justify-between rounded-[28px] border border-white/10 bg-white/70 px-5 py-4 shadow-glow backdrop-blur-xl dark:bg-slate-950/55">
      <div>
        <p className="text-xs uppercase tracking-[0.32em] text-teal-500">Medical Copilot</p>
        <h1 className="text-xl font-semibold text-slate-900 dark:text-white">{APP_NAME}</h1>
      </div>

      <div className="flex items-center gap-3">
        <button
          className="rounded-2xl border border-black/5 bg-black/[0.03] p-3 text-slate-700 transition hover:scale-[1.02] dark:border-white/10 dark:bg-white/5 dark:text-slate-200"
          onClick={() =>
            updateSettings({
              theme: settings.theme === "dark" ? "light" : "dark"
            })
          }
          type="button"
        >
          {settings.theme === "dark" ? <SunMedium className="h-4 w-4" /> : <MoonStar className="h-4 w-4" />}
        </button>
        <button
          className="rounded-2xl border border-black/5 bg-black/[0.03] p-3 text-slate-700 transition hover:scale-[1.02] dark:border-white/10 dark:bg-white/5 dark:text-slate-200"
          onClick={onOpenSettings}
          type="button"
        >
          <Settings2 className="h-4 w-4" />
        </button>
        <div className="flex items-center gap-3 rounded-2xl border border-black/5 bg-black/[0.03] px-4 py-2 dark:border-white/10 dark:bg-white/5">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-teal-400 to-cyan-500 text-slate-950">
            <UserRound className="h-4 w-4" />
          </div>
          <div className="hidden sm:block">
            <p className="text-sm font-medium text-slate-900 dark:text-white">Clinical Operator</p>
            <p className="text-xs text-slate-500">Session secured</p>
          </div>
        </div>
      </div>
    </header>
  );
}
