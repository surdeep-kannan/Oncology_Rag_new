"use client";

import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { useChatStore } from "@/store/chatStore";
import type { ChatSettings } from "@/types/chat";

export function SettingsModal({
  open,
  onClose
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { settings, updateSettings } = useChatStore();
  const toggles: Array<{
    key: keyof Pick<ChatSettings, "optimizePrompts" | "showOptimizationPanel" | "streamResponses">;
    title: string;
    description: string;
  }> = [
    {
      key: "optimizePrompts",
      title: "Prompt optimization",
      description: "Expand layman queries into clinical, medically normalized prompts."
    },
    {
      key: "showOptimizationPanel",
      title: "Show optimization panel",
      description: "Display original prompt, optimized prompt, confidence, and warnings."
    },
    {
      key: "streamResponses",
      title: "Stream responses",
      description: "Render answer tokens progressively as the backend returns them."
    }
  ];

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          animate={{ opacity: 1 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-4"
          initial={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            animate={{ opacity: 1, y: 0, scale: 1 }}
            className="w-full max-w-xl rounded-[32px] border border-white/10 bg-white/90 p-6 shadow-2xl backdrop-blur-2xl dark:bg-slate-950/90"
            initial={{ opacity: 0, y: 20, scale: 0.96 }}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="mb-5 flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.28em] text-teal-500">Settings</p>
                <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Experience controls</h2>
              </div>
              <button
                className="rounded-2xl border border-black/5 bg-black/[0.03] p-2 dark:border-white/10 dark:bg-white/5"
                onClick={onClose}
                type="button"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="space-y-4">
              {toggles.map((item) => (
                <label
                  key={item.key}
                  className="flex items-center justify-between rounded-3xl border border-black/5 bg-black/[0.03] p-4 dark:border-white/10 dark:bg-white/5"
                >
                  <div className="pr-4">
                    <p className="font-medium text-slate-900 dark:text-white">{item.title}</p>
                    <p className="text-sm text-slate-500">{item.description}</p>
                  </div>
                  <input
                    checked={settings[item.key]}
                    className="h-5 w-5 accent-teal-500"
                    onChange={(event) =>
                      updateSettings({
                        [item.key]: event.target.checked
                      })
                    }
                    type="checkbox"
                  />
                </label>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
