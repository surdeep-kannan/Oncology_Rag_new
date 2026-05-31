"use client";

import { AnimatePresence, motion } from "framer-motion";
import { MessageSquarePlus, Pencil, Search, Trash2 } from "lucide-react";
import { formatTimestamp } from "@/utils/date";
import { cn } from "@/utils/cn";
import { useChatStore } from "@/store/chatStore";

const categoryLabels = {
  general: "General",
  diagnostics: "Diagnostics",
  treatment: "Treatment",
  urgent: "Urgent"
} as const;

export function Sidebar() {
  const {
    conversations,
    activeConversationId,
    searchTerm,
    setSearchTerm,
    createConversation,
    deleteConversation,
    renameConversation,
    setActiveConversation
  } = useChatStore();

  const filtered = conversations.filter((conversation) => {
    const text = `${conversation.title} ${conversation.messages.map((m) => m.content).join(" ")}`.toLowerCase();
    return text.includes(searchTerm.toLowerCase());
  });

  return (
    <aside className="flex h-full w-full flex-col rounded-[28px] border border-white/10 bg-slate-950/55 p-4 shadow-glow backdrop-blur-xl dark:bg-slate-950/55">
      <button
        className="mb-4 flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-teal-500 to-cyan-400 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:scale-[1.01]"
        onClick={() => createConversation()}
        type="button"
      >
        <MessageSquarePlus className="h-4 w-4" />
        New Consultation
      </button>

      <label className="mb-4 flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-300">
        <Search className="h-4 w-4" />
        <input
          className="w-full bg-transparent outline-none placeholder:text-slate-500"
          onChange={(event) => setSearchTerm(event.target.value)}
          placeholder="Search conversations"
          value={searchTerm}
        />
      </label>

      <div className="mb-3 flex items-center justify-between px-1 text-xs uppercase tracking-[0.28em] text-slate-400">
        <span>History</span>
        <span>{filtered.length}</span>
      </div>

      <div className="scrollbar-thin flex-1 space-y-2 overflow-y-auto pr-1">
        <AnimatePresence initial={false}>
          {filtered.map((conversation) => (
            <motion.button
              key={conversation.id}
              animate={{ opacity: 1, y: 0 }}
              className={cn(
                "w-full rounded-2xl border p-3 text-left transition",
                activeConversationId === conversation.id
                  ? "border-teal-400/70 bg-teal-400/10"
                  : "border-white/8 bg-white/5 hover:border-white/15 hover:bg-white/8"
              )}
              exit={{ opacity: 0, y: -4 }}
              initial={{ opacity: 0, y: 8 }}
              onClick={() => setActiveConversation(conversation.id)}
              type="button"
            >
              <div className="mb-2 flex items-start justify-between gap-2">
                <div>
                  <p className="line-clamp-2 text-sm font-medium text-white">{conversation.title}</p>
                  <p className="mt-1 text-xs text-slate-400">{categoryLabels[conversation.category]}</p>
                </div>
                <div className="flex gap-1">
                  <button
                    className="rounded-lg p-1 text-slate-400 transition hover:bg-white/10 hover:text-white"
                    onClick={(event) => {
                      event.stopPropagation();
                      const title = window.prompt("Rename conversation", conversation.title);
                      if (title?.trim()) {
                        renameConversation(conversation.id, title.trim());
                      }
                    }}
                    type="button"
                  >
                    <Pencil className="h-3.5 w-3.5" />
                  </button>
                  <button
                    className="rounded-lg p-1 text-slate-400 transition hover:bg-rose-500/10 hover:text-rose-300"
                    onClick={(event) => {
                      event.stopPropagation();
                      deleteConversation(conversation.id);
                    }}
                    type="button"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
              <p className="line-clamp-2 text-xs text-slate-400">
                {conversation.messages[conversation.messages.length - 1]?.content ?? "No messages yet"}
              </p>
              <p className="mt-2 text-[11px] text-slate-500">{formatTimestamp(conversation.updatedAt)}</p>
            </motion.button>
          ))}
        </AnimatePresence>
      </div>
    </aside>
  );
}
