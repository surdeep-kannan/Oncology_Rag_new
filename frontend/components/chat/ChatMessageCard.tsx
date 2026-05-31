"use client";

import { motion } from "framer-motion";
import { Bot, Clipboard, RefreshCcw, TriangleAlert, User } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "@/types/chat";
import { formatTimestamp } from "@/utils/date";
import { cn } from "@/utils/cn";

export function ChatMessageCard({
  message,
  onRetry,
  onRegenerate
}: {
  message: ChatMessage;
  onRetry: (text: string) => void;
  onRegenerate: () => void;
}) {
  const isAssistant = message.role === "assistant";

  return (
    <motion.article
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "rounded-[28px] border p-5 shadow-sm backdrop-blur-xl",
        isAssistant
          ? "border-white/10 bg-white/80 dark:bg-slate-950/50"
          : "border-teal-500/20 bg-teal-500/10 dark:bg-teal-500/10"
      )}
      initial={{ opacity: 0, y: 12 }}
    >
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div
            className={cn(
              "flex h-10 w-10 items-center justify-center rounded-2xl",
              isAssistant
                ? "bg-gradient-to-br from-cyan-400 to-teal-500 text-slate-950"
                : "bg-slate-900 text-white dark:bg-white dark:text-slate-900"
            )}
          >
            {isAssistant ? <Bot className="h-4 w-4" /> : <User className="h-4 w-4" />}
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-900 dark:text-white">
              {isAssistant ? "Medical AI" : "You"}
            </p>
            <p className="text-xs text-slate-500">{formatTimestamp(message.timestamp)}</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            className="rounded-xl border border-black/5 bg-black/[0.03] p-2 text-slate-500 transition hover:text-slate-900 dark:border-white/10 dark:bg-white/5 dark:text-slate-300 dark:hover:text-white"
            onClick={() => navigator.clipboard.writeText(message.content)}
            type="button"
          >
            <Clipboard className="h-4 w-4" />
          </button>
          {isAssistant && (
            <button
              className="rounded-xl border border-black/5 bg-black/[0.03] p-2 text-slate-500 transition hover:text-slate-900 dark:border-white/10 dark:bg-white/5 dark:text-slate-300 dark:hover:text-white"
              onClick={onRegenerate}
              type="button"
            >
              <RefreshCcw className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      <div className="max-w-none space-y-3 text-sm leading-7 text-slate-700 dark:text-slate-200">
        <Markdown
          components={{
            code(props) {
              const { children, className } = props;
              return (
                <code className={cn("rounded-xl bg-slate-900/90 px-1.5 py-1 text-cyan-200", className)}>
                  {children}
                </code>
              );
            },
            pre(props) {
              return (
                <pre className="overflow-x-auto rounded-2xl bg-slate-950 p-4 text-sm text-slate-100">
                  {props.children}
                </pre>
              );
            }
          }}
          remarkPlugins={[remarkGfm]}
        >
          {message.content || (message.status === "streaming" ? "Synthesizing medical response..." : "")}
        </Markdown>
      </div>

      {message.optimization?.optimized && (
        <div className="mt-4 rounded-2xl border border-teal-500/15 bg-teal-500/10 p-3 text-xs text-slate-700 dark:text-slate-200">
          <span className="font-semibold text-teal-600 dark:text-teal-300">Optimized prompt:</span>{" "}
          {message.optimization.optimized}
        </div>
      )}

      {message.status === "error" && (
        <div className="mt-4 flex items-center justify-between rounded-2xl border border-rose-500/20 bg-rose-500/10 p-3 text-sm text-rose-100">
          <span className="flex items-center gap-2">
            <TriangleAlert className="h-4 w-4" />
            {message.error ?? "Response failed"}
          </span>
          <button
            className="rounded-xl border border-rose-400/20 px-3 py-1.5 text-xs font-medium text-rose-50"
            onClick={() => onRetry(message.content)}
            type="button"
          >
            Retry
          </button>
        </div>
      )}
    </motion.article>
  );
}
