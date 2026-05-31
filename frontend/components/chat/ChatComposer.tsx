"use client";

import { motion } from "framer-motion";
import { useEffect, useRef } from "react";
import { Mic, Paperclip, SendHorizonal, Square } from "lucide-react";
import type { FileAttachment } from "@/types/chat";

export function ChatComposer({
  value,
  onChange,
  attachments,
  onAttachments,
  onSend,
  onStop,
  isGenerating
}: {
  value: string;
  onChange: (value: string) => void;
  attachments: FileAttachment[];
  onAttachments: (items: FileAttachment[]) => void;
  onSend: () => void;
  onStop: () => void;
  isGenerating: boolean;
}) {
  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!textAreaRef.current) {
      return;
    }
    textAreaRef.current.style.height = "0px";
    textAreaRef.current.style.height = `${textAreaRef.current.scrollHeight}px`;
  }, [value]);

  const ingestFiles = (files: FileList | null) => {
    if (!files) {
      return;
    }
    const nextAttachments = Array.from(files).map((file) => ({
      id: `${file.name}-${file.lastModified}`,
      name: file.name,
      size: file.size,
      type: file.type
    }));
    onAttachments(nextAttachments);
  };

  return (
    <section className="rounded-[28px] border border-white/10 bg-white/70 p-4 shadow-glow backdrop-blur-xl dark:bg-slate-950/55">
      <div
        className="rounded-[24px] border border-dashed border-black/10 p-3 transition hover:border-teal-400/40 dark:border-white/10"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          ingestFiles(event.dataTransfer.files);
        }}
      >
        <textarea
          ref={textAreaRef}
          className="min-h-[120px] w-full resize-none bg-transparent text-base text-slate-900 outline-none placeholder:text-slate-500 dark:text-white"
          onChange={(event) => onChange(event.target.value)}
          placeholder="Describe symptoms, upload supporting records, or ask for guideline-grounded medical synthesis."
          value={value}
        />

        {attachments.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {attachments.map((attachment) => (
              <span
                key={attachment.id}
                className="rounded-2xl border border-black/5 bg-black/[0.03] px-3 py-2 text-xs text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-slate-200"
              >
                {attachment.name}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <label className="cursor-pointer rounded-2xl border border-black/5 bg-black/[0.03] px-3 py-2 text-sm text-slate-700 dark:border-white/10 dark:bg-white/5 dark:text-slate-200">
            <input
              className="hidden"
              multiple
              onChange={(event) => ingestFiles(event.target.files)}
              type="file"
            />
            <span className="flex items-center gap-2">
              <Paperclip className="h-4 w-4" />
              Attach file
            </span>
          </label>
          <button
            className="rounded-2xl border border-black/5 bg-black/[0.03] px-3 py-2 text-sm text-slate-500 dark:border-white/10 dark:bg-white/5 dark:text-slate-300"
            type="button"
          >
            <span className="flex items-center gap-2">
              <Mic className="h-4 w-4" />
              Voice soon
            </span>
          </button>
        </div>

        {isGenerating ? (
          <button
            className="rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm font-semibold text-rose-200"
            onClick={onStop}
            type="button"
          >
            <span className="flex items-center gap-2">
              <Square className="h-4 w-4" />
              Stop generation
            </span>
          </button>
        ) : (
          <motion.button
            className="rounded-2xl bg-gradient-to-r from-teal-500 to-cyan-400 px-4 py-3 text-sm font-semibold text-slate-950 shadow-glow"
            onClick={onSend}
            type="button"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <span className="flex items-center gap-2">
              <SendHorizonal className="h-4 w-4" />
              Send to medical LLM
            </span>
          </motion.button>
        )}
      </div>
    </section>
  );
}
