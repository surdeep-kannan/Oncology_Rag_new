"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ChatComposer } from "@/components/chat/ChatComposer";
import { ChatMessageCard } from "@/components/chat/ChatMessageCard";
import { PromptInspector } from "@/components/chat/PromptInspector";
import { Sidebar } from "@/components/layout/Sidebar";
import { SettingsModal } from "@/components/layout/SettingsModal";
import { TopBar } from "@/components/layout/TopBar";
import { SafetyBanner } from "@/components/safety/SafetyBanner";
import { useChatController } from "@/hooks/useChatController";
import { useActiveConversation } from "@/store/chatStore";

export function ChatShell() {
  const {
    conversation,
    draft,
    setDraft,
    attachments,
    setAttachments,
    sendMessage,
    stopGeneration,
    isGenerating,
    settings
  } = useChatController();
  const latestOptimization = useMemo(
    () =>
      conversation?.messages
        .slice()
        .reverse()
        .find((message) => message.role === "user" && message.optimization)?.optimization,
    [conversation]
  );
  const scrollRef = useRef<HTMLDivElement>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const activeConversation = useActiveConversation();

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth"
    });
  }, [conversation?.messages]);

  return (
    <>
      <div className="grid min-h-screen grid-cols-1 gap-4 p-4 lg:grid-cols-[320px,1fr]">
        <div className="h-[calc(100vh-2rem)]">
          <Sidebar />
        </div>

        <div className="grid h-[calc(100vh-2rem)] grid-rows-[auto,1fr] gap-4">
          <TopBar onOpenSettings={() => setSettingsOpen(true)} />

          <div className="grid min-h-0 grid-cols-1 gap-4 xl:grid-cols-[1fr,360px]">
            <main className="grid min-h-0 grid-rows-[auto,1fr,auto] gap-4">
              <SafetyBanner optimization={latestOptimization} />

              <section
                ref={scrollRef}
                className="min-h-0 space-y-4 overflow-y-auto rounded-[32px] border border-white/10 bg-white/55 p-4 backdrop-blur-xl dark:bg-slate-950/45"
              >
                {activeConversation?.messages.length ? (
                  activeConversation.messages.map((message, index) => (
                    <ChatMessageCard
                      key={message.id}
                      message={message}
                      onRegenerate={() => {
                        const previousUserMessage = activeConversation.messages
                          .slice(0, index)
                          .reverse()
                          .find((entry) => entry.role === "user");
                        if (previousUserMessage) {
                          void sendMessage(previousUserMessage.content);
                        }
                      }}
                      onRetry={(text) => {
                        const previousUserMessage = activeConversation.messages
                          .slice(0, index)
                          .reverse()
                          .find((entry) => entry.role === "user");
                        void sendMessage(previousUserMessage?.content ?? text);
                      }}
                    />
                  ))
                ) : (
                  <motion.div
                    animate={{ opacity: 1, y: 0 }}
                    className="rounded-[28px] border border-dashed border-black/10 bg-black/[0.02] p-10 text-center dark:border-white/10 dark:bg-white/[0.03]"
                    initial={{ opacity: 0, y: 10 }}
                  >
                    <p className="text-xs uppercase tracking-[0.32em] text-teal-500">Ready</p>
                    <h2 className="mt-2 text-2xl font-semibold text-slate-900 dark:text-white">
                      Premium medical AI workspace
                    </h2>
                    <p className="mx-auto mt-3 max-w-xl text-sm text-slate-500">
                      Stream guideline-grounded answers, inspect optimized prompts, and keep long-running medical
                      consultations organized with search-ready history.
                    </p>
                  </motion.div>
                )}
              </section>

              <ChatComposer
                attachments={attachments}
                isGenerating={isGenerating}
                onAttachments={setAttachments}
                onChange={setDraft}
                onSend={() => void sendMessage()}
                onStop={stopGeneration}
                value={draft}
              />
            </main>

            {settings.showOptimizationPanel && (
              <aside className="space-y-4">
                <PromptInspector enabled={settings.optimizePrompts} optimization={latestOptimization} />
                <div className="rounded-[28px] border border-white/10 bg-white/70 p-5 backdrop-blur-xl dark:bg-slate-950/55">
                  <p className="text-xs uppercase tracking-[0.28em] text-coral-500">Hallucination Guard</p>
                  <h2 className="mt-1 text-lg font-semibold text-slate-900 dark:text-white">Clinical reliability</h2>
                  <ul className="mt-4 space-y-3 text-sm text-slate-600 dark:text-slate-300">
                    <li>Flag unsupported advice when retrieved context is absent or low-confidence.</li>
                    <li>Escalate urgent symptoms instead of offering false reassurance.</li>
                    <li>Encourage clinician verification for treatment plans and medication changes.</li>
                  </ul>
                </div>
              </aside>
            )}
          </div>
        </div>
      </div>

      <SettingsModal onClose={() => setSettingsOpen(false)} open={settingsOpen} />
    </>
  );
}
