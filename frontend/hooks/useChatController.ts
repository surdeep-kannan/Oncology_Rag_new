"use client";

import { useRef, useState } from "react";
import { optimizeMedicalPrompt } from "@/features/prompt-optimizer/optimizeMedicalPrompt";
import { streamChatResponse } from "@/services/chatService";
import { useActiveConversation, useChatStore } from "@/store/chatStore";
import type { FileAttachment } from "@/types/chat";

function makeId(prefix: string) {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

export function useChatController() {
  const conversation = useActiveConversation();
  const {
    addMessage,
    attachOptimization,
    replaceMessage,
    settings,
    isGenerating,
    setIsGenerating
  } = useChatStore();
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<FileAttachment[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = async (override?: string) => {
    if (!conversation) {
      return;
    }

    const content = (override ?? draft).trim();
    if (!content || isGenerating) {
      return;
    }

    const optimization = settings.optimizePrompts
      ? optimizeMedicalPrompt(content)
      : {
          original: content,
          optimized: content,
          detectedMedicalTerms: [],
          confidence: 1,
          corrections: [],
          warnings: [],
          unsafe: false,
          emergency: false
        };

    const userMessageId = makeId("msg");
    const assistantMessageId = makeId("msg");
    const timestamp = new Date().toISOString();

    addMessage(conversation.id, {
      id: userMessageId,
      role: "user",
      content,
      optimizedPrompt: optimization.optimized,
      optimization,
      timestamp,
      attachments,
      status: "complete"
    });
    attachOptimization(conversation.id, userMessageId, optimization);

    addMessage(conversation.id, {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      timestamp: new Date().toISOString(),
      status: "streaming"
    });

    setDraft("");
    setAttachments([]);
    setIsGenerating(true);
    abortRef.current = new AbortController();

    try {
      await streamChatResponse(
        {
          conversationId: conversation.id,
          message: content,
          optimizedPrompt: optimization.optimized,
          optimization
        },
        {
          signal: abortRef.current.signal,
          onChunk: (chunk) => {
            const liveConversation = useChatStore
              .getState()
              .conversations.find((item) => item.id === conversation.id);
            const currentContent =
              liveConversation?.messages.find((item) => item.id === assistantMessageId)?.content ?? "";
            replaceMessage(conversation.id, assistantMessageId, {
              content: `${currentContent}${chunk}`,
              status: "streaming"
            });
          }
        }
      );

      replaceMessage(conversation.id, assistantMessageId, {
        status: "complete"
      });
    } catch (error) {
      replaceMessage(conversation.id, assistantMessageId, {
        status: "error",
        error: error instanceof Error ? error.message : "Unknown chat error",
        content:
          "I was unable to complete the medical response stream. Please retry, verify backend availability, or disable streaming for troubleshooting."
      });
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
    setIsGenerating(false);
  };

  return {
    conversation,
    draft,
    setDraft,
    attachments,
    setAttachments,
    sendMessage,
    stopGeneration,
    isGenerating,
    settings
  };
}
