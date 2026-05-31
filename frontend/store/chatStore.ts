"use client";

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { STORAGE_KEY } from "@/lib/constants";
import type {
  ChatMessage,
  ChatSettings,
  Conversation,
  PromptOptimizationResult
} from "@/types/chat";

function makeId(prefix: string) {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function buildStarterConversation(): Conversation {
  const timestamp = new Date().toISOString();
  return {
    id: makeId("conv"),
    title: "New medical consultation",
    createdAt: timestamp,
    updatedAt: timestamp,
    category: "general",
    messages: [
      {
        id: makeId("msg"),
        role: "assistant",
        content:
          "Ask a symptom, diagnosis, treatment, or guideline question. I’ll optimize the prompt before it reaches your medical LLM and surface safety warnings where needed.",
        timestamp,
        status: "complete"
      }
    ]
  };
}

interface ChatState {
  conversations: Conversation[];
  activeConversationId: string;
  searchTerm: string;
  isGenerating: boolean;
  settings: ChatSettings;
  setSearchTerm: (value: string) => void;
  setIsGenerating: (value: boolean) => void;
  updateSettings: (settings: Partial<ChatSettings>) => void;
  createConversation: () => string;
  setActiveConversation: (id: string) => void;
  renameConversation: (id: string, title: string) => void;
  deleteConversation: (id: string) => void;
  addMessage: (conversationId: string, message: ChatMessage) => void;
  replaceMessage: (conversationId: string, messageId: string, patch: Partial<ChatMessage>) => void;
  attachOptimization: (
    conversationId: string,
    messageId: string,
    optimization: PromptOptimizationResult
  ) => void;
}

const initialConversation = buildStarterConversation();

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      conversations: [initialConversation],
      activeConversationId: initialConversation.id,
      searchTerm: "",
      isGenerating: false,
      settings: {
        optimizePrompts: true,
        streamResponses: true,
        showOptimizationPanel: true,
        theme: "dark"
      },
      setSearchTerm: (searchTerm) => set({ searchTerm }),
      setIsGenerating: (isGenerating) => set({ isGenerating }),
      updateSettings: (settings) =>
        set((state) => ({
          settings: {
            ...state.settings,
            ...settings
          }
        })),
      createConversation: () => {
        const conversation = buildStarterConversation();
        set((state) => ({
          conversations: [conversation, ...state.conversations],
          activeConversationId: conversation.id
        }));
        return conversation.id;
      },
      setActiveConversation: (activeConversationId) => set({ activeConversationId }),
      renameConversation: (id, title) =>
        set((state) => ({
          conversations: state.conversations.map((conversation) =>
            conversation.id === id ? { ...conversation, title, updatedAt: new Date().toISOString() } : conversation
          )
        })),
      deleteConversation: (id) =>
        set((state) => {
          const conversations = state.conversations.filter((conversation) => conversation.id !== id);
          const nextConversation = conversations[0] ?? buildStarterConversation();
          return {
            conversations: conversations.length > 0 ? conversations : [nextConversation],
            activeConversationId:
              state.activeConversationId === id ? nextConversation.id : state.activeConversationId
          };
        }),
      addMessage: (conversationId, message) =>
        set((state) => ({
          conversations: state.conversations.map((conversation) =>
            conversation.id === conversationId
              ? {
                  ...conversation,
                  title:
                    conversation.messages.length <= 1 && message.role === "user"
                      ? message.content.slice(0, 52)
                      : conversation.title,
                  updatedAt: message.timestamp,
                  category: message.optimization?.emergency ? "urgent" : conversation.category,
                  messages: [...conversation.messages, message]
                }
              : conversation
          )
        })),
      replaceMessage: (conversationId, messageId, patch) =>
        set((state) => ({
          conversations: state.conversations.map((conversation) =>
            conversation.id === conversationId
              ? {
                  ...conversation,
                  updatedAt: new Date().toISOString(),
                  messages: conversation.messages.map((message) =>
                    message.id === messageId ? { ...message, ...patch } : message
                  )
                }
              : conversation
          )
        })),
      attachOptimization: (conversationId, messageId, optimization) =>
        get().replaceMessage(conversationId, messageId, {
          optimizedPrompt: optimization.optimized,
          optimization
        })
    }),
    {
      name: STORAGE_KEY,
      storage: createJSONStorage(() => localStorage)
    }
  )
);

export function useActiveConversation() {
  return useChatStore((state) =>
    state.conversations.find((conversation) => conversation.id === state.activeConversationId)
  );
}
