import type { PromptOptimizationResult } from "@/types/chat";

export interface BackendChatPayload {
  message: string;
  optimizedPrompt: string;
  conversationId: string;
  optimization?: PromptOptimizationResult;
}
