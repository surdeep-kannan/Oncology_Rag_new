import type { PromptOptimizationResult } from "@/types/chat";

export interface ChatRequest {
  conversationId: string;
  message: string;
  optimizedPrompt?: string;
  optimization?: PromptOptimizationResult;
}

export async function streamChatResponse(
  payload: ChatRequest,
  {
    onChunk,
    signal
  }: {
    onChunk: (chunk: string) => void;
    signal?: AbortSignal;
  }
) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload),
    signal
  });

  if (!response.ok || !response.body) {
    const error = await response.text();
    throw new Error(error || "Unable to stream medical response.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    onChunk(decoder.decode(value, { stream: true }));
  }
}
