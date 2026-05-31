export type MessageRole = "user" | "assistant" | "system";

export interface PromptOptimizationResult {
  original: string;
  optimized: string;
  detectedMedicalTerms: string[];
  confidence: number;
  corrections: string[];
  warnings: string[];
  unsafe: boolean;
  emergency: boolean;
}

export interface FileAttachment {
  id: string;
  name: string;
  size: number;
  type: string;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  optimizedPrompt?: string;
  optimization?: PromptOptimizationResult;
  timestamp: string;
  status?: "pending" | "streaming" | "complete" | "error";
  error?: string;
  attachments?: FileAttachment[];
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  category: "general" | "diagnostics" | "treatment" | "urgent";
  messages: ChatMessage[];
}

export interface ChatSettings {
  optimizePrompts: boolean;
  streamResponses: boolean;
  showOptimizationPanel: boolean;
  theme: "light" | "dark";
}
