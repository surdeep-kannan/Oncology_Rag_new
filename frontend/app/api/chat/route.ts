import { NextRequest } from "next/server";
import { optimizeMedicalPrompt } from "@/features/prompt-optimizer/optimizeMedicalPrompt";

export const runtime = "nodejs";

async function proxyBackend(payload: {
  originalPrompt: string;
  optimizedPrompt: string;
  conversationHistory: any[];
  attachments: any[];
  sessionId: string;
}, signal: AbortSignal) {
  const backendUrl = process.env.MEDICAL_BACKEND_URL || "http://127.0.0.1:8000";

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60000);

  signal.addEventListener("abort", () => {
    controller.abort();
    clearTimeout(timeoutId);
  });

  try {
    const response = await fetch(`${backendUrl.replace(/\/$/, "")}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok || !response.body) {
      throw new Error(`Backend request failed with status ${response.status}.`);
    }

    return response.body;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as {
      message?: string;
      optimizedPrompt?: string;
      conversationId?: string;
      conversationHistory?: any[];
      attachments?: any[];
    };

    if (!body.message || !body.conversationId) {
      return new Response("Missing message or conversationId.", { status: 400 });
    }

    const optimization = optimizeMedicalPrompt(body.message);
    const optimizedPrompt = body.optimizedPrompt?.trim() || optimization.optimized;

    const payload = {
      originalPrompt: body.message,
      optimizedPrompt: optimizedPrompt,
      conversationHistory: body.conversationHistory || [],
      attachments: body.attachments || [],
      sessionId: body.conversationId,
    };

    const stream = await proxyBackend(payload, request.signal);

    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
      },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return new Response("Request aborted", { status: 499 });
    }
    return new Response(error instanceof Error ? error.message : "Unexpected API error", {
      status: 500,
    });
  }
}
