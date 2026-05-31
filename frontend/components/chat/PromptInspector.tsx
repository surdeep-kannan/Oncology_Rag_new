import { Sparkles } from "lucide-react";
import type { PromptOptimizationResult } from "@/types/chat";

export function PromptInspector({
  optimization,
  enabled
}: {
  optimization?: PromptOptimizationResult;
  enabled: boolean;
}) {
  return (
    <section className="rounded-[28px] border border-white/10 bg-white/70 p-5 backdrop-blur-xl dark:bg-slate-950/55">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-teal-500">Prompt Optimization</p>
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Clinical prompt enhancer</h2>
        </div>
        <div className="rounded-2xl border border-black/5 bg-black/[0.03] px-3 py-2 text-xs text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-slate-300">
          {enabled ? "Enabled" : "Disabled"}
        </div>
      </div>

      {optimization ? (
        <div className="space-y-4 text-sm">
          <div className="rounded-2xl border border-black/5 bg-black/[0.03] p-4 dark:border-white/10 dark:bg-white/5">
            <p className="mb-2 text-xs uppercase tracking-[0.2em] text-slate-500">Original prompt</p>
            <p className="text-slate-800 dark:text-slate-100">{optimization.original}</p>
          </div>
          <div className="rounded-2xl border border-teal-500/20 bg-teal-500/10 p-4">
            <p className="mb-2 text-xs uppercase tracking-[0.2em] text-teal-600 dark:text-teal-300">
              Optimized prompt
            </p>
            <p className="text-slate-900 dark:text-teal-50">{optimization.optimized}</p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-2xl border border-black/5 bg-black/[0.03] p-4 dark:border-white/10 dark:bg-white/5">
              <p className="mb-2 text-xs uppercase tracking-[0.2em] text-slate-500">Detected terms</p>
              <p className="text-slate-700 dark:text-slate-200">
                {optimization.detectedMedicalTerms.join(", ") || "No strong term matches yet"}
              </p>
            </div>
            <div className="rounded-2xl border border-black/5 bg-black/[0.03] p-4 dark:border-white/10 dark:bg-white/5">
              <p className="mb-2 text-xs uppercase tracking-[0.2em] text-slate-500">Confidence</p>
              <p className="text-slate-700 dark:text-slate-200">{Math.round(optimization.confidence * 100)}%</p>
            </div>
          </div>
          {optimization.warnings.length > 0 && (
            <div className="rounded-2xl border border-amber-500/20 bg-amber-500/10 p-4 text-amber-100">
              <p className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-amber-200">
                <Sparkles className="h-3.5 w-3.5" />
                Warnings
              </p>
              <ul className="space-y-1 text-sm">
                {optimization.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ) : (
        <div className="rounded-3xl border border-dashed border-black/10 bg-black/[0.02] p-8 text-center text-sm text-slate-500 dark:border-white/10 dark:bg-white/[0.03] dark:text-slate-400">
          Submit a question to see the original prompt, optimized clinical expansion, and safety signals.
        </div>
      )}
    </section>
  );
}
